"""Job folder layout and stage runner."""
from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import JOBS_DIR, STAGES, Config
from .events import Events


class StageError(Exception):
    """Expected failure with a user-facing message (no traceback needed)."""


class StageWarn(Exception):
    """Stage finished but a quality gate tripped. Carries msg + stats; pipeline continues unless gate=True."""

    def __init__(self, msg: str, stats: Optional[dict] = None, stop: bool = False):
        super().__init__(msg)
        self.msg, self.stats, self.stop = msg, stats or {}, stop


@dataclass
class Job:
    dir: Path

    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def config_path(self) -> Path:
        return self.dir / "config.json"

    def work(self, stage: str) -> Path:
        p = self.dir / "work" / stage
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def outputs(self) -> Path:
        p = self.dir / "outputs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def load_config(self) -> Config:
        return Config.load(self.config_path)

    def save_json(self, stage: str, name: str, data) -> Path:
        p = self.work(stage) / name
        p.write_text(json.dumps(data, indent=2, default=str))
        return p

    def load_json(self, stage: str, name: str):
        p = self.dir / "work" / stage / name
        if not p.exists():
            raise StageError(f"missing {stage}/{name}: run stage '{stage}' first")
        return json.loads(p.read_text())


def new_job(video: str, cfg: Optional[Config] = None, name: str = "") -> Job:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(video).stem if video else "job"
    jid = name or f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}-{uuid.uuid4().hex[:4]}"
    job = Job(JOBS_DIR / jid)
    job.dir.mkdir(parents=True)
    cfg = cfg or Config()
    cfg.video = str(Path(video).resolve()) if video else ""
    cfg.save(job.config_path)
    return job


def stage_fn(name: str) -> Callable:
    import importlib
    return importlib.import_module(f".stages.{name}", __package__).run


def stage_range(from_stage: Optional[str], to_stage: Optional[str]) -> list[str]:
    a = STAGES.index(from_stage) if from_stage else 0
    b = STAGES.index(to_stage) if to_stage else len(STAGES) - 1
    if a > b:
        raise ValueError(f"from-stage {from_stage} is after to-stage {to_stage}")
    return STAGES[a:b + 1]


def pid_alive(job: Job) -> Optional[int]:
    """PID of a run currently working on this job (started from the CLI or the server), else None."""
    import os
    p = job.dir / "run.pid"
    try:
        pid = int(p.read_text())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def preload_torch() -> None:
    """Import torch before anything imports pycolmap.

    pycolmap ships its own OpenMP runtime. If it is loaded first, torch's multi-threaded CPU ops later crash the
    process with SIGSEGV (exit 139), e.g. LightGlue in the refine stage of a run started at 'sfm'."""
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001  (torch is optional: SIFT/GrabCut runs work without it)
        pass


def run(job: Job, from_stage: Optional[str] = None, to_stage: Optional[str] = None, echo: bool = True) -> bool:
    import os
    preload_torch()
    other = pid_alive(job)
    if other and other != os.getpid():
        raise RuntimeError(f"job {job.id} is already being processed by pid {other}")
    (job.dir / "run.pid").write_text(str(os.getpid()))
    ev = Events(job.dir, echo=echo)
    cfg = job.load_config()
    ok = True
    from . import scheduler
    sched = scheduler.mode(cfg)
    started = time.time()          # the job that started first is served first on every resource
    if os.environ.get("GSTURN_PRIORITY"):      # an explicit start time (seconds since the epoch) to jump the queue
        started = float(os.environ["GSTURN_PRIORITY"])
    job_lease = None
    try:
        if sched == "job":
            job_lease = scheduler.acquire([scheduler.JOB_LOCK], job.id, "run", started,
                                          on_wait=lambda m: ev.log(m))
        for stage in stage_range(from_stage, to_stage):
            ev.stage_status(stage, "running")
            lease = None
            if sched == "resource":
                need = scheduler.stage_resources(stage, cfg)
                if need:
                    try:
                        lease = scheduler.acquire(need, job.id, stage, started, on_wait=lambda m, st=stage: (
                            ev.log(m), ev.stage_status(st, "running", m)))
                    except KeyboardInterrupt:
                        ev.stage_status(stage, "error", "cancelled")
                        ok = False
                        break
            t0 = time.time()           # waiting for another job does not count as stage time
            try:
                result = stage_fn(stage)(job, cfg, ev) or {}
                msg = result.pop("_msg", "") if isinstance(result, dict) else ""
                result["seconds"] = round(time.time() - t0, 1)
                ev.stage_status(stage, "done", msg, stats=result)
                job.save_json(stage, "stats.json", result)
            except StageWarn as w:
                w.stats["seconds"] = round(time.time() - t0, 1)
                job.save_json(stage, "stats.json", w.stats)
                ev.stage_status(stage, "warn", w.msg, stats=w.stats)
                if w.stop:
                    ev.log(f"quality gate failed in {stage}; stopping before the next stage", "error")
                    ok = False
                    break
            except KeyboardInterrupt:
                ev.stage_status(stage, "error", "cancelled")
                ok = False
                break
            except StageError as e:
                ev.stage_status(stage, "error", str(e))
                ok = False
                break
            except Exception as e:  # noqa: BLE001
                ev.log(traceback.format_exc(), "error")
                ev.stage_status(stage, "error", f"{type(e).__name__}: {e}")
                ok = False
                break
            finally:
                if lease is not None:
                    lease.release()
            # config may have been changed by a stage (e.g. resolved defaults); reload for the next one
            cfg = job.load_config()
    finally:
        if job_lease is not None:
            job_lease.release()
        ev.emit(type="run_end", ok=ok)
        ev.close()
        (job.dir / "run.pid").unlink(missing_ok=True)
    return ok
