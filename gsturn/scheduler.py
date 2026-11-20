"""Cross-process resource scheduler: heavy stages of different jobs do not run on the same resource at once.

Why: two jobs that both run SuperPoint/LightGlue, Cutie, COLMAP or the renderer split the CPU between them, each
takes twice as long, and the memory use doubles. Running them one after the other is faster for both in the end.

Resources
  cpu   all CPU cores (torch on the CPU, COLMAP, the software rasteriser, x264)
  gpu   the GPU (Brush training, torch on MPS or CUDA)
Stages that need different resources do overlap (job A trains on the GPU while job B extracts on the CPU).

Order: the job that started first goes first, on every resource ("finish one before the next"). It is not
pre-emptive: a stage that is running is never interrupted. A waiting job may use a free resource when every job
ahead of it is blocked on something else (backfill), so a resource is not left idle for nothing.

Mechanics: one `fcntl.flock` file per resource (the kernel releases it when the process dies, so a crashed job
cannot block the others) plus a queue directory with one small JSON file per waiting stage. The directory is
shared by every process of the user (web UI, CLI, batch), in ~/.cache/gsturn/locks (GSTURN_LOCK_DIR overrides).
Modes (Config.scheduler or GSTURN_SCHEDULER): resource (default) | job (one job at a time, for the whole run) | off.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

RESOURCES = ("cpu", "gpu")
JOB_LOCK = "job"


def lock_dir() -> Path:
    d = Path(os.environ.get("GSTURN_LOCK_DIR") or Path.home() / ".cache" / "gsturn" / "locks")
    (d / "queue").mkdir(parents=True, exist_ok=True)
    (d / "holders").mkdir(parents=True, exist_ok=True)
    return d


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


class Lease:
    """The resources held by one stage. Release it (or use it as a context manager) when the stage is done."""

    def __init__(self, base: Path, fds: dict[str, int], info: dict):
        self.base, self.fds, self.info = base, fds, info

    def release(self) -> None:
        for r, fd in list(self.fds.items()):
            (self.base / "holders" / f"{r}.json").unlink(missing_ok=True)
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        self.fds = {}

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def holders(base: Optional[Path] = None) -> dict[str, dict]:
    """{resource: info of the stage that currently holds it} (live holders only)."""
    base = base or lock_dir()
    out = {}
    for r in (*RESOURCES, JOB_LOCK):
        h = _read(base / "holders" / f"{r}.json")
        if h and _alive(int(h.get("pid", 0))):
            out[r] = h
    return out


def queue(base: Optional[Path] = None) -> list[dict]:
    """Waiting stages, in the order they will be served."""
    base = base or lock_dir()
    out = []
    for p in sorted((base / "queue").glob("*.json")):
        e = _read(p)
        if not e or not _alive(int(e.get("pid", 0))):
            p.unlink(missing_ok=True)          # stale entry of a dead process
            continue
        e["_file"] = p.name
        out.append(e)
    return out


def acquire(resources: list[str], job: str, stage: str, priority: Optional[float] = None,
            on_wait: Optional[Callable[[str], None]] = None, poll: float = 1.0, notify_every: float = 60.0,
            base: Optional[Path] = None, timeout: Optional[float] = None) -> Lease:
    """Block until every resource in `resources` is free and no earlier job is ready for any of them.

    priority: when the job started (seconds since the epoch); smaller goes first. All resources are taken
    together or none, so two stages can never wait for each other. Returns a Lease."""
    base = base or lock_dir()
    res = sorted(set(resources))
    me = {"pid": os.getpid(), "job": job, "stage": stage, "resources": res,
          "priority": float(priority if priority is not None else time.time()), "since": time.time()}
    if not res:
        return Lease(base, {}, me)
    entry = base / "queue" / f"{me['priority']:020.6f}-{os.getpid()}-{uuid.uuid4().hex[:6]}.json"
    entry.write_text(json.dumps(me))
    t0 = last = time.time()
    told = False
    fds: dict[str, int] = {}
    try:
        while True:
            q = queue(base)
            held = holders(base)
            ahead = []
            for e in q:
                if e["_file"] == entry.name:
                    break
                ahead.append(e)
            blockers = []
            for e in ahead:                                     # earlier jobs that could start right now
                if set(e["resources"]) & set(res) and not any(r in held and held[r].get("pid") != os.getpid()
                                                              for r in e["resources"]):
                    blockers.append(e)
            if not blockers:
                got = {}
                ok = True
                for r in res:
                    fd = os.open(base / f"{r}.lock", os.O_CREAT | os.O_RDWR, 0o644)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        got[r] = fd
                    except OSError:
                        os.close(fd)
                        ok = False
                        break
                if ok:
                    fds = got
                    for r in res:
                        (base / "holders" / f"{r}.json").write_text(json.dumps(me))
                    entry.unlink(missing_ok=True)
                    if told and on_wait:
                        on_wait(f"got {', '.join(res)} after {time.time() - t0:.0f} s")
                    return Lease(base, fds, me)
                for fd in got.values():                         # all or nothing
                    os.close(fd)
            now = time.time()
            if on_wait and (not told or now - last >= notify_every):
                who = blockers[0] if blockers else next((held[r] for r in res if r in held), None)
                desc = f"job {who['job']} ({who['stage']})" if who else "another job"
                on_wait(f"waiting for {', '.join(res)}: {desc} goes first ({now - t0:.0f} s so far)")
                told, last = True, now
            if timeout is not None and now - t0 > timeout:
                raise TimeoutError(f"could not get {res} within {timeout:.0f} s")
            time.sleep(poll)
    finally:
        if not fds:
            entry.unlink(missing_ok=True)


# --------------------------------------------------------------------------- what each stage needs
def mode(cfg=None) -> str:
    m = os.environ.get("GSTURN_SCHEDULER") or (getattr(cfg, "scheduler", "resource") if cfg is not None else "resource")
    return m if m in ("resource", "job", "off") else "resource"


def stage_resources(stage: str, cfg) -> list[str]:
    """Resources a stage keeps busy. GPU work still needs a CPU thread, but not all of them, so it is not
    counted as cpu."""
    from .config import resolve_device
    dev = resolve_device(cfg.device)
    gpu = dev in ("mps", "cuda")
    if stage in ("downscale", "frames", "quality", "sfm", "refine"):
        return ["cpu"]
    if stage == "segment":
        return ["gpu"] if resolve_device(cfg.device, "segment") in ("mps", "cuda") and cfg.seg_backend == "cutie" \
            else ["cpu"]
    if stage == "features":
        if cfg.feature_backend == "sift":
            return ["cpu"]
        return ["cpu", "gpu"] if gpu else ["cpu"]       # keypoints on the GPU, matching on the CPU
    if stage == "train":
        return ["gpu"] if cfg.train_backend in ("brush", "opensplat") else []
    if stage == "render":
        return ["cpu"] if getattr(cfg, "render_enabled", False) else []
    return []                                          # export: a few file copies
