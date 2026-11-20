"""Bulk processing: one job per video, run one after another.

Every video needs its own first-frame boxes (the segmentation is seeded from them). Boxes come from, in order:
a per-video entry in a boxes file, the "*" entry of that file, or the boxes given on the command line / in the UI.
A video without an object box only runs up to the frames stage, so its box can be drawn afterwards in the web UI.

boxes file (JSON), keys are video file names (or paths):
    {"*":         {"object_box": [0.2, 0.2, 0.8, 0.8]},
     "IMG_8901.MOV": {"object_box": [0.3, 0.1, 0.7, 0.9], "hand_boxes": [[0, 0.5, 0.3, 1]], "bg_boxes": []}}
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .config import JOBS_DIR, ROOT, Config
from .events import read_events, summarize
from .pipeline import Job, new_job

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
BOX_KEYS = ("object_box", "hand_boxes", "bg_boxes")


def find_videos(paths: Iterable[str]) -> list[Path]:
    """Files as given, directories expanded to their videos (sorted; calibration videos skipped)."""
    out: list[Path] = []
    for s in paths:
        p = Path(s)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        if p.is_dir():
            out += [f for f in sorted(p.iterdir())
                    if f.suffix.lower() in VIDEO_EXT and not f.name.lower().startswith("calibration")]
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"no such video or folder: {s}")
    seen, uniq = set(), []
    for f in out:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(f)
    return uniq


def load_boxes(path: Optional[str]) -> dict[str, dict]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("boxes file must be a JSON object {video name: {object_box, ...}}")
    for name, entry in data.items():
        bad = set(entry) - set(BOX_KEYS)
        if bad:
            raise ValueError(f"{name}: unknown keys {sorted(bad)} (allowed: {list(BOX_KEYS)})")
    return data


def boxes_for(video: Path, boxes: dict[str, dict], default: Optional[dict] = None) -> dict:
    """Box config for one video: its own entry, else "*", else the default (command line / UI)."""
    for key in (str(video), str(video.resolve()), video.name, video.stem):
        if key in boxes:
            return dict(boxes[key])
    if "*" in boxes:
        return dict(boxes["*"])
    return dict(default or {})


def finished_job(video: Path, to_stage: str = "export") -> Optional[str]:
    """Id of an existing job for this video whose `to_stage` already finished, else None."""
    if not JOBS_DIR.exists():
        return None
    want = str(video.resolve())
    for d in sorted(JOBS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        try:
            if Config.load(cfg_path).video != want:
                continue
            st = summarize(read_events(d)).get(to_stage)
        except Exception:  # noqa: BLE001  (unreadable job folder: not a match)
            continue
        if st and st["status"] in ("done", "warn"):
            return d.name
    return None


@dataclass
class Item:
    video: str
    job: str
    from_stage: str
    to_stage: str
    state: str = "waiting"        # waiting | running | done | error | cancelled | skipped
    msg: str = ""
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"video": Path(self.video).name, "job": self.job, "from": self.from_stage, "to": self.to_stage,
                "state": self.state, "msg": self.msg, "seconds": round(self.seconds, 1)}


def create_jobs(videos: list[Path], base: dict, boxes: dict[str, dict], default_boxes: Optional[dict] = None,
                to_stage: str = "export") -> list[Item]:
    """Create one job per video (config validated up front, so a typo fails before anything runs)."""
    from .config import STAGES
    if to_stage not in STAGES:
        raise ValueError(f"unknown stage {to_stage}")
    plans = []
    for v in videos:
        cfg = Config().update(base)
        cfg.update(boxes_for(v, boxes, default_boxes))
        plans.append((v, cfg))
    items = []
    for v, cfg in plans:
        job = new_job(str(v), cfg)
        # without an object box the run stops after frame selection; the box is drawn afterwards in the UI
        needs_box = STAGES.index(to_stage) >= STAGES.index("segment")
        to = "frames" if needs_box and not cfg.object_box else to_stage
        items.append(Item(str(v), job.id, "downscale", to))
    return items


def run_item(item: Item, log_name: str = "process.log") -> int:
    """Run one job in its own process (isolates torch / pycolmap state and memory between videos)."""
    job = Job(JOBS_DIR / item.job)
    cmd = [sys.executable, "-m", "gsturn", "run", "--job", str(job.dir), "--quiet",
           "--from-stage", item.from_stage, "--to-stage", item.to_stage]
    with open(job.dir / log_name, "ab") as log:
        log.write(f"\n=== {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n".encode())
        log.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode


def last_error(job_id: str) -> str:
    """The message of the stage that failed in this job."""
    try:
        for name, s in summarize(read_events(JOBS_DIR / job_id)).items():
            if s["status"] == "error":
                return f"{name}: {s['msg']}"
    except Exception:  # noqa: BLE001
        pass
    return ""


def run_batch(items: list[Item], keep_going: bool = True, echo=print) -> list[Item]:
    """Sequential run. Stops at the first failure unless keep_going."""
    for n, it in enumerate(items, 1):
        if it.state != "waiting":
            continue
        echo(f"[{n}/{len(items)}] {Path(it.video).name}  job {it.job}  {it.from_stage} -> {it.to_stage}")
        it.state = "running"
        t0 = time.time()
        try:
            rc = run_item(it)
        except KeyboardInterrupt:
            it.state, it.msg = "cancelled", "interrupted"
            for rest in items:
                if rest.state == "waiting":
                    rest.state = "cancelled"
            raise
        it.seconds = time.time() - t0
        it.state = "done" if rc == 0 else "error"
        if rc != 0:
            it.msg = last_error(it.job) or f"exit code {rc}"
        echo(f"      {it.state} in {it.seconds:.0f}s" + (f": {it.msg}" if it.msg else ""))
        if rc != 0 and not keep_going:
            for rest in items:
                if rest.state == "waiting":
                    rest.state, rest.msg = "cancelled", "stopped after an error"
            break
    return items


def write_report(items: list[Item]) -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    p = JOBS_DIR / f"batch-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps([i.to_dict() for i in items], indent=2))
    return p
