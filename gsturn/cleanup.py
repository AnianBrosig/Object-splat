"""Free disk space by deleting the large intermediate images/videos under jobs/<id>/work/.

Only image and video files below `work/` are removed. `outputs/` (splat.ply, turntables, preview, report), config,
logs, events and the small data files (json, npz, db, bin, ply, txt) are never touched, and jobs with a live run
are skipped. Re-running a stage regenerates what it needs; `export-model`, `delight` and `track-eval` read
work/frames, work/segment and work/downscale, so re-run those stages first if you still need them.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from .config import JOBS_DIR

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
CLEAN_EXT = IMAGE_EXT | VIDEO_EXT


def clean_job(job_dir: Path, dry_run: bool = False) -> dict:
    """Delete image/video files in job_dir/work. Returns {files, bytes, dirs_removed}."""
    work = Path(job_dir) / "work"
    files = freed = dirs = 0
    if not work.is_dir():
        return {"files": 0, "bytes": 0, "dirs_removed": 0}
    for root, _dirs, names in os.walk(work):
        for n in names:
            p = Path(root) / n
            if p.suffix.lower() not in CLEAN_EXT:
                continue
            try:
                st = p.lstat()                       # a symlink is unlinked, never followed
                size = 0 if p.is_symlink() else st.st_size
                if not dry_run:
                    p.unlink()
            except OSError:
                continue
            files += 1
            freed += size
    if not dry_run:                                  # drop folders that are now empty (bottom-up)
        for root, _dirs, _names in os.walk(work, topdown=False):
            r = Path(root)
            if r != work and not any(r.iterdir()):
                try:
                    r.rmdir()
                    dirs += 1
                except OSError:
                    pass
    return {"files": files, "bytes": freed, "dirs_removed": dirs}


def clean_jobs(jobs_dir: Optional[Path] = None, only: Optional[list[str]] = None, dry_run: bool = False,
               log=print) -> dict:
    """Run clean_job on every job folder (or just the ids in `only`). Jobs with a live run are skipped."""
    from .pipeline import Job, pid_alive
    jobs_dir = Path(jobs_dir or JOBS_DIR)
    total = {"jobs": 0, "skipped": 0, "files": 0, "bytes": 0}
    for d in sorted(p for p in jobs_dir.iterdir() if p.is_dir() and (p / "config.json").exists()):
        if only and d.name not in only:
            continue
        if pid_alive(Job(d)) is not None:
            log(f"skip {d.name}: run in progress")
            total["skipped"] += 1
            continue
        r = clean_job(d, dry_run)
        total["jobs"] += 1
        total["files"] += r["files"]
        total["bytes"] += r["bytes"]
        log(f"{'would free' if dry_run else 'freed'} {r['bytes'] / 1e6:9.1f} MB  {r['files']:6d} files  {d.name}")
    log(f"{'would free' if dry_run else 'freed'} {total['bytes'] / 1e9:.2f} GB in {total['files']} files "
        f"across {total['jobs']} jobs ({total['skipped']} skipped)")
    return total
