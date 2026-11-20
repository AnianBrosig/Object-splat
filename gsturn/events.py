"""Append-only JSON-lines event stream (events.jsonl) plus a plain-text log."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional


class Events:
    def __init__(self, job_dir: Optional[Path], echo: bool = True):
        self.job_dir = job_dir
        self.echo = echo
        self.stage: str = ""
        self._last_progress = 0.0
        self._last_pct = -1.0
        if job_dir is not None:
            job_dir.mkdir(parents=True, exist_ok=True)
            self._ev = open(job_dir / "events.jsonl", "a", buffering=1)
            self._log = open(job_dir / "log.txt", "a", buffering=1)
        else:
            self._ev = self._log = None

    # -- low level -------------------------------------------------------
    def emit(self, **ev: Any) -> None:
        ev.setdefault("t", round(time.time(), 3))
        if self._ev:
            self._ev.write(json.dumps(ev, default=_json_default) + "\n")

    def _text(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"{stamp} {line}".replace(" [] ", " ")
        if self._log:
            self._log.write(line + "\n")
        if self.echo:
            print(line, file=sys.stderr, flush=True)

    # -- high level ------------------------------------------------------
    def stage_status(self, stage: str, status: str, msg: str = "", stats: Optional[dict] = None) -> None:
        self.stage = stage
        ev: dict[str, Any] = {"type": "stage", "stage": stage, "status": status}
        if status == "running":
            ev["pct"] = 0
            self._last_pct = -1.0
        if msg:
            ev["msg"] = msg
        if stats is not None:
            ev["stats"] = stats
        self.emit(**ev)
        self._text(f"[{stage}] {status.upper()} {msg}")

    def progress(self, done: float, total: float, msg: str = "", force: bool = False) -> None:
        pct = 100.0 * done / total if total else 0.0
        now = time.time()
        # throttle: at most ~4 events/s unless the percentage moved a lot
        if not force and now - self._last_progress < 0.25 and abs(pct - self._last_pct) < 5 and done < total:
            return
        self._last_progress, self._last_pct = now, pct
        self.emit(type="progress", stage=self.stage, pct=round(pct, 1), done=done, total=total, msg=msg)

    def artifact(self, path: Path | str, caption: str = "", kind: str = "image") -> None:
        rel = str(path)
        if self.job_dir is not None:
            try:
                rel = str(Path(path).resolve().relative_to(self.job_dir.resolve()))
            except ValueError:
                pass
        self.emit(type="artifact", stage=self.stage, path=rel, caption=caption, kind=kind)
        self._text(f"[{self.stage}] artifact {rel} {caption}")

    def log(self, msg: str, level: str = "info") -> None:
        self.emit(type="log", stage=self.stage, level=level, msg=msg)
        self._text(f"[{self.stage}] {level}: {msg}")

    def warn(self, msg: str) -> None:
        self.log(msg, "warn")

    def close(self) -> None:
        for f in (self._ev, self._log):
            if f:
                f.close()


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_events(job_dir: Path) -> list[dict]:
    p = job_dir / "events.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def summarize(events: list[dict]) -> dict[str, dict]:
    """Replay events into per-stage status (what the UI shows)."""
    from .config import STAGES
    st: dict[str, dict] = {s: {"status": "pending", "pct": 0, "msg": "", "artifacts": [], "stats": {},
                               "started": None, "ended": None} for s in STAGES}
    for ev in events:
        s = st.get(ev.get("stage", ""))
        if s is None:
            continue
        typ = ev.get("type")
        if typ == "stage":
            status = ev["status"]
            if status == "running":
                s.update(status="running", pct=0, msg=ev.get("msg", ""), artifacts=[], stats={},
                         started=ev.get("t"), ended=None)
            else:
                s["status"] = status
                s["ended"] = ev.get("t")
                if status in ("done", "warn", "skipped"):
                    s["pct"] = 100
                if "msg" in ev:
                    s["msg"] = ev["msg"]
                if "stats" in ev:
                    s["stats"] = ev["stats"]
        elif typ == "progress":
            s["pct"] = ev.get("pct", s["pct"])
            if ev.get("msg"):
                s["msg"] = ev["msg"]
        elif typ == "artifact":
            arts = [a for a in s["artifacts"] if a["path"] != ev["path"]]
            arts.append({"path": ev["path"], "caption": ev.get("caption", ""), "kind": ev.get("kind", "image")})
            s["artifacts"] = arts
    return st
