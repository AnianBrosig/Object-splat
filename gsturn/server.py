"""FastAPI server: job management, stage subprocesses, SSE event relay, artifact files."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from .config import JOBS_DIR, ROOT, STAGES, VIDEOS_DIR, Config
from . import batch as batchmod
from .events import Events, read_events, summarize
from .pipeline import Job, new_job, pid_alive, stage_range
from .util import find_tool

WEB = Path(__file__).parent / "web"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}

app = FastAPI(title="turntable2splat")


class FreshStatic(StaticFiles):
    """The UI files change with every update: make the browser always revalidate instead of using a stale copy."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/static", FreshStatic(directory=WEB), name="static")

PROCS: dict[str, subprocess.Popen] = {}
LOCK = threading.Lock()
_doctor_cache: dict[str, Any] = {}


# --------------------------------------------------------------------------- helpers
def get_job(job_id: str) -> Job:
    if "/" in job_id or job_id.startswith("."):
        raise HTTPException(400, "bad job id")
    d = JOBS_DIR / job_id
    if not (d / "config.json").exists():
        raise HTTPException(404, f"no job {job_id}")
    return Job(d)


def is_running(job_id: str) -> bool:
    p = PROCS.get(job_id)
    if p is not None and p.poll() is None:
        return True
    return pid_alive(Job(JOBS_DIR / job_id)) is not None


def spawn(job: Job, from_stage: Optional[str], to_stage: Optional[str]) -> None:
    with LOCK:
        if is_running(job.id):
            raise HTTPException(409, "a run is already in progress for this job")
        stage_range(from_stage, to_stage)  # validate
        cmd = [sys.executable, "-m", "gsturn", "run", "--job", str(job.dir), "--quiet"]
        if from_stage:
            cmd += ["--from-stage", from_stage]
        if to_stage:
            cmd += ["--to-stage", to_stage]
        log = open(job.dir / "process.log", "ab")
        log.write(f"\n=== {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n".encode())
        PROCS[job.id] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()


def job_state(job: Job) -> dict:
    ev_path = job.dir / "events.jsonl"
    offset = ev_path.stat().st_size if ev_path.exists() else 0
    events = read_events(job.dir)
    stages = summarize(events)
    running = is_running(job.id)
    if not running:
        # a run that died without reporting (crash/kill): mark its running stage as error
        for s in stages.values():
            if s["status"] == "running":
                s["status"] = "error"
                s["msg"] = (s["msg"] + " - process ended unexpectedly").strip(" -")
    cfg = job.load_config()
    return {"id": job.id, "config": cfg.to_dict(), "stages": stages, "order": STAGES, "running": running,
            "events_offset": offset,
            "recent_events": events[-300:],
            "video": Path(cfg.video).name if cfg.video else "", "created": job.dir.stat().st_ctime}


# --------------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB / "index.html").read_text()


@app.get("/api/doctor")
def doctor(refresh: bool = False) -> dict:
    from .doctor import report
    if refresh or "r" not in _doctor_cache:
        _doctor_cache["r"] = report()
    return _doctor_cache["r"]


@app.get("/api/defaults")
def defaults() -> dict:
    return Config().to_dict()


@app.get("/api/stages")
def stages() -> list[str]:
    return STAGES


@app.get("/api/videos")
def videos() -> list[dict]:
    VIDEOS_DIR.mkdir(exist_ok=True)
    return [{"name": p.name, "path": str(p), "mb": round(p.stat().st_size / 1e6, 1)}
            for p in sorted(VIDEOS_DIR.iterdir())
            if p.suffix.lower() in VIDEO_EXT and not p.name.lower().startswith("calibration")]


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    if not JOBS_DIR.exists():
        return []
    out = []
    for d in sorted(JOBS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (d / "config.json").exists():
            st = job_state(Job(d))
            out.append({"id": st["id"], "video": st["video"], "running": st["running"],
                        "status": {k: v["status"] for k, v in st["stages"].items()}})
    return out


async def save_upload(file: UploadFile) -> Path:
    name = Path(file.filename).name
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, f"not a video file: {name}")
    VIDEOS_DIR.mkdir(exist_ok=True)
    dst = VIDEOS_DIR / name
    if dst.exists():
        dst = VIDEOS_DIR / f"{Path(name).stem}-{int(time.time())}{Path(name).suffix}"
    with open(dst, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return dst


def resolve_video(path: str) -> Path:
    video = Path(path)
    if not video.is_absolute():
        video = ROOT / video
    if not video.is_file() or video.suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, f"video not found: {path}")
    return video


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Store a video in videos/ before a job exists, so the boxes can be drawn on it first."""
    dst = await save_upload(file)
    return {"name": dst.name, "path": str(dst)}


@app.get("/api/preview-frame")
def preview_frame(path: str, t: float = 0.0, height: int = 720) -> Response:
    """One frame of the input video (rotation applied) for drawing the boxes before any processing ran.
    Boxes are relative, so the preview size does not matter."""
    video = resolve_video(path)
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise HTTPException(500, "ffmpeg not found (brew install ffmpeg)")
    height = max(120, min(int(height), 2160))
    cmd = [ffmpeg, "-v", "error", "-ss", str(max(0.0, t)), "-i", str(video), "-frames:v", "1",
           "-vf", f"scale=-2:'min({height},ih)'", "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    if r.returncode != 0 or not r.stdout:
        raise HTTPException(422, "could not decode a frame: " + r.stderr.decode(errors="replace")[-300:])
    return Response(r.stdout, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/jobs")
async def create_job(file: Optional[UploadFile] = File(None), path: str = Form(""),
                     config: str = Form("{}"), to: str = Form("")) -> dict:
    """Create a job and start it. Boxes may already be in `config` (object_box etc.); then `to` can run past
    the frames stage straight away. Without `to` only downscale + frames run and the boxes are drawn afterwards."""
    cfg = Config()
    try:
        cfg.update(json.loads(config or "{}"))
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))
    if to and to not in STAGES:
        raise HTTPException(400, f"unknown stage {to}")
    if to and STAGES.index(to) >= STAGES.index("segment") and not cfg.object_box:
        raise HTTPException(400, "draw the object box before running past the frames stage")
    if file is not None and file.filename:
        video = await save_upload(file)
    elif path:
        video = resolve_video(path)
    else:
        raise HTTPException(400, "upload a file or give a path")
    job = new_job(str(video), cfg)
    spawn(job, "downscale", to or "frames")
    return {"id": job.id}


# --------------------------------------------------------------------------- bulk queue
# One job per queued video, run one after another by a background thread (compute is the bottleneck, so no
# parallel runs). The jobs are created when the queue starts and show up in the job list straight away.
BATCH: dict[str, Any] = {"items": [], "worker": None, "stop": False}
BLOCK = threading.Lock()


def _batch_worker() -> None:
    while True:
        with BLOCK:
            item = next((i for i in BATCH["items"] if i.state == "waiting"), None)
            if item is None or BATCH["stop"]:
                BATCH["worker"] = None
                return
            item.state = "running"
        t0 = time.time()
        try:
            spawn(Job(JOBS_DIR / item.job), item.from_stage, item.to_stage)
            rc = PROCS[item.job].wait()
            item.state = "done" if rc == 0 else "error"
            if rc != 0:
                item.msg = batchmod.last_error(item.job) or f"exit code {rc}"
        except Exception as e:  # noqa: BLE001
            item.state, item.msg = "error", getattr(e, "detail", None) or str(e)
        item.seconds = time.time() - t0
        if item.state == "error" and item.msg.endswith("cancelled"):
            item.state = "cancelled"


@app.get("/api/batch")
def batch_state() -> dict:
    with BLOCK:
        return {"items": [i.to_dict() for i in BATCH["items"]], "running": BATCH["worker"] is not None}


@app.post("/api/batch")
async def batch_start(request: Request) -> dict:
    """Body: {"items": [{"path": video, "config": {object_box, hand_boxes, bg_boxes}}], "config": {shared params},
    "to": stage}. Items without an object box stop after frame selection."""
    body = await request.json()
    to = body.get("to") or "export"
    try:
        entries = body.get("items") or []
        if not entries:
            raise ValueError("the queue is empty")
        vids = [resolve_video(e["path"]) for e in entries]
        boxes = {str(v): {k: e.get("config", {})[k] for k in batchmod.BOX_KEYS if k in e.get("config", {})}
                 for v, e in zip(vids, entries)}
        items = batchmod.create_jobs(vids, body.get("config", {}), boxes, None, to)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))
    with BLOCK:
        BATCH["items"] = [i for i in BATCH["items"] if i.state in ("waiting", "running")] + items
        BATCH["stop"] = False
        if BATCH["worker"] is None:
            BATCH["worker"] = threading.Thread(target=_batch_worker, daemon=True)
            BATCH["worker"].start()
    return {"ok": True, "jobs": [i.job for i in items]}


@app.post("/api/batch/cancel")
def batch_cancel() -> dict:
    """Drop everything still waiting and cancel the video that is running."""
    with BLOCK:
        BATCH["stop"] = True
        running = [i for i in BATCH["items"] if i.state == "running"]
        for i in BATCH["items"]:
            if i.state == "waiting":
                i.state, i.msg = "cancelled", "queue cancelled"
    for i in running:
        cancel(i.job)
    return {"ok": True}


@app.get("/api/jobs/{job_id}")
def get_job_state(job_id: str) -> dict:
    return job_state(get_job(job_id))


@app.post("/api/jobs/{job_id}/run")
async def run_job(job_id: str, request: Request) -> dict:
    job = get_job(job_id)
    body = await request.json() if (await request.body()) else {}
    if is_running(job.id):
        raise HTTPException(409, "a run is already in progress for this job")
    cfg = job.load_config()
    try:
        cfg.update(body.get("config", {}))
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))
    cfg.save(job.config_path)
    frm, to = body.get("from") or None, body.get("to") or None
    for s in (frm, to):
        if s and s not in STAGES:
            raise HTTPException(400, f"unknown stage {s}")
    spawn(job, frm, to)
    return {"ok": True, "from": frm, "to": to}


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    job = get_job(job_id)
    p = PROCS.get(job.id)
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(8)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    elif (pid := pid_alive(job)) is not None:  # run started from the CLI
        os.kill(pid, signal.SIGTERM)
        for _ in range(40):
            if pid_alive(job) is None:
                break
            time.sleep(0.2)
        else:
            os.kill(pid, signal.SIGKILL)
        (job.dir / "run.pid").unlink(missing_ok=True)
    else:
        return {"ok": False, "msg": "nothing running"}
    # make sure the stream records the cancel even if the process was hard-killed
    st = summarize(read_events(job.dir))
    ev = Events(job.dir, echo=False)
    for name, s in st.items():
        if s["status"] == "running":
            ev.stage_status(name, "error", "cancelled")
    ev.close()
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    job = get_job(job_id)
    if is_running(job.id):
        raise HTTPException(409, "cancel the run first")
    shutil.rmtree(job.dir)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
async def events(job_id: str, request: Request, offset: int = 0):
    job = get_job(job_id)
    path = job.dir / "events.jsonl"
    last = request.headers.get("last-event-id")
    pos = int(last) if last and last.isdigit() else offset

    async def gen():
        nonlocal pos
        idle = 0
        yield "retry: 1500\n\n"
        while True:
            if await request.is_disconnected():
                return
            sent = False
            if path.exists():
                with open(path, "rb") as f:
                    f.seek(pos)
                    while True:
                        line = f.readline()
                        if not line or not line.endswith(b"\n"):
                            break
                        pos += len(line)
                        yield f"id: {pos}\ndata: {line.decode().strip()}\n\n"
                        sent = True
            if not sent:
                idle += 1
                if idle % 30 == 0:
                    yield ": keepalive\n\n"
            else:
                idle = 0
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/files/{path:path}")
def files(job_id: str, path: str):
    job = get_job(job_id)
    p = (job.dir / path).resolve()
    if not p.is_relative_to(job.dir.resolve()) or not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(p)


SPLAT_FILES = [("Original", "outputs/splat.ply"), ("De-lit (neutral light)", "outputs/splat_delit.ply"),
               ("Trainer output", "work/train/splat.ply")]


@app.get("/api/jobs/{job_id}/splats")
def splats(job_id: str) -> list[dict]:
    """Splat files of a job that the browser viewer can show, with a suggested camera."""
    from .mesh import view_info
    job = get_job(job_id)
    out, seen = [], set()
    for label, rel in SPLAT_FILES:
        p = job.dir / rel
        if not p.is_file():
            continue
        key = p.stat().st_size                                  # outputs/ and work/ hold the same file
        if key in seen:
            continue
        seen.add(key)
        try:
            info = view_info(p)
        except Exception:  # noqa: BLE001
            info = {"centre": [0, 0, 0], "radius": 1.0, "up": [0, -1, 0]}
        out.append({"label": label, "path": rel, "mb": round(p.stat().st_size / 1e6, 1), "view": info})
    return out


@app.get("/api/jobs/{job_id}/list/{path:path}")
def list_dir(job_id: str, path: str) -> list[str]:
    job = get_job(job_id)
    p = (job.dir / path).resolve()
    if not p.is_relative_to(job.dir.resolve()) or not p.is_dir():
        raise HTTPException(404, "not found")
    return sorted(x.name for x in p.iterdir())


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def log(job_id: str, tail: int = 400) -> str:
    job = get_job(job_id)
    out = []
    for name in ("log.txt", "process.log"):
        p = job.dir / name
        if p.exists():
            lines = p.read_text(errors="replace").splitlines()
            out.append(f"===== {name} =====")
            out += lines[-tail:]
    return "\n".join(out)


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, full: bool = False):
    """Zip of the job folder. By default skips the bulky per-frame folders; ?full=1 includes everything."""
    job = get_job(job_id)
    tmp = Path(tempfile.mkdtemp())
    zpath = tmp / f"{job.id}.zip"
    import zipfile
    skip = set() if full else {"images", "masks", "hands", "overlays", "import"}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in job.dir.rglob("*"):
            rel = f.relative_to(job.dir)
            if f.is_file() and not (set(rel.parts[:-1]) & skip) and f.suffix != ".db" and (full or not f.is_symlink()):
                z.write(f, rel)
    return FileResponse(zpath, filename=zpath.name, background=BackgroundTask(shutil.rmtree, tmp))
