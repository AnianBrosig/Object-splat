"""Built-in webcam (or any camera / video file): threaded capture, live preview, chessboard calibration.

    python -m gsturn camera                 # preview: r = start/stop recording, s = snapshot, q = quit
    python -m gsturn camera --record        # record right away to videos/webcam0_<time>.mp4 (q / r stops)
    python -m gsturn camera --calibrate --square-mm 22   # hold the printed chessboard in view, tilt it around

macOS asks for camera access the first time; allow it for the app that runs Python (VS Code / Terminal) under
System Settings > Privacy & Security > Camera, then restart that app.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import ROOT

CALIB_DIR = ROOT / "calibration"
PERMISSION_HELP = ("could not open the camera. On macOS allow camera access for the app running Python "
                   "(System Settings > Privacy & Security > Camera > Visual Studio Code / Terminal), then restart it.")


def open_camera(index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(PERMISSION_HELP)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(50):                 # the first reads can fail while the camera starts
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        time.sleep(0.05)
    cap.release()
    raise RuntimeError(PERMISSION_HELP)


class Grabber:
    """Reads frames in a thread and keeps only the newest, so slow processing never builds up latency."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.frame: Optional[np.ndarray] = None
        self.stamp = 0.0
        self.count = 0
        self.ok = True
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                with self.cond:
                    self.cond.notify_all()
                break
            with self.cond:
                self.frame, self.stamp, self.count = f, time.perf_counter(), self.count + 1
                self.cond.notify_all()

    def read(self, last_count: int = -1, timeout: float = 2.0):
        """Newest frame (waits for one newer than last_count). Returns (frame, count, timestamp)."""
        with self.cond:
            self.cond.wait_for(lambda: (self.frame is not None and self.count > last_count) or not self.ok, timeout)
            return self.frame, self.count, self.stamp

    def close(self):
        self.ok = False
        self.thread.join(timeout=1)
        self.cap.release()


# --------------------------------------------------------------------------- intrinsics
def calib_path(width: int, height: int, index: int = 0) -> Path:
    """Default calibration file of a live camera (same folder and JSON format as `gsturn calibrate`)."""
    return CALIB_DIR / f"webcam{index}_{width}x{height}.json"


def intrinsics(calib_file: Path, width: int, height: int) -> tuple[np.ndarray, Optional[tuple], dict]:
    """K for frames of this size (after undistortion, if the calibration has distortion) and the undistort maps."""
    from ..calibrate import has_distortion, load_calibration, scaled_intrinsics, undistort_maps
    calib = load_calibration(calib_file)
    if has_distortion(calib):
        m1, m2, K = undistort_maps(calib, width, height)
        return K, (m1, m2), calib
    fx, fy, cx, cy, _ = scaled_intrinsics(calib, width, height)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), None, calib


def guess_K(width: int, height: int, hfov_deg: float) -> np.ndarray:
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.0]])


# --------------------------------------------------------------------------- live calibration
def _diverse(c: np.ndarray, kept: list[np.ndarray], size) -> bool:
    """New view only if the board moved/tilted noticeably compared to all kept views."""
    diag = np.hypot(*size)
    return all(np.mean(np.linalg.norm(c - k, axis=1)) >= 0.05 * diag for k in kept)


def find_board(gray: np.ndarray, pattern: Optional[tuple[int, int]]):
    """Inner corners (N, 2); with pattern=None the largest known pattern that is found."""
    from ..calibrate import PATTERN_CANDIDATES, detect
    for p in ([pattern] if pattern else sorted(PATTERN_CANDIDATES, key=lambda p: -p[0] * p[1])):
        c = detect(gray, p)
        if c is not None:
            return p, c
    return pattern, None


def collect_views(frames, pattern=None, n_views=25, show=True, log=print):
    """Diverse chessboard views from a frame stream: (pattern, list of corner arrays, image size)."""
    views, size = [], None
    last_try = 0.0
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        c = None
        if not show or time.perf_counter() - last_try > 0.25:   # detection is slow; keep the preview fluid
            last_try = time.perf_counter()
            pattern, c = find_board(gray, pattern)
            if c is not None and _diverse(c, views, size):
                views.append(c)
                log(f"  view {len(views)}/{n_views} (pattern {pattern[0]}x{pattern[1]})")
        if show:
            vis = frame.copy()
            if c is not None:
                cv2.drawChessboardCorners(vis, pattern, c.reshape(-1, 1, 2), True)
            msg = (f"views {len(views)}/{n_views}: move and tilt the board, q = finish" if pattern
                   else "hold the chessboard in view (detecting its size)")
            cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("calibrate", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        if len(views) >= n_views:
            break
    if show:
        cv2.destroyWindow("calibrate")
    return pattern, views, size


def solve(pattern, views, size, square_mm: float, camera: str, source: str, fit_distortion: bool = True) -> dict:
    """Robust calibration (gsturn.calibrate) -> dict in the same JSON format as `gsturn calibrate`."""
    from ..calibrate import CalibrationError, board_points, calibrate_robust
    if len(views) < 8:
        raise CalibrationError(f"only {len(views)} chessboard views; need at least 8")
    w, h = size
    obj = board_points(pattern, square_mm)
    vs = [{"corners": c, "frame": i} for i, c in enumerate(views)]
    res = calibrate_robust(vs, obj, w, h, fix_dist=not fit_distortion, f_guess=float(w))
    K, D = res["K"], res["D"]
    return {"camera": camera, "source": source, "width": w, "height": h, "model": "PINHOLE",
            "fx": round(float(K[0, 0]), 3), "fy": round(float(K[1, 1]), 3), "cx": round(float(K[0, 2]), 3),
            "cy": round(float(K[1, 2]), 3), "dist": [round(float(x), 6) for x in D[:4]], "rms_px": round(res["rms"], 4),
            "views": len(res["views"]), "pattern": list(pattern), "square_mm": square_mm,
            "hfov_deg": round(float(np.degrees(2 * np.arctan(w / 2 / K[0, 0]))), 2),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}


def video_frames(path: str, step: int = 5, width: int = 0):
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            _, f = cap.retrieve()
            if width and f.shape[1] != width:
                f = cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA)
            yield f
        i += 1


def camera_frames(grabber: Grabber):
    n = -1
    while grabber.ok:
        f, n, _ = grabber.read(n)
        if f is None:
            break
        yield f


# --------------------------------------------------------------------------- recording
class VideoRecorder:
    """Constant-frame-rate H.264 recording of a live stream.

    Frames go through a queue to an ffmpeg process (hardware encoder on macOS, libx264 otherwise; OpenCV's
    mp4v writer only if ffmpeg is missing), so encoding never stalls the camera loop. Timestamps decide how
    often each frame is written: when the camera delivers fewer frames than `fps` (it drops to ~15 fps in dim
    light), frames are repeated, so the video always plays at real speed.
    """

    def __init__(self, path: Path, width: int, height: int, fps: float = 30.0, bitrate: str = "",
                 log=print):
        import queue
        import subprocess
        from ..util import find_tool
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size, self.fps, self.log = (width, height), fps, log
        self.t0: Optional[float] = None
        self.slots = 0                       # output frames written so far
        self.received = 0
        self.q: "queue.Queue" = queue.Queue(maxsize=int(fps * 2))
        self.dropped = 0
        ffmpeg = find_tool("ffmpeg")
        self.proc = self.cv_writer = None
        if ffmpeg:
            # ~0.6 bit per pixel per second at 30 fps: plenty for SfM / feature matching
            br = bitrate or f"{max(4, round(width * height * fps * 0.6 / 1e6))}M"
            enc = (["-c:v", "h264_videotoolbox", "-b:v", br, "-maxrate", br, "-allow_sw", "1"]
                   if sys.platform == "darwin" else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16"])
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-", *enc, "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(self.path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            self.encoder = enc[1]
        else:
            self.cv_writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            self.encoder = "opencv mp4v"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, frame: np.ndarray, stamp: Optional[float] = None) -> None:
        stamp = time.perf_counter() if stamp is None else stamp
        if self.t0 is None:
            self.t0 = stamp
        self.received += 1
        due = int((stamp - self.t0) * self.fps) + 1          # output frames that should exist by now
        n = due - self.slots
        if n <= 0:
            return                                            # camera faster than fps: skip this frame
        self.slots = due
        try:
            self.q.put_nowait((frame, min(n, int(self.fps))))
        except Exception:                                     # encoder fell behind: drop rather than stall
            self.dropped += 1

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            frame, n = item
            if frame.shape[1::-1] != self.size:
                frame = cv2.resize(frame, self.size)
            for _ in range(n):
                if self.proc is not None:
                    try:
                        self.proc.stdin.write(frame.tobytes())
                    except (BrokenPipeError, OSError):
                        return
                else:
                    self.cv_writer.write(frame)

    @property
    def seconds(self) -> float:
        return self.slots / self.fps

    def close(self) -> Path:
        self.q.put(None)
        self.thread.join()
        if self.proc is not None:
            self.proc.stdin.close()
            err = self.proc.stderr.read().decode(errors="replace").strip()
            if self.proc.wait() != 0:
                self.log(f"ffmpeg failed: {err[-500:]}")
        else:
            self.cv_writer.release()
        msg = (f"saved {self.path}  {self.seconds:.1f} s at {self.fps:g} fps ({self.encoder}), "
               f"{self.received} camera frames")
        if self.dropped:
            msg += f", {self.dropped} dropped (encoder too slow)"
        self.log(msg)
        return self.path


def default_recording(index: int = 0) -> Path:
    return ROOT / "videos" / f"webcam{index}_{time.strftime('%Y%m%d-%H%M%S')}.mp4"


# --------------------------------------------------------------------------- preview
def preview(index: int = 0, width: int = 1280, height: int = 720, out_dir: Path = Path("captures"),
            record: Optional[Path] = None, seconds: float = 0.0, fps: float = 30.0) -> Optional[Path]:
    """Live view. r starts/stops a recording, s saves a snapshot, q quits.
    record: start recording to this file right away (stops after `seconds` if > 0, else on q / r)."""
    cap = open_camera(index, width, height)
    g = Grabber(cap)
    out_dir = Path(out_dir)
    rec: Optional[VideoRecorder] = None
    saved = None
    n, cam_fps, t_prev = -1, 0.0, time.perf_counter()
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {index}: {W}x{H}  keys: r = record on/off, s = snapshot, q = quit")

    def start(path: Path):
        nonlocal rec
        rec = VideoRecorder(path, W, H, fps)
        print(f"recording {path} ({rec.encoder}) ...")

    def stop():
        nonlocal rec, saved
        saved = rec.close()
        rec = None

    if record is not None:
        start(Path(record))
    try:
        while True:
            f, n, stamp = g.read(n)
            if f is None or not g.ok:
                break
            cam_fps = 0.9 * cam_fps + 0.1 / max(stamp - t_prev, 1e-3)
            t_prev = stamp
            if rec is not None:
                rec.write(f, stamp)
                if seconds and rec.seconds >= seconds:
                    stop()
                    if record is not None:
                        break
            vis = f.copy()
            label = f"{cam_fps:4.1f} fps  {f.shape[1]}x{f.shape[0]}"
            if rec is not None:
                label += f"  REC {rec.seconds:5.1f} s"
                cv2.circle(vis, (vis.shape[1] - 30, 30), 12, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_8)
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if rec is not None else (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("camera", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / f"snap_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                cv2.imwrite(str(p), f)
                print(f"saved {p}")
            if k == ord("r"):
                if rec is None:
                    start(default_recording(index))
                else:
                    stop()
    except KeyboardInterrupt:
        pass
    finally:
        if rec is not None:
            stop()
        g.close()
        cv2.destroyAllWindows()
    return saved


def calibrate_camera(index: int = 0, width: int = 1280, height: int = 720, pattern=None, n_views: int = 25,
                     square_mm: float = 22.0, out: Optional[Path] = None, fit_distortion: bool = True) -> Path:
    cap = open_camera(index, width, height)
    g = Grabber(cap)
    try:
        pattern, views, size = collect_views(camera_frames(g), pattern, n_views)
    finally:
        g.close()
    res = solve(pattern, views, size, square_mm, camera=f"webcam {index}", source="live",
                fit_distortion=fit_distortion)
    out = Path(out) if out else calib_path(res["width"], res["height"], index)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"fx {res['fx']:.1f}px  hfov {res['hfov_deg']:.1f} deg  rms {res['rms_px']:.2f}px  "
          f"dist {res['dist']}  ({res['views']} views) -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--square-mm", type=float, default=22.0)
    a = ap.parse_args()
    calibrate_camera(a.index, square_mm=a.square_mm) if a.calibrate else preview(a.index)
