"""Live object pose: webcam (or video file) -> pose, mask and axes overlay, optional pose log / video.

Keys: q quit, space pause, m mask, a axes, p points, r relocalise now, s snapshot.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .camera import Grabber, VideoRecorder, calib_path, guess_K, intrinsics, open_camera
from .draw import draw, euler_xyz, hull_mask, quat
from .model import ObjectModel, find_model
from .tracker import PoseTracker, TrackerConfig

DEFAULT_HFOV = 60.0   # FaceTime HD (2019 MacBook Pro) guess; run "gsturn camera --calibrate" for the real value


def scaled_K(K: np.ndarray, src_size, dst_size) -> np.ndarray:
    sx, sy = dst_size[0] / src_size[0], dst_size[1] / src_size[1]
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K2[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K2


def run(model_arg: str, source: str = "0", width: int = 1280, height: int = 720, calib: str = "",
        hfov: float = 0.0, display: bool = True, save_video: str = "", poses: str = "", seconds: float = 0.0,
        realtime: bool = True, cfg: Optional[TrackerConfig] = None, record_raw: str = "", learn: bool = False,
        learn_save: bool = True, log=print) -> dict:
    model = ObjectModel.load(find_model(model_arg))
    log(f"model {model.path} ({len(model.points)} points, {len(model.kf_names)} keyframes, units: {model.units})")
    is_cam = source.isdigit()
    grabber = cap = None
    if is_cam:
        cap = open_camera(int(source), width, height)
        grabber = Grabber(cap)
        frame, _, _ = grabber.read()
    else:
        cap = cv2.VideoCapture(source)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read {source}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    H, W = frame.shape[:2]
    resize = None
    if not is_cam and W > width:                     # e.g. 4K phone videos: work at `width`
        resize = (width, round(H * width / W))
        W, H = resize

    # ---- intrinsics
    undist = None
    cp = Path(calib) if calib else (calib_path(W, H, int(source)) if is_cam else None)
    if cp is not None and (cp.exists() or calib):
        K, undist, c = intrinsics(cp, W, H)
        log(f"intrinsics from {cp} (fx {K[0, 0]:.0f}, rms {c.get('rms_px', 0):.2f}px"
            f"{', undistorting frames' if undist else ''})")
    elif hfov or is_cam:
        K = guess_K(W, H, hfov or DEFAULT_HFOV)
        log(f"intrinsics guessed from a {hfov or DEFAULT_HFOV:.0f} deg horizontal field of view (fx {K[0, 0]:.0f}); "
            f"for metric accuracy run: python -m gsturn camera --calibrate --square-mm <board square size>")
    else:
        K = scaled_K(model.K, model.image_size, (W, H))
        log(f"intrinsics of the capture camera scaled to {W}x{H} (fx {K[0, 0]:.0f})")

    tracker = PoseTracker(model, K, cfg or TrackerConfig())
    log(f"warming up SuperPoint on {tracker.loc.sp.device} ...")
    t0 = time.perf_counter()
    tracker.warmup((H, W))
    log(f"ready ({time.perf_counter() - t0:.1f} s)")
    learner = None
    if learn:
        from .online import OnlineLearner
        learner = OnlineLearner(tracker, log=log)
        log("live learning on: the map adapts to this camera once the object is found"
            + (" (saved to the package on exit)" if learn_save else ""))

    writer = None
    pose_f = open(poses, "w") if poses else None
    ui = {"mask": True, "axes": True, "points": True, "paused": False, "stop": False, "snap": False}
    info: dict = {}
    start = time.perf_counter()
    stats = {"frames": 0, "track": 0, "coast": 0, "lost": 0, "track_ms": []}
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
    latest = {"vis": None, "n": 0}
    cond = threading.Condition()

    def work():
        """Capture -> track -> draw -> log, at the camera rate. Runs off the main thread because the macOS
        window (cv2.imshow) costs ~23 ms per frame and would otherwise cap the tracking rate."""
        nonlocal writer
        n_cam, fps, last = -1, 0.0, time.perf_counter()
        raw = None
        try:
            while not ui["stop"]:
                if ui["paused"]:
                    time.sleep(0.02)
                    continue
                stamp = None
                if is_cam:
                    frame, n_cam, stamp = grabber.read(n_cam)
                    if frame is None or not grabber.ok:
                        break
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if realtime:                      # play at the video's own frame rate, like a camera
                        wait = start + stats["frames"] / fps_v - time.perf_counter()
                        if wait > 0:
                            time.sleep(wait)
                if resize:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                if record_raw:                        # the plain camera image, before undistortion
                    if raw is None:
                        raw = VideoRecorder(Path(record_raw), frame.shape[1], frame.shape[0],
                                            30 if is_cam else fps_v, log=log)
                    raw.write(frame, stamp if is_cam else start + stats["frames"] / fps_v)
                if undist is not None:
                    frame = cv2.remap(frame, *undist, cv2.INTER_LINEAR)
                st = tracker.process(frame)
                now = time.perf_counter()
                if learner is not None:
                    learner.offer(tracker.frames[-1][1], st, now)
                    info["learned"] = learner.stats
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-3) if fps else 1 / max(now - last, 1e-3)
                last = now
                stats["frames"] += 1
                stats[st.status] += 1
                stats["track_ms"].append(st.track_ms)
                if st.reloc is not None:
                    info.update(reloc_ms=st.reloc.ms, reloc_kp=st.reloc.n_kp, reloc_matches=st.reloc.n_matches,
                                reloc_inliers=st.reloc.n_inliers)
                info.update(fps=fps, track_ms=st.track_ms)
                if pose_f is not None:
                    rec = {"frame": st.fid, "time": round(now - start, 4), "status": st.status,
                           "inliers": st.n_inliers}
                    if st.R is not None:
                        rec.update(t=[round(float(x), 5) for x in st.t], q_wxyz=[round(x, 6) for x in quat(st.R)],
                                   euler_xyz_deg=[round(float(x), 2) for x in euler_xyz(st.R)],
                                   R=np.round(st.R, 6).tolist())
                    pose_f.write(json.dumps(rec) + "\n")
                vis = None
                if display or save_video or ui["snap"]:
                    vis = draw(frame, st, model, K, info, ui["mask"], ui["points"], ui["axes"])
                if save_video:
                    if writer is None:
                        writer = VideoRecorder(Path(save_video), vis.shape[1], vis.shape[0],
                                               30 if is_cam else fps_v, log=log)
                    writer.write(vis, stamp if is_cam else start + stats["frames"] / fps_v)
                if ui["snap"]:
                    ui["snap"] = False
                    Path("captures").mkdir(exist_ok=True)
                    p = Path("captures") / f"pose_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                    cv2.imwrite(str(p), vis)
                    if st.R is not None:
                        m, _ = hull_mask(model.hull, st.R, st.t, K, frame.shape)
                        cv2.imwrite(str(p.with_name(p.stem + "_mask.png")), m)
                    log(f"saved {p} (+ mask)")
                with cond:
                    latest["vis"], latest["n"] = vis, latest["n"] + 1
                    cond.notify_all()
                if seconds and now - start > seconds:
                    break
        finally:
            if raw is not None:
                raw.close()
            ui["stop"] = True
            with cond:
                cond.notify_all()

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        shown = 0
        while not ui["stop"]:
            if not display:
                worker.join(timeout=0.2)
                continue
            with cond:
                cond.wait_for(lambda: latest["n"] > shown or ui["stop"], 0.1)
                vis, shown = latest["vis"], latest["n"]
            if vis is not None:
                cv2.imshow("object pose", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                ui["stop"] = True
            elif k == ord(" "):
                ui["paused"] = not ui["paused"]
            elif k in (ord("m"), ord("a"), ord("p")):
                key = {"m": "mask", "a": "axes", "p": "points"}[chr(k)]
                ui[key] = not ui[key]
            elif k == ord("r"):
                tracker.force = True
            elif k == ord("s"):
                ui["snap"] = True
    except KeyboardInterrupt:
        ui["stop"] = True
    finally:
        ui["stop"] = True
        worker.join(timeout=5)
        if learner is not None:
            learner.close()
            log(f"live learning: {json.dumps(learner.stats)}")
            if learn_save:
                learner.save()
        tracker.close()
        if grabber is not None:
            grabber.close()
        elif cap is not None:
            cap.release()
        if writer is not None:
            writer.close()
        if pose_f is not None:
            pose_f.close()
        if display:
            cv2.destroyAllWindows()
    tm = np.array(stats.pop("track_ms") or [0])
    el = time.perf_counter() - start
    stats.update(seconds=round(el, 1), fps=round(stats["frames"] / max(el, 1e-3), 1),
                 track_ms_median=round(float(np.median(tm)), 1), track_ms_p95=round(float(np.percentile(tm, 95)), 1),
                 relocalisations=tracker.stats, learned=learner.stats if learner is not None else None)
    log(json.dumps(stats))
    return stats
