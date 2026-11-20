"""Stage 2: pick sharp, well-spaced frames."""
from __future__ import annotations

import shutil

import cv2
import numpy as np

from ..config import Config
from ..events import Events
from ..util import contact_sheet, imwrite_rgb, pick_evenly


def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    half = max(1, win // 2)
    return np.array([np.median(x[max(0, i - half):i + half + 1]) for i in range(len(x))])


def select(sharp: np.ndarray, thumbs: np.ndarray, min_motion: float, sharp_ratio: float, win: int,
           max_frames: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Greedy selection. Returns kept indices, rolling-median threshold, and motion-vs-last-kept per frame."""
    thr = sharp_ratio * rolling_median(sharp, win)
    sharp_ok = sharp >= thr
    kept: list[int] = []
    motion = np.zeros(len(sharp))
    last = None
    for i in range(len(sharp)):
        if last is not None:
            motion[i] = float(np.mean(np.abs(thumbs[i] - thumbs[last])))
        if not sharp_ok[i]:
            continue
        if last is None or motion[i] >= min_motion:
            kept.append(i)
            last = i
    if len(kept) > max_frames:
        kept = pick_evenly(kept, max_frames)
    return kept, thr, motion


def cap_fps(path) -> float:
    c = cv2.VideoCapture(str(path))
    f = c.get(cv2.CAP_PROP_FPS) or 30.0
    c.release()
    return float(f)


def add_second_camera(job, cfg: Config, ev: Events, frames: list[dict], img_dir):
    """For every selected frame of camera 1, the sharpest frame of camera 2 within one frame of the same moment
    (camera 2's clock = camera 1's + the offset found by the sync). Written as c1_<k>.jpg, k = index of the partner.
    Returns (frame entries, camera info)."""
    src = job.work("downscale") / "video2_small.mp4"
    sync = job.load_json("downscale", "sync.json")
    off, t2start = float(sync["offset"]), float(sync["trim_start2"])
    cap = cv2.VideoCapture(str(src))
    sharp2, t2 = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        sharp2.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        t2.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()
    n2 = len(sharp2)
    if n2 == 0:
        raise RuntimeError("could not decode video2")
    fps2 = cap_fps(src)
    t2 = np.array(t2) if t2[-1] > 0 else np.arange(n2) / fps2
    chosen: dict[int, int] = {}                      # k (camera 1 frame index) -> frame index in video2
    for k, f in enumerate(frames):
        target = f["t"] + off - t2start
        j = int(np.argmin(np.abs(t2 - target)))
        if abs(t2[j] - target) > max(0.05, 1.5 / fps2):
            continue                                 # camera 2 was not recording at that moment
        lo, hi = max(0, j - 1), min(n2 - 1, j + 1)
        chosen[k] = lo + int(np.argmax(sharp2[lo:hi + 1]))
    if not chosen:
        raise RuntimeError("no frame of video2 matches the selected frames of video: check video2_offset")
    undist, intr = None, None
    if cfg.calibration2:
        from ..calibrate import has_distortion, load_calibration, undistort_maps
        calib = load_calibration(cfg.calibration2)
        if has_distortion(calib):
            c0 = cv2.VideoCapture(str(src))
            vw, vh = int(c0.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c0.get(cv2.CAP_PROP_FRAME_HEIGHT))
            c0.release()
            m1, m2, newK = undistort_maps(calib, vw, vh)
            undist = (m1, m2)
            intr = {"fx": float(newK[0, 0]), "fy": float(newK[1, 1]), "cx": float(newK[0, 2]),
                    "cy": float(newK[1, 2]), "undistorted": True, "calibration": cfg.calibration2}
    want = {j: k for k, j in chosen.items()}
    out, size = [], None
    cap = cv2.VideoCapture(str(src))
    for j in range(n2):
        ok, fr = cap.read()
        if not ok:
            break
        if j in want:
            k = want[j]
            if undist is not None:
                fr = cv2.remap(fr, undist[0], undist[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            name = f"c1_{k:05d}.jpg"
            cv2.imwrite(str(img_dir / name), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            size = (fr.shape[1], fr.shape[0])
            out.append({"name": name, "src_index": int(j), "sharpness": round(sharp2[j], 2), "cam": 1,
                        "t": round(float(t2[j] + t2start - off), 4), "partner": frames[k]["name"]})
    cap.release()
    out.sort(key=lambda e: e["name"])
    cam = {"width": size[0], "height": size[1], "video": cfg.video2, "offset": off}
    if intr:
        cam["intrinsics"] = intr
    return out, cam


def run(job, cfg: Config, ev: Events) -> dict:
    src = job.work("downscale") / "video_small.mp4"
    if not src.exists():
        raise RuntimeError("run stage 'downscale' first")
    out = job.work("frames")
    img_dir = out / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir()

    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sharp, thumbs, times0 = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        times0.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        th = cv2.resize(gray, (96, max(1, round(96 * gray.shape[0] / gray.shape[1]))), interpolation=cv2.INTER_AREA)
        thumbs.append(th.astype(np.float32) / 255.0)
        i += 1
        ev.progress(i, total * 2, f"scoring frame {i}/{total}")
    cap.release()
    n = len(sharp)
    if n == 0:
        raise RuntimeError("could not decode any frame")
    sharp_a, thumbs_a = np.array(sharp), np.stack(thumbs)
    kept, thr, motion = select(sharp_a, thumbs_a, cfg.min_motion, cfg.sharp_keep_ratio, cfg.sharp_window,
                               cfg.max_frames)

    # optional lens undistortion (only when the calibration kept k1/k2): COLMAP, Brush and the renderer then
    # all work on plain pinhole images
    undist = None
    if cfg.calibration:
        from ..calibrate import has_distortion, load_calibration, undistort_maps
        calib = load_calibration(cfg.calibration)
        if has_distortion(calib):
            vw = int(cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FRAME_HEIGHT))
            m1, m2, newK = undistort_maps(calib, vw, vh)
            undist = (m1, m2)
            intrinsics = {"fx": float(newK[0, 0]), "fy": float(newK[1, 1]), "cx": float(newK[0, 2]),
                          "cy": float(newK[1, 2]), "undistorted": True, "calibration": cfg.calibration,
                          "dist": calib["dist"]}
            ev.log(f"undistorting frames with k = {calib['dist'][:2]}; new fx {newK[0, 0]:.1f}")

    # write kept frames
    keep_set = {k: j for j, k in enumerate(kept)}
    names: list[str] = []
    thumbs_rgb = []
    cap = cv2.VideoCapture(str(src))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if i in keep_set:
            if undist is not None:
                frame = cv2.remap(frame, undist[0], undist[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            name = f"{keep_set[i]:05d}.jpg"
            cv2.imwrite(str(img_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            names.append(name)
            thumbs_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ev.progress(n + i + 1, n * 2, f"writing frames ({len(names)}/{len(kept)})")
    cap.release()
    h, w = thumbs_rgb[0].shape[:2]
    fps0 = cap_fps(src)
    t_of = lambda k: cfg.trim_start + (times0[k] if times0 and times0[-1] > 0 else k / fps0)  # noqa: E731
    frames = [{"name": nm, "src_index": int(k), "sharpness": round(float(sharp_a[k]), 2), "cam": 0,
               "t": round(float(t_of(k)), 4)} for nm, k in zip(names, kept)]
    fmeta = {"width": w, "height": h, "frames": frames}
    if undist is not None:
        fmeta["intrinsics"] = intrinsics
    if cfg.video2:
        cams = [{"width": w, "height": h, "video": cfg.video}]
        if undist is not None:
            cams[0]["intrinsics"] = intrinsics
        extra, cam1 = add_second_camera(job, cfg, ev, frames, img_dir)
        fmeta["frames"] = frames + extra
        fmeta["cameras"] = cams + [cam1]
        ev.log(f"camera 2: {len(extra)} synchronised frames of {len(frames)}")
    job.save_json("frames", "frames.json", fmeta)

    # debug plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax[0].plot(sharp_a, lw=1, label="sharpness (Laplacian var)")
    ax[0].plot(thr, lw=1, ls="--", label=f"{cfg.sharp_keep_ratio} x rolling median")
    ax[0].scatter(kept, sharp_a[kept], s=10, c="tab:red", zorder=3, label="kept")
    ax[0].legend(fontsize=8)
    ax[1].plot(motion, lw=1, label="motion vs last kept")
    ax[1].axhline(cfg.min_motion, ls="--", c="gray", lw=1, label="min_motion")
    ax[1].scatter(kept, motion[kept], s=10, c="tab:red", zorder=3)
    ax[1].set_xlabel("video frame")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"{len(kept)} of {n} frames kept")
    fig.tight_layout()
    fig.savefig(out / "selection.png", dpi=110)
    plt.close(fig)
    ev.artifact(out / "selection.png", "sharpness / motion, kept frames in red")

    sheet_sel = pick_evenly(list(range(len(thumbs_rgb))), 48)
    sheet = contact_sheet([thumbs_rgb[j] for j in sheet_sel], cols=8, thumb_w=200,
                          labels=[names[j][:-4] for j in sheet_sel])
    imwrite_rgb(out / "contact_sheet.jpg", sheet)
    ev.artifact(out / "contact_sheet.jpg", "contact sheet (evenly sampled kept frames)")
    ev.artifact(img_dir / names[0], "first selected frame (draw the object box on this)")

    stats = {"video_frames": n, "kept": len(kept), "width": w, "height": h, "undistorted": undist is not None,
             "median_sharpness": round(float(np.median(sharp_a)), 2),
             "median_motion_kept": round(float(np.median(motion[kept][1:])) if len(kept) > 1 else 0.0, 4),
             "_msg": f"{len(kept)} of {n} frames kept"}
    if len(kept) < 30:
        ev.warn(f"only {len(kept)} frames kept; lower min_motion or sharp_keep_ratio")
    return stats
