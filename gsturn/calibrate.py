"""Camera calibration from a checkerboard video (`python -m gsturn calibrate`).

The video is downscaled with the same code path as the pipeline (stages.downscale), so the intrinsics live on
exactly the pixel grid the jobs use. Result: a JSON file that `Config.calibration` points to.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")   # OpenCL on this Mac only prints cache errors
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .events import Events
from .util import contact_sheet, find_tool, imwrite_rgb, pick_evenly

PATTERN_CANDIDATES = [(9, 7), (9, 6), (8, 7), (8, 6), (7, 7), (7, 6), (7, 5), (6, 6), (6, 5), (6, 4),
                      (5, 5), (5, 4), (4, 4), (4, 3)]
SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
TILT_BINS = [0, 15, 30, 45, 180]          # degrees between board normal and optical axis


class CalibrationError(Exception):
    """Calibration failed a quality check; the message says why."""


# --------------------------------------------------------------------------- metadata
def video_meta(path: Path) -> dict:
    """Camera model tag and coded size of a video (ffprobe). Missing fields are None."""
    ffprobe = find_tool("ffprobe")
    if not ffprobe or not Path(path).exists():
        return {"model": None, "width": None, "height": None}
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_side_data=rotation:format_tags", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    try:
        d = json.loads(out)
        s = d["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError):
        return {"model": None, "width": None, "height": None}
    tags = d.get("format", {}).get("tags", {}) or {}
    model = tags.get("com.apple.quicktime.model") or tags.get("model") or tags.get("com.android.model")
    w, h = int(s["width"]), int(s["height"])
    for sd in s.get("side_data_list", []) or []:
        if abs(int(sd.get("rotation", 0) or 0)) in (90, 270):
            w, h = h, w
    return {"model": model, "width": w, "height": h}


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- loading for the pipeline
def load_calibration(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        from .config import ROOT
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"calibration file not found: {path}")
    return json.loads(p.read_text())


def scaled_intrinsics(calib: dict, w: int, h: int) -> tuple[float, float, float, float, list[float]]:
    """fx, fy, cx, cy, dist for an image of size w x h (the calibration may be at a different resolution).

    The scale uses pixel-centre coordinates, so the principal point stays on the same physical spot."""
    sx, sy = w / calib["width"], h / calib["height"]
    fx, fy = calib["fx"] * sx, calib["fy"] * sy
    cx = (calib["cx"] + 0.5) * sx - 0.5
    cy = (calib["cy"] + 0.5) * sy - 0.5
    return fx, fy, cx, cy, list(calib.get("dist", [0.0, 0.0, 0.0, 0.0]))


def has_distortion(calib: dict) -> bool:
    return any(abs(float(d)) > 0 for d in calib.get("dist", []))


def undistort_maps(calib: dict, w: int, h: int):
    """(map1, map2, new_K) to undistort frames of size w x h; new_K is the pinhole matrix of the output.

    new_K = K: same focal and principal point, square pixels. getOptimalNewCameraMatrix(alpha=0) would crop to
    the valid region, but with a k1/k2 fit that is extrapolated beyond the board's coverage it scales x and y
    differently (4 % on the iPhone calibration). The few border pixels without a source are replicated."""
    fx, fy, cx, cy, dist = scaled_intrinsics(calib, w, h)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    D = np.array(dist, np.float64)
    newK = K.copy()
    m1, m2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_16SC2)
    return m1, m2, newK


# --------------------------------------------------------------------------- detection
def parse_pattern(s: str | None) -> Optional[tuple[int, int]]:
    if not s:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*[xX,]\s*(\d+)\s*", s)
    if not m:
        raise ValueError(f"pattern must look like 6x6 (inner corners), got {s!r}")
    return int(m.group(1)), int(m.group(2))


def detect(gray: np.ndarray, pattern: tuple[int, int]) -> Optional[np.ndarray]:
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=SB_FLAGS)
    return corners.reshape(-1, 2).astype(np.float32) if ok else None


def auto_pattern(grays: list[np.ndarray], min_share: float = 0.3) -> tuple[int, int]:
    """Largest inner-corner pattern found in at least min_share of the sample frames."""
    sample = pick_evenly(grays, 15)
    for pat in sorted(PATTERN_CANDIDATES, key=lambda p: -p[0] * p[1]):
        hits = 0
        for i, g in enumerate(sample):
            if detect(g, pat) is not None:
                hits += 1
            if hits >= min_share * len(sample):
                return pat
            if hits + (len(sample) - i - 1) < min_share * len(sample):
                break   # cannot reach the share any more
    raise CalibrationError("no checkerboard found; pass --pattern WxH (inner corners)")


def board_points(pattern: tuple[int, int], square_mm: float) -> np.ndarray:
    nx, ny = pattern
    obj = np.zeros((nx * ny, 3), np.float32)
    obj[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square_mm
    return obj


# --------------------------------------------------------------------------- view selection
def select_views(dets: list[dict], obj: np.ndarray, w: int, h: int, max_views: int) -> list[dict]:
    """Spread views over board-tilt bins and image regions; inside a bin prefer views reaching the border."""
    f0 = 1.2 * w
    K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]], np.float64)
    for d in dets:
        ok, rvec, tvec = cv2.solvePnP(obj, d["corners"], K0, None)
        R = cv2.Rodrigues(rvec)[0]
        d["tilt"] = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))) if ok else 0.0
        c = d["corners"].mean(0)
        d["cell"] = (min(2, int(3 * c[0] / w)), min(2, int(3 * c[1] / h)))
        tb = int(np.searchsorted(TILT_BINS, d["tilt"], side="right") - 1)
        d["bin"] = (min(tb, len(TILT_BINS) - 2), d["cell"])
        # how far the corners reach towards the image border (0 centre .. 1 corner)
        r = np.abs(d["corners"] - [w / 2, h / 2]) / [w / 2, h / 2]
        d["reach"] = float(np.max(np.hypot(r[:, 0], r[:, 1])) / np.sqrt(2))
    if len(dets) <= max_views:
        return list(dets)
    bins: dict = {}
    for d in dets:
        bins.setdefault(d["bin"], []).append(d)
    for b in bins.values():
        b.sort(key=lambda d: -d["reach"])
    out: list[dict] = []
    while len(out) < max_views:
        added = False
        for b in sorted(bins, key=lambda k: -len(bins[k])):
            if bins[b] and len(out) < max_views:
                out.append(bins[b].pop(0))
                added = True
        if not added:
            break
    return sorted(out, key=lambda d: d["frame"])


# --------------------------------------------------------------------------- calibration
def calibrate_views(views: list[dict], obj: np.ndarray, w: int, h: int, fix_dist: bool, f_guess: float):
    flags = (cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
             | cv2.CALIB_USE_INTRINSIC_GUESS)
    if fix_dist:
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    K = np.array([[f_guess, 0, (w - 1) / 2], [0, f_guess, (h - 1) / 2], [0, 0, 1]], np.float64)
    D = np.zeros(5, np.float64)
    objs = [obj] * len(views)
    imgs = [v["corners"].reshape(-1, 1, 2) for v in views]
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9)
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, (w, h), K, D, flags=flags, criteria=crit)
    per_view, resid = [], []
    for o, im, r, t in zip(objs, imgs, rvecs, tvecs):
        proj = cv2.projectPoints(o, r, t, K, D)[0].reshape(-1, 2)
        e = im.reshape(-1, 2) - proj
        resid.append(e)
        per_view.append(float(np.sqrt((e ** 2).sum(1).mean())))
    return {"rms": float(rms), "K": K, "D": D.ravel(), "rvecs": rvecs, "tvecs": tvecs,
            "per_view": np.array(per_view), "resid": resid}


def calibrate_robust(views: list[dict], obj: np.ndarray, w: int, h: int, fix_dist: bool, f_guess: float,
                     ev: Optional[Events] = None):
    views = list(views)
    res = calibrate_views(views, obj, w, h, fix_dist, f_guess)
    for _ in range(2):
        thr = 2.0 * float(np.median(res["per_view"]))
        keep = res["per_view"] <= thr
        if keep.all() or keep.sum() < 15:
            break
        if ev:
            ev.log(f"dropping {int((~keep).sum())} views with RMS > {thr:.2f}px")
        views = [v for v, k in zip(views, keep) if k]
        res = calibrate_views(views, obj, w, h, fix_dist, f_guess)
    res["views"] = views
    return res


def distortion_shift_px(K: np.ndarray, D: np.ndarray, w: int, h: int, central: float = 0.8) -> float:
    """Largest pixel shift the distortion model causes inside the central share of the image."""
    mx, my = (1 - central) / 2 * w, (1 - central) / 2 * h
    xs, ys = np.meshgrid(np.linspace(mx, w - 1 - mx, 25), np.linspace(my, h - 1 - my, 15))
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)
    return float(np.max(np.linalg.norm(und - pts.reshape(-1, 2), axis=1)))


# --------------------------------------------------------------------------- debug images
def save_views_sheet(views: list[dict], frames: dict[int, np.ndarray], pattern, path: Path) -> None:
    imgs, labels = [], []
    for v in views:
        img = cv2.cvtColor(frames[v["frame"]], cv2.COLOR_GRAY2RGB)
        cv2.drawChessboardCorners(img, pattern, v["corners"].reshape(-1, 1, 2), True)
        imgs.append(img)
        labels.append(f"#{v['frame']} tilt {v.get('tilt', 0):.0f}")
    imwrite_rgb(path, contact_sheet(imgs, cols=8, thumb_w=240, labels=labels), 85)


def save_residuals(res: dict, w: int, h: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 1.4]})
    pv = res["per_view"]
    ax[0].bar(range(len(pv)), pv, color="tab:blue")
    ax[0].axhline(res["rms"], ls="--", c="gray", lw=1, label=f"overall RMS {res['rms']:.2f}px")
    ax[0].set_xticks(range(len(pv)))
    ax[0].set_xticklabels([str(v["frame"]) for v in res["views"]], rotation=90, fontsize=6)
    ax[0].set_xlabel("video frame")
    ax[0].set_ylabel("RMS reprojection error (px)")
    ax[0].legend(fontsize=8)
    pts = np.concatenate([v["corners"] for v in res["views"]])
    err = np.concatenate(res["resid"])
    scale = 0.05 * w / max(1e-6, float(np.percentile(np.linalg.norm(err, axis=1), 95)))
    ax[1].quiver(pts[:, 0], pts[:, 1], err[:, 0] * scale, err[:, 1] * scale, np.linalg.norm(err, axis=1),
                 angles="xy", scale_units="xy", scale=1, cmap="viridis", width=0.002)
    ax[1].set_xlim(0, w)
    ax[1].set_ylim(h, 0)
    ax[1].set_aspect("equal")
    ax[1].set_title(f"residuals over the image (arrows x{scale:.0f})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- main entry
def default_output(video: Path, meta: dict, fps: float) -> Path:
    from .config import ROOT
    slug = re.sub(r"[^a-z0-9]+", "", (meta.get("model") or video.stem).lower()) or "camera"
    size = meta.get("height") or 0
    return ROOT / "calibration" / f"{slug}_{size}p{round(fps)}.json"


def calibrate(video: Path, square_mm: float, pattern: Optional[tuple[int, int]] = None, height: int = 0,
              output: Optional[Path] = None, check_job: Optional[Path] = None, step: int = 3,
              max_views: int = 40, hwaccel: str = "auto", max_rms: float = 1.5, min_views: int = 15,
              distortion: str = "auto", ev: Optional[Events] = None) -> dict:
    """distortion: auto (decide from the data), zero (always pinhole), fit (always keep k1, k2)."""
    from .stages.downscale import downscale, needs_transcode, probe
    cv2.ocl.setUseOpenCL(False)
    ev = ev or Events(None)
    video = Path(video)
    meta = video_meta(video)
    info = probe(video)
    output = Path(output) if output else default_output(video, meta, info["fps"])
    output.parent.mkdir(parents=True, exist_ok=True)

    # 1. same pixel grid as the pipeline
    with tempfile.TemporaryDirectory() as tmp:
        small = Path(tmp) / "calib_small.mp4"
        if needs_transcode(info, height):
            ev.log(f"downscaling {video.name} to height {height or 'native'}")
            downscale(video, small, height, 0.0, hwaccel, ev=None)
        else:
            small = video   # the pipeline reads the original too
        cap = cv2.VideoCapture(str(small))
        grays: dict[int, np.ndarray] = {}
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                grays[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            i += 1
        cap.release()
    if not grays:
        raise CalibrationError("could not decode any frame")
    h, w = next(iter(grays.values())).shape

    # 2. drop blurred frames
    sharp = {k: float(cv2.Laplacian(g, cv2.CV_64F).var()) for k, g in grays.items()}
    med = float(np.median(list(sharp.values())))
    cand = [k for k in grays if sharp[k] >= 0.5 * med]
    ev.log(f"{len(cand)} of {len(grays)} sampled frames sharp enough (>= 0.5 x median Laplacian variance)")

    # 3. detect corners
    if pattern is None:
        pattern = auto_pattern([grays[k] for k in cand])
        ev.log(f"auto-detected pattern {pattern[0]}x{pattern[1]} inner corners")
    obj = board_points(pattern, square_mm)
    dets = []
    for n, k in enumerate(cand):
        c = detect(grays[k], pattern)
        if c is not None:
            dets.append({"frame": k, "corners": c})
        ev.progress(n + 1, len(cand), f"corners {n + 1}/{len(cand)} ({len(dets)} found)")
    ev.log(f"board found in {len(dets)} of {len(cand)} frames")
    if len(dets) < min_views:
        raise CalibrationError(f"board found in only {len(dets)} frames (need {min_views})")

    # 4. diverse views
    views = select_views(dets, obj, w, h, max_views)

    # 5./6. calibrate with free and with fixed distortion, outlier rejection inside
    f_guess = 1100.0 * w / 1280.0
    free = calibrate_robust(views, obj, w, h, False, f_guess, ev)
    fixed = calibrate_robust(views, obj, w, h, True, f_guess, ev)

    # distortion decision
    shift = distortion_shift_px(free["K"], free["D"], w, h)
    keep_zero = (fixed["rms"] - free["rms"] < 0.15) or shift < 0.5
    if distortion != "auto":
        keep_zero = distortion == "zero"
    use = fixed if keep_zero else free
    K = use["K"]
    dist = [0.0, 0.0, 0.0, 0.0] if keep_zero else [float(free["D"][0]), float(free["D"][1]), 0.0, 0.0]
    ev.log(f"distortion: free RMS {free['rms']:.3f}, fixed RMS {fixed['rms']:.3f}, max shift in central 80 % "
           f"{shift:.2f}px -> {'pinhole (dist = 0)' if keep_zero else 'keep k1, k2 (frames get undistorted)'}"
           + ("" if distortion == "auto" else f" [forced: --distortion {distortion}]"))

    # 7. checks
    dists = [float(np.linalg.norm(t)) for t in use["tvecs"]]
    problems = []
    if use["rms"] > max_rms:
        problems.append(f"RMS {use['rms']:.2f}px > {max_rms}px")
    if len(use["views"]) < min_views:
        problems.append(f"only {len(use['views'])} views left after outlier rejection (need {min_views})")
    sfm_f = None
    if check_job:
        st = Path(check_job) / "work" / "sfm" / "stats.json"
        fr = Path(check_job) / "work" / "frames" / "frames.json"
        if not st.exists() or not fr.exists():
            problems.append(f"--check-job {check_job}: no sfm stats")
        else:
            sfm_f = float(json.loads(st.read_text())["camera"]["params"][0])
            jw = json.loads(fr.read_text())["width"]
            f_job = float(K[0, 0]) * jw / w
            dev = abs(f_job - sfm_f) / sfm_f
            ev.log(f"focal at job resolution {f_job:.1f}px vs SfM {sfm_f:.1f}px ({100 * dev:.1f} %)")
            if dev > 0.05:
                problems.append(f"focal {f_job:.0f}px differs from the job's SfM focal {sfm_f:.0f}px by "
                                f"{100 * dev:.0f} % (> 5 %)")

    # 8. outputs
    stem = output.with_suffix("")
    save_views_sheet(use["views"], grays, pattern, Path(f"{stem}_views.jpg"))
    save_residuals(use, w, h, Path(f"{stem}_residuals.png"))
    result = {
        "camera": meta.get("model") or "", "source": str(video), "source_sha1": sha1(video),
        "source_size": [info["width"], info["height"]], "fps": info["fps"], "width": w, "height": h,
        "model": "PINHOLE", "fx": round(float(K[0, 0]), 3), "fy": round(float(K[1, 1]), 3),
        "cx": round(float(K[0, 2]), 3), "cy": round(float(K[1, 2]), 3), "dist": dist,
        "dist_fitted": [round(float(free["D"][0]), 5), round(float(free["D"][1]), 5), 0.0, 0.0],
        "rms_px": round(use["rms"], 4), "rms_free_px": round(free["rms"], 4), "rms_fixed_px": round(fixed["rms"], 4),
        "dist_shift_central_px": round(shift, 3), "distortion_mode": distortion, "views": len(use["views"]), "detections": len(dets),
        "pattern": list(pattern), "square_mm": float(square_mm),
        "board_distance_mm": [round(float(x)) for x in np.percentile(dists, [0, 50, 100])],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if sfm_f is not None:
        result["check_job_sfm_focal"] = sfm_f
    if problems:
        result["problems"] = problems
        Path(f"{stem}_rejected.json").write_text(json.dumps(result, indent=2))
        raise CalibrationError("calibration rejected: " + "; ".join(problems)
                               + f" (details in {stem}_rejected.json)")
    output.write_text(json.dumps(result, indent=2))
    result["output"] = str(output)
    return result
