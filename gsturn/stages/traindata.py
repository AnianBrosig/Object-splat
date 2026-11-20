"""Training dataset for Brush: per-frame crops, black background, soft loss weights, one camera per image.

Brush (v0.3) facts this relies on (read from its source):
  * the training background is black, and with a masks/ folder the loss is (L1 + SSIM) * mask, so a mask is a soft
    per-pixel weight and the image is used as is (no alpha premultiplication);
  * every image has its own COLMAP camera, including its own principal point, so a crop is just a shifted cx, cy.

train_mask_mode
  weighted        object + empty background trained (the background as black = "nothing"), hand, unknown band and
                  untrusted pixels ignored with soft edges            (default; no floaters, no carving by fingers)
  alpha           RGBA, background and hand forced transparent (the old default)
  ignore          colour images + masks/ (object only): fingers are not carved, but nothing pushes the
                  background to transparent
  ignore_quality  like ignore, minus the pixels the quality stage does not trust
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..crop import robust_bbox, train_crop
from ..events import Events
from ..handmask import loss_weights
from ..multicam import stem_of
from ..pipeline import StageError


def _lines(p: Path):
    with open(p) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                yield line.rstrip("\n")


def read_cameras(txt: Path) -> dict[int, dict]:
    """cameras.txt -> {id: {w, h, fx, fy, cx, cy}} (PINHOLE and the SIMPLE_* / RADIAL models, pinhole part only)."""
    out = {}
    for line in _lines(txt / "cameras.txt"):
        a = line.split()
        cid, model, w, h = int(a[0]), a[1], int(a[2]), int(a[3])
        p = [float(x) for x in a[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = p[:4]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        elif model in ("OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
            fx, fy, cx, cy = p[:4]
        else:
            raise StageError(f"unsupported camera model {model}")
        out[cid] = {"w": w, "h": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return out


def read_image_headers(txt: Path) -> list[list[str]]:
    """images.txt header lines (every other data line) split into fields."""
    lines = list(_lines(txt / "images.txt"))
    return [l.split() for l in lines[::2]]


def write_model(dst: Path, txt: Path, cams: dict[int, dict], crops: dict[str, tuple], ev: Events) -> int:
    """Per-image cameras for the crops, images without points2D, points3D without tracks. Returns the image count."""
    sparse = dst / "sparse" / "0"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    heads = read_image_headers(txt)
    cam_lines, img_lines = [], []
    k = 0
    for a in heads:
        stem = stem_of(a[9])
        if stem not in crops:
            continue
        k += 1
        x0, y0, x1, y1 = crops[stem]
        c = cams[int(a[8])]
        cam_lines.append(f"{k} PINHOLE {x1 - x0} {y1 - y0} {c['fx']:.6f} {c['fy']:.6f} "
                         f"{c['cx'] - x0:.6f} {c['cy'] - y0:.6f}")
        img_lines.append(f"{k} {' '.join(a[1:8])} {k} {stem}.png\n")
    (sparse / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
    (sparse / "images.txt").write_text("".join(l + "\n" for l in img_lines))
    with open(sparse / "points3D.txt", "w") as out:
        for line in _lines(txt / "points3D.txt"):
            a = line.split()
            out.write(f"{a[0]} {a[1]} {a[2]} {a[3]} {a[4]} {a[5]} {a[6]} {a[7]}\n")
    return k


def build(job, cfg: Config, ev: Events, out: Path, extra_ignore: dict[str, np.ndarray] | None = None) -> dict:
    """Write out/images, out/masks (weighted, ignore*) and out/sparse/0. extra_ignore: {stem: bool mask in full
    frame coordinates} of pixels not to train (from the residual pass)."""
    from ..track.colmap_txt import ensure_text
    from .quality import TRUST_TRAIN, load_quality

    sfm = job.dir / "work" / "sfm"
    txt = ensure_text(sfm / "dataset" / "sparse" / "0", sfm / "model_txt")
    cams = read_cameras(txt)
    heads = read_image_headers(txt)
    stems = [stem_of(a[9]) for a in heads]
    meta = job.load_json("frames", "frames.json")
    fname = {stem_of(f["name"]): f["name"] for f in meta["frames"]}
    img_dir = job.dir / "work" / "frames" / "images"
    seg = job.dir / "work" / "segment"
    quality = load_quality(job, cfg)
    mode = cfg.train_mask_mode
    if mode not in ("weighted", "alpha", "ignore", "ignore_quality"):
        raise StageError(f"unknown train_mask_mode {mode}")

    for sub in ("images", "masks"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if mode != "alpha":
        (out / "masks").mkdir(parents=True, exist_ok=True)

    crops: dict[str, tuple] = {}
    full_px = crop_px = 0
    wsum = []
    for k, (a, stem) in enumerate(zip(heads, stems)):
        bgr = cv2.imread(str(img_dir / fname[stem]), cv2.IMREAD_COLOR)
        obj = cv2.imread(str(seg / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or obj is None:
            raise StageError(f"missing frame or mask for {stem}: run 'frames' and 'segment' first")
        H, W = obj.shape
        hand = cv2.imread(str(seg / "hands" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if hand is None:
            hand = np.zeros_like(obj)
        untrusted = None
        if quality is not None and mode in ("weighted", "ignore_quality"):
            q = cv2.imread(str(job.dir / "work" / "quality" / "maps" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            if q is not None and q.shape == obj.shape:
                untrusted = (obj > 0) & (q < TRUST_TRAIN)
        if extra_ignore and stem in extra_ignore:
            untrusted = extra_ignore[stem] if untrusted is None else (untrusted | extra_ignore[stem])
        if cfg.train_crop:
            box = robust_bbox(obj)
            x0, y0, x1, y1 = train_crop(box, W, H, cfg.train_crop_margin, cfg.train_crop_pad_px)
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        crops[stem] = (x0, y0, x1, y1)
        full_px += W * H
        crop_px += (x1 - x0) * (y1 - y0)
        sl = (slice(y0, y1), slice(x0, x1))
        o = obj[sl] > 0
        name = f"{stem}.png"
        flags = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        if mode == "weighted":
            w = loss_weights(obj, hand, untrusted, cfg.train_hand_margin_px, cfg.train_edge_ramp_px)[sl]
            img = bgr[sl].copy()
            img[~o] = 0                                         # empty space = black = what Brush renders there
            cv2.imwrite(str(out / "images" / name), img, flags)
            cv2.imwrite(str(out / "masks" / name), np.round(w * 255).astype(np.uint8), flags)
            wsum.append(float(w.mean()))
        elif mode == "alpha":
            cv2.imwrite(str(out / "images" / name), np.dstack([bgr[sl], obj[sl]]), flags)
        else:
            m = o.copy()
            if mode == "ignore_quality" and untrusted is not None:
                m &= ~untrusted[sl]
            cv2.imwrite(str(out / "images" / name), bgr[sl], flags)
            cv2.imwrite(str(out / "masks" / name), (m * 255).astype(np.uint8), flags)
        ev.progress(k + 1, len(stems), f"training images {k + 1}/{len(stems)}")
    n = write_model(out, txt, cams, crops, ev)
    return {"crops": crops, "images": n, "crop_pixels_share": round(crop_px / max(1, full_px), 3),
            "mean_loss_weight": round(float(np.mean(wsum)), 3) if wsum else None, "mode": mode,
            "crop": bool(cfg.train_crop)}


def residual_ignore(job, cfg: Config, ds: Path, ply: Path, crops: dict[str, tuple], ev: Events) -> dict[str, np.ndarray]:
    """Pixels where a first, short training disagrees with the photograph even though they count as object.

    Whatever the other views agree on is in the splat; a finger that leaked into the object mask in one frame, a
    highlight that moved, a smudge on the lens are not, so they stand out as local colour error. The returned
    masks (full frame, per stem) go to build() as extra ignore regions for the real training.
    crops: {stem: (x0, y0, x1, y1)} of the dataset images in full-frame pixels (from build())."""
    from ..splat_render import load_ply, render
    from ..track.colmap_txt import qvec_to_R
    s = load_ply(ply)
    sparse = ds / "sparse" / "0"
    cams = read_cameras(sparse)
    heads = read_image_headers(sparse)
    seg = job.dir / "work" / "segment" / "masks"
    out: dict[str, np.ndarray] = {}
    share = []
    for k, a in enumerate(heads):
        stem = stem_of(a[9])
        c = cams[int(a[8])]
        gt = cv2.imread(str(ds / "images" / f"{stem}.png"), cv2.IMREAD_COLOR)
        obj = cv2.imread(str(seg / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if gt is None or obj is None or stem not in crops:
            continue
        x0, y0, x1, y1 = crops[stem]
        Rwc = qvec_to_R(np.array([float(x) for x in a[1:5]]))
        center = -Rwc.T @ np.array([float(x) for x in a[5:8]])
        img = render(s, Rwc, center, c["fx"], c["w"], c["h"], (0, 0, 0), c["cx"], c["cy"])[..., ::-1]
        err = cv2.blur(np.abs(img.astype(np.float32) - gt.astype(np.float32)).mean(2) / 255.0, (9, 9))
        bad = (err > cfg.train_residual_thresh) & (obj[y0:y1, x0:x1] > 0)
        bad = cv2.dilate(bad.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        full = np.zeros(obj.shape, bool)
        full[y0:y1, x0:x1] = bad & (obj[y0:y1, x0:x1] > 0)
        out[stem] = full
        share.append(float(full.sum() / max(1, (obj > 0).sum())))
        ev.progress(k + 1, len(heads), f"residual pass: view {k + 1}/{len(heads)}")
    ev.log(f"residual pass: {100 * float(np.mean(share)):.1f} % of the object pixels flagged on average "
           f"(max {100 * float(np.max(share)):.1f} %)" if share else "residual pass: nothing rendered")
    return out
