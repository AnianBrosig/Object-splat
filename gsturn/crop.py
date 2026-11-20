"""Crop windows that follow the visible part of the object.

The object mask already excludes the hand (and a margin around it), so a box around the mask is a box around what
the camera can actually use. Three things keep it stable when fingers come and go:
  * extents are percentiles of the mask pixels of the large components, so a stray speck of leaked skin or a
    far-away false positive does not stretch the box;
  * for feature extraction the window has one fixed size per camera (the largest visible extent over the whole
    video), so the scale seen by the extractor is the same in every frame and the matcher compares like with
    like; only its centre follows the object, smoothed over a few frames and clamped so the visible mask always
    fits inside;
  * for training the crop is per frame (the loss ignores the hand anyway) and its size is rounded up to a
    multiple of `quant` so there are only a handful of image shapes.
"""
from __future__ import annotations

import cv2
import numpy as np


def robust_bbox(mask: np.ndarray, pct: float = 0.2, min_comp_frac: float = 0.02):
    """(x0, y0, x1, y1) of the mask (x1, y1 exclusive), ignoring small components and 0.2 % outlier pixels.
    None for an empty mask."""
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.flatnonzero(areas >= max(8, min_comp_frac * areas.max())) + 1
        m = np.isin(lab, keep).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (int(np.floor(np.percentile(xs, pct))), int(np.floor(np.percentile(ys, pct))),
            int(np.ceil(np.percentile(xs, 100 - pct))) + 1, int(np.ceil(np.percentile(ys, 100 - pct))) + 1)


def _round_up(v: float, q: int) -> int:
    return int(-(-int(np.ceil(v)) // q) * q)


def fit_window(cx: float, cy: float, w: int, h: int, W: int, H: int, must_hold=None) -> tuple[int, int, int, int]:
    """Window of size w x h around (cx, cy), shifted (not shrunk) to stay inside the W x H image. If must_hold
    (x0, y0, x1, y1) is given the window is shifted as little as needed to contain it (when it can)."""
    w, h = min(w, W), min(h, H)
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    if must_hold is not None:
        bx0, by0, bx1, by1 = must_hold
        if bx1 - bx0 <= w:
            x0 = min(max(x0, bx1 - w), bx0)
        if by1 - by0 <= h:
            y0 = min(max(y0, by1 - h), by0)
    x0 = int(np.clip(x0, 0, W - w))
    y0 = int(np.clip(y0, 0, H - h))
    return x0, y0, x0 + w, y0 + h


def feature_windows(boxes: list, W: int, H: int, margin: float = 0.10, quant: int = 16, smooth: int = 2,
                    outlier: float = 1.6):
    """One fixed window size for all frames and a centre per frame.

    boxes: per frame (x0, y0, x1, y1) of the visible object, or None. Frames whose extent is more than `outlier`
    times the median (a mask that leaked into the background) do not set the size. Returns
    (list of (x0, y0, x1, y1) windows, (w, h))."""
    ok = [b for b in boxes if b is not None]
    if not ok:
        return [(0, 0, W, H)] * len(boxes), (W, H)
    ws = np.array([b[2] - b[0] for b in ok], float)
    hs = np.array([b[3] - b[1] for b in ok], float)
    good = (ws <= outlier * np.median(ws)) & (hs <= outlier * np.median(hs))
    w = _round_up(ws[good].max() * (1 + 2 * margin) + 8, quant)
    h = _round_up(hs[good].max() * (1 + 2 * margin) + 8, quant)
    w, h = min(w, W), min(h, H)
    c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] if b is not None else [np.nan, np.nan] for b in boxes])
    for k in range(2):                                      # fill frames without a mask from the neighbours
        v = c[:, k]
        idx = np.flatnonzero(~np.isnan(v))
        v[np.isnan(v)] = np.interp(np.flatnonzero(np.isnan(v)), idx, v[idx]) if len(idx) else (W, H)[k] / 2
    sm = c.copy()
    for k in range(2):
        for i in range(len(c)):
            sm[i, k] = np.median(c[max(0, i - smooth):i + smooth + 1, k])
    wins = [fit_window(sm[i, 0], sm[i, 1], w, h, W, H, boxes[i]) for i in range(len(boxes))]
    return wins, (w, h)


def train_crop(box, W: int, H: int, margin: float = 0.06, pad_px: int = 24, quant: int = 32, min_size: int = 128):
    """Per-frame crop (x0, y0, x1, y1) around the visible object: its extent, plus `margin` of it on each side,
    plus `pad_px` of empty background, rounded up to a multiple of quant. Whole image for an empty mask."""
    if box is None:
        return 0, 0, W, H
    x0, y0, x1, y1 = box
    w = (x1 - x0) * (1 + 2 * margin) + 2 * pad_px
    h = (y1 - y0) * (1 + 2 * margin) + 2 * pad_px
    w, h = _round_up(max(w, min_size), quant), _round_up(max(h, min_size), quant)
    return fit_window((x0 + x1) / 2, (y0 + y1) / 2, w, h, W, H, box)
