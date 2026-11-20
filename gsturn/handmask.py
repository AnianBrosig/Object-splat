"""Keeping fingers out of the object: skin leak detection and loss weights for training.

Three kinds of pixels matter to the splat trainer:
  object   - trained to the photographed colour (weight 1)
  empty    - background that is certainly not the object: trained to black = "nothing here" (weight 1), which is
             what stops floaters
  unknown  - the hand and a margin around it, the soft band along the object's silhouette, blurred or specular
             object pixels: not trained (weight 0, ramping up to 1 with distance)
"""
from __future__ import annotations

import cv2
import numpy as np


def disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# --------------------------------------------------------------------------- skin leaks
def skin_leak(img_rgb: np.ndarray, obj: np.ndarray, hand: np.ndarray, near_frac: float = 0.05,
              thr: float = 9.0, max_far_frac: float = 0.15, work_side: int = 480) -> np.ndarray:
    """Object-mask pixels that are really skin: colour like the hand's and connected to the hand.

    The hand's own pixels give the colour model (Gaussian on the Lab a/b chroma, so shadows on the hand do not
    matter much). A candidate must lie within near_frac of the image height from the hand and be connected to it.
    If a noticeable share of the object far from the hand also looks like skin (a pink or wooden object) the
    colour test cannot tell them apart and nothing is removed. Works on a reduced image (work_side px on the long
    side), returns a bool mask at full size (a subset of obj)."""
    H, W = obj.shape
    empty = np.zeros((H, W), bool)
    s = min(1.0, work_side / max(H, W))
    sz = (max(1, int(W * s)), max(1, int(H * s)))
    o = cv2.resize((obj > 0).astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
    hd = cv2.resize((hand > 0).astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
    if hd.sum() < 60 or o.sum() < 60:
        return empty
    small = cv2.resize(img_rgb, sz, interpolation=cv2.INTER_AREA)
    ab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 1:3] - 128.0
    ref = cv2.erode(hd.astype(np.uint8), disk(2)) > 0
    if ref.sum() < 50:
        ref = hd
    x = ab[ref]
    mu = np.median(x, 0)
    inv = np.linalg.inv(np.cov(x.T) + np.eye(2) * 0.5)
    d = ab - mu
    skin = np.einsum("...i,ij,...j->...", d, inv, d) < thr
    near_px = max(4, int(near_frac * sz[1]))
    near = cv2.dilate(hd.astype(np.uint8), disk(near_px)) > 0
    far = o & ~near
    if far.sum() > 100 and (skin & far).sum() / far.sum() > max_far_frac:
        return empty
    cand = (o & skin & near).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, disk(1))
    # the object mask stops hand_dilate_px short of the hand, so "connected" means within half of near_px
    touch = cv2.dilate(hd.astype(np.uint8), disk(max(3, near_px // 2))) > 0
    n, lab_, _, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros(cand.shape, np.uint8)
    for c in range(1, n):
        comp = lab_ == c
        if (comp & touch).any():
            keep[comp] = 1
    if not keep.any():
        return empty
    keep = cv2.dilate(keep, disk(1))                       # the open/resize lose a pixel or two at the border
    full = cv2.resize(keep, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return full & (obj > 0)


# --------------------------------------------------------------------------- loss weights
def ramp(ignore: np.ndarray, px: int) -> np.ndarray:
    """0 inside the ignored region, rising linearly to 1 at px pixels away from it (float32)."""
    if not ignore.any():
        return np.ones(ignore.shape, np.float32)
    if px <= 0:
        return (~ignore).astype(np.float32)
    dt = cv2.distanceTransform((~ignore).astype(np.uint8), cv2.DIST_L2, 3)
    return np.clip(dt / px, 0, 1).astype(np.float32)


def loss_weights(obj: np.ndarray, hand: np.ndarray, untrusted: np.ndarray | None, hand_margin_px: int,
                 edge_ramp_px: int) -> np.ndarray:
    """Per-pixel training weight in [0, 1].

    obj: object mask; hand: hand mask; untrusted: object pixels to skip (blur, highlights) or None.
    Ignored: the hand dilated by hand_margin_px, the untrusted object pixels, and the unknown band just outside
    the object silhouette (the mask is eroded and may have lost a pixel or two of real object).
    The weight ramps from 0 to 1 over edge_ramp_px from each of those regions."""
    o = obj > 0
    hand_zone = cv2.dilate((hand > 0).astype(np.uint8), disk(hand_margin_px)) > 0 if hand.any() else \
        np.zeros(o.shape, bool)
    w = ramp(hand_zone, edge_ramp_px)
    if untrusted is not None and untrusted.any():
        w = np.minimum(w, ramp(untrusted & o, edge_ramp_px))
    # unknown band outside the silhouette: distance to the object, only for pixels that are not object
    if o.any():
        dist = cv2.distanceTransform((~o).astype(np.uint8), cv2.DIST_L2, 3)
        band = np.clip(dist / max(1, edge_ramp_px // 2), 0, 1).astype(np.float32)
        w = np.where(o, w, np.minimum(w, band))
    return w


def keypoint_mask(mask: np.ndarray, hand: np.ndarray | None, margin_px: int) -> np.ndarray:
    """Object mask without the pixels within margin_px of the hand: a SuperPoint descriptor sees about 16-32 px
    around its keypoint, so near the fingers it describes the finger as much as the object."""
    if hand is None or margin_px <= 0 or not hand.any():
        return mask
    zone = cv2.dilate((hand > 0).astype(np.uint8), disk(margin_px)) > 0
    out = mask.copy()
    out[zone] = 0
    return out
