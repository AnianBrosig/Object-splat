"""Relocalisation (SuperPoint -> 2D-3D matches -> PnP RANSAC) plus KLT tracking between relocalisations.

Per camera frame (PoseTracker.process):
  1. KLT: follow the current 2D-3D inliers from the previous frame, PnP with the previous pose as guess (~5 ms).
  2. If a relocalisation finished in the background, carry its inliers forward to the current frame with KLT
     (it was computed on an older frame) and merge them with the tracked points; RANSAC on the union picks
     whichever pose has more support.
  3. Start a new relocalisation when lost, when the tracked inliers run low, or every `reloc_interval` seconds,
     on a crop around the predicted object box (or the whole frame when lost).
  4. Smooth the pose with a constant-velocity alpha-beta filter.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

from .model import ObjectModel
from .superpoint import SuperPointDense

ROW_BUCKET = 2048        # candidate point sets are padded to a multiple of this
SCALES_PER_QUERY = 3     # map scales compared per relocalisation while tracking
LK_FB_ERR = 2.0          # forward-backward KLT consistency (px)
LK = dict(winSize=(25, 25), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


@dataclass
class TrackerConfig:
    device: str = "auto"
    max_keypoints: int = 1024        # SuperPoint keypoints when lost (whole frame)
    max_keypoints_roi: int = 512     # ... and on the crop around the tracked object
    reloc_side: int = 640            # longest side of the crop SuperPoint sees while tracking
    lost_side: int = 960             # longest side of the whole frame SuperPoint sees when lost
    roi_margin: float = 1.3          # crop = object box * margin
    prior_angle: float = 40.0        # while tracking, match only points seen from keyframes within this angle
    lost_angles: int = 8             # in-plane rotations tried when lost (SuperPoint tolerates only ~+-30 deg)
    lost_angles_per_try: int = 2     # ... per relocalisation attempt (each ~200 ms)
    ratio: float = 0.95              # descriptor distance ratio test (best / second-best point)
    min_sim: float = 0.55            # minimum cosine similarity of a match
    pnp_px: float = 6.0              # RANSAC reprojection threshold at full resolution
    min_inliers: int = 15            # to accept a relocalisation
    min_inlier_ratio: float = 0.2    # inliers / matches
    track_px: float = 4.0            # PnP threshold while tracking
    min_track: int = 12              # below this many KLT inliers the object counts as lost
    want_track: int = 60             # relocalise early when fewer inliers than this
    reloc_interval: float = 0.3      # seconds between relocalisations while tracking
    max_points: int = 500
    smooth: float = 0.25             # pose filter alpha at rest (1 = raw pose, lower = smoother)
    coast_frames: int = 6            # keep the last pose this many frames after KLT fails, while relocalising
    sync: bool = False               # run relocalisation inline (deterministic, slower) instead of in a thread


@dataclass
class Reloc:
    ok: bool
    fid: int = -1
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    n_kp: int = 0
    n_matches: int = 0
    n_inliers: int = 0
    ms: float = 0.0
    roi: Optional[tuple] = None


@dataclass
class State:
    fid: int
    status: str                       # "track" | "coast" (pose kept briefly after losing it) | "lost"
    R: Optional[np.ndarray] = None    # object -> camera, filtered
    t: Optional[np.ndarray] = None
    R_raw: Optional[np.ndarray] = None
    t_raw: Optional[np.ndarray] = None
    n_inliers: int = 0
    pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # 3D point rows of pts
    reloc: Optional[Reloc] = None     # relocalisation that finished on this frame, if any
    track_ms: float = 0.0


# --------------------------------------------------------------------------- relocalisation
class Localizer:
    def __init__(self, model: ObjectModel, K: np.ndarray, cfg: TrackerConfig):
        self.model, self.K, self.cfg = model, K.astype(np.float64), cfg
        self.sp = SuperPointDense(cfg.device)
        dev = self.sp.device
        S, P, D = model.desc.shape
        self.S, self.P = S, P
        self.dtype = torch.float16 if dev != "cpu" else torch.float32
        # (S, P + 1, 256): the extra all-zero row pads candidate sets to fixed sizes. MPS compiles kernels per
        # tensor shape (~200 ms each), so every shape seen at run time must come from a small fixed set.
        self.M = torch.from_numpy(np.concatenate([model.desc, np.zeros((S, 1, D), np.float32)], 1)).to(dev, self.dtype)
        self.pad_row = P
        self.pts3d = model.points.astype(np.float64)
        self.kf_dirs = model.kf_dirs
        # in-plane reference: only keyframes of the capture video. Descriptors are always sampled in their
        # orientation (refine/online learning rotate webcam crops to match them), so rotating a crop to match a
        # webcam keyframe held sideways would rotate it away from the descriptors.
        self.ref_kf = np.flatnonzero(np.asarray(model.kf_src_index) >= 0)
        if len(self.ref_kf) == 0:
            self.ref_kf = np.arange(len(model.kf_names))
        self.scales = np.asarray(model.scales, np.float64)
        self.lock = threading.RLock()    # the map (M, keyframes) may be updated live by OnlineLearner
        self.last_angle = 0.0            # in-plane rotation that worked last (lost search starts there)
        self._sweep: list = []           # rotations still to try in the current lost search
        self._sweep_from = 0.0

    def warmup(self, frame_shape: tuple[int, int]):
        H, W = frame_shape[:2]
        rng = np.random.default_rng(0)
        g = rng.integers(0, 255, (H, W), np.uint8)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        for a in self.lost_angle_list():
            self.localize(g, -1, angles=[a])
        self.localize(g, -1, (W * 0.25, H * 0.25, W * 0.75, H * 0.75))
        n_sc = min(SCALES_PER_QUERY, len(self.scales))
        for nq in (512, 1024):
            d = torch.nn.functional.normalize(torch.randn(nq, 256, device=self.sp.device), dim=1)
            for b in range(ROW_BUCKET, self.P + ROW_BUCKET + 1, ROW_BUCKET):
                self.match(d, rng.integers(0, self.P, b - 7), np.arange(n_sc))
            self.match(d)

    def lost_angle_list(self) -> list[float]:
        n = max(1, self.cfg.lost_angles)
        return [i * 360.0 / n for i in range(n)]

    def inplane_angle_kf(self, k: int, R: np.ndarray) -> float:
        Rr = R @ self.model.kf_R[k].T
        return float(np.degrees(np.arctan2(Rr[1, 0], Rr[0, 0])))

    def inplane_angle(self, R: np.ndarray, t: np.ndarray) -> float:
        """Image rotation (degrees, OpenCV sign) that makes the object look as in the nearest keyframe."""
        c = -R.T @ t
        with self.lock:
            ref = self.ref_kf
            k = int(ref[np.argmax(self.kf_dirs[ref] @ (c / np.linalg.norm(c)))])
            return self.inplane_angle_kf(k, R)

    @staticmethod
    def _canvas(gray, box, f: float, angle: float, size=None):
        """Scale box (x0, y0, x1, y1) of gray by f, rotate by angle (deg, OpenCV sign) about its centre.
        Returns canvas, keypoint mask and the 2x3 affine full-frame -> canvas. size=None: tight canvas."""
        x0, y0, x1, y1 = box
        crop = gray[y0:y1, x0:x1]
        small = crop if f == 1.0 else cv2.resize(crop, None, fx=f, fy=f,
                                                 interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        h, w = small.shape
        a = np.radians(angle)
        cw = abs(w * np.cos(a)) + abs(h * np.sin(a))
        ch = abs(w * np.sin(a)) + abs(h * np.cos(a))
        if size is None:
            size = (int(np.ceil(cw / 8)) * 8, int(np.ceil(ch / 8)) * 8)
        Wc, Hc = size
        Rm = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
        Rm[:, 2] += [(Wc - 1) / 2 - (w - 1) / 2, (Hc - 1) / 2 - (h - 1) / 2]
        if angle % 360 == 0 and (w, h) == (Wc, Hc):
            canvas = small
        else:
            canvas = cv2.warpAffine(small, Rm, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine(np.ones((h, w), np.uint8), Rm, (Wc, Hc), flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
        S = np.array([[f, 0, -f * x0 + (f - 1) * 0.5], [0, f, -f * y0 + (f - 1) * 0.5], [0, 0, 1]])
        A = (np.vstack([Rm, [0, 0, 1]]) @ S)[:2]
        return canvas, valid, A

    def candidates(self, R, t, obj_px: float):
        """Point rows seen from keyframes close to the prior viewing direction, and scales near the object size."""
        c = -R.T @ t
        d = c / np.linalg.norm(c)
        with self.lock:
            cos = self.kf_dirs @ d
            near = np.flatnonzero(cos > np.cos(np.radians(self.cfg.prior_angle)))
            if len(near) < 4:
                near = np.argsort(-cos)[:4]
            rows = np.unique(np.concatenate([self.model.kf_points[k] for k in near]))
        rel = obj_px / self.model.ref_size_px            # apparent size relative to the keyframes
        dist = np.abs(np.log(self.scales / max(rel, 1e-3)))
        n = min(SCALES_PER_QUERY, len(self.scales))
        sc = np.sort(np.argsort(dist)[:n])
        return rows, sc

    def match(self, desc: torch.Tensor, rows=None, scales=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query descriptors -> (query idx, point row, similarity): max over scales, ratio test, one query per point."""
        if len(desc) < 2:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        n = len(desc)
        with torch.inference_mode(), self.lock:
            M = self.M
            if scales is not None:
                M = M[torch.from_numpy(scales).to(M.device)]
            if rows is not None:
                pad = -len(rows) % ROW_BUCKET
                rows = np.concatenate([rows, np.full(pad, self.pad_row, rows.dtype)])
                M = M[:, torch.from_numpy(rows).to(M.device)]
            else:
                rows = np.arange(self.M.shape[1])
            S, P, D = M.shape
            q = desc.to(M.dtype)
            nq = next(b for b in (512, 1024, 2048, 4096, 1 << 30) if n <= b)
            if nq > n:
                q = torch.cat([q, torch.zeros(nq - n, D, device=q.device, dtype=q.dtype)])
            sim = (q @ M.reshape(S * P, D).T).view(nq, S, P).amax(1)
            top, idx = sim.float().topk(2, dim=1)
        top, idx = top[:n].cpu().numpy(), idx[:n].cpu().numpy()
        d1 = np.sqrt(np.clip(2 - 2 * top[:, 0], 0, None))
        d2 = np.sqrt(np.clip(2 - 2 * top[:, 1], 0, None))
        ok = (d1 < self.cfg.ratio * d2) & (top[:, 0] > self.cfg.min_sim)
        q = np.flatnonzero(ok)
        p = idx[q, 0]
        order = np.argsort(-top[q, 0])
        _, first = np.unique(p[order], return_index=True)
        keep = order[first]
        p = rows[p[keep]]
        good = p != self.pad_row
        return q[keep][good], p[good], top[q[keep][good], 0]

    def localize(self, gray: np.ndarray, fid: int, roi: Optional[tuple] = None, prior=None,
                 angles: Optional[list] = None) -> Reloc:
        """roi: (x0, y0, x1, y1) crop around the predicted object; prior: (R, t) to restrict the map and to
        rotate the crop upright. Without roi the whole frame is searched at several in-plane rotations
        (`angles`, default: all, starting from the last one that worked) until one gives a pose."""
        t0 = time.perf_counter()
        H, W = gray.shape
        if roi is None:
            if angles is None:
                # a few rotations per call, on the newest frame; the sweep continues on the next call and
                # restarts from the last angle that worked once something was found
                if not self._sweep or self._sweep_from != self.last_angle:
                    n = len(self.lost_angle_list())
                    step = 360.0 / n
                    start = int(round(self.last_angle / step)) % n
                    order = [start] + [(start + s * d) % n for d in range(1, n) for s in (1, -1)]
                    self._sweep = [self.lost_angle_list()[i] for i in dict.fromkeys(order)]
                    self._sweep_from = self.last_angle
                k = max(1, self.cfg.lost_angles_per_try)
                angles, self._sweep = self._sweep[:k], self._sweep[k:]
            best = None
            for ang in angles:
                r = self._localize_once(gray, fid, (0, 0, W, H), None, ang, t0)
                if r.ok:
                    self.last_angle = ang
                    self._sweep = []
                    return r
                best = r if best is None or r.n_matches > best.n_matches else best
            best.ms = (time.perf_counter() - t0) * 1000
            return best
        x0, y0, x1, y1 = (int(round(v)) for v in roi)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 32 or y1 - y0 < 32:
            return Reloc(False, fid, roi=roi)
        ang = 0.0
        if prior is not None:
            ang = self.inplane_angle(*prior)
            step = 360.0 / max(1, self.cfg.lost_angles)
            self.last_angle = (round(ang / step) * step) % 360
        return self._localize_once(gray, fid, (x0, y0, x1, y1), prior, ang, t0, roi)

    def _localize_once(self, gray, fid, box, prior, angle, t0, roi=None) -> Reloc:
        x0, y0, x1, y1 = box
        cw, chh = x1 - x0, y1 - y0
        a = np.radians(angle)
        rw = abs(cw * np.cos(a)) + abs(chh * np.sin(a))       # rotated box size before scaling
        rh = abs(cw * np.sin(a)) + abs(chh * np.cos(a))
        rows = scales = None
        if roi is None:
            f = min(1.0, self.cfg.lost_side / max(rw, rh))
            size, nkp = None, self.cfg.max_keypoints
        else:
            # scale so the object appears at ~0.7x its keyframe size, within the fixed canvas
            obj_px = max(roi[2] - roi[0], roi[3] - roi[1]) / self.cfg.roi_margin
            f = float(np.clip(0.7 * self.model.ref_size_px / max(obj_px, 1), 0.25, 2.0))
            side = self.cfg.reloc_side
            f = min(f, (side - 1) / max(rw, rh))
            size, nkp = (side, side), self.cfg.max_keypoints_roi
            if prior is not None:
                rows, scales = self.candidates(prior[0], prior[1], obj_px * f)
        canvas, valid, A = self._canvas(gray, box, f, angle, size)
        kp, _, desc = self.sp.detect(canvas, nkp, valid)
        qi, pi, _ = self.match(desc, rows, scales)
        res = Reloc(False, fid, n_kp=len(kp), n_matches=len(qi), roi=box)
        if len(qi) >= max(6, self.cfg.min_inliers):
            Ai = cv2.invertAffineTransform(A)
            img = (kp[qi] @ Ai[:, :2].T + Ai[:, 2]).astype(np.float32)
            obj = self.pts3d[pi]
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img.astype(np.float64), self.K, None,
                                                     iterationsCount=1000, reprojectionError=self.cfg.pnp_px,
                                                     confidence=0.999, flags=cv2.SOLVEPNP_AP3P)
            if ok and inl is not None and len(inl) >= self.cfg.min_inliers:
                inl = inl.ravel()
                rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl].astype(np.float64), self.K, None, rvec, tvec)
                inl = inliers(obj, img, rvec, tvec, self.K, self.cfg.pnp_px * 0.75)
                ratio = len(inl) / len(qi)
                if len(inl) >= self.cfg.min_inliers and ratio >= self.cfg.min_inlier_ratio and tvec[2, 0] > 0:
                    res.ok = True
                    res.R, res.t = cv2.Rodrigues(rvec)[0], tvec.ravel()
                    res.pts, res.ids = img[inl].astype(np.float32), pi[inl]
                    res.n_inliers = len(inl)
        res.ms = (time.perf_counter() - t0) * 1000
        return res


def inliers(obj, img, rvec, tvec, K, thr) -> np.ndarray:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    return np.flatnonzero(err < thr)


def lk_forward_backward(g0, g1, p0, max_err=None):
    if len(p0) == 0:
        return p0, np.zeros(0, bool)
    max_err = LK_FB_ERR if max_err is None else max_err
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0.reshape(-1, 1, 2), None, **LK)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **LK)
    fb = np.linalg.norm(pb.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < max_err)
    return p1.reshape(-1, 2), ok


# --------------------------------------------------------------------------- pose filter
class PoseFilter:
    """Adaptive exponential smoothing (in the spirit of the One-Euro filter): strong smoothing while the object is
    nearly still (removes PnP jitter), none during fast motion (no lag). `smooth` is the alpha used at rest."""

    def __init__(self, smooth=0.25, rot_fast_deg=3.0, trans_fast_rel=0.01):
        self.a0, self.rf, self.tf = smooth, np.radians(rot_fast_deg), trans_fast_rel
        self.R = self.t = None

    def reset(self):
        self.R = self.t = None

    def update(self, R, t, now=None, scale=None):
        if self.R is None:
            self.R, self.t = R.copy(), t.copy()
            return self.R, self.t
        dr = cv2.Rodrigues(R @ self.R.T)[0].ravel()
        dt = t - self.t
        motion = max(np.linalg.norm(dr) / self.rf, np.linalg.norm(dt) / (self.tf * np.linalg.norm(t)))
        a = self.a0 + (1 - self.a0) * min(1.0, motion)
        self.R = cv2.Rodrigues(a * dr)[0] @ self.R
        self.t = self.t + a * dt
        return self.R, self.t


# --------------------------------------------------------------------------- orchestration
class PoseTracker:
    def __init__(self, model: ObjectModel, K: np.ndarray, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.model, self.K = model, K.astype(np.float64)
        self.loc = Localizer(model, self.K, self.cfg)
        self.pts3d = model.points.astype(np.float64)
        self.scale = float(np.linalg.norm(model.extent))
        self.filter = PoseFilter(self.cfg.smooth)
        self.coast = 0                       # frames since tracking failed (pose kept for coast_frames)
        self.R_last = self.t_last = None
        self.pool = ThreadPoolExecutor(1)
        self.pending: Optional[Future] = None
        self.frames: deque = deque(maxlen=120)
        self.fid = -1
        self.pts = np.zeros((0, 2), np.float32)
        self.ids = np.zeros(0, np.int64)
        self.R = self.t = None
        self.last_reloc = -1e9
        self.force = False
        self.stats = {"reloc": 0, "reloc_ok": 0}

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def warmup(self, frame_shape: tuple[int, int]):
        """Compile every MPS kernel shape used at run time (~2-4 s once instead of stalls while tracking)."""
        self.loc.warmup(frame_shape)

    # ---- helpers
    def _pnp(self, pts, ids, R0=None, t0=None, thr=None, iters=100):
        thr = thr or self.cfg.track_px
        if len(pts) < 6:
            return None
        obj = self.pts3d[ids]
        img = pts.astype(np.float64)
        if R0 is not None:
            rvec, tvec = cv2.Rodrigues(R0)[0], t0.reshape(3, 1).astype(np.float64)
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, self.K, None, rvec.copy(), tvec.copy(), True,
                                                     iters, thr, 0.999, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, self.K, None, iterationsCount=iters,
                                                     reprojectionError=thr, confidence=0.999,
                                                     flags=cv2.SOLVEPNP_SQPNP)
        if not ok or inl is None or len(inl) < 6:
            return None
        inl = inl.ravel()
        rvec, tvec = cv2.Rodrigues(cv2.Rodrigues(rvec)[0])[0], tvec   # normalise
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
        inl = inliers(obj, img, rvec, tvec, self.K, thr)
        if tvec[2, 0] <= 0:
            return None
        return cv2.Rodrigues(rvec)[0], tvec.ravel(), inl

    def object_box(self, R, t, margin=1.3):
        proj, _ = cv2.projectPoints(self.model.hull.astype(np.float64), cv2.Rodrigues(R)[0], t, self.K, None)
        p = proj.reshape(-1, 2)
        c, half = (p.max(0) + p.min(0)) / 2, (p.max(0) - p.min(0)) / 2 * margin
        half = np.maximum(half, 24)
        return (c[0] - half[0], c[1] - half[1], c[0] + half[0], c[1] + half[1])

    def _forward(self, res: Reloc, gray) -> tuple[np.ndarray, np.ndarray]:
        """Carry a relocalisation's inliers from its frame to the current frame with chained KLT."""
        chain = [g for f, g in self.frames if f >= res.fid]
        pts, ids = res.pts.copy(), res.ids.copy()
        if not chain or self.frames[0][0] > res.fid:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.int64)
        for g0, g1 in zip(chain[:-1], chain[1:]):
            pts, ok = lk_forward_backward(g0, g1, pts)
            pts, ids = pts[ok], ids[ok]
        return pts, ids

    # ---- main entry
    def process(self, bgr: np.ndarray, now: Optional[float] = None) -> State:
        now = time.perf_counter() if now is None else now
        t0 = time.perf_counter()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        prev = self.frames[-1][1] if self.frames else None
        self.fid += 1
        self.frames.append((self.fid, gray))
        cfg = self.cfg
        n_inl = 0

        # 1. KLT tracking
        if self.R is not None and prev is not None and len(self.pts):
            pts, ok = lk_forward_backward(prev, gray, self.pts)
            pts, ids = pts[ok], self.ids[ok]
            r = self._pnp(pts, ids, self.R, self.t)
            if r is None or len(r[2]) < cfg.min_track:
                r = self._pnp(pts, ids, iters=200)          # the prior may be off after a fast move
            if r is not None and len(r[2]) >= cfg.min_track:
                self.R, self.t, inl = r
                self.pts, self.ids = pts[inl].astype(np.float32), ids[inl]
                n_inl = len(inl)
            else:
                self.R_last, self.t_last = self.R, self.t
                self.R = self.t = None
                self.coast = 0
                self.pts, self.ids = np.zeros((0, 2), np.float32), np.zeros(0, np.int64)

        # 2. collect a finished relocalisation
        done = None
        if self.pending is not None and self.pending.done():
            done = self.pending.result()
            self.pending = None
            self._merge(done, gray)
            n_inl = len(self.ids) if self.R is not None else 0

        # 3. schedule the next one
        due = self.R is None or n_inl < cfg.want_track or now - self.last_reloc > cfg.reloc_interval or self.force
        if self.pending is None and due:
            if self.R is not None:
                roi = self.object_box(self.R, self.t, cfg.roi_margin)
                prior = (self.R.copy(), self.t.copy())
            elif self.R_last is not None and self.coast < cfg.coast_frames:
                roi = self.object_box(self.R_last, self.t_last, cfg.roi_margin * 1.3)   # just lost: search nearby
                prior = (self.R_last.copy(), self.t_last.copy())
            else:
                roi = prior = None
            self.last_reloc = now
            self.force = False
            self.stats["reloc"] += 1
            if cfg.sync:
                done = self.loc.localize(gray, self.fid, roi, prior)
                self._merge(done, gray)
                n_inl = len(self.ids) if self.R is not None else 0
            else:
                self.pending = self.pool.submit(self.loc.localize, gray, self.fid, roi, prior)

        # 4. filter + state
        st = State(self.fid, "track" if self.R is not None else "lost", reloc=done)
        if self.R is not None:
            self.coast = 0
            self.R_last = None
            st.R_raw, st.t_raw = self.R.copy(), self.t.copy()
            st.R, st.t = self.filter.update(self.R, self.t)
            st.n_inliers, st.pts, st.ids = len(self.ids), self.pts.copy(), self.ids.copy()
        elif self.R_last is not None and self.coast < cfg.coast_frames:
            self.coast += 1
            st.status = "coast"
            st.R, st.t = self.filter.R, self.filter.t
            st.R_raw, st.t_raw = self.R_last, self.t_last
        else:
            self.filter.reset()
        st.track_ms = (time.perf_counter() - t0) * 1000
        return st

    def _merge(self, res: Reloc, gray):
        if not res.ok:
            return
        self.stats["reloc_ok"] += 1
        pts, ids = self._forward(res, gray)
        if self.R is not None and len(self.ids):
            new = ~np.isin(self.ids, ids)
            pts = np.vstack([pts, self.pts[new]])
            ids = np.concatenate([ids, self.ids[new]])
        # pose guess: the relocalised pose, unless the tracker has a better-supported one (RANSAC decides)
        cands = [(res.R, res.t)] + ([(self.R, self.t)] if self.R is not None else [])
        best = None
        for R0, t0 in cands:
            r = self._pnp(pts, ids, R0, t0)
            if r is not None and (best is None or len(r[2]) > len(best[2])):
                best = r
        if best is None or len(best[2]) < self.cfg.min_track:
            return
        R, t, inl = best
        if len(inl) > self.cfg.max_points:
            inl = inl[np.random.default_rng(0).choice(len(inl), self.cfg.max_points, replace=False)]
        self.R, self.t = R, t
        self.pts, self.ids = pts[inl].astype(np.float32), ids[inl]
