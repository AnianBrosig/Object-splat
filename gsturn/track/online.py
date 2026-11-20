"""Live learning: once the object is tracked, teach the map how it looks to this camera, during the session.

A background thread takes well-tracked, sharp frames (at most every `interval` seconds) and
  * samples SuperPoint descriptors at the visible 3D points from an upright crop of the object (descriptors
    depend on the in-plane rotation), at each map scale the object fits, and keeps the samples that agree
    with the map (points hidden by a hand do not);
  * blends them into the points' mean descriptors (webcam samples weigh `weight` times a capture sample) and
    writes the new means straight into the matcher, so relocalisation uses them from the next attempt on;
  * adds the frame as a keyframe when its view differs from the known ones, so the tracker's view-based
    candidate selection covers it.
`save()` writes the collected keyframes into the package (descriptors + new triangulated points, via
gsturn.track.refine), so the next session starts from the improved map.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .refine import _object_box, _sharpness, _visible_rows, apply, learn
from .tracker import Localizer, PoseTracker, State


class OnlineLearner:
    def __init__(self, tracker: PoseTracker, interval: float = 0.5, min_inliers: int = 40, weight: float = 2.0,
                 accept_sim: float = 0.5, novel_deg: float = 10.0, max_keyframes: int = 120, background: bool = True,
                 log=print):
        self.tr, self.loc, self.model, self.K = tracker, tracker.loc, tracker.model, tracker.K
        self.interval, self.min_inliers, self.weight = interval, min_inliers, weight
        self.accept_sim, self.novel_deg, self.max_keyframes, self.log = accept_sim, novel_deg, max_keyframes, log
        dev = self.loc.sp.device
        z = np.load(self.model.path)
        # one tensor per scale: index_add_ into a slice of an MPS tensor is silently dropped (torch 2.2)
        self.sum = [torch.from_numpy(x.astype(np.float32)).to(dev) for x in z["desc_sum"]]
        self.lock = self.loc.lock
        self.q: "queue.Queue" = queue.Queue(maxsize=1)
        self.last_offer = -1e9
        self.sharp: list[float] = []
        self.kfs: list[dict] = []                 # learned keyframes (kept for save())
        self.stats = {"frames": 0, "samples": 0, "keyframes": 0, "rejected_blur": 0}
        self.stop = False
        self.background = background
        self.thread = threading.Thread(target=self._run, daemon=True)
        if background:
            self.thread.start()

    # ---- called from the tracking loop (cheap)
    def offer(self, gray: np.ndarray, st: State, now: float) -> None:
        if (st.status != "track" or st.n_inliers < self.min_inliers or st.R_raw is None
                or now - self.last_offer < self.interval or not self.q.empty()):
            return
        self.last_offer = now
        if not self.background:                    # inline (deterministic tests)
            try:
                self._learn_one(gray, st.R_raw.copy(), st.t_raw.copy(), st.fid)
            except Exception as e:  # noqa: BLE001
                self.log(f"online learning skipped a frame: {type(e).__name__}: {e}")
            return
        try:
            self.q.put_nowait((gray, st.R_raw.copy(), st.t_raw.copy(), st.fid))
        except queue.Full:
            pass

    def close(self):
        self.stop = True
        if not self.background:
            return
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=5)

    # ---- background
    def _run(self):
        while not self.stop:
            item = self.q.get()
            if item is None:
                break
            try:
                self._learn_one(*item)
            except Exception as e:  # noqa: BLE001 - learning must never break tracking
                self.log(f"online learning skipped a frame: {type(e).__name__}: {e}")

    def _learn_one(self, gray, R, t, fid):
        m, K, loc, sp = self.model, self.K, self.loc, self.loc.sp
        box = _object_box(m, R, t, K, gray.shape)
        sh = _sharpness(gray, box)
        self.sharp.append(sh)
        if len(self.sharp) > 5 and sh < 0.6 * float(np.median(self.sharp[-50:])):
            self.stats["rejected_blur"] += 1
            return
        with self.lock:
            rows, uv = _visible_rows(m, R, t, K, gray.shape)
            angle = loc.inplane_angle(R, t)
        if len(rows) < 10:
            return
        obj_px = max(box[2] - box[0], box[3] - box[1]) / 1.15
        side = 640
        a = np.radians(angle)
        cw, ch = box[2] - box[0], box[3] - box[1]
        rot = max(abs(cw * np.cos(a)) + abs(ch * np.sin(a)), abs(cw * np.sin(a)) + abs(ch * np.cos(a)))
        dev = sp.device
        rel = np.array([obj_px / m.ref_size_px / s for s in m.scales])
        base = int(np.argmin(np.abs(np.log(rel))))
        samples = {}
        for si, s in enumerate(m.scales):
            f = s * m.ref_size_px / max(obj_px, 1)
            if f * rot > side - 1 or f > 2.0 or f < 0.2:
                continue                                # does not fit the fixed canvas (or pointless upsampling)
            canvas, valid, A = Localizer._canvas(gray, box, f, angle, (side, side))
            q = (uv @ A[:, :2].T + A[:, 2]).astype(np.float32)
            ok = (q[:, 0] > 4) & (q[:, 1] > 4) & (q[:, 0] < side - 5) & (q[:, 1] < side - 5)
            ok &= valid[np.clip(q[:, 1].astype(int), 0, side - 1), np.clip(q[:, 0].astype(int), 0, side - 1)] > 0
            if ok.sum() < 10:
                continue
            _, dm = sp.dense(canvas)
            samples[si] = (np.flatnonzero(ok), sp.sample(dm, torch.from_numpy(q[ok]).to(dev)))
        if not samples:
            return
        # accept points whose sample agrees with the map at the scale closest to the apparent size
        ref_si = base if base in samples else min(samples, key=lambda s: abs(s - base))
        idx, d = samples[ref_si]
        r_t = torch.from_numpy(rows[idx]).to(dev)
        with self.lock:
            cur = loc.M[ref_si].index_select(0, r_t).float()
        sim = (d * cur).sum(1).cpu().numpy()
        accepted = set(rows[idx][sim > self.accept_sim].tolist())
        if len(accepted) < 10:
            return
        touched = []
        for si, (idx_s, d_s) in samples.items():
            keep = np.array([rows[i] in accepted for i in idx_s])
            if not keep.any():
                continue
            rr = torch.from_numpy(rows[idx_s][keep]).to(dev)
            self.sum[si].index_add_(0, rr, self.weight * d_s[torch.from_numpy(np.flatnonzero(keep)).to(dev)])
            touched.append((si, rr))
        with self.lock:
            for si, rr in touched:
                new = torch.nn.functional.normalize(self.sum[si].index_select(0, rr), dim=1).to(loc.M.dtype)
                loc.M.index_put_((torch.full_like(rr, si), rr), new)
        self.stats["frames"] += 1
        self.stats["samples"] += int(sum(len(rr) for _, rr in touched))
        # a new keyframe if this view is new
        c = -R.T @ t
        d_view = c / np.linalg.norm(c)
        with self.lock:
            cos = loc.kf_dirs @ d_view
            near = np.flatnonzero(cos > np.cos(np.radians(self.novel_deg)))
            novel = not any(abs((loc.inplane_angle_kf(k, R) + 180) % 360 - 180) < 2 * self.novel_deg for k in near)
            if novel and len(self.kfs) < self.max_keyframes:
                pts = np.array(sorted(accepted), np.int64)
                m.kf_R = np.concatenate([m.kf_R, R[None]])
                m.kf_t = np.concatenate([m.kf_t, t[None]])
                m.kf_points.append(pts)
                loc.kf_dirs = np.vstack([loc.kf_dirs, d_view])
                self.kfs.append({"i": fid, "R": R, "t": t, "box": box, "dir": d_view, "angle": angle,
                                 "gray": gray})
                self.stats["keyframes"] += 1

    # ---- persist
    def save(self, path: Optional[Path] = None, tag: str = "live") -> Optional[dict]:
        """Write the learned keyframes into the package (offline-quality descriptors + new 3D points)."""
        if not self.kfs:
            self.log("online learning: no new keyframes to save")
            return None
        from .model import ObjectModel
        path = Path(path or self.model.path)
        base = path.with_name("object_model_base.npz")
        if not base.exists():
            import shutil
            shutil.copy(path, base)
        model = ObjectModel.load(path)                    # the stored package, without the live changes
        grays = {}
        kfs = []
        for n, f in enumerate(self.kfs):
            g = dict(f)
            g["i"] = n
            grays[n] = f["gray"]
            kfs.append(g)
        from .tracker import TrackerConfig
        loc = Localizer(model, self.K, TrackerConfig(device=self.loc.sp.device))
        add = learn(model, kfs, grays, self.K, loc, self.log)
        r = apply(path, path, kfs, add, f"{tag}@{time.strftime('%Y%m%d-%H%M%S')}")
        self.log(f"online learning saved to {path}: {r['keyframes']} keyframes, {r['new_points']} new points, "
                 f"{r['observations']} descriptor samples")
        return r
