"""Refine an object package with footage from the live camera (webcam).

The capture video taught the map how the object looks to the phone, in the orientations it was held in. A
webcam recording of the same object adds what the live camera sees:

1. track the recording with the current map (poses for every frame where it works);
2. optionally estimate the camera focal length from the object itself (2D-3D inliers of many frames: the object
   is a known 3D shape, so this needs no chessboard);
3. for well-tracked, sharp, diverse frames ("webcam keyframes"): sample SuperPoint descriptors at the projected
   3D points from an upright crop (descriptors depend on the in-plane rotation) at every map scale, and add them
   to the points' descriptor sums when they agree with the map (occluded points do not);
4. triangulate new 3D points from keypoints the map does not explain, matched between webcam keyframes with
   known poses (fills surfaces the capture video saw poorly or never);
5. add the webcam keyframes (pose + visible points), so the tracker's view-based candidate selection covers them.

Repeat (passes): the refined map tracks more of the recording, which gives more keyframes to learn from.
The original package is kept as object_model_base.npz.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from .model import ObjectModel, rot_err_deg
from .tracker import Localizer, PoseTracker, TrackerConfig, inliers


# --------------------------------------------------------------------------- 1. tracking a recording
def track_video(model: ObjectModel, video: Path, K: np.ndarray, use: Callable[[int], bool] = lambda i: True,
                log=print) -> list[dict]:
    """Per frame: status, pose, and the fresh SuperPoint inliers of relocalisations (exact detections)."""
    cfg = TrackerConfig(sync=True, reloc_interval=0.0)       # relocalise every frame: best poses, slower
    tr = PoseTracker(model, K, cfg)
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tr.warmup((H, W))
    out, i, t0 = [], 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not use(i):
            tr.R = tr.t = tr.R_last = None                       # held-out frame: start fresh after it
            tr.pts, tr.ids = np.zeros((0, 2), np.float32), np.zeros(0, np.int64)
            out.append({"i": i, "status": "skip"})
            i += 1
            continue
        st = tr.process(frame, now=i / 30.0)
        rec = {"i": i, "status": st.status}
        r = st.reloc
        if st.status == "track" and r is not None and r.ok and r.fid == st.fid:
            rec.update(R=r.R, t=r.t, pts=r.pts, ids=r.ids)
        out.append(rec)
        i += 1
        if i % 200 == 0:
            log(f"  tracked {i} frames ({sum(o['status'] == 'track' for o in out)} with pose, "
                f"{(time.perf_counter() - t0) / i * 1000:.0f} ms/frame)")
    tr.close()
    return out


# --------------------------------------------------------------------------- 2. focal from the object
def estimate_focal(model: ObjectModel, frames: list[dict], size: tuple[int, int], K0: np.ndarray,
                   n: int = 60, log=print) -> Optional[dict]:
    """Pinhole focal length (principal point fixed at the centre) from 2D-3D inliers of many frames."""
    W, H = size
    cand = [f for f in frames if "pts" in f and len(f["pts"]) >= 40]
    if len(cand) < 10:
        log("  focal: too few well-tracked frames")
        return None
    sel = [cand[i] for i in np.linspace(0, len(cand) - 1, min(n, len(cand))).astype(int)]
    obj = [model.points[f["ids"]].astype(np.float32) for f in sel]
    img = [f["pts"].reshape(-1, 1, 2).astype(np.float32) for f in sel]
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    Kc = np.array([[K0[0, 0], 0, (W - 1) / 2], [0, K0[0, 0], (H - 1) / 2], [0, 0, 1.0]])

    def solve(idx):
        rms, K, _, _, _ = cv2.calibrateCamera([obj[i] for i in idx], [img[i] for i in idx], (W, H), Kc.copy(),
                                              np.zeros(5), flags=flags)
        return rms, K
    rms, K = solve(range(len(sel)))
    rng = np.random.default_rng(0)
    boot = [solve(rng.choice(len(sel), len(sel), replace=True))[1][0, 0] for _ in range(10)]
    spread = float(np.std(boot) / K[0, 0])
    log(f"  focal from the object: {K[0, 0]:.0f} px (hfov {np.degrees(2 * np.arctan(W / 2 / K[0, 0])):.1f} deg, "
        f"rms {rms:.2f} px, bootstrap spread {100 * spread:.1f} %, {len(sel)} frames)")
    return {"K": K, "rms": float(rms), "spread": spread, "frames": len(sel)}


# --------------------------------------------------------------------------- 3-5. learning
def _sharpness(gray, box) -> float:
    x0, y0, x1, y1 = box
    c = gray[y0:y1, x0:x1]
    return float(cv2.Laplacian(c, cv2.CV_32F).var()) if c.size else 0.0


def select_keyframes(model: ObjectModel, frames: list[dict], grays: dict, K, loc: Localizer,
                     max_n: int = 80, min_inl: int = 30, min_angle: float = 8.0) -> list[dict]:
    """Well-tracked, sharp frames whose viewing direction or in-plane angle differs from those already chosen."""
    cand = [f for f in frames if "pts" in f and len(f["pts"]) >= min_inl and f["i"] in grays]
    for f in cand:
        box = _object_box(model, f["R"], f["t"], K, grays[f["i"]].shape)
        f["box"], f["sharp"] = box, _sharpness(grays[f["i"]], box)
        c = -f["R"].T @ f["t"]
        f["dir"] = c / np.linalg.norm(c)
        f["angle"] = loc.inplane_angle(f["R"], f["t"])
    if not cand:
        return []
    thr = 0.5 * np.median([f["sharp"] for f in cand])
    cand = sorted([f for f in cand if f["sharp"] >= thr], key=lambda f: -len(f["pts"]))
    chosen: list[dict] = []
    for f in cand:
        if all(np.degrees(np.arccos(np.clip(f["dir"] @ g["dir"], -1, 1))) > min_angle
               or abs((f["angle"] - g["angle"] + 180) % 360 - 180) > 2 * min_angle for g in chosen):
            chosen.append(f)
        if len(chosen) >= max_n:
            break
    return sorted(chosen, key=lambda f: f["i"])


def _object_box(model, R, t, K, shape, margin=1.15):
    proj, _ = cv2.projectPoints(model.hull.astype(np.float64), cv2.Rodrigues(R)[0], t, K, None)
    p = proj.reshape(-1, 2)
    c, h = (p.max(0) + p.min(0)) / 2, (p.max(0) - p.min(0)) / 2 * margin
    H, W = shape
    return (int(max(0, c[0] - h[0])), int(max(0, c[1] - h[1])), int(min(W, c[0] + h[0])), int(min(H, c[1] + h[1])))


def _visible_rows(model: ObjectModel, R, t, K, shape, angle_deg=55.0) -> tuple[np.ndarray, np.ndarray]:
    """Map points that should be visible: seen from keyframes with a similar view, in front, inside the image."""
    c = -R.T @ t
    d = c / np.linalg.norm(c)
    near = np.flatnonzero(model.kf_dirs @ d > np.cos(np.radians(angle_deg)))
    if len(near) == 0:
        return np.zeros(0, int), np.zeros((0, 2))
    rows = np.unique(np.concatenate([model.kf_points[k] for k in near]))
    pc = model.points[rows] @ R.T + t
    uv = pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + K[:2, 2]
    H, W = shape
    ok = (pc[:, 2] > 0) & (uv[:, 0] > 4) & (uv[:, 1] > 4) & (uv[:, 0] < W - 5) & (uv[:, 1] < H - 5)
    return rows[ok], uv[ok]


def learn(model: ObjectModel, kfs: list[dict], grays: dict, K: np.ndarray, loc: Localizer, log=print,
          accept_sim: float = 0.5) -> dict:
    """Descriptor observations for existing points + new triangulated points. Returns the additions."""
    sp, dev = loc.sp, loc.sp.device
    S = len(model.scales)
    desc_map = torch.from_numpy(model.desc).to(dev)               # (S, P, 256) normalised means
    obs_rows, new_kp = [], []
    hull_lo, hull_hi = model.hull.min(0), model.hull.max(0)
    pad = 0.08 * (hull_hi - hull_lo)

    def canvases(f):
        """Upright crops of the object at each map scale: (scale index, canvas, affine)."""
        g, box = grays[f["i"]], f["box"]
        obj_px = max(box[2] - box[0], box[3] - box[1]) / 1.15
        out = []
        for si, s in enumerate(model.scales):
            fs = float(np.clip(s * model.ref_size_px / max(obj_px, 1), 0.2, 2.0))
            canvas, valid, A = Localizer._canvas(g, box, fs, f["angle"], None)
            out.append((si, canvas, valid, A, fs * obj_px / model.ref_size_px))
        return out

    # ---- pass 1: associations with existing points, candidate keypoints for new points
    for n, f in enumerate(kfs):
        rows, uv = _visible_rows(model, f["R"], f["t"], K, grays[f["i"]].shape)
        cv = canvases(f)
        # nearest map scale to the apparent size decides acceptance
        base = int(np.argmin([abs(np.log(model.scales[si] / max(rel, 1e-3))) for si, _, _, _, rel in cv]))
        si, canvas, valid, A, _ = cv[base]
        q = (uv @ A[:, :2].T + A[:, 2]).astype(np.float32)
        hc, wc = canvas.shape
        inside = (q[:, 0] > 4) & (q[:, 1] > 4) & (q[:, 0] < wc - 5) & (q[:, 1] < hc - 5)
        inside &= valid[np.clip(q[:, 1].astype(int), 0, hc - 1), np.clip(q[:, 0].astype(int), 0, wc - 1)] > 0
        rows, q, uv = rows[inside], q[inside], uv[inside]
        _, dm = sp.dense(canvas)
        d = sp.sample(dm, torch.from_numpy(q).to(dev))
        sim = (d * desc_map[si, torch.from_numpy(rows).to(dev)]).sum(1).cpu().numpy()
        good = sim > accept_sim
        f["obs_rows"], f["obs_uv"] = rows[good], uv[good]
        obs_rows.append(int(good.sum()))
        # keypoints on the object that no visible map point explains -> candidates for new points
        hull_m = np.zeros(canvas.shape, np.uint8)
        hp, _ = cv2.projectPoints(model.hull.astype(np.float64), cv2.Rodrigues(f["R"])[0], f["t"], K, None)
        hq = hp.reshape(-1, 2) @ A[:, :2].T + A[:, 2]
        cv2.fillConvexPoly(hull_m, np.round(cv2.convexHull(hq.astype(np.float32))).astype(np.int32).reshape(-1, 2), 1)
        kp, _, kd = sp.detect(canvas, 1024, (hull_m & valid).astype(np.uint8))
        if len(kp) and len(q):
            dist = np.min(np.linalg.norm(kp[:, None] - q[None], axis=2), 1)
            free = dist > 4
        else:
            free = np.ones(len(kp), bool)
        Ai = cv2.invertAffineTransform(A)
        f["new_uv"] = kp[free] @ Ai[:, :2].T + Ai[:, 2]
        f["new_desc"] = kd[torch.from_numpy(np.flatnonzero(free)).to(dev)] if free.any() else kd[:0]
        new_kp.append(int(free.sum()))
    log(f"  {len(kfs)} webcam keyframes: {int(np.median(obs_rows)) if obs_rows else 0} map points confirmed "
        f"per frame (median), {int(np.median(new_kp)) if new_kp else 0} unexplained keypoints per frame")

    # ---- new points: match unexplained keypoints between keyframes with known poses, triangulate
    Kinv = np.linalg.inv(K)
    tracks = []                                                    # list of {(frame idx, kp idx)}
    owner: dict = {}
    for a in range(len(kfs)):
        for b in range(a + 1, len(kfs)):
            fa, fb = kfs[a], kfs[b]
            ang = np.degrees(np.arccos(np.clip(fa["dir"] @ fb["dir"], -1, 1)))
            if not (5 <= ang <= 40) or len(fa["new_uv"]) < 5 or len(fb["new_uv"]) < 5:
                continue
            sim = (fa["new_desc"] @ fb["new_desc"].T).float().cpu().numpy()
            ia = sim.argmax(1)
            ib = sim.argmax(0)
            mutual = np.flatnonzero((ib[ia] == np.arange(len(ia))) & (sim[np.arange(len(ia)), ia] > 0.8))
            if len(mutual) < 5:
                continue
            ka, kb = mutual, ia[mutual]
            Pa = K @ np.column_stack([fa["R"], fa["t"]])
            Pb = K @ np.column_stack([fb["R"], fb["t"]])
            X = cv2.triangulatePoints(Pa, Pb, fa["new_uv"][ka].T.astype(np.float64),
                                      fb["new_uv"][kb].T.astype(np.float64))
            X = (X[:3] / X[3]).T
            ok = np.ones(len(X), bool)
            for f_, k_ in ((fa, ka), (fb, kb)):
                pc = X @ f_["R"].T + f_["t"]
                uv = pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + K[:2, 2]
                ok &= (pc[:, 2] > 0) & (np.linalg.norm(uv - f_["new_uv"][k_], axis=1) < 2.0)
            ok &= np.all((X > hull_lo - pad) & (X < hull_hi + pad), axis=1)      # on the object
            ra = X - (-fa["R"].T @ fa["t"])
            rb = X - (-fb["R"].T @ fb["t"])
            tri = np.degrees(np.arccos(np.clip(np.sum(ra * rb, 1) / (np.linalg.norm(ra, axis=1)
                                                                      * np.linalg.norm(rb, axis=1)), -1, 1)))
            ok &= tri > 3
            for m in np.flatnonzero(ok):
                ea, eb = (a, int(ka[m])), (b, int(kb[m]))
                ta, tb = owner.get(ea), owner.get(eb)
                if ta is None and tb is None:
                    tracks.append({ea, eb})
                    owner[ea] = owner[eb] = len(tracks) - 1
                elif ta is not None and tb is None:
                    tracks[ta].add(eb)
                    owner[eb] = ta
                elif tb is not None and ta is None:
                    tracks[tb].add(ea)
                    owner[ea] = tb
                elif ta != tb:                                     # merge
                    for e in tracks[tb]:
                        owner[e] = ta
                    tracks[ta] |= tracks[tb]
                    tracks[tb] = set()
    new_pts, new_obs = [], []
    for tr in tracks:
        frames_in = {e[0] for e in tr}
        if len(tr) < 3 or len(frames_in) != len(tr):              # >= 3 views, one keypoint per frame
            continue
        # multi-view linear triangulation, then a reprojection check in every view
        A_ = []
        for fi, ki in tr:
            f = kfs[fi]
            P = K @ np.column_stack([f["R"], f["t"]])
            u, v = f["new_uv"][ki]
            A_ += [u * P[2] - P[0], v * P[2] - P[1]]
        X = np.linalg.svd(np.array(A_))[2][-1]
        X = X[:3] / X[3]
        errs = []
        for fi, ki in tr:
            f = kfs[fi]
            pc = f["R"] @ X + f["t"]
            errs.append(np.linalg.norm(pc[:2] / pc[2] * [K[0, 0], K[1, 1]] + K[:2, 2] - f["new_uv"][ki]))
        if max(errs) < 2.5:
            new_pts.append(X)
            new_obs.append(sorted(tr))
    # drop new points that duplicate existing ones
    if new_pts:
        from scipy.spatial import cKDTree
        P_new = np.array(new_pts)
        dmin, _ = cKDTree(model.points).query(P_new)
        keep = dmin > 0.004 * np.linalg.norm(model.extent)
        new_pts = [p for p, k in zip(new_pts, keep) if k]
        new_obs = [o for o, k in zip(new_obs, keep) if k]
    log(f"  {len(new_pts)} new 3D points triangulated (>= 3 webcam views each)")

    # ---- pass 2: descriptors at every scale for confirmed and new points
    P0 = len(model.points)
    P1 = P0 + len(new_pts)
    add_sum = [torch.zeros(P1, 256, device=dev) for _ in range(S)]
    add_cnt = np.zeros((S, P1), np.int32)
    per_frame_new = {fi: [] for fi in range(len(kfs))}
    for j, obs in enumerate(new_obs):
        for fi, ki in obs:
            per_frame_new[fi].append((P0 + j, ki))
    kf_rows = []
    for fi, f in enumerate(kfs):
        rows = list(f["obs_rows"]) + [r for r, _ in per_frame_new[fi]]
        uv = np.vstack([f["obs_uv"].reshape(-1, 2)] + [f["new_uv"][[k for _, k in per_frame_new[fi]]].reshape(-1, 2)])
        rows = np.array(rows, np.int64)
        kf_rows.append(rows)
        if not len(rows):
            continue
        for si, canvas, valid, A, _ in canvases(f):
            q = (uv @ A[:, :2].T + A[:, 2]).astype(np.float32)
            hc, wc = canvas.shape
            ok = (q[:, 0] > 2) & (q[:, 1] > 2) & (q[:, 0] < wc - 3) & (q[:, 1] < hc - 3)
            if not ok.any():
                continue
            _, dm = sp.dense(canvas)
            d = sp.sample(dm, torch.from_numpy(q[ok]).to(dev))
            add_sum[si].index_add_(0, torch.from_numpy(rows[ok]).to(dev), d)
            np.add.at(add_cnt[si], rows[ok], 1)
    return {"points": np.array(new_pts, np.float32).reshape(-1, 3), "add_sum": torch.stack(add_sum).cpu().numpy(),
            "add_cnt": add_cnt, "kf_rows": kf_rows}


def apply(npz_path: Path, out_path: Path, kfs: list[dict], add: dict, tag: str) -> dict:
    """Write a new package with the additions merged in."""
    z = dict(np.load(npz_path, allow_pickle=False))
    P0 = len(z["points"])
    n_new = len(add["points"])
    S = z["desc_sum"].shape[0]
    dsum = np.concatenate([z["desc_sum"].astype(np.float32), np.zeros((S, n_new, 256), np.float32)], 1)
    dcnt = np.concatenate([z["desc_count"], np.zeros((S, n_new), z["desc_count"].dtype)], 1)
    dsum += add["add_sum"]
    dcnt += add["add_cnt"]
    z["desc_sum"], z["desc_count"] = dsum.astype(np.float16), dcnt
    z["points"] = np.concatenate([z["points"], add["points"]]).astype(np.float32)
    z["point_clean"] = np.concatenate([z["point_clean"], np.zeros(n_new, bool)])
    names = [f"{tag}:{f['i']}" for f in kfs]
    z["kf_names"] = np.concatenate([z["kf_names"].astype(str), np.array(names)])
    z["kf_R"] = np.concatenate([z["kf_R"], np.array([f["R"] for f in kfs]).reshape(-1, 3, 3)])
    z["kf_t"] = np.concatenate([z["kf_t"], np.array([f["t"] for f in kfs]).reshape(-1, 3)])
    z["kf_src_index"] = np.concatenate([z["kf_src_index"], -np.ones(len(kfs), z["kf_src_index"].dtype)])
    old = np.split(z["kf_pid"], z["kf_ptr"][1:-1])
    allp = old + [r.astype(z["kf_pid"].dtype) for r in add["kf_rows"]]
    z["kf_ptr"] = np.cumsum([0] + [len(x) for x in allp])
    z["kf_pid"] = np.concatenate(allp)
    meta = json.loads(str(z["meta"]))
    meta.setdefault("refined", []).append({"source": tag, "keyframes": len(kfs), "new_points": int(n_new),
                                           "observations": int(add["add_cnt"].sum()),
                                           "date": time.strftime("%Y-%m-%d %H:%M:%S")})
    meta["points"] = int(P0 + n_new)
    z["meta"] = json.dumps(meta)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **z)
    tmp.replace(out_path)
    return meta["refined"][-1]


# --------------------------------------------------------------------------- driver
def refine(model_path: Path, video: Path, K: np.ndarray, passes: int = 2, calibrate: bool = True,
           use: Callable[[int], bool] = lambda i: True, calib_out: Optional[Path] = None,
           max_keyframes: int = 80, log=print) -> dict:
    model_path = Path(model_path)
    base = model_path.with_name("object_model_base.npz")
    if not base.exists():
        shutil.copy(model_path, base)
        log(f"original package kept as {base}")
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    summary = {"video": str(video), "passes": []}
    for p in range(passes):
        model = ObjectModel.load(model_path)
        log(f"pass {p + 1}/{passes}: tracking {video.name} with {len(model.points)} points, "
            f"{len(model.kf_names)} keyframes")
        frames = track_video(model, video, K, use, log)
        n_used = sum(f["status"] != "skip" for f in frames)
        n_track = sum(f["status"] == "track" for f in frames)
        log(f"  pose in {n_track}/{n_used} frames")
        info = {"tracked": n_track, "frames": n_used}
        if calibrate and p == 0:
            fc = estimate_focal(model, frames, (W, H), K, log=log)
            if fc is not None and fc["spread"] < 0.05:
                K = fc["K"]
                info["focal"] = float(K[0, 0])
                if calib_out is not None:
                    calib_out.parent.mkdir(parents=True, exist_ok=True)
                    calib_out.write_text(json.dumps({
                        "camera": "webcam (focal from the tracked object)", "source": str(video),
                        "width": W, "height": H, "model": "PINHOLE", "fx": round(float(K[0, 0]), 3),
                        "fy": round(float(K[1, 1]), 3), "cx": round(float(K[0, 2]), 3), "cy": round(float(K[1, 2]), 3),
                        "dist": [0.0, 0.0, 0.0, 0.0], "rms_px": round(fc["rms"], 4), "views": fc["frames"],
                        "method": "object self-calibration (gsturn refine-model)",
                        "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
                    log(f"  wrote {calib_out}")
                frames = track_video(model, video, K, use, log)    # re-track with the better focal
                log(f"  pose in {sum(f['status'] == 'track' for f in frames)}/{n_used} frames with the new focal")
            elif fc is not None:
                log("  focal estimate not stable enough; keeping the current intrinsics")
        # grey frames of the candidates only (memory)
        want = {f["i"] for f in frames if "pts" in f}
        grays = {}
        cap = cv2.VideoCapture(str(video))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                grays[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            i += 1
        cap.release()
        loc = Localizer(model, K, TrackerConfig())
        kfs = select_keyframes(model, frames, grays, K, loc, max_n=max_keyframes)
        if not kfs:
            log("  nothing to learn from")
            break
        add = learn(model, kfs, grays, K, loc, log)
        r = apply(model_path, model_path, kfs, add, f"{video.name}#p{p + 1}")
        log(f"  added {r['keyframes']} keyframes, {r['new_points']} points, {r['observations']} descriptor samples")
        info.update(r)
        summary["passes"].append(info)
        del loc
    summary["K"] = K.tolist()
    return summary
