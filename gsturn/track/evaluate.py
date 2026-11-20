"""Accuracy checks against the SfM poses.

held-out:  relocalise keyframe k with a map that excludes k and its neighbours (+-window keyframes), compare to
           the SfM pose. Run at several query scales (smaller = object further away / lower resolution camera).
sequence:  run the full live tracker over the job's 720p video and compare every keyframe frame to its SfM pose.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import ObjectModel, rot_err_deg, sample_image
from .tracker import Localizer, PoseTracker, TrackerConfig


def _errors(R, t, Rg, tg):
    return rot_err_deg(R, Rg), float(np.linalg.norm(t - tg) / np.linalg.norm(tg))


def held_out(model: ObjectModel, job_dir: Path, every: int = 10, window: int = 2, query_scales=(1.0, 0.5),
             cfg: TrackerConfig | None = None, log=print) -> dict:
    from .colmap_txt import read_model
    cfg = cfg or TrackerConfig()
    work = Path(job_dir) / "work"
    rec = read_model(work / "sfm" / "model_txt")
    by_stem = {im.name.rsplit(".", 1)[0]: im for im in rec.images.values()}
    row = {int(p): i for i, p in enumerate(rec.point_ids)}
    z = np.load(model.path)
    dsum0 = torch.from_numpy(z["desc_sum"].astype(np.float32))
    dcnt0 = z["desc_count"].astype(np.int64)
    loc = Localizer(model, model.K, cfg)
    for qs in query_scales:
        loc.warmup((round(model.image_size[1] * qs), round(model.image_size[0] * qs)))
    sp, dev = loc.sp, loc.sp.device
    frames = json.loads((work / "frames" / "frames.json").read_text())
    name_of = {f["name"].rsplit(".", 1)[0]: f["name"] for f in frames["frames"]}
    grays = {}

    def gray(stem):
        if stem not in grays:
            grays[stem] = cv2.imread(str(work / "frames" / "images" / name_of[stem]), cv2.IMREAD_GRAYSCALE)
        return grays[stem]

    rows = []
    n = len(model.kf_names)
    capture = [k for k in range(n) if model.kf_src_index[k] >= 0]   # skip keyframes added by refine-model
    for k in capture[::every]:
        excl = model.kf_names[max(0, k - window):k + window + 1]
        dsum, dcnt = dsum0.clone(), dcnt0.copy()
        for stem in excl:
            im = by_stem[stem]
            has = im.pid >= 0
            xy = im.xy[has] - 0.5
            rid = np.array([row[int(p)] for p in im.pid[has]])
            for si, (ok, d) in enumerate(sample_image(sp, gray(stem), xy, model.scales)):
                dsum[si].index_add_(0, torch.from_numpy(rid[ok]), -d.cpu().to(dsum.dtype))
                np.add.at(dcnt[si], rid[ok], -1)
        nrm = dsum.norm(dim=2, keepdim=True)
        desc = torch.where((torch.from_numpy(dcnt)[..., None] > 0) & (nrm > 1e-6), dsum / nrm.clamp_min(1e-8), 0)
        loc.M = torch.cat([desc, torch.zeros(desc.shape[0], 1, desc.shape[2])], 1).to(dev, loc.dtype)
        Rg, tg = model.kf_R[k], model.kf_t[k]
        for qs in query_scales:
            g = gray(model.kf_names[k])
            g = g if qs == 1.0 else cv2.resize(g, None, fx=qs, fy=qs, interpolation=cv2.INTER_AREA)
            Kq = model.K.copy()
            Kq[:2] *= qs
            Kq[0, 2] = (model.K[0, 2] + 0.5) * qs - 0.5
            Kq[1, 2] = (model.K[1, 2] + 0.5) * qs - 0.5
            loc.K = Kq
            for mode in ("full", "roi"):
                roi = None
                if mode == "roi":
                    proj, _ = cv2.projectPoints(model.hull.astype(np.float64), cv2.Rodrigues(Rg)[0], tg, Kq, None)
                    p = proj.reshape(-1, 2)
                    c, h = (p.max(0) + p.min(0)) / 2, (p.max(0) - p.min(0)) / 2 * 1.3
                    roi = (c[0] - h[0], c[1] - h[1], c[0] + h[0], c[1] + h[1])
                # roi mode: prior = ground truth perturbed by ~10 deg (what the tracker would predict)
                prior = (cv2.Rodrigues(np.radians([6.0, -6.0, 5.0]))[0] @ Rg, tg) if mode == "roi" else None
                r = loc.localize(g, 0, roi, prior)
                e = _errors(r.R, r.t, Rg, tg) if r.ok else (np.nan, np.nan)
                rows.append({"kf": model.kf_names[k], "scale": qs, "mode": mode, "ok": r.ok, "rot_deg": e[0],
                             "trans_rel": e[1], "inliers": r.n_inliers, "matches": r.n_matches, "kp": r.n_kp,
                             "ms": r.ms})
        last = [x for x in rows if x["kf"] == model.kf_names[k]]
        log("  " + model.kf_names[k] + "  " + "  ".join(
            f"{x['scale']}/{x['mode']}: " + (f"{x['rot_deg']:.2f}deg {100 * x['trans_rel']:.1f}% {x['inliers']}inl"
                                             if x["ok"] else f"FAIL ({x['matches']} m)") for x in last))
    summary = {}
    for qs in query_scales:
        for mode in ("full", "roi"):
            sel = [x for x in rows if x["scale"] == qs and x["mode"] == mode]
            good = [x for x in sel if x["ok"]]
            rot = np.array([x["rot_deg"] for x in good])
            tr = np.array([x["trans_rel"] for x in good])
            summary[f"scale {qs} {mode}"] = {
                "success": f"{len(good)}/{len(sel)}",
                "correct (<5deg, <5%)": int(np.sum((rot < 5) & (tr < 0.05))),
                "rot_deg_median": round(float(np.median(rot)), 2) if len(rot) else None,
                "trans_rel_median_pct": round(100 * float(np.median(tr)), 2) if len(tr) else None,
                "ms_median": round(float(np.median([x["ms"] for x in sel])), 1)}
    return {"summary": summary, "rows": rows}


def sequence(model: ObjectModel, video: Path, cfg: TrackerConfig | None = None, out_video: Path | None = None,
             src_index_offset: int = 0, log=print) -> dict:
    """Run the tracker on every frame of the job video; ground truth at frames that are keyframes."""
    from .draw import draw
    cfg = cfg or TrackerConfig()
    cap = cv2.VideoCapture(str(video))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
    tr = PoseTracker(model, model.K, cfg)
    tr.warmup((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    gt = {int(s): i for i, s in enumerate(model.kf_src_index)}
    writer = None
    rows, times, info = [], [], {}
    i = 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # pace like a live camera so the background relocalisation sees realistic latencies
        target = t_start + i / fps_v
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)
        st = tr.process(frame)
        times.append(st.track_ms)
        if st.reloc is not None:
            info.update(reloc_ms=st.reloc.ms, reloc_kp=st.reloc.n_kp, reloc_matches=st.reloc.n_matches,
                        reloc_inliers=st.reloc.n_inliers)
        k = gt.get(i + src_index_offset)
        if k is not None:
            if st.R is not None:
                er, et = _errors(st.R, st.t, model.kf_R[k], model.kf_t[k])
                er_raw, et_raw = _errors(st.R_raw, st.t_raw, model.kf_R[k], model.kf_t[k])
            else:
                er = et = er_raw = et_raw = np.nan
            rows.append({"frame": i, "kf": model.kf_names[k], "status": st.status, "rot_deg": er, "trans_rel": et,
                         "rot_deg_raw": er_raw, "trans_rel_raw": et_raw, "inliers": st.n_inliers})
        if out_video is not None:
            info.update(fps=fps_v, track_ms=st.track_ms)
            vis = draw(frame, st, model, model.K, info)
            if writer is None:
                writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps_v,
                                         (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        i += 1
    if writer is not None:
        writer.release()
    tr.close()
    rot = np.array([r["rot_deg"] for r in rows])
    trl = np.array([r["trans_rel"] for r in rows])
    tracked = ~np.isnan(rot)
    summary = {"frames": i, "gt_frames": len(rows), "tracked_at_gt": int(tracked.sum()),
               "rot_deg_median": round(float(np.nanmedian(rot)), 2) if tracked.any() else None,
               "rot_deg_p90": round(float(np.nanpercentile(rot, 90)), 2) if tracked.any() else None,
               "trans_rel_median_pct": round(100 * float(np.nanmedian(trl)), 2) if tracked.any() else None,
               "rot_deg_raw_median": round(float(np.nanmedian([r["rot_deg_raw"] for r in rows])), 2)
               if tracked.any() else None,
               "track_ms_median": round(float(np.median(times)), 2),
               "track_ms_p95": round(float(np.percentile(times, 95)), 2),
               "relocalisations": tr.stats}
    return {"summary": summary, "rows": rows}
