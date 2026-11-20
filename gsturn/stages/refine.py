"""Stage 5b: pose-guided re-matching (two-pass SfM).

After the first SfM pass every frame has a pose. Frames that see the object from a similar direction but are far
apart in time are matched now (loop closures), filtered with the epipolar geometry of the known poses, added to a
copy of the COLMAP database, and the model is re-triangulated and bundle-adjusted. The refined model replaces
work/sfm/dataset/sparse/0 only if it is not worse; the first pass is kept in work/sfm/sparse_first_pass/0.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import Config, refine_focal, resolve_device
from ..events import Events
from ..pipeline import StageError, StageWarn
from ..util import imread_rgb, imwrite_rgb


# --------------------------------------------------------------------------- geometry
def rotation_angle_deg(Ri: np.ndarray, Rj: np.ndarray) -> float:
    """Angle of the relative rotation Ri Rj^T (covers viewing direction and in-plane roll)."""
    c = (np.trace(Ri @ Rj.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def skew(t: np.ndarray) -> np.ndarray:
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], np.float64)


def fundamental_from_poses(Ri, ti, Rj, tj, K: np.ndarray) -> np.ndarray:
    """F with x_j^T F x_i = 0 for world->camera poses (R, t) and shared intrinsics K."""
    R = Rj @ Ri.T
    t = tj - R @ ti
    Kinv = np.linalg.inv(K)
    return Kinv.T @ skew(t) @ R @ Kinv


def sampson_px(F: np.ndarray, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    """Sampson distance (pixels) of correspondences xi <-> xj (N x 2)."""
    a = np.hstack([xi, np.ones((len(xi), 1))])
    b = np.hstack([xj, np.ones((len(xj), 1))])
    Fa = a @ F.T          # rows: F x_i
    Fb = b @ F            # rows: F^T x_j
    num = np.sum(b * Fa, 1) ** 2
    den = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Fb[:, 0] ** 2 + Fb[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def object_centre(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) == 0:
        return np.zeros(3)
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    keep = np.all((xyz >= lo) & (xyz <= hi), 1)
    return np.median(xyz[keep] if keep.any() else xyz, 0)


def candidate_pairs(R: dict[int, np.ndarray], C: dict[int, np.ndarray], O: np.ndarray, existing: set,
                    min_gap: int, max_angle: float, per_image: int, ratio=(0.7, 1.4)):
    """Returns (selected [(i, j, theta)], all [(i, j, theta)] that pass gap and distance ratio)."""
    idx = sorted(R)
    dist = {i: float(np.linalg.norm(C[i] - O)) for i in idx}
    allc = []
    per: dict[int, list] = {i: [] for i in idx}
    for a, i in enumerate(idx):
        for j in idx[a + 1:]:
            if j - i <= min_gap:
                continue
            rho = dist[i] / max(dist[j], 1e-12)
            if not ratio[0] <= rho <= ratio[1]:
                continue
            th = rotation_angle_deg(R[i], R[j])
            allc.append((i, j, th))
            if th < max_angle and (i, j) not in existing:
                per[i].append((th, i, j))
                per[j].append((th, i, j))
    sel = set()
    for i in idx:
        for th, a, b in sorted(per[i])[:per_image]:
            sel.add((a, b, th))
    return sorted(sel), allc


# --------------------------------------------------------------------------- model helpers
def model_metrics(rec) -> dict:
    rec.update_point_3d_errors()
    pts = [int(im.num_points3D) for im in rec.images.values() if im.has_pose]
    return {"registered": int(rec.num_reg_images()), "points3D": int(rec.num_points3D()),
            "mean_reproj_error": round(float(rec.compute_mean_reprojection_error()), 4),
            "mean_track_length": round(float(rec.compute_mean_track_length()), 3),
            "min_points_per_image": int(min(pts)) if pts else 0,
            "median_points_per_image": int(np.median(pts)) if pts else 0}


def poses_by_index(rec, stems: list[str]):
    pos = {f"{s}.png": k for k, s in enumerate(stems)}
    R, t, C = {}, {}, {}
    for im in rec.images.values():
        if im.has_pose and im.name in pos:
            k = pos[im.name]
            T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            R[k] = np.asarray(T.rotation.matrix(), np.float64)
            t[k] = np.asarray(T.translation, np.float64)
            C[k] = np.asarray(im.projection_center(), np.float64)
    return R, t, C


def intrinsics_matrix(rec) -> np.ndarray:
    cam = next(iter(rec.cameras.values()))
    return np.asarray(cam.calibration_matrix(), np.float64)


def bundle_adjust(rec, cfg: Config) -> None:
    import pycolmap
    o = pycolmap.BundleAdjustmentOptions()
    o.refine_focal_length = refine_focal(cfg)
    o.refine_principal_point = False
    o.refine_extra_params = False
    o.solver_options.max_num_iterations = 100
    pycolmap.bundle_adjustment(rec, o)


# --------------------------------------------------------------------------- plots
def plot_pairs(allc, matched: dict, added: set, max_angle: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if allc:
        a = np.array([(abs(j - i), th) for i, j, th in allc])
        ax.scatter(a[:, 0], a[:, 1], s=2, c="lightgray", label="all pairs (gap and distance ratio ok)")
    if matched:
        m = np.array([(abs(j - i), th, n) for (i, j), (th, n) in matched.items()])
        sc = ax.scatter(m[:, 0], m[:, 1], s=10, c=m[:, 2], cmap="viridis", label="matched candidates")
        fig.colorbar(sc, ax=ax, label="matches after epipolar filter")
        acc = np.array([(abs(j - i), th) for (i, j), (th, _) in matched.items() if (i, j) in added])
        if len(acc):
            ax.scatter(acc[:, 0], acc[:, 1], s=26, facecolors="none", edgecolors="red", lw=0.7, label="added")
    ax.axhline(max_angle, ls="--", c="gray", lw=1)
    ax.set_xlabel("time gap |i - j| (frames)")
    ax.set_ylabel("relative rotation (deg)")
    ax.set_ylim(0, min(180, 3 * max_angle))
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"pose-guided candidates: {len(matched)} matched, {len(added)} added")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_match_matrix(n: int, old: dict, new: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = np.zeros((n, n), np.float32)
    for (i, j), m in old.items():
        counts[i, j] = counts[j, i] = len(m)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(np.log1p(counts), cmap="magma")
    if new:
        p = np.array([(i, j) for i, j in new])
        ax.scatter(p[:, 1], p[:, 0], s=3, c="cyan", marker="s", lw=0, label="new loop pairs")
        ax.scatter(p[:, 0], p[:, 1], s=3, c="cyan", marker="s", lw=0)
        ax.legend(fontsize=7, loc="lower left")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(f"matches per pair (log) + {len(new)} new pairs (cyan)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def stack_images(paths: list[Path], out: Path) -> None:
    imgs = [cv2.imread(str(p)) for p in paths if p.exists()]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return
    w = max(i.shape[1] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, 0, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in imgs]
    cv2.imwrite(str(out), np.concatenate(imgs, 0))


# --------------------------------------------------------------------------- stage
def run(job, cfg: Config, ev: Events) -> dict:
    if not cfg.refine_enabled:
        return {"skipped": True, "_msg": "disabled"}
    from ..multicam import n_cameras
    if n_cameras(job.load_json("frames", "frames.json")) > 1:
        return {"skipped": True, "_msg": "skipped: refine works on one camera"}
    import pycolmap

    from .features import load_features, load_matcher, match_lightglue, match_sift
    from .sfm import load_features as load_sfm_features
    from .sfm import mapper_options, plot_model

    sfm = job.dir / "work" / "sfm"
    model = sfm / "dataset" / "sparse" / "0"
    first = sfm / "sparse_first_pass" / "0"
    src_model = first if first.exists() else model     # re-running refine starts again from the first pass
    if not src_model.exists() or not (sfm / "database.db").exists():
        raise StageError("no SfM model: run stage 'sfm' first")
    idx = job.load_json("features", "index.json")
    backend = idx["backend"]
    stems, kpts, old_matches = load_sfm_features(job)
    meta = job.load_json("frames", "frames.json")
    w, h = meta["width"], meta["height"]
    device = resolve_device(cfg.device) if backend != "sift" else "cpu"
    feats = load_features(job, stems, backend, device, sizes=[(w, h)] * len(stems))
    matcher = load_matcher(backend, device) if backend != "sift" else None

    out = job.work("refine")
    for p in ("database.db", "model", "remap", "pairs_new.txt"):
        q = out / p
        if q.is_dir():
            shutil.rmtree(q)
        elif q.exists():
            q.unlink()
    shutil.copy(sfm / "database.db", out / "database.db")
    db_path = out / "database.db"
    image_dir = sfm / "dataset" / "images"
    names = [f"{s}.png" for s in stems]

    rec0 = pycolmap.Reconstruction(str(src_model))
    before = model_metrics(rec0)
    n = len(stems)
    mode = cfg.refine_mode
    if mode == "triangulate" and before["registered"] < n:   # unregistered frames need the mapper
        ev.log(f"first pass registered {before['registered']}/{n}: switching to refine_mode=remap")
        mode = "remap"
    min_gap = cfg.refine_min_gap or cfg.match_window

    existing = set(old_matches)
    all_new: dict[tuple[int, int], np.ndarray] = {}
    matched: dict[tuple[int, int], tuple[float, int]] = {}
    allc_first: list = []
    n_cand = 0
    rec = rec0
    t_start = time.time()
    for it in range(max(1, cfg.refine_iterations)):
        R, t, C = poses_by_index(rec, stems)
        O = object_centre(np.array([p.xyz for p in rec.points3D.values()]) if rec.num_points3D() else np.zeros((0, 3)))
        K = intrinsics_matrix(rec)
        cand, allc = candidate_pairs(R, C, O, existing | set(all_new) | set(matched), min_gap,
                                     cfg.refine_max_angle, cfg.refine_pairs_per_image)
        if it == 0:
            allc_first = allc
        n_cand += len(cand)
        ev.log(f"round {it + 1}: {len(cand)} candidate pairs (angle < {cfg.refine_max_angle} deg, gap > {min_gap})")
        if not cand:
            break
        new_round = {}
        t0 = time.time()
        for k, (i, j, th) in enumerate(cand):
            if len(kpts[i]) < 8 or len(kpts[j]) < 8:
                continue
            m, _ = match_sift(feats, i, j) if backend == "sift" else match_lightglue(feats, matcher, i, j)
            if len(m):
                F = fundamental_from_poses(R[i], t[i], R[j], t[j], K)
                d = sampson_px(F, kpts[i][m[:, 0]] + 0.5, kpts[j][m[:, 1]] + 0.5)   # COLMAP pixel convention
                m = m[d <= cfg.refine_max_epipolar_px]
            matched[(i, j)] = (th, len(m))
            if len(m) >= cfg.min_matches:
                new_round[(i, j)] = m
            if k % 5 == 0 or k == len(cand) - 1:
                eta = (time.time() - t0) / (k + 1) * (len(cand) - k - 1)
                ev.progress(k + 1, len(cand), f"round {it + 1}: pair {k + 1}/{len(cand)}, "
                                              f"{len(new_round)} added, eta {eta / 60:.1f} min")
        ev.log(f"round {it + 1}: {len(new_round)} of {len(cand)} pairs kept (>= {cfg.min_matches} matches "
               f"within {cfg.refine_max_epipolar_px}px of the epipolar lines)")
        if not new_round:
            break
        all_new.update(new_round)

        # write the new matches and verify only them
        db = pycolmap.Database(str(db_path))
        ids = {im.name: im.image_id for im in db.read_all_images()}
        for (i, j), m in new_round.items():
            db.write_matches(ids[names[i]], ids[names[j]], m.astype(np.uint32))
        db.close()
        pairs_txt = out / "pairs_new.txt"
        pairs_txt.write_text("\n".join(f"{names[i]} {names[j]}" for i, j in new_round) + "\n")
        pycolmap.verify_matches(str(db_path), str(pairs_txt), pycolmap.TwoViewGeometryOptions())

        ev.progress(0, 1, f"round {it + 1}: {'re-triangulating' if mode == 'triangulate' else 're-mapping'}",
                    force=True)
        if mode == "triangulate":
            tri_dir = out / "model"
            tri_dir.mkdir(exist_ok=True)
            po = mapper_options(cfg)
            # clear_points=False: keep the mapper's tracks and extend/merge them through the new pairs.
            # Rebuilding all tracks (clear_points=True) fragmented them on IMG_8651 (track length 8.95 -> 7.45).
            rec = pycolmap.triangulate_points(pycolmap.Reconstruction(str(src_model)) if it == 0 else rec,
                                              str(db_path), str(image_dir), str(tri_dir), clear_points=False,
                                              options=po, refine_intrinsics=False)
            bundle_adjust(rec, cfg)
        else:
            rdir = out / "remap"
            if rdir.exists():
                shutil.rmtree(rdir)
            rdir.mkdir()
            recs = pycolmap.incremental_mapping(str(db_path), str(image_dir), str(rdir), mapper_options(cfg))
            if not recs:
                raise StageWarn("refinement rejected: re-mapping produced no model", {"accepted": False})
            rec = max(recs.values(), key=lambda r: r.num_reg_images())
        ev.log(f"round {it + 1}: {model_metrics(rec)}")
        if len(new_round) < 20:
            break

    after = model_metrics(rec) if all_new else dict(before)
    # artifacts
    plot_pairs(allc_first, matched, set(all_new), cfg.refine_max_angle, out / "pairs.png")
    ev.artifact(out / "pairs.png", "candidate pairs: time gap vs. relative rotation (red: added)")
    plot_match_matrix(n, old_matches, all_new, out / "match_matrix_refined.png")
    ev.artifact(out / "match_matrix_refined.png", "match matrix with the new loop pairs (cyan)")
    if all_new:
        from .features import draw_matches
        img_dir = job.dir / "work" / "frames" / "images"
        frame_names = {s: n_["name"] for s in stems for n_ in meta["frames"] if n_["name"].rsplit(".", 1)[0] == s}
        best = sorted(all_new, key=lambda p: (-(abs(p[1] - p[0]) > 2 * min_gap), -len(all_new[p])))[:2]
        for i, j in best:
            canvas = draw_matches(imread_rgb(img_dir / frame_names[stems[i]]), imread_rgb(img_dir / frame_names[stems[j]]),
                                  kpts[i], kpts[j], all_new[(i, j)],
                                  f"{stems[i]} - {stems[j]}: {len(all_new[(i, j)])} loop matches")
            p = out / f"loop_{stems[i]}_{stems[j]}.jpg"
            imwrite_rgb(p, cv2.resize(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2)))
            ev.artifact(p, f"new long-range pair {stems[i]} - {stems[j]}")

    accepted = bool(all_new) and after["registered"] >= before["registered"] and \
        after["mean_reproj_error"] <= before["mean_reproj_error"] + 0.1
    stats = {"mode": mode, "candidates": n_cand, "pairs_matched": len(matched), "pairs_added": len(all_new),
             "before": before, "after": after, "accepted": accepted,
             "match_seconds": round(time.time() - t_start, 1),
             "max_gap_added": max((j - i for i, j in all_new), default=0),
             "top_pairs": [[stems[i], stems[j], len(all_new[(i, j)])]
                           for i, j in sorted(all_new, key=lambda p: -len(all_new[p]))[:10]]}

    if accepted:
        plot_model(rec0, out / "cameras_before.png", f"first pass: {before['registered']} registered, "
                                                     f"track {before['mean_track_length']:.1f}, "
                                                     f"reproj {before['mean_reproj_error']:.2f}px")
        plot_model(rec, out / "cameras_after.png", f"refined: {after['registered']} registered, "
                                                   f"track {after['mean_track_length']:.1f}, "
                                                   f"reproj {after['mean_reproj_error']:.2f}px")
        stack_images([out / "cameras_before.png", out / "cameras_after.png"], out / "cameras_before_after.png")
        for p in ("cameras_before.png", "cameras_after.png"):
            (out / p).unlink(missing_ok=True)
        ev.artifact(out / "cameras_before_after.png", "cameras and points: first pass (top) vs. refined (bottom)")
        if not first.exists():
            first.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(model), str(first))
        if model.exists():
            shutil.rmtree(model)
        model.mkdir(parents=True)
        rec.write(str(model))
        txt = sfm / "model_txt"
        if txt.exists():
            shutil.rmtree(txt)
        txt.mkdir()
        rec.write_text(str(txt))
        stats["_msg"] = (f"+{len(all_new)} loop pairs, track length {before['mean_track_length']:.1f} -> "
                         f"{after['mean_track_length']:.1f}, reproj {before['mean_reproj_error']:.2f} -> "
                         f"{after['mean_reproj_error']:.2f} px, min pts/img {before['min_points_per_image']} -> "
                         f"{after['min_points_per_image']}")
        return stats

    # rejected or nothing to add: make sure sparse/0 holds the first pass again
    if first.exists():
        if model.exists():
            shutil.rmtree(model)
        shutil.move(str(first), str(model))
        shutil.rmtree(first.parent, ignore_errors=True)
        txt = sfm / "model_txt"
        if txt.exists():
            shutil.rmtree(txt)
        txt.mkdir()
        rec0.write_text(str(txt))
    if not all_new:
        stats["_msg"] = f"no new loop pairs ({n_cand} candidates); model unchanged"
        return stats
    reason = []
    if after["registered"] < before["registered"]:
        reason.append(f"registered {before['registered']} -> {after['registered']}")
    if after["mean_reproj_error"] > before["mean_reproj_error"] + 0.1:
        reason.append(f"reproj {before['mean_reproj_error']:.2f} -> {after['mean_reproj_error']:.2f}px")
    raise StageWarn("refinement rejected: " + ", ".join(reason) + "; kept the first-pass model", stats)
