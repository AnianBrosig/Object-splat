"""Stage 5: Structure-from-Motion with COLMAP (pycolmap or colmap CLI) on the masked keypoints/matches."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..config import Config, refine_focal
from ..events import Events
from ..pipeline import StageError, StageWarn
from ..util import find_tool, run_streaming


def build_rgba_images(job, stems: list[str], names: list[str], ev: Events) -> Path:
    """dataset/images/<stem>.png with the object mask as alpha.

    stems: frames used by SfM (may skip frames the quality stage excluded); names: all frame file names. Each
    stem is matched to its own file by name, never by position."""
    by_stem = {n.rsplit(".", 1)[0]: n for n in names}
    missing = [s for s in stems if s not in by_stem]
    if missing:
        raise StageError(f"features reference frames that no longer exist ({missing[:5]}): re-run from 'features'")
    ds = job.dir / "work" / "sfm" / "dataset" / "images"
    if ds.exists():
        shutil.rmtree(ds)
    ds.mkdir(parents=True)
    img_dir = job.dir / "work" / "frames" / "images"
    mask_dir = job.dir / "work" / "segment" / "masks"
    for k, s in enumerate(stems):
        bgr = cv2.imread(str(img_dir / by_stem[s]))
        m = cv2.imread(str(mask_dir / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(ds / f"{s}.png"), np.dstack([bgr, m]))
        ev.progress(k + 1, len(stems), f"writing RGBA images {k + 1}/{len(stems)}")
    return ds


def load_features(job):
    idx = job.load_json("features", "index.json")
    stems = idx["stems"]
    kp = np.load(job.dir / "work" / "features" / "keypoints.npz")
    mt = np.load(job.dir / "work" / "features" / "matches.npz")
    kpts = [kp[s] for s in stems]
    matches = {tuple(int(x) for x in k.split("_")): mt[k] for k in mt.files}
    return stems, kpts, matches


def camera_params(cfg: Config, w: int, h: int, meta: dict | None = None, cam: int = 0) -> tuple[float, float, float, float]:
    """PINHOLE fx, fy, cx, cy at working resolution of camera `cam`.

    Order: intrinsics written by the frames stage (undistorted frames) > calibration file > focal prior."""
    intr = (meta or {}).get("intrinsics") if cam == 0 else \
        ((meta or {}).get("cameras") or [{}, {}])[cam].get("intrinsics")
    if intr:
        return float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    calibration = cfg.calibration if cam == 0 else cfg.calibration2
    if calibration:
        from ..calibrate import load_calibration, scaled_intrinsics
        fx, fy, cx, cy, _ = scaled_intrinsics(load_calibration(calibration), w, h)
        return fx, fy, cx, cy
    f = (cfg.focal_prior or 1.2 * max(w, h)) if cam == 0 else 1.2 * max(w, h)
    return f, f, w / 2.0, h / 2.0


def calibration_checks(job, cfg: Config, w: int, h: int, meta: dict) -> tuple[dict, list[str]]:
    """Info about the calibration in use, and warnings when it may not match this video."""
    from ..calibrate import has_distortion, load_calibration, video_meta
    calib = load_calibration(cfg.calibration)
    fx, fy, cx, cy = camera_params(cfg, w, h, meta)
    info = {"file": cfg.calibration, "fx": round(fx, 2), "fy": round(fy, 2), "cx": round(cx, 2), "cy": round(cy, 2),
            "rms_px": calib.get("rms_px"), "undistorted": bool((meta.get("intrinsics") or {}).get("undistorted"))}
    warns = []
    src = None
    try:
        src = job.load_json("downscale", "stats.json")["input"]
    except Exception:  # noqa: BLE001
        pass
    if src and calib.get("source_size"):
        a_job = src["width"] / src["height"]
        a_cal = calib["source_size"][0] / calib["source_size"][1]
        if abs(a_job - a_cal) > 0.01:
            warns.append(f"aspect ratio of the calibration video ({a_cal:.3f}) differs from this video ({a_job:.3f})")
    model = video_meta(Path(cfg.video)).get("model") if cfg.video else None
    if model and calib.get("camera") and model != calib["camera"]:
        warns.append(f"camera model differs: calibration '{calib['camera']}', video '{model}'")
    if (calib.get("width"), calib.get("height")) != (w, h):
        warns.append(f"working resolution differs: calibration {calib.get('width')}x{calib.get('height')}, "
                     f"job {w}x{h} (scaling is exact only for the same sensor crop)")
    if has_distortion(calib) and not info["undistorted"]:
        warns.append("calibration has lens distortion but the frames are not undistorted: re-run from 'frames'")
    return info, warns


# --------------------------------------------------------------------------- backends
def mapper_options(cfg: Config):
    import pycolmap
    po = pycolmap.IncrementalPipelineOptions()
    po.ba_refine_focal_length = refine_focal(cfg)
    po.ba_refine_principal_point = False
    po.ba_refine_extra_params = False
    po.min_model_size = 5
    return po


def run_pycolmap(db_path: Path, image_dir: Path, out_dir: Path, stems, kpts, matches, cfg: Config,
                 w: int, h: int, ev: Events, meta: dict | None = None):
    import pycolmap
    fx, fy, cx, cy = camera_params(cfg, w, h, meta)
    names = [f"{s}.png" for s in stems]
    opts = pycolmap.ImageReaderOptions()
    opts.camera_model = "PINHOLE"
    opts.camera_params = f"{fx},{fy},{cx},{cy}"
    pycolmap.Database(str(db_path)).close()  # creates the schema; import_images needs an existing file
    pycolmap.import_images(str(db_path), str(image_dir), pycolmap.CameraMode.SINGLE, names, opts)
    db = pycolmap.Database(str(db_path))
    ids = {im.name: im.image_id for im in db.read_all_images()}
    for s, kp in zip(stems, kpts):
        db.write_keypoints(ids[f"{s}.png"], (kp + 0.5).astype(np.float32))  # COLMAP: pixel centre at +0.5
    for (i, j), m in matches.items():
        db.write_matches(ids[names[i]], ids[names[j]], m.astype(np.uint32))
    db.close()
    pairs_txt = out_dir / "pairs.txt"
    pairs_txt.write_text("\n".join(f"{names[i]} {names[j]}" for i, j in matches) + "\n")
    ev.log(f"geometric verification of {len(matches)} pairs")
    pycolmap.verify_matches(str(db_path), str(pairs_txt), pycolmap.TwoViewGeometryOptions())

    po = mapper_options(cfg)
    registered = [0]

    def on_image():
        registered[0] += 1
        ev.progress(registered[0], len(stems), f"registered {registered[0]}/{len(stems)} images")

    sparse = out_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    ev.log("incremental mapping (pycolmap)")
    recs = pycolmap.incremental_mapping(str(db_path), str(image_dir), str(sparse), po,
                                        next_image_callback=on_image)
    if not recs:
        raise StageError("COLMAP could not initialise a model (too few verified matches?)")
    return max(recs.values(), key=lambda r: r.num_reg_images())


def run_pycolmap_multi(db_path: Path, image_dir: Path, out_dir: Path, stems, kpts, matches, cfg: Config,
                      cams: list[int], cam_params: list[tuple], ev: Events):
    """Several cameras (one per video): a PINHOLE camera each in the database. Camera 2 is not tied to camera 1 as a
    COLMAP rig (its frame grouping did not pair the images), but the rigid mounting is checked afterwards from the
    result: see rig_consistency."""
    import pycolmap
    pycolmap.Database(str(db_path)).close()
    db = pycolmap.Database(str(db_path))
    cid = {}
    for c, (w, h, fx, fy, cx, cy) in enumerate(cam_params):
        cid[c] = db.write_camera(pycolmap.Camera(model="PINHOLE", width=w, height=h, params=[fx, fy, cx, cy],
                                                 camera_id=c + 1), True)
    names = [f"{s}.png" for s in stems]
    ids = {}
    for nm, c in zip(names, cams):
        ids[nm] = db.write_image(pycolmap.Image(name=nm, camera_id=cid[c]))
    for nm, kp in zip(names, kpts):
        db.write_keypoints(ids[nm], (kp + 0.5).astype(np.float32))
    for (i, j), m in matches.items():
        db.write_matches(ids[names[i]], ids[names[j]], m.astype(np.uint32))
    rig_used = False
    db.close()
    pairs_txt = out_dir / "pairs.txt"
    pairs_txt.write_text("\n".join(f"{names[i]} {names[j]}" for i, j in matches) + "\n")
    ev.log(f"geometric verification of {len(matches)} pairs")
    pycolmap.verify_matches(str(db_path), str(pairs_txt), pycolmap.TwoViewGeometryOptions())
    po = mapper_options(cfg)
    registered = [0]

    def on_image():
        registered[0] += 1
        ev.progress(registered[0], len(stems), f"registered {registered[0]}/{len(stems)} images")
    sparse = out_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    ev.log("incremental mapping (pycolmap, %d cameras%s)" % (len(cam_params), ", rig" if rig_used else ""))
    recs = pycolmap.incremental_mapping(str(db_path), str(image_dir), str(sparse), po, next_image_callback=on_image)
    if not recs:
        raise StageError("COLMAP could not initialise a model (too few verified matches?)")
    return max(recs.values(), key=lambda r: r.num_reg_images())


def write_colmap_features(import_dir: Path, stems, kpts) -> None:
    import_dir.mkdir(parents=True, exist_ok=True)
    for s, kp in zip(stems, kpts):
        lines = [f"{len(kp)} 128"]
        zeros = " ".join(["0"] * 128)
        for x, y in kp + 0.5:
            lines.append(f"{x:.3f} {y:.3f} 1 0 {zeros}")
        (import_dir / f"{s}.png.txt").write_text("\n".join(lines) + "\n")


def run_colmap_cli(db_path: Path, image_dir: Path, out_dir: Path, stems, kpts, matches, cfg: Config,
                   w: int, h: int, ev: Events, meta: dict | None = None):
    import pycolmap
    colmap = find_tool("colmap", cfg.colmap_bin)
    if not colmap:
        raise StageError("colmap not found (set colmap_bin or use sfm_backend=pycolmap)")
    fx, fy, cx, cy = camera_params(cfg, w, h, meta)
    names = [f"{s}.png" for s in stems]
    imp = out_dir / "import"
    write_colmap_features(imp, stems, kpts)
    match_txt = out_dir / "matches.txt"
    with open(match_txt, "w") as fh:
        for (i, j), m in matches.items():
            fh.write(f"{names[i]} {names[j]}\n")
            fh.writelines(f"{a} {b}\n" for a, b in m)
            fh.write("\n")

    def call(args: list[str], what: str, on_line=None):
        tail: list[str] = []

        def cb(line: str):
            tail.append(line)
            del tail[:-40]
            if on_line:
                on_line(line)
        ev.log(f"colmap {what}")
        rc = run_streaming([colmap, *args], cb)
        if rc != 0:
            raise StageError(f"colmap {what} failed:\n" + "\n".join(tail[-15:]))

    call(["feature_importer", "--database_path", str(db_path), "--image_path", str(image_dir),
          "--import_path", str(imp), "--ImageReader.camera_model", "PINHOLE",
          "--ImageReader.single_camera", "1", "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}"],
         "feature_importer")
    call(["matches_importer", "--database_path", str(db_path), "--match_list_path", str(match_txt),
          "--match_type", "raw"], "matches_importer (geometric verification)")
    sparse = out_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    reg = [0]

    def on_map(line: str):
        if re.search(r"Registering image #\d+", line):
            reg[0] += 1
            ev.progress(reg[0], len(stems), f"registered {reg[0]}/{len(stems)} images")
        elif "Initializing with image pair" in line:
            reg[0] = 2
    call(["mapper", "--database_path", str(db_path), "--image_path", str(image_dir), "--output_path", str(sparse),
          "--Mapper.ba_refine_focal_length", "1" if refine_focal(cfg) else "0",
          "--Mapper.ba_refine_principal_point", "0", "--Mapper.ba_refine_extra_params", "0",
          "--Mapper.min_model_size", "5"], "mapper", on_map)
    models = [p for p in sparse.iterdir() if p.is_dir()]
    if not models:
        raise StageError("COLMAP could not initialise a model (too few verified matches?)")
    recs = [pycolmap.Reconstruction(str(p)) for p in models]
    return max(recs, key=lambda r: r.num_reg_images())


def rig_consistency(rec, meta: dict, stems: list[str]) -> dict:
    """If camera 2 is rigidly mounted to camera 1, its pose relative to camera 1 is the same at every moment. The
    spread of that relative pose over all synchronised pairs measures the quality of the registration and of the time
    sync (an offset error of dt shows up as a spread of about angular speed x dt)."""
    from scipy.spatial.transform import Rotation as Rot

    from ..multicam import partners_for
    pos = {s_: k for k, s_ in enumerate(stems)}
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    rels = []
    for j, i0 in partners_for(meta, stems).items():
        a, b = by_name.get(f"{stems[i0]}.png"), by_name.get(f"{stems[j]}.png")
        if a is None or b is None:
            continue
        Ta = a.cam_from_world() if callable(a.cam_from_world) else a.cam_from_world
        Tb = b.cam_from_world() if callable(b.cam_from_world) else b.cam_from_world
        Ra, Rb = np.asarray(Ta.rotation.matrix()), np.asarray(Tb.rotation.matrix())
        rels.append((Rb @ Ra.T, np.asarray(Tb.translation) - Rb @ Ra.T @ np.asarray(Ta.translation)))
    if len(rels) < 5:
        return {"pairs": len(rels)}
    rot = Rot.from_matrix(np.stack([r[0] for r in rels]))
    mean = rot.mean()
    dev = np.degrees((rot * mean.inv()).magnitude())
    tr = np.stack([r[1] for r in rels])
    med = np.median(tr, 0)
    base = float(np.linalg.norm(med))
    return {"pairs": len(rels), "rotation_spread_deg_median": round(float(np.median(dev)), 3),
            "rotation_spread_deg_p90": round(float(np.percentile(dev, 90)), 3),
            "baseline": round(base, 4),
            "translation_spread_pct_median": round(100 * float(np.median(np.linalg.norm(tr - med, axis=1))) / max(base, 1e-9), 2),
            "relative_rotation_deg": round(float(np.degrees(mean.magnitude())), 1)}


# --------------------------------------------------------------------------- continuity (Kalman) and focal retry
def pose_track(rec, stems: list[str], meta: dict):
    """kalman.Track of the registered frames of camera 0, in frame order. Time from the frame timestamps."""
    from scipy.spatial.transform import Rotation as Rot

    from ..kalman import Track
    t_of = {f["name"].rsplit(".", 1)[0]: f.get("t") for f in meta["frames"] if int(f.get("cam", 0)) == 0}
    rows = []
    for k, s_ in enumerate(stems):
        if s_ not in t_of:
            continue
        im = next((x for x in rec.images.values() if x.name == f"{s_}.png" and x.has_pose), None)
        if im is None:
            continue
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        t = t_of[s_] if t_of[s_] is not None else float(k)
        rows.append((t, np.asarray(T.rotation.matrix()), np.asarray(im.projection_center()), s_))
    if len(rows) < 6:
        return None, []
    rows.sort(key=lambda r: r[0])
    tr = Track(np.array([r[0] for r in rows]), Rot.from_matrix(np.stack([r[1] for r in rows])),
               np.stack([r[2] for r in rows]))
    return tr, [r[3] for r in rows]


def continuity_report(rec, stems, meta) -> dict:
    from ..kalman import continuity_score, find_breaks
    tr, order = pose_track(rec, stems, meta)
    if tr is None:
        return {"breaks": 0, "max_step_deg": 0.0, "median_step_deg": 0.0, "list": []}
    br = find_breaks(tr)
    sc = continuity_score(tr, br)
    sc["list"] = [dict(b, frame=order[b["index"]]) for b in br]
    return sc


def focal_plausible(f: float, w: int, h: int) -> bool:
    return 0.55 <= f / max(w, h) <= 1.25        # phone main cameras: 65-80 degrees across the long side


def retry_with_fixed_focal(first_rec, first_rep, fn, args, cfg: Config, out: Path, ev: Events, meta, stems, w, h):
    """Re-map with a fixed focal length from cfg.sfm_focal_candidates. Returns (best rec, report, log rows)."""
    import dataclasses
    rows = [{"focal": round(float(next(iter(first_rec.cameras.values())).params[0]), 1), "source": "free",
             "breaks": first_rep["breaks"], "reproj": round(float(first_rec.compute_mean_reprojection_error()), 3),
             "registered": int(first_rec.num_reg_images())}]
    best = (first_rep["breaks"], float(first_rec.compute_mean_reprojection_error()), -first_rec.num_reg_images())
    best_rec, best_rep = first_rec, first_rep
    for frac in [float(x) for x in getattr(cfg, "sfm_focal_candidates", "0.85,0.95,0.75").split(",") if x.strip()]:
        f = frac * max(w, h)
        c2 = dataclasses.replace(cfg, calibration="", focal_prior=f, refine_focal=False)
        for p_ in ("database.db", "sparse", "import"):
            q = out / p_
            if q.is_dir():
                shutil.rmtree(q)
            elif q.exists():
                q.unlink()
        ev.log(f"continuity: re-mapping with the focal fixed at {f:.0f} px ({frac} x long side)")
        try:
            rec2 = fn(out / "database.db", *args[:1], out, *args[1:], c2, w, h, ev, meta)
        except Exception as e:  # noqa: BLE001
            ev.log(f"  failed: {e}")
            continue
        rep2 = continuity_report(rec2, stems, meta)
        err2 = float(rec2.compute_mean_reprojection_error())
        rows.append({"focal": round(f, 1), "source": "fixed", "breaks": rep2["breaks"], "reproj": round(err2, 3),
                     "registered": int(rec2.num_reg_images())})
        ev.log(f"  -> {rep2['breaks']} orientation jumps, reproj {err2:.3f} px, {rec2.num_reg_images()} registered")
        key = (rep2["breaks"], err2, -rec2.num_reg_images())
        if rec2.num_reg_images() >= 0.9 * first_rec.num_reg_images() and key < best:
            best, best_rec, best_rep = key, rec2, rep2
        if rep2["breaks"] == 0 and err2 <= float(first_rec.compute_mean_reprojection_error()) + 0.05:
            break                                   # clean and no worse: no need to try the others
    return best_rec, best_rep, rows


# --------------------------------------------------------------------------- plots
def plot_model(rec, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers, order = [], []
    for iid, im in rec.images.items():
        if im.has_pose:
            centers.append(im.projection_center())
            order.append(im.name)
    c = np.array(centers)[np.argsort(order)] if centers else np.zeros((0, 3))
    pts = np.array([p.xyz for p in rec.points3D.values()]) if rec.num_points3D() else np.zeros((0, 3))
    cols = np.array([p.color for p in rec.points3D.values()]) / 255.0 if len(pts) else None
    if len(pts) > 20000:
        sel = np.random.default_rng(0).choice(len(pts), 20000, replace=False)
        pts, cols = pts[sel], cols[sel]
    # robust view limits from the points
    if len(pts):
        lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
        keep = np.all((pts >= lo) & (pts <= hi), 1)
        pts, cols = pts[keep], cols[keep]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (a, b), lab in zip(axes, [(0, 2), (0, 1), (2, 1)], ["top (x-z)", "front (x-y)", "side (z-y)"]):
        if len(pts):
            ax.scatter(pts[:, a], pts[:, b], s=0.5, c=cols, alpha=0.6)
        if len(c):
            ax.plot(c[:, a], c[:, b], "-", c="tab:blue", lw=0.6, alpha=0.5)
            ax.scatter(c[:, a], c[:, b], s=8, c=np.linspace(0, 1, len(c)), cmap="viridis", zorder=3)
        ax.set_aspect("equal", "datalim")
        ax.set_title(lab)
        ax.invert_yaxis() if b == 1 else None
    fig.suptitle(title + "  (dots: cameras coloured by frame order)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- stage
def run(job, cfg: Config, ev: Events) -> dict:
    meta = job.load_json("frames", "frames.json")
    names = [f["name"] for f in meta["frames"]]
    w, h = meta["width"], meta["height"]
    stems, kpts, matches = load_features(job)
    if len(matches) < 5:
        raise StageError(f"only {len(matches)} matched pairs: check the features stage")
    out = job.work("sfm")
    for p in ("database.db", "sparse", "import", "sparse_first_pass"):   # first pass belongs to the old run
        q = out / p
        if q.is_dir():
            shutil.rmtree(q)
        elif q.exists():
            q.unlink()
    image_dir = build_rgba_images(job, stems, names, ev)
    ev.progress(0, len(stems), "importing features")

    backend = cfg.sfm_backend
    if backend == "pycolmap":
        try:
            import pycolmap  # noqa: F401
        except Exception as e:  # noqa: BLE001
            ev.warn(f"pycolmap unavailable ({e}); trying colmap CLI")
            backend = "colmap-cli"
    from ..multicam import cams_for, n_cameras, size_for
    cams = cams_for(meta, stems)
    ncam = n_cameras(meta)
    calib_info, calib_warns = (calibration_checks(job, cfg, w, h, meta) if cfg.calibration else (None, []))
    for wmsg in calib_warns:
        ev.warn(wmsg)
    if ncam > 1:
        if backend != "pycolmap":
            raise StageError("two cameras need sfm_backend=pycolmap")
        cam_params = [(*size_for(meta, c), *camera_params(cfg, *size_for(meta, c), meta, c)) for c in range(ncam)]
        rec = run_pycolmap_multi(out / "database.db", image_dir, out, stems, kpts, matches, cfg, cams, cam_params, ev)
    else:
        fn = run_pycolmap if backend == "pycolmap" else run_colmap_cli
        rec = fn(out / "database.db", image_dir, out, stems, kpts, matches, cfg, w, h, ev, meta)

    continuity = None
    focal_rows = []
    if ncam == 1 and getattr(cfg, "continuity_check", True):   # getattr: old Config objects of runs started earlier
        continuity = continuity_report(rec, stems, meta)
        f0 = float(next(iter(rec.cameras.values())).params[0])
        suspicious = continuity["breaks"] > 0 or (refine_focal(cfg) and not focal_plausible(f0, w, h))
        if continuity["breaks"]:
            ev.warn(f"continuity: {continuity['breaks']} orientation jump(s) no hand motion explains: "
                    + ", ".join(f"{b['frame']} ({b['step_deg']} deg)" for b in continuity["list"][:5]))
        if suspicious and getattr(cfg, "sfm_focal_retry", True) and refine_focal(cfg) and backend == "pycolmap":
            if continuity["breaks"] == 0:
                ev.log(f"continuity: the free focal ({f0:.0f} px = {f0 / max(w, h):.2f} x long side) is implausible; "
                       f"trying fixed focal lengths")
            rec, continuity, focal_rows = retry_with_fixed_focal(
                rec, continuity, run_pycolmap, (image_dir, stems, kpts, matches), cfg, out, ev, meta, stems, w, h)

    model = out / "dataset" / "sparse" / "0"
    if model.exists():
        shutil.rmtree(model)
    model.mkdir(parents=True)
    rec.write(str(model))
    txt = out / "model_txt"
    txt.mkdir(exist_ok=True)
    rec.write_text(str(txt))

    n_reg = rec.num_reg_images()
    ratio = n_reg / len(stems)
    err = float(rec.compute_mean_reprojection_error())
    cam = next(iter(rec.cameras.values()))
    reg_names = {im.name for im in rec.images.values() if im.has_pose}
    table = []
    for s in stems:
        im = next((x for x in rec.images.values() if x.name == f"{s}.png"), None)
        table.append({"image": s, "registered": f"{s}.png" in reg_names,
                      "points3D": int(im.num_points3D) if im is not None and im.has_pose else 0})
    job.save_json("sfm", "registration.json", table)
    plot_model(rec, out / "cameras.png", f"{n_reg}/{len(stems)} registered, {rec.num_points3D()} points, "
                                         f"reproj {err:.2f}px" + (", fixed intrinsics" if not refine_focal(cfg) else ""))
    ev.artifact(out / "cameras.png", "camera positions and sparse points")
    ev.artifact(out / "registration.json", "per-image registration table", kind="json")

    stats = {"backend": backend, "registered": n_reg, "frames": len(stems), "registered_ratio": round(ratio, 3),
             "points3D": rec.num_points3D(), "mean_reproj_error": round(err, 3),
             "mean_track_length": round(float(rec.compute_mean_track_length()), 2),
             "camera": {"model": cam.model.name, "params": [round(float(p), 2) for p in cam.params]},
             "focal_prior": camera_params(cfg, w, h, meta)[0], "refine_focal": refine_focal(cfg),
             "unregistered": [t["image"] for t in table
                                                                           if not t["registered"]][:100]}
    if continuity is not None:
        stats["continuity"] = {k: v for k, v in continuity.items()}
        if focal_rows:
            stats["focal_retry"] = focal_rows
    if ncam > 1:
        per = {}
        for c in range(ncam):
            idx = [s_ for s_, cc in zip(stems, cams) if cc == c]
            per[f"camera{c + 1}"] = {"frames": len(idx), "registered": sum(1 for s_ in idx if f"{s_}.png" in reg_names)}
        stats["per_camera"] = per
        stats["cameras"] = {int(k): {"model": c_.model.name, "params": [round(float(p_), 2) for p_ in c_.params]}
                            for k, c_ in rec.cameras.items()}
        stats["rig"] = rig_consistency(rec, meta, stems)
        rg = stats["rig"]
        if rg.get("pairs", 0) >= 5:
            ev.log(f"rig check over {rg['pairs']} synchronised pairs: the relative pose of camera 2 varies by "
                   f"{rg['rotation_spread_deg_median']} deg (median) and {rg['translation_spread_pct_median']} % of "
                   f"the baseline; if the cameras are rigid this is registration/sync error")
        if any(v["registered"] < 0.5 * v["frames"] for v in per.values()):
            ev.warn("one camera registered less than half of its frames: the two cameras were not linked well "
                    "(few cross-camera matches); check the features stage")
    if calib_info:
        stats["calibration"] = calib_info
    msg = (f"{n_reg}/{len(stems)} registered ({100 * ratio:.0f} %), {rec.num_points3D()} points, "
           f"reproj {err:.2f}px, f={cam.params[0]:.0f}" + (" (calibrated)" if calib_info else "")
           + ("; " + ", ".join(f"cam{k[-1]}: {v['registered']}/{v['frames']}" for k, v in stats["per_camera"].items())
              if ncam > 1 else ""))
    if ratio < cfg.min_registered_ratio or err > cfg.max_reproj_error:
        raise StageWarn(msg + " - quality gate failed", stats, stop=cfg.train_backend != "none")
    if calib_warns:
        stats["calibration_warnings"] = calib_warns
        raise StageWarn(msg + "; calibration: " + "; ".join(calib_warns), stats)
    stats["_msg"] = msg
    return stats
