"""Object package for live tracking, built from a finished job.

outputs/object_model.npz holds
  points      (P, 3)  3D points in the object frame (metres if a size was given, else model units)
  desc_sum    (S, P, 256) sum of SuperPoint descriptors per point and image scale (normalised at load)
  desc_count  (S, P)  number of observations summed
  scales      (S,)    image scales the descriptors were sampled at (1.0 = capture resolution)
  hull        (H, 3)  convex hull vertices of the cleaned points (for the live mask)
  T_obj_world (4, 4)  SfM (world) -> object frame, including the scale
  K           (3, 3)  capture intrinsics at the frame resolution (OpenCV pixel convention)
  image_size  (2,)    capture width, height
  kf_names, kf_R, kf_t, kf_src_index  keyframe poses (object -> camera) and their source video frame
  ref_size_px         median apparent object size (hull bbox diagonal) in the keyframes, in pixels
plus object_model.json (summary) and object_model.png (points + object axes).

Object frame: origin at the median of the cleaned points, x = longest principal axis, y = middle, z = shortest,
right-handed. "--object-size" is the object's extent along x in metres (e.g. the long edge of a box).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SCALES = (1.0, 0.7, 0.5, 0.35)


# --------------------------------------------------------------------------- geometry helpers
def clean_points(xyz: np.ndarray, errors: np.ndarray, track_len: np.ndarray) -> np.ndarray:
    """Boolean mask of points that are reliable enough to define the object frame and hull."""
    keep = (errors < 2.0) & (track_len >= 3)
    p = xyz[keep]
    c = np.median(p, 0)
    d = np.linalg.norm(p - c, axis=1)
    keep_idx = np.flatnonzero(keep)
    keep[keep_idx[d > np.percentile(d, 98) * 1.05]] = False
    # local density: drop isolated points (mean distance to 8 neighbours far above typical)
    from scipy.spatial import cKDTree
    p = xyz[keep]
    dist, _ = cKDTree(p).query(p, 9)
    md = dist[:, 1:].mean(1)
    keep[np.flatnonzero(keep)[md > np.median(md) * 3]] = False
    return keep


def object_frame(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, rotation (rows = object axes in world coords), extents (1-99 percentile range per axis)."""
    c = np.median(p, 0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    # deterministic signs: the largest component of each axis is positive, then make it right-handed
    for i in range(3):
        if vt[i, np.argmax(np.abs(vt[i]))] < 0:
            vt[i] *= -1
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    q = (p - c) @ vt.T
    ext = np.percentile(q, 99, 0) - np.percentile(q, 1, 0)
    # recentre on the box centre along each axis (median is biased toward the more textured side)
    mid = (np.percentile(q, 99, 0) + np.percentile(q, 1, 0)) / 2
    c = c + mid @ vt
    return c, vt, ext


def rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# --------------------------------------------------------------------------- loading
@dataclass
class ObjectModel:
    points: np.ndarray
    desc: np.ndarray          # (S, P, 256) normalised, zero rows where a point has no observation at that scale
    valid: np.ndarray         # (S, P) bool
    scales: np.ndarray
    hull: np.ndarray
    K: np.ndarray
    image_size: tuple[int, int]
    ref_size_px: float
    kf_names: list[str]
    kf_R: np.ndarray
    kf_t: np.ndarray
    kf_src_index: np.ndarray
    units: str
    extent: np.ndarray
    path: Path
    kf_points: list            # per keyframe: rows of the points it observes

    @property
    def kf_dirs(self) -> np.ndarray:
        """Unit vectors from the object origin to each keyframe camera centre (object frame)."""
        c = np.einsum("kji,kj->ki", self.kf_R, -self.kf_t)
        return c / np.linalg.norm(c, axis=1, keepdims=True)

    @classmethod
    def load(cls, path: Path) -> "ObjectModel":
        z = np.load(path, allow_pickle=False)
        cnt = z["desc_count"]
        s = z["desc_sum"].astype(np.float32)
        n = np.linalg.norm(s, axis=2, keepdims=True)
        desc = np.where(n > 0, s / np.maximum(n, 1e-8), 0).astype(np.float32)
        meta = json.loads(str(z["meta"]))
        return cls(points=z["points"].astype(np.float32), desc=desc, valid=cnt > 0, scales=z["scales"],
                   hull=z["hull"].astype(np.float32), K=z["K"], image_size=tuple(int(x) for x in z["image_size"]),
                   ref_size_px=float(z["ref_size_px"]), kf_names=[str(x) for x in z["kf_names"]], kf_R=z["kf_R"],
                   kf_t=z["kf_t"], kf_src_index=z["kf_src_index"], units=meta["units"],
                   extent=np.array(meta["extent"]), path=Path(path),
                   kf_points=np.split(z["kf_pid"], z["kf_ptr"][1:-1]))


def find_model(arg: str) -> Path:
    """Accept a job folder, its outputs folder or the npz itself."""
    p = Path(arg)
    for cand in (p, p / "object_model.npz", p / "outputs" / "object_model.npz"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no object_model.npz at {arg}: run 'python -m gsturn export-model --job {arg}' first")


# --------------------------------------------------------------------------- building
def sample_image(sp, gray: np.ndarray, xy: np.ndarray, scales):
    """SuperPoint descriptors at pixel positions xy (OpenCV convention) for each scale: [(valid mask, desc)]."""
    import torch
    H, W = gray.shape
    out = []
    for s in scales:
        g = gray if s == 1.0 else cv2.resize(gray, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        _, dm = sp.dense(g)
        q = (xy + 0.5) * s - 0.5
        ok = (q[:, 0] < dm.shape[2] * 8 - 1) & (q[:, 1] < dm.shape[1] * 8 - 1)
        out.append((ok, sp.sample(dm, torch.from_numpy(np.ascontiguousarray(q[ok])).to(sp.device))))
    return out


def build(job_dir: Path, object_size: float = 0.0, scales=DEFAULT_SCALES, device: str = "auto",
          log=print) -> Path:
    import torch
    from scipy.spatial import ConvexHull

    from .colmap_txt import ensure_text, read_model
    from .superpoint import SuperPointDense

    job_dir = Path(job_dir)
    work = job_dir / "work"
    rec = read_model(ensure_text(work / "sfm" / "dataset" / "sparse" / "0", work / "sfm" / "model_txt"))
    frames = json.loads((work / "frames" / "frames.json").read_text())
    src_index = {f["name"].rsplit(".", 1)[0]: f["src_index"] for f in frames["frames"]}
    name_of = {f["name"].rsplit(".", 1)[0]: f["name"] for f in frames["frames"]}
    W, H = frames["width"], frames["height"]

    pids = rec.point_ids
    row = {int(p): i for i, p in enumerate(pids)}
    P = len(pids)
    xyz, err, tl = rec.xyz, rec.error, rec.track_len
    log(f"{len(rec.images)} keyframes, {P} points")

    # ---- object frame + scale
    keep = clean_points(xyz, err, tl)
    c, R_ow, ext = object_frame(xyz[keep])
    scale = object_size / ext[0] if object_size > 0 else 1.0
    T = np.eye(4)
    T[:3, :3] = scale * R_ow
    T[:3, 3] = -scale * R_ow @ c
    pts_obj = (xyz - c) @ R_ow.T * scale
    hull_src = pts_obj[keep]
    hull = hull_src[ConvexHull(hull_src).vertices]
    units = "m" if object_size > 0 else "model units"
    log(f"object extent {ext[0] * scale:.3f} x {ext[1] * scale:.3f} x {ext[2] * scale:.3f} {units}, "
        f"{keep.sum()} clean points, hull {len(hull)} vertices")

    # ---- descriptors per point and scale
    fx, fy, cx, cy = rec.params
    Kc = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    Kc[0, 2] -= 0.5   # COLMAP pixel centres are at +0.5
    Kc[1, 2] -= 0.5
    sp = SuperPointDense(device)
    log(f"sampling SuperPoint descriptors at scales {list(scales)} on {sp.device}")
    S = len(scales)
    # one tensor per scale: index_add_ into a slice of an MPS tensor is silently dropped in torch 2.2
    dsum = [torch.zeros(P, 256, device=sp.device) for _ in scales]
    dcnt = np.zeros((S, P), np.int32)
    kf_names, kf_R, kf_t, kf_src, sizes, kf_pid = [], [], [], [], [], []
    images = sorted(rec.images.values(), key=lambda im: im.name)
    for k, im in enumerate(images):
        stem = im.name.rsplit(".", 1)[0]
        gray = cv2.imread(str(work / "frames" / "images" / name_of[stem]), cv2.IMREAD_GRAYSCALE)
        has = im.pid >= 0
        xy = im.xy[has] - 0.5                                     # -> OpenCV convention
        rid = np.array([row[int(p)] for p in im.pid[has]], np.int64)
        for si, (ok, d) in enumerate(sample_image(sp, gray, xy, scales)):
            dsum[si].index_add_(0, torch.from_numpy(rid[ok]).to(sp.device), d)
            np.add.at(dcnt[si], rid[ok], 1)
        # pose object -> camera
        Rw, tw = im.R, im.t                   # world -> cam
        R_co = Rw @ R_ow.T                    # object -> cam  (x_world = R_ow.T x_obj / scale + c)
        t_co = (Rw @ c + tw) * scale
        kf_names.append(stem)
        kf_pid.append(np.unique(rid))
        kf_R.append(R_co)
        kf_t.append(t_co)
        kf_src.append(src_index.get(stem, -1))
        proj, _ = cv2.projectPoints(hull, cv2.Rodrigues(R_co)[0], t_co, Kc, None)
        bb = proj.reshape(-1, 2)
        sizes.append(np.linalg.norm(bb.max(0) - bb.min(0)))
        if (k + 1) % 20 == 0 or k + 1 == len(images):
            log(f"  {k + 1}/{len(images)} keyframes")

    out = job_dir / "outputs"
    out.mkdir(exist_ok=True)
    meta = {"job": job_dir.name, "units": units, "scale_model_to_units": scale, "object_size": object_size,
            "extent": [float(e * scale) for e in ext], "points": int(P), "clean_points": int(keep.sum()),
            "keyframes": len(kf_names), "scales": list(scales), "image_size": [W, H],
            "K": Kc.round(3).tolist(), "ref_size_px": float(np.median(sizes)),
            "observations_per_point_scale1": float(dcnt[0].mean())}
    path = out / "object_model.npz"
    np.savez_compressed(path, points=pts_obj.astype(np.float32), desc_sum=torch.stack(dsum).cpu().numpy().astype(np.float16),
                        desc_count=dcnt, scales=np.array(scales, np.float32), hull=hull.astype(np.float32),
                        T_obj_world=T, K=Kc, image_size=np.array([W, H]), ref_size_px=np.float32(np.median(sizes)),
                        kf_names=np.array(kf_names), kf_R=np.array(kf_R), kf_t=np.array(kf_t),
                        kf_src_index=np.array(kf_src), point_clean=keep,
                        kf_ptr=np.cumsum([0] + [len(x) for x in kf_pid]), kf_pid=np.concatenate(kf_pid), meta=json.dumps(meta))
    (out / "object_model.json").write_text(json.dumps(meta, indent=2))
    preview(pts_obj[keep], hull, kf_R, kf_t, units, out / "object_model.png")
    log(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def preview(pts: np.ndarray, hull: np.ndarray, kf_R, kf_t, units: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cams = np.array([-R.T @ t for R, t in zip(kf_R, kf_t)])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = "xyz"
    for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.scatter(pts[:, a], pts[:, b], s=0.5, c="k", alpha=0.4)
        ax.scatter(hull[:, a], hull[:, b], s=6, c="tab:orange")
        ax.scatter(cams[:, a], cams[:, b], s=6, c=np.linspace(0, 1, len(cams)), cmap="viridis")
        L = np.abs(pts).max() * 0.6
        ax.arrow(0, 0, L, 0, color="r" if a == 0 else "g", width=L * 0.01)
        ax.arrow(0, 0, 0, L, color="g" if b == 1 else "b", width=L * 0.01)
        ax.set_xlabel(f"{names[a]} ({units})")
        ax.set_ylabel(f"{names[b]} ({units})")
        ax.set_aspect("equal", "datalim")
    fig.suptitle("object frame: points (black), hull (orange), keyframe cameras (coloured by order)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
