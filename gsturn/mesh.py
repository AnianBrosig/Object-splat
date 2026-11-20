"""Splat -> solid mesh (STL / OBJ / PLY with colours) and convex parts for a physics engine.

A Gaussian splat has no surface: it is a cloud of translucent ellipsoids. What it does have is a density: where the
photographed object is, many opaque Gaussians overlap. The mesh comes from that density:

  1. read the Gaussians (position, scale, rotation, opacity, colour), drop floaters (isolated splats);
  2. draw a few samples from every Gaussian, weighted by its opacity, and histogram them on a voxel grid;
  3. smooth, threshold, keep the largest connected blob, close small gaps and fill the inside (splats only cover
     the visible surface, the object is solid);
  4. marching cubes on the smoothed solid -> triangle mesh; vertex colours from the nearest splats;
  5. turn the mesh into its principal frame (longest axis x, shortest axis y = up, so it lies flat), centre it and
     scale it to a real size in metres (the reconstruction has no units: the longest side is given by the user);
  6. convex parts: one convex hull when the shape is nearly convex, otherwise k-means clusters of the solid, one hull
     each. Physics engines want convex pieces (a bowl modelled as one hull would be a solid lump).

Unseen sides of the object are not in the splat: a face that was always covered by the hand is a hole or a dent in
the density, and the solid-fill step can only guess. Check the result before you trust a thickness.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2
from scipy.spatial import ConvexHull, cKDTree

SH_C0 = 0.28209479177387814


# --------------------------------------------------------------------------- reading
def load_gaussians(path: Path) -> dict:
    """xyz (N,3), cov (N,3,3), opacity (N), rgb (N,3 in 0..1) of a Brush / 3DGS binary ply."""
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[2])
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(n * 4 * len(props)), dtype=np.dtype([(p, "<f4") for p in props]), count=n)
    g = lambda k: np.ascontiguousarray(data[k]).astype(np.float64)  # noqa: E731
    xyz = np.stack([g("x"), g("y"), g("z")], 1)
    opacity = 1 / (1 + np.exp(-g("opacity")))
    scale = np.exp(np.stack([g(f"scale_{i}") for i in range(3)], 1))
    q = np.stack([g(f"rot_{i}") for i in range(4)], 1)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    M = R * scale[:, None, :]
    rgb = np.clip(np.stack([g(f"f_dc_{i}") for i in range(3)], 1) * SH_C0 + 0.5, 0, 1)
    return {"xyz": xyz, "cov": M @ M.transpose(0, 2, 1), "opacity": opacity, "rgb": rgb}


# --------------------------------------------------------------------------- volume
def drop_floaters(xyz: np.ndarray, k: int = 12, sigma: float = 2.5) -> np.ndarray:
    """Boolean keep mask: splats whose mean distance to their k nearest neighbours is far above the typical one."""
    d, _ = cKDTree(xyz).query(xyz, k=k + 1)
    m = d[:, 1:].mean(1)
    return m < np.median(m) + sigma * 1.4826 * np.median(np.abs(m - np.median(m))) * 2


def density_grid(gs: dict, keep: np.ndarray, res: int, samples: int = 6, seed: int = 0):
    """Opacity-weighted sample density on a cubic-voxel grid. Returns (grid, origin, voxel size)."""
    rng = np.random.default_rng(seed)
    xyz, cov, op = gs["xyz"][keep], gs["cov"][keep], gs["opacity"][keep]
    lo, hi = np.percentile(xyz, 0.5, 0), np.percentile(xyz, 99.5, 0)
    pad = 0.06 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    vox = float((hi - lo).max() / res)
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    L = np.linalg.cholesky(cov + np.eye(3) * 1e-12)
    pts = xyz[:, None, :] + np.einsum("nij,nsj->nsi", L, rng.standard_normal((len(xyz), samples, 3)))
    w = np.repeat(op[:, None], samples, 1)
    idx = np.floor((pts.reshape(-1, 3) - lo) / vox).astype(int)
    ok = np.all((idx >= 0) & (idx < shape), 1)
    grid = np.zeros(shape, np.float32)
    np.add.at(grid, tuple(idx[ok].T), w.reshape(-1)[ok].astype(np.float32))
    return grid, lo, vox


def _fill(s: np.ndarray, close_vox: int) -> np.ndarray:
    if close_vox > 0:
        pad = close_vox + 1
        sp = ndi.binary_closing(np.pad(s, pad), structure=ndi.generate_binary_structure(3, 2), iterations=close_vox)
        s = sp[pad:-pad, pad:-pad, pad:-pad]
    # fill holes slice by slice along each axis as well (a shell with a small opening is still filled)
    filled = ndi.binary_fill_holes(s)
    for ax in range(3):
        f2 = np.moveaxis(np.stack([ndi.binary_fill_holes(sl) for sl in np.moveaxis(s, ax, 0)]), 0, ax)
        filled |= f2 & ndi.binary_dilation(s, iterations=max(2, close_vox * 2))
    return filled


def slice_hull_fill(s: np.ndarray, axis: int = 0) -> np.ndarray:
    """Convex hull of every slice perpendicular to `axis`: closes tubes and cups whose ends were never seen
    (the grid axes are the principal axes, so axis 0 is the longest side)."""
    from skimage.draw import polygon
    out = s.copy()
    for i in range(s.shape[axis]):
        sl = np.take(s, i, axis)
        pts = np.argwhere(sl)
        if len(pts) < 6:
            continue
        try:
            h = ConvexHull(pts)
        except Exception:  # noqa: BLE001
            continue
        rr, cc = polygon(pts[h.vertices, 0], pts[h.vertices, 1], sl.shape)
        m = np.zeros(sl.shape, bool)
        m[rr, cc] = True
        if axis == 0:
            out[i] |= m
        elif axis == 1:
            out[:, i] |= m
        else:
            out[:, :, i] |= m
    return out


def _solidity(solid: np.ndarray) -> float:
    idx = np.argwhere(solid)
    if len(idx) < 10:
        return 0.0
    try:
        return float(len(idx) / ConvexHull(idx).volume)
    except Exception:  # noqa: BLE001
        return 0.0


def solid_from_density(grid: np.ndarray, level_frac: float = 0.25, close_vox: int | None = None, log=print):
    """Smooth, threshold, largest blob, close gaps, fill the inside. Returns (bool solid, threshold).

    close_vox None: try 2, 5, 9, 14 voxels and stop at the first that makes the solid at least 80 % of its convex
    hull (a shell with open ends needs a big closing; a real concavity such as a bowl keeps its low solidity, and
    then the smallest closing that gave the highest solidity is used)."""
    sm = ndi.gaussian_filter(grid, 1.0)
    nz = sm[sm > 0]
    level = level_frac * float(np.percentile(nz, 95))
    s = sm > level
    lab, n = ndi.label(s, structure=np.ones((3, 3, 3)))
    if n > 1:
        sizes = ndi.sum(s, lab, range(1, n + 1))
        s = lab == (1 + int(np.argmax(sizes)))
    if close_vox is not None:
        return _fill(s, close_vox), level
    best, best_sol, best_r = None, -1.0, 0
    for r in (2, 5, 9, 14):
        f = _fill(s, r)
        sol = _solidity(f)
        if sol > best_sol + 0.03:                       # a larger closing must clearly help to be accepted
            best, best_sol, best_r = f, sol, r
        if sol >= 0.8:
            break
    log(f"solid: closing radius {best_r} voxels, solidity {best_sol:.2f}")
    if best_sol < 0.6:
        sl = slice_hull_fill(best, 0)
        sol2 = _solidity(sl)
        if sol2 > best_sol + 0.15:
            log(f"solid: slice-wise hull along the longest axis, solidity {best_sol:.2f} -> {sol2:.2f}")
            best = sl
    return best, level


# --------------------------------------------------------------------------- mesh
@dataclass
class Mesh:
    verts: np.ndarray                  # (V,3) metres, principal frame, y up, centred
    faces: np.ndarray                  # (F,3)
    colors: np.ndarray                 # (V,3) 0..1
    parts: list[dict] = field(default_factory=list)      # convex parts: {"verts", "faces"} in the same frame
    primitives: list[dict] = field(default_factory=list)  # box / cylinder / sphere fit of every part
    info: dict = field(default_factory=dict)


def decimate(verts: np.ndarray, faces: np.ndarray, colors: np.ndarray, target_faces: int):
    """Vertex clustering on a grid: vertices in one cell merge (mean position and colour), faces that collapse
    are dropped. The cell grows until the face count is at most target_faces."""
    if len(faces) <= target_faces:
        return verts, faces, colors
    size = float(np.ptp(verts, axis=0).max())
    cell = size / 200
    for _ in range(14):
        key = np.floor((verts - verts.min(0)) / cell).astype(np.int64)
        _, inv, cnt = np.unique(key[:, 0] * 1_000_003 ** 2 + key[:, 1] * 1_000_003 + key[:, 2], return_inverse=True,
                                return_counts=True)
        nv = np.zeros((len(cnt), 3))
        nc = np.zeros((len(cnt), 3))
        np.add.at(nv, inv, verts)
        np.add.at(nc, inv, colors)
        nv, nc = nv / cnt[:, None], nc / cnt[:, None]
        f = inv[faces]
        ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        f = f[ok]
        if len(f) <= target_faces:
            used = np.unique(f)
            remap = np.full(len(nv), -1)
            remap[used] = np.arange(len(used))
            return nv[used].astype(np.float32), remap[f].astype(np.int32), nc[used].astype(np.float32)
        cell *= 1.18
    return verts, faces, colors


def principal_frame(points: np.ndarray, weights: np.ndarray | None = None):
    """centre and rotation (rows = axes: longest, shortest = up, middle) of a point set; right-handed."""
    c = np.average(points, 0, weights=weights)
    u, s, vt = np.linalg.svd((points - c) * (1 if weights is None else np.sqrt(weights)[:, None]),
                             full_matrices=False)
    ax = np.stack([vt[0], vt[2], vt[1]])
    if np.linalg.det(ax) < 0:
        ax[2] = -ax[2]
    return c, ax


def hull_part(points: np.ndarray, max_verts: int = 24) -> dict | None:
    """Convex hull as outward triangles; the vertex count is limited by farthest-point sampling of the hull."""
    if len(points) < 4:
        return None
    try:
        h = ConvexHull(points)
    except Exception:  # noqa: BLE001
        return None
    v = points[h.vertices]
    if len(v) > max_verts:
        sel = [int(np.argmax(np.linalg.norm(v - v.mean(0), axis=1)))]
        d = np.linalg.norm(v - v[sel[0]], axis=1)
        while len(sel) < max_verts:
            j = int(np.argmax(d))
            sel.append(j)
            d = np.minimum(d, np.linalg.norm(v - v[j], axis=1))
        v = v[sel]
        h = ConvexHull(v)
    c = v.mean(0)
    f = h.simplices.copy()
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    flip = np.einsum("ij,ij->i", n, v[f[:, 0]] - c) < 0
    f[flip] = f[flip][:, ::-1]
    return {"verts": v, "faces": f, "volume": float(h.volume)}


def fit_primitive(pts: np.ndarray, part_volume: float) -> dict:
    """Box, cylinder or sphere that overlaps a convex part's solid best (IoU on its voxel centres), oriented along its
    principal axes. Physics engines collide these an order of magnitude faster than general hulls.
    pts: voxel centres of the part, part_volume: its solid volume in m3."""
    from scipy.spatial.transform import Rotation as Rot
    c0 = pts.mean(0)
    axes = np.linalg.svd(pts - c0, full_matrices=False)[2]
    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]
    proj = (pts - c0) @ axes.T
    lo, hi = proj.min(0), proj.max(0)
    ext = np.maximum(hi - lo, 1e-4)
    mid = (lo + hi) / 2
    centre = c0 + mid @ axes
    q = proj - mid                                            # points relative to the box centre, in the axes frame
    cands = []
    # box: every point is inside by construction
    v_box = float(np.prod(ext))
    cands.append((part_volume / max(v_box, 1e-12), "box"))
    # cylinder along the axis whose two other extents are most alike
    mism = [abs(ext[(a + 1) % 3] - ext[(a + 2) % 3]) / (ext[(a + 1) % 3] + ext[(a + 2) % 3]) for a in range(3)]
    a = int(np.argmin(mism))
    r = float((ext[(a + 1) % 3] + ext[(a + 2) % 3]) / 4)
    h = float(ext[a])
    rad = np.hypot(q[:, (a + 1) % 3], q[:, (a + 2) % 3])
    frac = float(np.mean((rad <= r) & (np.abs(q[:, a]) <= h / 2)))
    v_cyl = np.pi * r * r * h
    inter = part_volume * frac
    cands.append((inter / max(part_volume + v_cyl - inter, 1e-12), "cylinder"))
    # sphere
    rs = float(ext.mean() / 2)
    frac = float(np.mean(np.linalg.norm(q, axis=1) <= rs))
    v_sph = 4 / 3 * np.pi * rs ** 3
    inter = part_volume * frac
    cands.append((inter / max(part_volume + v_sph - inter, 1e-12), "sphere"))
    kind = max(cands)[1]
    if kind == "box":
        return {"type": "box", "center": centre.tolist(), "half": (ext / 2).tolist(),
                "quat": Rot.from_matrix(axes.T).as_quat().tolist(), "iou": round(max(cands)[0], 3)}
    if kind == "cylinder":
        x, y = axes[(a + 1) % 3], axes[a]
        R = np.stack([x, y, np.cross(x, y)], 1)                 # cannon cylinders run along Y
        return {"type": "cylinder", "center": centre.tolist(), "radius": r, "height": h,
                "quat": Rot.from_matrix(R).as_quat().tolist(), "iou": round(max(cands)[0], 3)}
    return {"type": "sphere", "center": centre.tolist(), "radius": rs, "iou": round(max(cands)[0], 3)}


def convex_parts(solid_pts: np.ndarray, vol_solid: float, max_parts: int = 6, seed: int = 0) -> tuple[list[dict], float]:
    """One hull if the solid is nearly convex, else k-means clusters with a hull each. Returns (parts, solidity)."""
    whole = hull_part(solid_pts, max_verts=24)
    if whole is None:
        return [], 1.0
    solidity = vol_solid / max(whole["volume"], 1e-12)
    whole["pts"] = solid_pts
    if solidity >= 0.86:
        return [whole], solidity
    k = int(np.clip(np.ceil((1 - solidity) * 14), 2, max_parts))
    rng = np.random.default_rng(seed)
    sample = solid_pts if len(solid_pts) <= 40000 else solid_pts[rng.choice(len(solid_pts), 40000, replace=False)]
    cent, lab = kmeans2(sample, k, minit="++", seed=seed)
    parts = []
    for i in range(k):
        p = hull_part(sample[lab == i], max_verts=20)
        if p is not None and p["volume"] > 0.01 * whole["volume"]:
            p["pts"] = sample[lab == i]
            parts.append(p)
    return (parts or [whole]), solidity


def extract_mesh(ply: Path, size_m: float, res: int = 112, level_frac: float = 0.25, samples: int = 6,
                 solid: str = "fill", log=print) -> Mesh:
    """solid: fill (fill the inside) | shell (surface only) | hull (convex hull of the object)."""
    from skimage import measure
    gs = load_gaussians(ply)
    n0 = len(gs["xyz"])
    keep = (gs["opacity"] > 0.1)
    keep[keep] = drop_floaters(gs["xyz"][keep])
    log(f"{keep.sum()} of {n0} splats kept (opacity > 0.1, floaters dropped)")
    # work in the principal frame from the start: grid axes = longest, shortest (up), middle axis of the object
    c0, ax0 = principal_frame(gs["xyz"][keep], gs["opacity"][keep])
    gs["xyz"] = (gs["xyz"] - c0) @ ax0.T
    gs["cov"] = np.einsum("ij,njk,lk->nil", ax0, gs["cov"], ax0)
    grid, lo, vox = density_grid(gs, keep, res, samples)
    s, level = solid_from_density(grid, level_frac, log=log)
    if solid == "shell":
        s = ndi.binary_dilation(s) & ~ndi.binary_erosion(s, iterations=2)
    vol = ndi.gaussian_filter(s.astype(np.float32), 1.0)
    pad = np.pad(vol, 1)
    verts, faces, _, _ = measure.marching_cubes(pad, 0.5, spacing=(vox, vox, vox))
    verts = verts - vox + lo                                     # undo the padding, back to splat coordinates
    idx = np.argwhere(s)
    solid_pts = lo + (idx + 0.5) * vox
    vol_solid = float(s.sum()) * vox ** 3
    # colours
    tree = cKDTree(gs["xyz"][keep])
    d, nn = tree.query(verts, k=4)
    wgt = 1.0 / np.maximum(d, 1e-9)
    wgt /= wgt.sum(1, keepdims=True)
    colors = np.einsum("vk,vkc->vc", wgt, gs["rgb"][keep][nn])
    # principal frame, metres
    c = np.array([(solid_pts[:, 0].min() + solid_pts[:, 0].max()) / 2, 0.0,
                  (solid_pts[:, 2].min() + solid_pts[:, 2].max()) / 2])
    to = lambda p: p - c  # noqa: E731
    verts_p, pts_p = to(verts), to(solid_pts)
    ext = np.ptp(pts_p, axis=0)
    scale = size_m / float(ext.max())
    verts_p, pts_p = verts_p * scale, pts_p * scale
    # rest on the floor: shift so the bottom is at y = 0, keep x, z centred
    lo_y = float(pts_p[:, 1].min())
    verts_p[:, 1] -= lo_y
    pts_p[:, 1] -= lo_y
    vol_m3 = vol_solid * scale ** 3
    parts, solidity = convex_parts(pts_p, vol_m3)
    if solid == "hull" and parts:
        parts = parts[:1]
    info = {"splats": int(keep.sum()), "grid": list(map(int, grid.shape)), "voxel_m": float(vox * scale),
            "size_m": [round(float(v), 4) for v in np.ptp(verts_p, axis=0)], "volume_cm3": round(vol_m3 * 1e6, 1),
            "solidity": round(float(solidity), 3), "convex_parts": len(parts),
            "vertices": int(len(verts_p)), "faces": int(len(faces)), "scale_to_m": float(scale),
            "density_level": float(level)}
    log(f"mesh: {info['vertices']} vertices, {info['faces']} faces, {info['size_m']} m, "
        f"solidity {info['solidity']} -> {len(parts)} convex part(s)")
    prims = [fit_primitive(p["pts"], vol_m3 * len(p["pts"]) / max(1, len(pts_p) if len(p["pts"]) == len(pts_p) else min(len(pts_p), 40000)))
             for p in parts]
    info["primitives"] = [p["type"] for p in prims]
    return Mesh(verts_p.astype(np.float32), faces.astype(np.int32), colors.astype(np.float32), parts, prims, info)


# --------------------------------------------------------------------------- writers
def write_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Binary STL in metres is not standard (STL has no unit); millimetres are what slicers and CAD expect, so the
    coordinates are written x 1000."""
    v = verts[faces] * 1000.0
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, v
    with open(path, "wb") as f:
        f.write(b"gsturn splat mesh, units: millimetres".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        f.write(rec.tobytes())


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray | None = None) -> None:
    with open(path, "w") as f:
        f.write("# gsturn splat mesh, units: metres\n")
        for i, v in enumerate(verts):
            c = f" {colors[i, 0]:.4f} {colors[i, 1]:.4f} {colors[i, 2]:.4f}" if colors is not None else ""
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}{c}\n")
        for t in faces + 1:
            f.write(f"f {t[0]} {t[1]} {t[2]}\n")


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    rgb = np.clip(colors * 255, 0, 255).astype(np.uint8)
    vd = np.zeros(len(verts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    vd["p"], vd["c"] = verts, rgb
    fd = np.zeros(len(faces), dtype=[("n", "u1"), ("i", "<i4", 3)])
    fd["n"], fd["i"] = 3, faces
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\nproperty float x\n"
                 f"property float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                 f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        f.write(vd.tobytes())
        f.write(fd.tobytes())


def write_physics_json(path: Path, mesh: Mesh, name: str, mass: float, extra: dict | None = None) -> None:
    """Everything the browser playground needs: the coloured visual mesh and the convex parts (y up, metres)."""
    q = lambda a, d=5: np.round(np.asarray(a, np.float64), d).reshape(-1).tolist()  # noqa: E731
    data = {"name": name, "mass_kg": mass, "size_m": mesh.info["size_m"], "solidity": mesh.info["solidity"],
            "visual": {"positions": q(mesh.verts), "indices": mesh.faces.reshape(-1).tolist(),
                       "colors": q(mesh.colors, 3)},
            "parts": [{"vertices": q(p["verts"]), "faces": p["faces"].tolist()} for p in mesh.parts],
            "primitives": mesh.primitives}
    data.update(extra or {})
    path.write_text(json.dumps(data, separators=(",", ":")))


def export_all(ply: Path, out: Path, name: str, size_m: float, density: float = 350.0, res: int = 112,
               level_frac: float = 0.25, solid: str = "fill", log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    m = extract_mesh(ply, size_m, res, level_frac, solid=solid, log=log)
    v, f, c = decimate(m.verts, m.faces, m.colors, 60000)         # files for CAD / viewers
    write_stl(out / f"{name}.stl", v, f)
    write_obj(out / f"{name}.obj", v, f, c)
    write_ply(out / f"{name}.ply", v, f, c)
    m.info["faces_written"] = int(len(f))
    mass = float(np.clip(m.info["volume_cm3"] * 1e-6 * density, 0.02, 5.0))
    vv, ff, cc = decimate(m.verts, m.faces, m.colors, 14000)       # the browser playground
    light = Mesh(vv, ff, cc, m.parts, m.primitives, m.info)
    write_physics_json(out / f"{name}.physics.json", light, name, mass)
    m.info.update({"name": name, "mass_kg": round(mass, 3), "files": [f"{name}{e}" for e in
                                                                      (".stl", ".obj", ".ply", ".physics.json")]})
    return m.info


def view_info(path: Path) -> dict:
    """Where to put a viewer camera for a splat file: centre and radius of the confident splats, up = -y (the
    COLMAP/SfM convention the splats are in). Cached next to the file, keyed by its size and mtime."""
    cache = path.with_suffix(".view.json")
    key = [path.stat().st_size, int(path.stat().st_mtime)]
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                return c
        except (OSError, ValueError):
            pass
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[2])
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(n * 4 * len(props)), dtype=np.dtype([(q, "<f4") for q in props]), count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-data["opacity"].astype(np.float64))) if "opacity" in props else np.ones(n)
    sel = xyz[op > 0.3] if (op > 0.3).sum() > 100 else xyz
    lo, hi = np.percentile(sel, 1, 0), np.percentile(sel, 99, 0)
    sel = sel[np.all((sel >= lo) & (sel <= hi), 1)]
    c = np.median(sel, 0)
    r = float(np.percentile(np.linalg.norm(sel - c, axis=1), 98))
    out = {"key": key, "centre": [round(float(v), 5) for v in c], "radius": round(r, 5), "up": [0, -1, 0],
           "splats": int(n)}
    try:
        cache.write_text(json.dumps(out))
    except OSError:
        pass
    return out
