"""De-lighting: remove the room light that is baked into a splat, so the object has neutral light.

The geometry of this capture makes the problem solvable without any extra data. The camera and the room lights
are fixed while the object turns in the hands. So for a surface point the *lighting* (a function of where the
surface faces relative to the camera and the lamps) changes from frame to frame, while its *albedo* stays the same.
Writing n_c for the surface normal in the camera frame of frame i:

        pixel colour  c(p, i)  =  albedo(p) * E(n_c(p, i))            E = low-order spherical-harmonic lighting

E is one function for the whole video (fixed lights), albedo(p) is one colour per splat. Both are unknown, but
every splat is seen in many frames with many different n_c, which pins them down (up to one scale per colour
channel, which is fixed by making the average lighting 1 and white: the room light's colour cast goes into E, so the
result is lit by a neutral white light).

Steps
  1. surface normals of the splats (PCA of the neighbours, oriented away from the object's centre);
  2. for every frame, the splats that are visible (front-facing, inside the trusted object mask, not hidden by a splat
     in front of them) and the photograph's colour at their projection;
  3. alternate: fit the 9 SH lighting coefficients (per colour channel) given the albedos, then re-estimate every albedo
     given the lighting, with a robust weight that discounts what the model cannot explain (specular highlights, cast
     shadows of the hand, mask leaks);
  4. write a splat whose colour is the albedo (SH degree 0, no view-dependent terms: highlights are view dependent).

What this does not do: remove shadows the object casts on itself (the model has no visibility term), model glossy
reflections (they are discounted, not explained), or recover colour where no frame saw the surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814
GAMMA = 2.2


# --------------------------------------------------------------------------- ply io
def read_ply_raw(path: Path):
    """Header text, property names and the structured float32 vertex array of a 3DGS ply."""
    with open(path, "rb") as f:
        header, props, n = b"", [], 0
        while True:
            line = f.readline()
            header += line
            t = line.decode("ascii", "replace").strip()
            if t.startswith("element vertex"):
                n = int(t.split()[-1])
            elif t.startswith("property"):
                props.append(t.split()[2])
            elif t == "end_header":
                break
        data = np.frombuffer(f.read(n * 4 * len(props)), dtype=np.dtype([(p, "<f4") for p in props]), count=n).copy()
    return header, props, data


def write_ply_raw(path: Path, header: bytes, data: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


# --------------------------------------------------------------------------- geometry
def surface_normals(xyz: np.ndarray, k: int = 18) -> np.ndarray:
    """Normals from the PCA of the k nearest splats, pointing away from the centre of the cloud."""
    _, nn = cKDTree(xyz).query(xyz, k=k)
    pts = xyz[nn] - xyz[nn].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", pts, pts)
    _, vec = np.linalg.eigh(cov)
    n = vec[:, :, 0]
    c = np.median(xyz, 0)
    flip = np.einsum("ij,ij->i", n, xyz - c) < 0
    n[flip] = -n[flip]
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def sh_basis(n: np.ndarray) -> np.ndarray:
    """Real spherical harmonics up to order 2 at unit vectors n (N, 3) -> (N, 9)."""
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    return np.stack([np.full_like(x, 0.282095), 0.488603 * y, 0.488603 * z, 0.488603 * x,
                     1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z,
                     0.546274 * (x * x - y * y)], 1)


def srgb_to_lin(x):
    return np.power(np.clip(x, 0, 1), GAMMA)


def lin_to_srgb(x):
    return np.power(np.clip(x, 0, 1), 1.0 / GAMMA)


# --------------------------------------------------------------------------- observations
def collect(job, xyz: np.ndarray, normals: np.ndarray, radius: float, max_per_frame: int = 25000,
            min_cos: float = 0.25, log=print, seed: int = 0):
    """For each frame, the visible splats and their photographed colour. Returns arrays over all samples:
    splat index, linear rgb (S, 3), camera-frame normal (S, 3), frame index (S)."""
    from .stages.traindata import read_cameras, read_image_headers
    from .track.colmap_txt import ensure_text, qvec_to_R
    sfm = job.dir / "work" / "sfm"
    txt = ensure_text(sfm / "dataset" / "sparse" / "0", sfm / "model_txt")
    cams = read_cameras(txt)
    heads = read_image_headers(txt)
    meta = json.loads((job.dir / "work" / "frames" / "frames.json").read_text())
    fname = {f["name"].rsplit(".", 1)[0]: f["name"] for f in meta["frames"]}
    seg = job.dir / "work" / "segment"
    rng = np.random.default_rng(seed)
    out_i, out_c, out_n, out_f = [], [], [], []
    for fi, a in enumerate(heads):
        stem = a[9].rsplit(".", 1)[0]
        c = cams[int(a[8])]
        W, H = c["w"], c["h"]
        R = qvec_to_R(np.array([float(v) for v in a[1:5]]))
        t = np.array([float(v) for v in a[5:8]])
        xc = xyz @ R.T + t
        z = xc[:, 2]
        u = c["fx"] * xc[:, 0] / np.maximum(z, 1e-9) + c["cx"]
        v = c["fy"] * xc[:, 1] / np.maximum(z, 1e-9) + c["cy"]
        ok = (z > 1e-3) & (u >= 1) & (u < W - 2) & (v >= 1) & (v < H - 2)
        nc = normals @ R.T
        view = -xc / np.maximum(np.linalg.norm(xc, axis=1, keepdims=True), 1e-12)
        cos = np.einsum("ij,ij->i", nc, view)
        ok &= cos > min_cos
        # hidden behind a splat in front: nearest depth per low-resolution pixel
        s = 0.25
        pi = np.clip((v * s).astype(int), 0, int(H * s) - 1) * int(W * s + 1) + np.clip((u * s).astype(int), 0, int(W * s))
        buf = np.full(int(H * s + 1) * int(W * s + 1) + 8, np.inf)
        np.minimum.at(buf, pi[ok], z[ok])
        ok &= z <= buf[pi] + 0.07 * radius
        obj = cv2.imread(str(seg / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(job.dir / "work" / "frames" / "images" / fname[stem]), cv2.IMREAD_COLOR)
        if obj is None or img is None or obj.shape != (H, W):
            continue
        obj = cv2.erode(obj, np.ones((7, 7), np.uint8))          # silhouette pixels mix in the background
        idx = np.flatnonzero(ok)
        idx = idx[obj[v[idx].astype(int), u[idx].astype(int)] > 0]
        if len(idx) > max_per_frame:
            idx = rng.choice(idx, max_per_frame, replace=False)
        if len(idx) == 0:
            continue
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
        col = img[v[idx].astype(int), u[idx].astype(int), ::-1].astype(np.float32) / 255.0
        out_i.append(idx)
        out_c.append(srgb_to_lin(col))
        out_n.append(nc[idx])
        out_f.append(np.full(len(idx), fi))
    if not out_i:
        raise RuntimeError("no visible splat samples: is the splat in the SfM coordinate frame of this job?")
    log(f"{sum(len(i) for i in out_i)} colour samples from {len(out_i)} frames")
    return (np.concatenate(out_i), np.concatenate(out_c), np.concatenate(out_n).astype(np.float64),
            np.concatenate(out_f))


# --------------------------------------------------------------------------- the fit
def fit_lighting(ip: np.ndarray, col: np.ndarray, nc: np.ndarray, n_splats: int, iters: int = 8, log=print):
    """Alternating robust least squares. Returns (L (3, 9) per-channel SH lighting normalised to mean 1,
    albedo (n_splats, 3) linear, count per splat, sample weights)."""
    Y = sh_basis(nc)                                              # (S, 9)
    S = len(ip)
    cnt = np.bincount(ip, minlength=n_splats).astype(np.float64)
    rho = np.stack([np.bincount(ip, col[:, k], n_splats) / np.maximum(cnt, 1) for k in range(3)], 1)
    L = np.zeros((3, 9))
    L[:, 0] = 1 / 0.282095
    w = np.ones(S)
    for it in range(iters):
        for k in range(3):
            A = rho[ip, k][:, None] * Y
            sw = np.sqrt(w)[:, None]
            L[k] = np.linalg.lstsq(A * sw, col[:, k] * sw[:, 0], rcond=None)[0]
        E = np.clip(Y @ L.T, 0.12, None)                          # (S, 3), floor keeps dark samples from blowing up
        mean_e = E.mean(0)
        L /= mean_e[:, None]                                      # mean lighting 1 per channel: white, scale-free
        E = np.clip(Y @ L.T, 0.12, None)
        for k in range(3):
            num = np.bincount(ip, w * E[:, k] * col[:, k], n_splats)
            den = np.bincount(ip, w * E[:, k] ** 2, n_splats)
            rho[:, k] = np.where(den > 1e-9, num / np.maximum(den, 1e-9), rho[:, k])
        pred = rho[ip] * E
        res = (col - pred)
        mad = np.median(np.abs(res)) * 1.4826 + 1e-4
        w = 1.0 / (1.0 + (np.abs(res).max(1) / (2.5 * mad)) ** 2)  # Huber-like: highlights, shadows, leaks fade out
        if it in (0, iters - 1):
            log(f"  iteration {it + 1}: robust residual {mad:.4f}, down-weighted samples {(w < 0.5).mean() * 100:.1f} %")
    return L, rho, cnt, w, Y


def view_variation(ip: np.ndarray, vals: np.ndarray, n_splats: int, min_obs: int = 4) -> float:
    """Mean over splats of std/mean of a colour across the frames that see it (luminance)."""
    y = vals.mean(1)
    cnt = np.bincount(ip, minlength=n_splats)
    m = np.bincount(ip, y, n_splats) / np.maximum(cnt, 1)
    v = np.bincount(ip, y * y, n_splats) / np.maximum(cnt, 1) - m * m
    sel = cnt >= min_obs
    return float(np.mean(np.sqrt(np.maximum(v[sel], 0)) / np.maximum(m[sel], 1e-4)))


# --------------------------------------------------------------------------- entry
def delight(job, ply_in: Path, ply_out: Path, log=print) -> dict:
    header, props, data = read_ply_raw(ply_in)
    n = len(data)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
    conf = opacity > 0.3
    c0 = np.median(xyz[conf], 0)
    radius = float(np.percentile(np.linalg.norm(xyz[conf] - c0, axis=1), 98))
    log(f"{n} splats, radius {radius:.3f}")
    normals = np.zeros((n, 3))
    normals[conf] = 0
    # normals from the confident splats; the rest borrow the nearest confident splat's normal
    idx_c = np.flatnonzero(conf)
    nrm = surface_normals(xyz[idx_c])
    normals[idx_c] = nrm
    if (~conf).any():
        _, nn = cKDTree(xyz[idx_c]).query(xyz[~conf], k=1)
        normals[~conf] = nrm[nn]
    ip, col, nc, fr = collect(job, xyz, normals, radius, log=log)
    L, rho, cnt, w, Y = fit_lighting(ip, col, nc, n, log=log)
    E = np.clip(Y @ L.T, 0.12, None)
    before = view_variation(ip, col, n)
    after = view_variation(ip, col / E, n)
    log(f"colour variation between views (std/mean per splat): {before:.3f} -> {after:.3f}")
    seen = cnt >= 2
    albedo = np.clip(lin_to_srgb(np.clip(rho, 0, 1.5)), 0, 1)
    out = data.copy()
    for k in range(3):
        out[f"f_dc_{k}"] = np.where(seen, (albedo[:, k] - 0.5) / SH_C0, data[f"f_dc_{k}"]).astype(np.float32)
    for p in props:
        if p.startswith("f_rest_"):
            out[p] = 0.0                                   # no view-dependent colour: highlights go
    write_ply_raw(ply_out, header, out)
    # a human-readable summary of the recovered light: direction of the strongest lobe in the camera frame
    lum = L.mean(0)
    dirs = np.array([lum[3], lum[1], lum[2]])
    ambient_frac = float(abs(lum[0] * 0.282095) / (abs(lum[0] * 0.282095) + np.linalg.norm(lum[1:4]) * 0.488603 + 1e-9))
    info = {"splats": int(n), "splats_with_albedo": int(seen.sum()), "samples": int(len(ip)),
            "view_variation_before": round(before, 4), "view_variation_after": round(after, 4),
            "variation_reduction_pct": round(100 * (1 - after / max(before, 1e-9)), 1),
            "light_direction_camera_frame": [round(float(v), 3) for v in (dirs / (np.linalg.norm(dirs) + 1e-12))],
            "ambient_share": round(ambient_frac, 3), "sh_lighting": np.round(L, 4).tolist(),
            "output": str(ply_out)}
    log(json.dumps({k: v for k, v in info.items() if k != "sh_lighting"}))
    return info


# --------------------------------------------------------------------------- demo renders
def lighting_sphere(L: np.ndarray, size: int = 220) -> np.ndarray:
    """The recovered lighting E(n) on a sphere as seen by the camera (x right, y down, z toward the scene), sRGB."""
    yy, xx = np.mgrid[0:size, 0:size]
    x, y = (xx - size / 2 + 0.5) / (size / 2), (yy - size / 2 + 0.5) / (size / 2)
    r2 = x * x + y * y
    inside = r2 < 1
    z = -np.sqrt(np.clip(1 - r2, 0, 1))                         # normals facing the camera have z < 0 here
    n = np.stack([x, y, z], -1).reshape(-1, 3)
    E = np.clip(sh_basis(n) @ L.T, 0, None).reshape(size, size, 3)
    img = np.where(inside[..., None], lin_to_srgb(E * 0.6), 0.12)
    return (img * 255).astype(np.uint8)


def demo(job, ply_orig: Path, ply_delit: Path, out_dir: Path, info: dict, n: int = 48, size: int = 480,
         log=print) -> list[Path]:
    """Side-by-side turntable (original | de-lit), the de-lit splat under a rotating white light, a still sheet and
    the recovered lighting. Returns the files written."""
    import subprocess

    import torch

    from .splat_render import Splats, load_ply, look_at, principal_frame, render
    from .stages.render import orbit
    from .util import find_tool

    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = load_ply(ply_orig), load_ply(ply_delit)
    ctr, axes, radius = principal_frame(b)
    ext = [float(np.ptp(b.xyz.numpy() @ axes[i])) for i in range(3)]
    ai = 0 if ext[0] / max(ext[2], 1e-9) > 1.8 else 2
    axis = axes[ai]
    start = axes[(ai + 1) % 3] + 0.35 * axes[(ai + 2) % 3]
    fov = np.deg2rad(40.0)
    f = 0.5 * size / np.tan(fov / 2)
    dist = 1.15 * radius / np.sin(fov / 2)
    bg = (0.10, 0.10, 0.10)
    nrm = torch.from_numpy(surface_normals(b.xyz.numpy().astype(np.float64)).astype(np.float32))

    def relit(light: np.ndarray) -> Splats:
        alb = b.sh[:, 0, :] * SH_C0 + 0.5
        shade = 0.30 + 0.70 * torch.clamp(nrm @ torch.from_numpy(light.astype(np.float32)), min=0)
        shade = shade / 0.65                                      # keep the average brightness comparable
        col = torch.pow(torch.pow(alb.clamp(0, 1), GAMMA) * shade[:, None], 1 / GAMMA).clamp(0, 1)
        return Splats(b.xyz, b.cov, b.opacity, ((col - 0.5) / SH_C0)[:, None, :])

    ff = find_tool("ffmpeg")
    paths = []

    def writer(name):
        p = out_dir / name
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
               f"{2 * size}x{size}", "-r", "24", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(p)]
        return p, subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def label(img, text):
        img = np.ascontiguousarray(img)
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    p1, w1 = writer("delight_turntable.mp4")
    p2, w2 = writer("delight_relit.mp4")
    stills = []
    for i in range(n):
        cam = orbit(ctr, axis, start, dist, 2 * np.pi * i / n)
        R = look_at(cam, ctr, axis)
        ia = render(a, R, cam, f, size, size, bg)
        ib = render(b, R, cam, f, size, size, bg)
        w1.stdin.write(np.concatenate([label(ia, "original (baked light)"), label(ib, "de-lit (albedo)")], 1).tobytes())
        # light circling the object, expressed in the world frame of the splat
        th = 2 * np.pi * i / n
        l = np.cos(th) * axes[(ai + 1) % 3] + np.sin(th) * axes[(ai + 2) % 3] + 0.6 * axis
        l = l / np.linalg.norm(l)
        ic = render(relit(l), R, cam, f, size, size, bg)
        w2.stdin.write(np.concatenate([label(ic, "de-lit, white light circling"), label(ia, "original")], 1).tobytes())
        if i % (n // 6) == 0:
            stills.append(np.concatenate([ia, ib, ic], 1))
        log(f"  demo frame {i + 1}/{n}")
    for w in (w1, w2):
        w.stdin.close()
        w.wait()
    sheet = np.concatenate(stills, 0)
    from .util import imwrite_rgb
    imwrite_rgb(out_dir / "delight_sheet.jpg", cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2)), 90)
    L = np.array(info["sh_lighting"])
    sph = lighting_sphere(L)
    cv2.putText(sph, "recovered room light", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    imwrite_rgb(out_dir / "lighting.png", sph)
    (out_dir / "delight.json").write_text(json.dumps({k: v for k, v in info.items()}, indent=1))
    paths += [p1, p2, out_dir / "delight_sheet.jpg", out_dir / "lighting.png", out_dir / "delight.json"]
    return paths
