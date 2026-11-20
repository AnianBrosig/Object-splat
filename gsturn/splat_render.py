"""Small CPU Gaussian-splat rasterizer (torch), enough for preview turntable videos.

Standard 3DGS model: EWA projection of 3-D covariances, SH colour up to degree 3, tile-based front-to-back
alpha compositing. No CUDA needed; ~1 s per 720x720 frame for ~30k splats on a laptop CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
         1.445305721320277, -0.5900435899266435]
TILE = 16


@dataclass
class Splats:
    xyz: torch.Tensor        # (N,3)
    cov: torch.Tensor        # (N,3,3)
    opacity: torch.Tensor    # (N,)
    sh: torch.Tensor         # (N,K,3)  K = (deg+1)^2

    @property
    def degree(self) -> int:
        return int(round(self.sh.shape[1] ** 0.5)) - 1


def load_ply(path: Path, min_opacity: float = 0.005) -> Splats:
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
    g = lambda k: torch.from_numpy(np.ascontiguousarray(data[k]).astype(np.float32))  # noqa: E731
    xyz = torch.stack([g("x"), g("y"), g("z")], 1)
    opacity = torch.sigmoid(g("opacity"))
    scale = torch.exp(torch.stack([g(f"scale_{i}") for i in range(3)], 1))
    q = torch.stack([g(f"rot_{i}") for i in range(4)], 1)
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).view(-1, 3, 3)
    M = R * scale[:, None, :]
    cov = M @ M.transpose(1, 2)
    dc = torch.stack([g(f"f_dc_{i}") for i in range(3)], 1)[:, None, :]
    n_rest = sum(1 for p in props if p.startswith("f_rest_"))
    if n_rest:
        rest = torch.stack([g(f"f_rest_{i}") for i in range(n_rest)], 1).view(-1, 3, n_rest // 3).transpose(1, 2)
        sh = torch.cat([dc, rest], 1)
    else:
        sh = dc
    keep = opacity > min_opacity
    return Splats(xyz[keep], cov[keep], opacity[keep], sh[keep])


def eval_sh(sh: torch.Tensor, dirs: torch.Tensor, deg: int) -> torch.Tensor:
    res = SH_C0 * sh[:, 0]
    if deg >= 1:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        res = res - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
        if deg >= 2:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            res = (res + SH_C2[0] * xy * sh[:, 4] + SH_C2[1] * yz * sh[:, 5]
                   + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6] + SH_C2[3] * xz * sh[:, 7]
                   + SH_C2[4] * (xx - yy) * sh[:, 8])
            if deg >= 3:
                res = (res + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9] + SH_C3[1] * xy * z * sh[:, 10]
                       + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                       + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                       + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + SH_C3[5] * z * (xx - yy) * sh[:, 14]
                       + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return (res + 0.5).clamp_min(0.0)


def look_at(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World->camera rotation (rows x right, y down, z forward), COLMAP/OpenCV convention."""
    z = target - cam_pos
    z = z / np.linalg.norm(z)
    x = np.cross(-up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


@torch.no_grad()
def render(s: Splats, Rwc: np.ndarray, cam_pos: np.ndarray, f: float, W: int, H: int,
           bg=(0.1, 0.1, 0.1), cx: float | None = None, cy: float | None = None) -> np.ndarray:
    """Returns an RGB uint8 image. cx, cy: principal point in pixels (default: the image centre)."""
    Rt = torch.from_numpy(Rwc.astype(np.float32))
    c = torch.from_numpy(cam_pos.astype(np.float32))
    pc = (s.xyz - c) @ Rt.T
    front = pc[:, 2] > 0.02
    pc, cov3, op, sh, xyz = pc[front], s.cov[front], s.opacity[front], s.sh[front], s.xyz[front]
    x, y, z = pc.unbind(1)
    cx = W / 2 if cx is None else cx
    cy = H / 2 if cy is None else cy
    u, v = f * x / z + cx, f * y / z + cy
    J = torch.zeros(len(z), 2, 3)
    J[:, 0, 0] = f / z
    J[:, 0, 2] = -f * x / (z * z)
    J[:, 1, 1] = f / z
    J[:, 1, 2] = -f * y / (z * z)
    T = J @ Rt
    cov2 = T @ cov3 @ T.transpose(1, 2)
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    d = cov2[:, 1, 1] + 0.3
    det = a * d - b * b
    ok = det > 1e-12
    mid = 0.5 * (a + d)
    lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
    rad = torch.ceil(3 * torch.sqrt(lam))
    ok &= (u + rad > 0) & (u - rad < W) & (v + rad > 0) & (v - rad < H) & (rad < 4 * max(W, H))
    idx = torch.nonzero(ok).squeeze(1)
    if len(idx) == 0:
        return np.full((H, W, 3), np.array(bg) * 255, np.uint8)
    u, v, z, rad = u[idx], v[idx], z[idx], rad[idx]
    ca, cb, cc = (d / det)[idx], (-b / det)[idx], (a / det)[idx]
    op = op[idx]
    dirs = xyz[idx] - c
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    col = eval_sh(sh[idx], dirs, s.degree)

    tw, th = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    tx0 = ((u - rad) / TILE).floor().clamp(0, tw - 1).long()
    tx1 = ((u + rad) / TILE).floor().clamp(0, tw - 1).long()
    ty0 = ((v - rad) / TILE).floor().clamp(0, th - 1).long()
    ty1 = ((v + rad) / TILE).floor().clamp(0, th - 1).long()
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    cnt = nx * ny
    gid = torch.repeat_interleave(torch.arange(len(u)), cnt)
    start = torch.cumsum(cnt, 0) - cnt
    local = torch.arange(len(gid)) - start[gid]
    tile = (ty0[gid] + local // nx[gid]) * tw + (tx0[gid] + local % nx[gid])
    order = torch.argsort(z)                       # depth rank, then stable sort by tile
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    key = tile * len(u) + rank[gid]
    srt = torch.argsort(key)
    tile, gid = tile[srt], gid[srt]
    bounds = torch.searchsorted(tile, torch.arange(tw * th + 1))

    out = torch.empty(H, W, 3)
    bgt = torch.tensor(bg, dtype=torch.float32)
    py, px = torch.meshgrid(torch.arange(TILE, dtype=torch.float32), torch.arange(TILE, dtype=torch.float32),
                            indexing="ij")
    px, py = px.reshape(-1) + 0.5, py.reshape(-1) + 0.5
    for t in range(tw * th):
        y0, x0 = (t // tw) * TILE, (t % tw) * TILE
        hh, ww = min(TILE, H - y0), min(TILE, W - x0)
        s0, s1 = bounds[t].item(), bounds[t + 1].item()
        if s0 == s1:
            out[y0:y0 + hh, x0:x0 + ww] = bgt
            continue
        g = gid[s0:s1]
        dx = (px + x0)[None, :] - u[g][:, None]
        dy = (py + y0)[None, :] - v[g][:, None]
        power = -0.5 * (ca[g][:, None] * dx * dx + cc[g][:, None] * dy * dy) - cb[g][:, None] * dx * dy
        alpha = (op[g][:, None] * torch.exp(power.clamp_max(0))).clamp_max(0.99)
        alpha = torch.where(alpha < 1 / 255, torch.zeros_like(alpha), alpha)
        Tm = torch.cumprod(torch.cat([torch.ones(1, alpha.shape[1]), 1 - alpha[:-1]], 0), 0)
        w_ = Tm * alpha
        rgb = w_.T @ col[g] + (Tm[-1] * (1 - alpha[-1]))[:, None] * bgt
        out[y0:y0 + hh, x0:x0 + ww] = rgb.view(TILE, TILE, 3)[:hh, :ww]
    return (out.clamp(0, 1) * 255).round().byte().numpy()


def principal_frame(s: Splats) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre, principal axes (rows, longest first) and bounding radius of the confident splats."""
    sel = s.opacity > 0.3
    p = s.xyz[sel].numpy() if sel.sum() > 100 else s.xyz.numpy()
    lo, hi = np.percentile(p, 1, 0), np.percentile(p, 99, 0)
    p = p[np.all((p >= lo) & (p <= hi), 1)]
    ctr = np.median(p, 0)
    _, _, vt = np.linalg.svd(p - ctr, full_matrices=False)
    radius = float(np.percentile(np.linalg.norm(p - ctr, axis=1), 99))
    return ctr, vt, radius
