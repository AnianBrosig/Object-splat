"""Stage 7: copy results to outputs/, write report.json and a preview image."""
from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from ..config import STAGES, Config
from ..events import Events
from .train import read_ply_gaussians


def preview_points(xyz: np.ndarray, rgb: np.ndarray, path: Path, title: str, alpha=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(xyz) == 0:
        return
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    keep = np.all((xyz >= lo) & (xyz <= hi), 1)
    if alpha is not None:
        keep &= alpha > 0.3
    xyz, rgb = xyz[keep], rgb[keep]
    # principal axes so the views are object-aligned
    c = xyz - xyz.mean(0)
    _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 20000)], full_matrices=False)
    p = c @ vt.T
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#202020")
    for ax, (a, b), lab in zip(axes, [(0, 1), (0, 2), (1, 2)], ["axis 1-2", "axis 1-3", "axis 2-3"]):
        order = np.argsort(p[:, 3 - a - b])  # paint back to front along the remaining axis
        ax.scatter(p[order, a], p[order, b], s=0.6, c=rgb[order], linewidths=0)
        ax.set_aspect("equal", "datalim")
        ax.set_facecolor("#202020")
        ax.set_title(lab, color="w")
        ax.tick_params(colors="gray")
    fig.suptitle(title, color="w")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def run(job, cfg: Config, ev: Events) -> dict:
    out = job.outputs
    stats_all = {}
    for st in (s for s in STAGES if s != "export"):
        p = job.dir / "work" / st / "stats.json"
        if p.exists():
            stats_all[st] = json.loads(p.read_text())
    ply = job.dir / "work" / "train" / "splat.ply"
    result = {}
    if ply.exists() and not stats_all.get("train", {}).get("skipped"):
        shutil.copy(ply, out / "splat.ply")
        xyz, rgb, op = read_ply_gaussians(ply)
        preview_points(xyz, rgb, out / "preview.png", f"splat centres ({len(xyz)} shown)", op)
        ev.artifact(out / "splat.ply", "Gaussian splat (open in SuperSplat or Brush)", kind="file")
        result["splat"] = "outputs/splat.ply"
    else:
        # no splat: export the sparse cloud so there is still something to look at
        import pycolmap
        rec = pycolmap.Reconstruction(str(job.dir / "work" / "sfm" / "dataset" / "sparse" / "0"))
        rec.export_PLY(str(out / "sparse.ply"))
        pts = np.array([p.xyz for p in rec.points3D.values()])
        cols = np.array([p.color for p in rec.points3D.values()]) / 255.0
        preview_points(pts, cols, out / "preview.png", f"sparse SfM points ({len(pts)})")
        ev.artifact(out / "sparse.ply", "sparse point cloud (no splat trained)", kind="file")
        result["sparse"] = "outputs/sparse.ply"
    for v in sorted((job.dir / "work" / "render").glob("turntable_*.mp4")):
        shutil.copy(v, out / v.name)
        ev.artifact(out / v.name, v.stem.replace("_", " "), kind="video")
    if (out / "preview.png").exists():
        ev.artifact(out / "preview.png", "preview (object-aligned projections)")

    import importlib.metadata as md
    versions = {}
    for pkg in ("torch", "numpy", "opencv-python-headless", "pycolmap", "lightglue", "cutie"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    report = {"job": job.id, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg.to_dict(),
              "stages": stats_all, "versions": versions, "python": sys.version.split()[0],
              "platform": f"{platform.system()} {platform.mac_ver()[0]} {platform.machine()}", **result}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    ev.artifact(out / "report.json", "report.json (all stage stats, config, versions)", kind="json")
    result["_msg"] = "outputs/" + ", ".join(sorted(p.name for p in out.iterdir()))
    return result
