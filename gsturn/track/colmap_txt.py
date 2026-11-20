"""Minimal reader for COLMAP text models.

pycolmap is deliberately not imported here: loading it in the same process breaks torch's MPS backend (segfault
on this Mac), and the tracker needs MPS. If only a binary model exists, pycolmap converts it in a subprocess.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Image:
    id: int
    name: str
    R: np.ndarray            # world -> camera
    t: np.ndarray
    xy: np.ndarray           # (N, 2) COLMAP pixel convention (centre of first pixel at 0.5)
    pid: np.ndarray          # (N,) point3D id, -1 if none


@dataclass
class Model:
    width: int
    height: int
    params: list[float]      # PINHOLE fx fy cx cy (or SIMPLE_* converted)
    images: dict[int, Image] = field(default_factory=dict)
    point_ids: np.ndarray = None
    xyz: np.ndarray = None
    rgb: np.ndarray = None
    error: np.ndarray = None
    track_len: np.ndarray = None


def qvec_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def ensure_text(binary_dir: Path, text_dir: Path) -> Path:
    if (text_dir / "images.txt").exists():
        return text_dir
    text_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-c", "import pycolmap,sys; pycolmap.Reconstruction(sys.argv[1]).write_text("
                    "sys.argv[2])", str(binary_dir), str(text_dir)], check=True)
    return text_dir


def _lines(p: Path):
    with open(p) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                yield line.rstrip("\n")


def read_model(text_dir: Path) -> Model:
    cam = next(_lines(text_dir / "cameras.txt")).split()
    model_name, w, h, prm = cam[1], int(cam[2]), int(cam[3]), [float(x) for x in cam[4:]]
    if model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        prm = [prm[0], prm[0], prm[1], prm[2]]
    elif model_name not in ("PINHOLE", "OPENCV"):
        raise ValueError(f"unsupported camera model {model_name}")
    m = Model(w, h, prm[:4])
    it = _lines(text_dir / "images.txt")
    for head in it:
        pts = next(it, "")
        a = head.split()
        q, t = [float(x) for x in a[1:5]], np.array([float(x) for x in a[5:8]])
        v = np.array(pts.split(), dtype=np.float64).reshape(-1, 3) if pts.strip() else np.zeros((0, 3))
        m.images[int(a[0])] = Image(int(a[0]), a[9], qvec_to_R(q), t, v[:, :2].astype(np.float32),
                                    v[:, 2].astype(np.int64))
    ids, xyz, rgb, err, tl = [], [], [], [], []
    for line in _lines(text_dir / "points3D.txt"):
        a = line.split()
        ids.append(int(a[0]))
        xyz.append([float(x) for x in a[1:4]])
        rgb.append([int(x) for x in a[4:7]])
        err.append(float(a[7]))
        tl.append((len(a) - 8) // 2)
    m.point_ids, m.xyz, m.rgb = np.array(ids), np.array(xyz), np.array(rgb, np.uint8)
    m.error, m.track_len = np.array(err), np.array(tl)
    return m
