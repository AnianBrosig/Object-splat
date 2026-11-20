"""Calibration on a synthetic checkerboard video with known intrinsics."""
import json

import cv2
import numpy as np
import pytest

from gsturn.calibrate import board_points, calibrate, scaled_intrinsics

W, H, F, CX, CY = 640, 480, 600.0, 322.0, 238.0
PATTERN, SQUARE = (6, 6), 22.0


def board_image(px_per_mm=8):
    n = PATTERN[0] + 1, PATTERN[1] + 1
    margin = SQUARE
    w = int((n[0] * SQUARE + 2 * margin) * px_per_mm)
    h = int((n[1] * SQUARE + 2 * margin) * px_per_mm)
    img = np.full((h, w), 255, np.uint8)
    for r in range(n[1]):
        for c in range(n[0]):
            if (r + c) % 2 == 0:
                x0 = int((margin + c * SQUARE) * px_per_mm)
                y0 = int((margin + r * SQUARE) * px_per_mm)
                img[y0:y0 + int(SQUARE * px_per_mm), x0:x0 + int(SQUARE * px_per_mm)] = 0
    return img, px_per_mm, margin


def render(tex, ppm, margin, rvec, tvec):
    K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1]])
    th, tw = tex.shape
    # board plane in mm: inner-corner (0,0) at the origin
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    mm = (src / ppm) - margin - SQUARE
    obj = np.hstack([mm, np.zeros((4, 1))]).astype(np.float64)
    dst = cv2.projectPoints(obj, rvec, tvec, K, None)[0].reshape(-1, 2).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_AREA, borderValue=110)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    tex, ppm, margin = board_image()
    rng = np.random.default_rng(3)
    centre = np.array([SQUARE * (PATTERN[0] - 1) / 2, SQUARE * (PATTERN[1] - 1) / 2, 0])
    path = tmp_path_factory.mktemp("cal") / "calibration.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for i in range(60):
        rvec = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.4, 0.4)])
        R = cv2.Rodrigues(rvec)[0]
        z = rng.uniform(250, 420)
        off = np.array([rng.uniform(-0.25, 0.25) * z, rng.uniform(-0.2, 0.2) * z, z])
        tvec = off - R @ centre
        vw.write(render(tex, ppm, margin, rvec, tvec))
    vw.release()
    return path


def test_calibrate_synthetic(video, tmp_path):
    out = tmp_path / "cal.json"
    res = calibrate(video, SQUARE, None, height=0, output=out, step=1, hwaccel="none", distortion="zero")
    assert res["pattern"] == list(PATTERN)
    assert abs(res["fx"] - F) / F < 0.01, res
    assert abs(res["cx"] - CX) < 2 and abs(res["cy"] - CY) < 2, res
    assert res["dist"] == [0, 0, 0, 0]
    saved = json.loads(out.read_text())
    assert saved["width"] == W and (tmp_path / "cal_views.jpg").exists() and (tmp_path / "cal_residuals.png").exists()
    fx, fy, cx, cy, _ = scaled_intrinsics(saved, 2 * W, 2 * H)
    assert abs(fx - 2 * saved["fx"]) < 1e-6 and abs(cx - (2 * saved["cx"] + 0.5)) < 1e-6


def test_board_points():
    p = board_points((3, 2), 10.0)
    assert p.shape == (6, 3) and p[-1].tolist() == [20.0, 10.0, 0.0]
