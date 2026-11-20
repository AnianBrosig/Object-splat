"""End-to-end on a synthetic rotating textured cube in front of a fixed camera (torch-free backends)."""
import json

import cv2
import numpy as np
import pytest

from gsturn.config import Config
from gsturn.events import read_events, summarize
from gsturn.pipeline import Job, run

W, H, F = 640, 480, 520.0


def face_textures(rng, n=6, size=256):
    tex = []
    for _ in range(n):
        t = np.full((size, size, 3), rng.integers(60, 200, 3), np.uint8)
        for _ in range(60):
            c = tuple(int(x) for x in rng.integers(0, 255, 3))
            p = tuple(int(x) for x in rng.integers(0, size, 2))
            if rng.random() < 0.5:
                cv2.circle(t, p, int(rng.integers(4, 22)), c, -1)
            else:
                q = tuple(int(x) for x in rng.integers(0, size, 2))
                cv2.rectangle(t, p, q, c, int(rng.integers(-1, 4)) or -1)
        tex.append(t)
    return tex


def rot(ax, ay):
    cx, sx, cy, sy = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rx @ ry


FACES = [  # corner indices (counter-clockwise seen from outside), outward normal
    ([0, 1, 2, 3], [0, 0, -1]), ([5, 4, 7, 6], [0, 0, 1]), ([4, 0, 3, 7], [-1, 0, 0]),
    ([1, 5, 6, 2], [1, 0, 0]), ([4, 5, 1, 0], [0, -1, 0]), ([3, 2, 6, 7], [0, 1, 0]),
]
CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float) * 0.9


def render(tex, R, t=np.array([0, 0, 6.0])):
    img = np.full((H, W, 3), 128, np.uint8)
    cam = (R @ CORNERS.T).T + t
    uv = np.stack([F * cam[:, 0] / cam[:, 2] + W / 2, F * cam[:, 1] / cam[:, 2] + H / 2], 1)
    order = sorted(range(6), key=lambda f: -cam[FACES[f][0]].mean(0)[2])  # back to front
    s = tex[0].shape[0]
    src = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
    for f in order:
        idx, n = FACES[f]
        nc = R @ np.array(n, float)
        if nc @ cam[idx].mean(0) >= 0:      # facing away
            continue
        Hm = cv2.getPerspectiveTransform(src, uv[idx].astype(np.float32))
        warped = cv2.warpPerspective(tex[f], Hm, (W, H))
        m = cv2.warpPerspective(np.full((s, s), 255, np.uint8), Hm, (W, H))
        img[m > 128] = warped[m > 128]
    return img


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    rng = np.random.default_rng(1)
    tex = face_textures(rng)
    path = tmp_path_factory.mktemp("vid") / "cube.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    n = 90
    for i in range(n):
        a = 2 * np.pi * i / n
        vw.write(cv2.cvtColor(render(tex, rot(0.35 * np.sin(a), a)), cv2.COLOR_RGB2BGR))
    vw.release()
    return path


def test_pipeline_on_synthetic_cube(video, tmp_path):
    job = Job(tmp_path / "job")
    job.dir.mkdir()
    cfg = Config(video=str(video), target_height=0, hwaccel="none", max_frames=90, min_motion=0.0,
                 seg_backend="grabcut", object_box=[0.22, 0.15, 0.78, 0.85], feature_backend="sift",
                 max_keypoints=1500, sfm_backend="pycolmap", focal_prior=F, train_backend="none")
    cfg.save(job.config_path)
    ok = run(job, echo=False)
    st = summarize(read_events(job.dir))
    assert ok, {k: (v["status"], v["msg"]) for k, v in st.items()}
    sfm = json.loads((job.dir / "work" / "sfm" / "stats.json").read_text())
    assert sfm["registered_ratio"] >= 0.7, sfm
    assert sfm["mean_reproj_error"] <= 1.5, sfm
    assert (job.dir / "outputs" / "report.json").exists()
    assert (job.dir / "work" / "sfm" / "dataset" / "images").is_dir()
    assert all(st[s]["status"] in ("done", "warn") for s in st)


@pytest.mark.parametrize("mode", ["remap", "triangulate"])
def test_refine_closes_the_loop(video, tmp_path, mode):
    """Only sequential pairs are matched, so the seam (frame 0 vs 89, same side of the cube) is never matched
    in the first pass. refine must find and add those loop pairs."""
    job = Job(tmp_path / "job")
    job.dir.mkdir()
    cfg = Config(video=str(video), target_height=0, hwaccel="none", max_frames=90, min_motion=0.0,
                 seg_backend="grabcut", object_box=[0.22, 0.15, 0.78, 0.85], feature_backend="sift",
                 max_keypoints=1500, sfm_backend="pycolmap", focal_prior=F, train_backend="none",
                 exhaustive_below=0, match_stride_pairs=0, match_window=6, refine_mode=mode)
    cfg.save(job.config_path)
    ok = run(job, to_stage="refine", echo=False)
    st = summarize(read_events(job.dir))
    assert ok, {k: (v["status"], v["msg"]) for k, v in st.items()}
    q = json.loads((job.dir / "work" / "quality" / "quality.json").read_text())
    assert len(q["frames"]) == 90 and not q["excluded"]           # clean synthetic frames: nothing excluded
    assert (job.dir / "work" / "features" / "descriptors.npz").exists()
    r = json.loads((job.dir / "work" / "refine" / "stats.json").read_text())
    assert r["pairs_added"] > 0 and r["accepted"] and r["mode"] == mode, r
    assert r["max_gap_added"] > 70, r                              # pairs across the seam
    assert r["after"]["mean_track_length"] > r["before"]["mean_track_length"], r
    assert r["after"]["mean_reproj_error"] <= r["before"]["mean_reproj_error"] + 0.1, r
    assert (job.dir / "work" / "sfm" / "sparse_first_pass" / "0").is_dir()
    assert (job.dir / "work" / "sfm" / "dataset" / "sparse" / "0").is_dir()
