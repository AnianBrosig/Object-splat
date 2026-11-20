"""Two rigidly mounted cameras on a synthetic rotating cube: sync offset, per-camera frames, one SfM model."""
import json

import cv2
import numpy as np
import pytest

from gsturn.config import Config
from gsturn.events import read_events, summarize
from gsturn.pipeline import Job, run
from test_synthetic import F, H, W, face_textures, render, rot  # noqa: F401


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    rng = np.random.default_rng(2)
    tex = face_textures(rng)
    d = tmp_path_factory.mktemp("two")
    n, fps, off_frames = 80, 30, 15                      # camera 2 started 0.5 s before camera 1: video2 = video + 0.5 s
    Rc = rot(0.0, np.radians(35))                        # camera 2 looks at the object from 35 degrees to the side
    paths = []
    for name, nframes, shift, R_cam in (("cam1.mp4", n, 0, np.eye(3)), ("cam2.mp4", n + off_frames, off_frames, Rc)):
        p = d / name
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for j in range(nframes):
            i = max(0, min(n - 1, j - shift))
            a = 2 * np.pi * i / n
            vw.write(cv2.cvtColor(render(tex, R_cam @ rot(0.35 * np.sin(a), a)), cv2.COLOR_RGB2BGR))
        vw.release()
        paths.append(p)
    return paths


def test_two_cameras_one_model(videos, tmp_path):
    job = Job(tmp_path / "job")
    job.dir.mkdir()
    cfg = Config(video=str(videos[0]), video2=str(videos[1]), video2_sync="manual", video2_offset=0.5,
                 target_height=0, hwaccel="none", max_frames=50, min_motion=0.0, seg_backend="grabcut",
                 object_box=[0.22, 0.15, 0.78, 0.85], object_box2=[0.22, 0.15, 0.78, 0.85],
                 feature_backend="sift", max_keypoints=1500, sfm_backend="pycolmap", focal_prior=F,
                 train_backend="none", refine_enabled=False, scheduler="off", match_window=6, cross_window=1)
    cfg.save(job.config_path)
    ok = run(job, to_stage="sfm", echo=False)
    st = summarize(read_events(job.dir))
    assert ok, {k: (v["status"], v["msg"]) for k, v in st.items()}
    meta = json.loads((job.dir / "work" / "frames" / "frames.json").read_text())
    assert len(meta["cameras"]) == 2
    cams = [f["cam"] for f in meta["frames"]]
    assert cams.count(0) > 20 and cams.count(1) > 20
    assert all(f.get("partner") for f in meta["frames"] if f["cam"] == 1)
    # synchronised: the partner frame of camera 2 must be the one 0.5 s later in its own clock
    f1 = next(f for f in meta["frames"] if f["cam"] == 1)
    f0 = next(f for f in meta["frames"] if f["name"] == f1["partner"])
    assert abs(f1["t"] - f0["t"]) < 0.06, (f0, f1)
    assert (job.dir / "work" / "frames" / "images" / f1["name"]).exists()
    feats = json.loads((job.dir / "work" / "features" / "index.json").read_text())
    assert len(set(feats["cams"])) == 2
    sfm = json.loads((job.dir / "work" / "sfm" / "stats.json").read_text())
    per = sfm["per_camera"]
    assert per["camera1"]["registered"] >= 0.7 * per["camera1"]["frames"], sfm
    assert per["camera2"]["registered"] >= 0.7 * per["camera2"]["frames"], sfm
    assert sfm["mean_reproj_error"] <= 1.6, sfm
