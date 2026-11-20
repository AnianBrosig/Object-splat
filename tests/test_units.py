import json

import numpy as np
import pytest

from gsturn.config import Config, parse_box
from gsturn.events import Events, read_events, summarize
from gsturn.stages.features import filter_in_mask, make_pairs, match_sift
from gsturn.stages.frames import select
from gsturn.stages.segment import clean_object


def test_config_roundtrip_and_coercion(tmp_path):
    c = Config().update({"max-frames": "120", "refine_focal": "false", "object_box": "0.8,0.1,0.2,0.9",
                         "hand_boxes": "0,0,0.1,0.1;0.5,0.5,0.6,0.6"})
    assert c.max_frames == 120 and c.refine_focal is False
    assert c.object_box == [0.2, 0.1, 0.8, 0.9]          # normalised order
    assert len(c.hand_boxes) == 2
    c.save(tmp_path / "c.json")
    assert Config.load(tmp_path / "c.json") == c
    with pytest.raises(KeyError):
        Config().update({"nope": 1})


def test_config_load_ignores_old_fields(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({"max_frames": 50, "removed_field": True}))
    assert Config.load(tmp_path / "c.json").max_frames == 50


def test_parse_box_rejects_pixels():
    with pytest.raises(ValueError):
        parse_box("10,10,200,200")


def test_events_summary(tmp_path):
    ev = Events(tmp_path, echo=False)
    ev.stage_status("frames", "running")
    ev.progress(5, 10, "half", force=True)
    ev.artifact(tmp_path / "a.png", "plot")
    ev.stage_status("frames", "done", "ok", stats={"kept": 3})
    ev.stage_status("segment", "running")
    ev.close()
    st = summarize(read_events(tmp_path))
    assert st["frames"]["status"] == "done" and st["frames"]["pct"] == 100
    assert st["frames"]["artifacts"][0]["path"] == "a.png"
    assert st["frames"]["stats"] == {"kept": 3}
    assert st["segment"]["status"] == "running"
    assert st["sfm"]["status"] == "pending"


def test_frame_selection_motion_and_cap():
    n = 100
    thumbs = np.zeros((n, 10, 10), np.float32)
    for i in range(n):
        thumbs[i] += i * 0.005                     # steady change
    sharp = np.full(n, 100.0)
    sharp[50] = 1.0                                 # one blurry frame
    kept, _, _ = select(sharp, thumbs, min_motion=0.012, sharp_ratio=0.55, win=15, max_frames=1000)
    assert 50 not in kept
    assert all(b - a >= 3 for a, b in zip(kept, kept[1:]))   # 0.005/frame -> every 3rd frame
    kept2, _, _ = select(sharp, thumbs, 0.0, 0.55, 15, max_frames=10)
    assert len(kept2) == 10


def test_pairs():
    assert len(make_pairs(5, 2, 0, 100)) == 10                      # exhaustive
    p = make_pairs(200, 8, 20, 100)
    assert (0, 1) in p and (0, 8) in p and (0, 9) not in p          # window
    assert any(b - a > 100 for a, b in p)                           # long-range pairs exist
    assert all(a < b for a, b in p) and len(set(p)) == len(p)


def test_clean_object_removes_hand_and_specks():
    obj = np.zeros((100, 100), bool)
    obj[20:80, 20:80] = True
    obj[2:4, 2:4] = True                       # speck
    hand = np.zeros((100, 100), np.uint8)
    hand[40:60, 0:30] = 255                    # hand overlapping the left edge
    m = clean_object(obj, hand, erode_px=1, hand_dilate_px=3)
    assert m[3, 3] == 0
    assert m[50, 25] == 0 and m[50, 35] == 255 and m[30, 30] == 255


def test_filter_in_mask_keeps_best_inside():
    mask = np.zeros((50, 50), np.uint8)
    mask[10:40, 10:40] = 255
    kp = np.array([[5, 5], [20, 20], [30, 30], [25, 25]], np.float32)
    sc = np.array([9, 1, 3, 2], np.float32)
    sel = filter_in_mask(kp, sc, mask, keep=2)
    assert set(sel.tolist()) == {2, 3}


def test_sift_mutual_ratio_matching():
    rng = np.random.default_rng(0)
    d = rng.random((50, 128)).astype(np.float32)
    d /= d.sum(1, keepdims=True)
    d = np.sqrt(d)
    perm = rng.permutation(50)
    feats = [{"desc": d}, {"desc": d[perm]}]
    m, _ = match_sift(feats, 0, 1)
    assert len(m) == 50
    assert all(perm[j] == i for i, j in m)


def test_splat_renderer_single_gaussian():
    import torch
    from gsturn.splat_render import SH_C0, Splats, look_at, render
    # one red, opaque, isotropic gaussian at the origin; camera on -z looking at it
    sh = torch.zeros(1, 16, 3)
    sh[0, 0] = torch.tensor([0.5, -0.5, -0.5]) / SH_C0     # colour (1, 0, 0) after +0.5
    s = Splats(xyz=torch.zeros(1, 3), cov=torch.eye(3)[None] * 0.05 ** 2, opacity=torch.tensor([0.99]), sh=sh)
    cam = np.array([0.0, 0.0, -2.0])
    R = look_at(cam, np.zeros(3), np.array([0.0, -1.0, 0.0]))
    img = render(s, R, cam, f=200.0, W=64, H=48, bg=(0, 0, 0))
    assert img.shape == (48, 64, 3)
    cy, cx = np.unravel_index(img[..., 0].argmax(), img.shape[:2])
    assert abs(cx - 32) <= 1 and abs(cy - 24) <= 1
    assert img[24, 32, 0] > 240 and img[24, 32, 1] < 10
    assert img[0, 0].max() == 0                              # background untouched


# --------------------------------------------------------------------------- hole filling
def test_fill_small_holes():
    from gsturn.stages.segment import fill_small_holes
    obj = np.zeros((200, 200), np.uint8)
    obj[20:180, 20:180] = 1                    # object area 25600
    obj[40:46, 40:46] = 0                      # small hole (36 px < 0.5 %): filled
    obj[100:130, 100:130] = 0                  # large hole (900 px > 0.5 %): kept
    obj[40:46, 140:146] = 0                    # small hole under a hand blob: kept
    obj[90:96, 170:200] = 0                    # notch open to the border side: kept
    obj[90:96, 180:200] = 0
    hand = np.zeros_like(obj)
    hand[38:48, 138:148] = 1
    out, filled = fill_small_holes(obj, hand, max_frac=0.005)
    expect = np.zeros_like(obj)
    expect[40:46, 40:46] = 1
    assert np.array_equal(filled, expect)
    assert out[43, 43] == 1 and out[115, 115] == 0 and out[43, 143] == 0 and out[93, 175] == 0


def test_clean_object_reports_filled_pixels():
    obj = np.zeros((100, 100), bool)
    obj[10:90, 10:90] = True
    obj[50:53, 50:53] = False
    m, f = clean_object(obj, np.zeros((100, 100), np.uint8), erode_px=1, hand_dilate_px=3,
                        fill_holes_frac=0.005, return_filled=True)
    assert m[51, 51] == 255 and f[51, 51] == 255 and f.sum() == 9 * 255


# --------------------------------------------------------------------------- quality
def _texture(h=256, w=256, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 255, (h // 4, w // 4), np.uint8)
    return cv2.resize(t, (w, h), interpolation=cv2.INTER_NEAREST)


def test_quality_flags_blur_only_where_blurred():
    import cv2
    from gsturn.config import Config
    from gsturn.stages.quality import sharpness_score, tile_energies
    sharp = _texture(256, 320)
    sharp[:, 96:160] = 128                     # blank grey band: cannot be judged, must not be flagged
    blurred = sharp.copy()
    blurred[:, 224:] = cv2.GaussianBlur(sharp, (0, 0), 3)[:, 224:]   # right part blurred
    mask = np.full(sharp.shape, 255, np.uint8)
    cfg = Config()
    a, b = tile_energies(blurred, mask, 32), tile_energies(sharp, mask, 32)
    s, _, _ = sharpness_score(a, [b], cfg)
    assert s[:, 8:88].mean() > 0.95            # sharp left part
    assert s[:, 104:152].mean() > 0.95         # blank band
    assert s[:, 168:200].mean() > 0.95         # sharp part next to the blur
    assert s[:, 256:].mean() < 0.1             # blurred right part
    s2, _, _ = sharpness_score(b, [a], cfg)    # the sharp frame itself is fine everywhere
    assert s2.min() > 0.95


def test_quality_specular_vs_white_print():
    import cv2
    from gsturn.stages.quality import bad_light_mask
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = (180, 40, 40)                     # red surface
    cv2.circle(img, (50, 50), 4, (255, 255, 255), -1)       # small saturated highlight
    img[100:190, 100:190] = (235, 235, 235)                  # large white print, not clipped
    for x in range(110, 180, 12):                            # with dark lettering on it
        cv2.putText(img, "a", (x, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    img[20:30, 140:190] = (255, 40, 40)                      # bright red print: one channel saturated
    bad, _ = bad_light_mask(img, 230, 40)
    assert bad[50, 50] and not bad[20, 150] and not bad[25, 160]
    assert bad[110:180, 110:180].mean() < 0.01             # lettering area (square corners may flag)


# --------------------------------------------------------------------------- refine geometry
def test_rotation_angle_and_sampson_filter():
    import cv2
    from gsturn.stages.refine import fundamental_from_poses, rotation_angle_deg, sampson_px
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    Ri = np.eye(3)
    ti = np.array([0, 0, 5.0])
    Rj = cv2.Rodrigues(np.array([0, np.radians(20), 0]))[0]
    tj = np.array([0.3, 0, 5.0])
    assert abs(rotation_angle_deg(Ri, Rj) - 20) < 1e-6
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (50, 3))

    def proj(R, t):
        c = X @ R.T + t
        return (c @ K.T)[:, :2] / c[:, 2:]
    xi, xj = proj(Ri, ti), proj(Rj, tj)
    F = fundamental_from_poses(Ri, ti, Rj, tj, K)
    assert sampson_px(F, xi, xj).max() < 1e-6
    wrong = xj[rng.permutation(50)]
    assert np.median(sampson_px(F, xi, wrong)) > 4


def test_rgba_images_match_frames_when_frames_are_excluded(tmp_path):
    """Regression: with quality-excluded frames, every RGBA image must still hold its own frame and mask."""
    import cv2
    from gsturn.events import Events
    from gsturn.pipeline import Job
    from gsturn.stages.sfm import build_rgba_images
    job = Job(tmp_path / "job")
    img_dir, mask_dir = job.work("frames") / "images", job.work("segment") / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    names = [f"{i:05d}.jpg" for i in range(4)]
    for i, n in enumerate(names):
        cv2.imwrite(str(img_dir / n), np.full((8, 8, 3), 40 * (i + 1), np.uint8))
        cv2.imwrite(str(mask_dir / n.replace(".jpg", ".png")), np.full((8, 8), 10 * (i + 1), np.uint8))
    stems = ["00000", "00002", "00003"]                      # 00001 excluded by the quality stage
    ds = build_rgba_images(job, stems, names, Events(None))
    for s in stems:
        rgba = cv2.imread(str(ds / f"{s}.png"), cv2.IMREAD_UNCHANGED)
        i = int(s)
        assert abs(int(rgba[0, 0, 0]) - 40 * (i + 1)) <= 2 and rgba[0, 0, 3] == 10 * (i + 1), s


def test_torch_after_pycolmap_does_not_crash():
    """Regression: pycolmap's OpenMP runtime loaded first made LightGlue segfault (refine stage, exit 139)."""
    import subprocess
    import sys
    code = ("from gsturn.pipeline import preload_torch; preload_torch(); import pycolmap, torch; "
            "from lightglue import LightGlue, SuperPoint; x = torch.rand(1, 3, 240, 320); "
            "f = SuperPoint(max_num_keypoints=256).eval().extract(x); "
            "LightGlue(features='superpoint').eval()({'image0': f, 'image1': f})")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=300)
    assert r.returncode == 0, r.stderr.decode()[-2000:]


def test_bg_boxes_are_never_object():
    from gsturn.stages.segment import first_frame_labels
    rng = np.random.default_rng(0)
    img = rng.integers(60, 90, (120, 160, 3)).astype(np.uint8)
    img[30:90, 40:120] = (210, 190, 150)                      # brightly lit table inside the box
    for k in range(60):                                       # thin dark diagonal "pen"
        img[85 - k:89 - k, 45 + k:47 + k] = (30, 30, 120)
    obj_box = [40 / 160, 25 / 120, 112 / 160, 95 / 120]
    bg = [[42 / 160, 27 / 120, 70 / 160, 50 / 120], [80 / 160, 65 / 120, 110 / 160, 92 / 120]]
    lab = first_frame_labels(img, obj_box, [], bg)
    assert lab[27:50, 42:70].max() == 0 and lab[65:92, 80:110].max() == 0
    assert lab[70, 60] == 1 or lab[71, 61] == 1 or lab[72, 60] == 1   # the pen itself survives
