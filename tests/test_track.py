"""Live tracking: geometry helpers, pose filter, COLMAP text reader, KLT + PnP on a synthetic scene."""
import cv2
import numpy as np

from gsturn.track.colmap_txt import qvec_to_R, read_model
from gsturn.track.draw import euler_xyz, hull_mask, quat
from gsturn.track.live import scaled_K
from gsturn.track.model import object_frame, rot_err_deg
from gsturn.track.tracker import PoseFilter, inliers, lk_forward_backward


def _rot(rx, ry, rz):
    return cv2.Rodrigues(np.radians([rx, ry, rz]))[0]


def test_object_frame_is_right_handed_and_aligned():
    rng = np.random.default_rng(0)
    box = rng.uniform(-1, 1, (4000, 3)) * [3.0, 1.5, 0.5]        # long x, middle y, short z
    R = _rot(20, -35, 50)
    world = box @ R.T + [5, -2, 1]
    c, axes, ext = object_frame(world)
    assert np.isclose(np.linalg.det(axes), 1.0)
    assert np.allclose(c, [5, -2, 1], atol=0.1)
    assert ext[0] > ext[1] > ext[2]
    assert abs(abs(axes[0] @ R[:, 0]) - 1) < 1e-2                 # first axis = long box edge


def test_quat_and_euler_roundtrip():
    for angles in ([10, 20, 30], [-170, 5, 90], [0, 89, 0]):
        R = _rot(*angles)
        w, x, y, z = quat(R)
        Rq = qvec_to_R([w, x, y, z])
        assert np.allclose(Rq, R, atol=1e-6)
        r, p, yw = np.radians(euler_xyz(R))
        Re = _rot(0, 0, np.degrees(yw)) @ _rot(0, np.degrees(p), 0) @ _rot(np.degrees(r), 0, 0)
        assert np.allclose(Re, R, atol=1e-6)


def test_pose_filter_smooths_jitter_but_follows_motion():
    f = PoseFilter(smooth=0.25)
    R0, t0 = np.eye(3), np.array([0, 0, 1.0])
    rng = np.random.default_rng(1)
    out = [f.update(_rot(*rng.normal(0, 0.2, 3)) @ R0, t0 + rng.normal(0, 1e-3, 3))[0] for _ in range(50)]
    raw_jitter = 0.2 * np.sqrt(3)
    assert np.mean([rot_err_deg(R, R0) for R in out[10:]]) < 0.6 * raw_jitter
    Rb = _rot(0, 30, 0)                                            # a fast 30 deg move is followed at once
    R, _ = f.update(Rb, t0)
    assert rot_err_deg(R, Rb) < 0.5


def test_scaled_K_keeps_principal_point_physical():
    K = np.array([[1000, 0, 639.5], [0, 1000, 359.5], [0, 0, 1.0]])
    K2 = scaled_K(K, (1280, 720), (640, 360))
    assert np.allclose(K2[0, 0], 500) and np.allclose(K2[:2, 2], [319.5, 179.5])


def test_read_colmap_text_model(tmp_path):
    (tmp_path / "cameras.txt").write_text("# c\n1 PINHOLE 1280 720 1000 1001 640 360\n")
    (tmp_path / "images.txt").write_text("# i\n7 1 0 0 0 0.1 0.2 0.3 1 00007.png\n10.5 20.5 3 30.5 40.5 -1\n")
    (tmp_path / "points3D.txt").write_text("# p\n3 1 2 3 255 0 0 0.5 7 0 9 4\n")
    m = read_model(tmp_path)
    assert m.params == [1000, 1001, 640, 360]
    im = m.images[7]
    assert im.name == "00007.png" and np.allclose(im.R, np.eye(3)) and np.allclose(im.t, [0.1, 0.2, 0.3])
    assert im.pid.tolist() == [3, -1] and np.allclose(im.xy[0], [10.5, 20.5])
    assert m.point_ids.tolist() == [3] and m.track_len.tolist() == [2]


def test_klt_and_pnp_recover_a_synthetic_pose():
    """Textured plane seen by a camera: KLT follows points through a small motion, PnP recovers the pose."""
    rng = np.random.default_rng(3)
    K = np.array([[800, 0, 319.5], [0, 800, 239.5], [0, 0, 1.0]])
    tex = cv2.GaussianBlur(rng.integers(0, 255, (600, 600), dtype=np.uint8), (0, 0), 2)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)

    def render(R, t):
        # plane z=0, 1 unit = 300 texture px, centred
        H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ np.array([[1 / 300, 0, -1], [0, 1 / 300, -1], [0, 0, 1]])
        return cv2.warpPerspective(tex, H, (640, 480))

    R0, t0 = _rot(10, -5, 3), np.array([0.0, 0.0, 4.0])
    R1, t1 = _rot(12, -3, 4), np.array([0.03, -0.02, 4.05])
    g0, g1 = render(R0, t0), render(R1, t1)
    obj = np.column_stack([rng.uniform(-0.8, 0.8, (300, 2)), np.zeros(300)])
    p0 = cv2.projectPoints(obj, cv2.Rodrigues(R0)[0], t0, K, None)[0].reshape(-1, 2).astype(np.float32)
    p1, ok = lk_forward_backward(g0, g1, p0)
    assert ok.mean() > 0.8
    _, rvec, tvec, inl = cv2.solvePnPRansac(obj[ok], p1[ok].astype(np.float64), K, None,
                                            reprojectionError=2.0, flags=cv2.SOLVEPNP_SQPNP)
    assert rot_err_deg(cv2.Rodrigues(rvec)[0], R1) < 0.5
    assert np.linalg.norm(tvec.ravel() - t1) < 0.05
    assert len(inliers(obj[ok], p1[ok], rvec, tvec, K, 2.0)) > 0.8 * ok.sum()


def test_hull_mask_covers_projected_box():
    K = np.array([[500, 0, 159.5], [0, 500, 119.5], [0, 0, 1.0]])
    box = np.array([[x, y, z] for x in (-1, 1) for y in (-0.5, 0.5) for z in (-0.2, 0.2)], np.float64)
    m, poly = hull_mask(box, np.eye(3), np.array([0, 0, 10.0]), K, (240, 320))
    assert m[120, 160] == 255 and m[0, 0] == 0
    assert abs((m > 0).sum() - 100 * 50 * 1.04) < 0.15 * 100 * 50     # ~ (2*50) x (1*50) px front face, a bit more


def test_video_recorder_keeps_real_time_with_a_slow_camera(tmp_path):
    """A 15 fps camera recorded at 30 fps: every frame is written twice, so 2 s of stamps give a 2 s video."""
    from gsturn.track.camera import VideoRecorder
    rec = VideoRecorder(tmp_path / "r.mp4", 64, 48, fps=30, log=lambda *a: None)
    for i in range(30):
        rec.write(np.full((48, 64, 3), i * 8, np.uint8), stamp=100.0 + i / 15)
    rec.close()
    cap = cv2.VideoCapture(str(tmp_path / "r.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert abs(n - 59) <= 1 and rec.received == 30


def test_refine_apply_merges_points_keyframes_and_descriptors(tmp_path):
    import json
    from gsturn.track.model import ObjectModel
    from gsturn.track.refine import apply
    S, P = 2, 5
    rng = np.random.default_rng(0)
    z = dict(points=rng.normal(size=(P, 3)).astype(np.float32),
             desc_sum=rng.normal(size=(S, P, 256)).astype(np.float16), desc_count=np.ones((S, P), np.int32),
             scales=np.array([1.0, 0.5], np.float32), hull=rng.normal(size=(8, 3)).astype(np.float32),
             T_obj_world=np.eye(4), K=np.eye(3), image_size=np.array([640, 480]), ref_size_px=np.float32(300),
             kf_names=np.array(["00000"]), kf_R=np.eye(3)[None], kf_t=np.array([[0, 0, 5.0]]),
             kf_src_index=np.array([0]), point_clean=np.ones(P, bool), kf_ptr=np.array([0, 3]),
             kf_pid=np.array([0, 1, 2]), meta=json.dumps({"units": "model units", "extent": [1, 1, 1]}))
    path = tmp_path / "object_model.npz"
    np.savez(path, **z)
    add = {"points": np.ones((2, 3), np.float32), "add_sum": np.ones((S, P + 2, 256), np.float32),
           "add_cnt": np.ones((S, P + 2), np.int32), "kf_rows": [np.array([1, 5, 6])]}
    kfs = [{"i": 7, "R": np.eye(3), "t": np.array([0, 0, 4.0])}]
    r = apply(path, path, kfs, add, "test")
    m = ObjectModel.load(path)
    assert r == {**r, "keyframes": 1, "new_points": 2}
    assert len(m.points) == P + 2 and m.desc.shape == (S, P + 2, 256)
    assert np.allclose(np.linalg.norm(m.desc[:, -1], axis=1), 1)            # new points got descriptors
    assert m.kf_names[-1] == "test:7" and m.kf_src_index[-1] == -1
    assert [k.tolist() for k in m.kf_points] == [[0, 1, 2], [1, 5, 6]]
