"""Orientation-continuity test (Kalman-style) on synthetic hand-turned trajectories."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from gsturn.kalman import Track, find_breaks, interpolate, smooth


def trajectory(n=120, seed=0, jitter_deg=0.4, regrip=True):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 20, n))                      # frames are chosen by motion: uneven times
    t[0] = 0
    w = np.zeros((n, 3))
    axis = np.array([0.3, 1.0, 0.2])
    rate = np.radians(30) + np.radians(12) * np.sin(t / 2.3)        # 30 +- 12 deg/s
    ang = np.cumsum(np.r_[0, rate[1:] * np.diff(t)])
    if regrip:                                               # re-grips: the axis changes within ~0.6 s
        for tg in (6.0, 13.0):
            ang = ang + np.where(t > tg, np.radians(25), 0.0)
    rot = R.from_rotvec(np.outer(ang, axis / np.linalg.norm(axis)))
    noise = R.from_rotvec(rng.normal(0, np.radians(jitter_deg), (n, 3)))
    pos = np.stack([np.cos(t / 3), np.sin(t / 3), 0.1 * t], 1) * 0.3
    return Track(t, noise * rot, pos)


def test_clean_trajectory_has_no_breaks():
    for seed in range(5):
        assert find_breaks(trajectory(seed=seed)) == []


def test_a_150_degree_flip_is_found_once():
    tr = trajectory(seed=1)
    k = 61
    rot = tr.rot.as_matrix()
    flip = R.from_rotvec(np.radians(150) * np.array([0.0, 0.0, 1.0])).as_matrix()
    rot[k:] = flip @ rot[k:]                                 # everything after k is registered rotated: a block flip
    br = find_breaks(Track(tr.t, R.from_matrix(rot), tr.pos))
    assert [b["index"] for b in br] == [k], br
    assert br[0]["step_deg"] > 100


def test_a_single_bad_frame_is_found():
    tr = trajectory(seed=2)
    rot = tr.rot.as_matrix()
    rot[40] = R.from_rotvec(np.radians(120) * np.array([1.0, 0, 0])).as_matrix() @ rot[40]
    br = find_breaks(Track(tr.t, R.from_matrix(rot), tr.pos))
    assert br and br[0]["index"] == 40


def test_smoother_reduces_jitter_and_keeps_the_motion():
    tr = trajectory(seed=3, jitter_deg=1.5, regrip=False)
    clean = trajectory(seed=3, jitter_deg=0.0, regrip=False)
    sm = smooth(tr)
    err_raw = np.mean([(tr.rot[i] * clean.rot[i].inv()).magnitude() for i in range(len(tr.t))])
    err_sm = np.mean([(sm.rot[i] * clean.rot[i].inv()).magnitude() for i in range(len(tr.t))])
    assert err_sm < 0.8 * err_raw, (np.degrees(err_raw), np.degrees(err_sm))


def test_interpolate_fills_a_gap():
    tr = trajectory(seed=4, jitter_deg=0.0, regrip=False)
    keep = np.r_[0:40, 50:len(tr.t)]
    part = Track(tr.t[keep], tr.rot[keep], tr.pos[keep])
    est = interpolate(part, tr.t[40:50])
    err = [(est.rot[i] * tr.rot[40 + i].inv()).magnitude() for i in range(10)]
    assert np.degrees(max(err)) < 3.0
