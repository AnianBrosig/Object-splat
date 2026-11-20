"""Constant-velocity Kalman filter / RTS smoother on camera poses, used as a continuity test for SfM.

A hand-turned object cannot teleport: between two selected frames its orientation changes by a few degrees. SfM
sometimes breaks that (symmetric or edge-on objects such as a round bowl are ambiguous when the focal length is
wrong), and the result is a block of frames registered with a consistent but wrong orientation. A smoother
cannot repair such a block, but the filter's *innovation* (how far the next pose is from what the motion so
far predicted, in units of the expected process noise) flags it reliably, and that is what `find_breaks` returns.
The smoother (local polynomial fit) and `interpolate` are used to fill in unregistered frames and to damp pose jitter.

Rotation is handled as an error state: the filter keeps an angular velocity (rad/s) and a rotation vector
residual, the position part is a plain 3-D constant-velocity filter.

Time comes from the frame timestamps (frames are selected by motion, so the spacing is not uniform).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Track:
    t: np.ndarray            # (N,) seconds
    rot: R                   # (N,) rotations (world -> camera or camera -> world, consistently)
    pos: np.ndarray          # (N, 3) camera centres


def _angles(rot: R, i: int, j: int) -> float:
    return float(np.degrees((rot[j] * rot[i].inv()).magnitude()))


def robust_rate(track: Track) -> float:
    """Typical angular speed (rad/s) from the median of the step rates: robust against the breaks themselves."""
    dt = np.maximum(np.diff(track.t), 1e-3)
    ang = np.array([(track.rot[i + 1] * track.rot[i].inv()).magnitude() for i in range(len(dt))])
    return float(np.median(ang / dt))


def find_breaks(track: Track, k: float = 6.0, min_deg: float = 45.0) -> list[dict]:
    """Orientation jumps that no hand motion explains.

    A constant-velocity Kalman filter over the rotation vectors gives, for every frame, the innovation (angle
    between the predicted and the registered orientation). The expected size is the process noise: the median
    absolute innovation (robust, so one break does not inflate it) times 1.4826, scaled by the time step. A frame
    is a break when its innovation exceeds k sigmas and min_deg degrees. Frames after a break are re-seeded, so
    one jump gives one report, not a chain."""
    n = len(track.t)
    if n < 6:
        return []
    inn = np.zeros(n)
    w = np.zeros(3)                                    # angular velocity (rad/s)
    for i in range(1, n):
        dt = max(track.t[i] - track.t[i - 1], 1e-3)
        pred = R.from_rotvec(w * dt) * track.rot[i - 1]
        err = (track.rot[i] * pred.inv())
        inn[i] = np.degrees(err.magnitude())
        meas_w = (track.rot[i] * track.rot[i - 1].inv()).as_rotvec() / dt
        w = 0.6 * w + 0.4 * meas_w if inn[i] < max(min_deg, 3 * np.median(inn[1:i + 1]) + 1) else w   # alpha-beta update
    scale = 1.4826 * float(np.median(np.abs(inn[1:] - np.median(inn[1:])))) + 1e-6
    thr = max(min_deg, k * scale + float(np.median(inn[1:])))
    out = []
    for i in np.flatnonzero(inn > thr):
        step = _angles(track.rot, i - 1, i)
        if step > min_deg:
            out.append({"index": int(i), "step_deg": round(step, 1), "innovation_deg": round(float(inn[i]), 1),
                        "threshold_deg": round(float(thr), 1)})
    # one report per jump: a break at i also makes i + 1 look odd for a moment
    merged = []
    for b in out:
        if not merged or b["index"] - merged[-1]["index"] > 2:
            merged.append(b)
    return merged


def continuity_score(track: Track, breaks: list[dict] | None = None) -> dict:
    """Numbers for choosing between several registrations of the same frames."""
    dt = np.maximum(np.diff(track.t), 1e-3)
    ang = np.array([np.degrees((track.rot[i + 1] * track.rot[i].inv()).magnitude()) for i in range(len(dt))])
    br = find_breaks(track) if breaks is None else breaks
    return {"breaks": len(br), "max_step_deg": round(float(ang.max()), 1) if len(ang) else 0.0,
            "median_step_deg": round(float(np.median(ang)), 1) if len(ang) else 0.0}


def smooth(track: Track, sigma_frames: float = 1.6, deg: int = 2) -> Track:
    """Damp pose jitter: every pose becomes the value of a Gaussian-weighted local polynomial fit (degree `deg`) over
    its neighbours in time. The rotation is fitted in the local frame of the pose itself (the neighbours' rotations
    relative to it, as rotation vectors), so a long video that turns many times around never wraps. The width is
    sigma_frames times the median frame spacing, so uneven timestamps are handled and a quick re-grip is not smeared
    over more than a frame or two. Meant for jitter and interpolation, not for repairing breaks."""
    n = len(track.t)
    if n < 5:
        return track
    sig = sigma_frames * float(np.median(np.diff(track.t)))
    rots, pos = [], np.zeros_like(track.pos)
    for i in range(n):
        dt = track.t - track.t[i]
        w = np.exp(-0.5 * (dt / sig) ** 2)
        sel = np.flatnonzero(w > 1e-3)
        if len(sel) < deg + 1:
            rots.append(track.rot[i])
            pos[i] = track.pos[i]
            continue
        A = np.vander(dt[sel], deg + 1, increasing=True)
        sw = np.sqrt(w[sel])[:, None]
        rel = (track.rot[sel] * track.rot[i].inv()).as_rotvec()
        c_rot = np.linalg.lstsq(A * sw, rel * sw, rcond=None)[0]
        c_pos = np.linalg.lstsq(A * sw, track.pos[sel] * sw, rcond=None)[0]
        rots.append(R.from_rotvec(c_rot[0]) * track.rot[i])
        pos[i] = c_pos[0]
    return Track(track.t.copy(), R.concatenate(rots), pos)


def interpolate(track: Track, t_new: np.ndarray) -> Track:
    """Poses at other times (for frames SfM could not register): slerp for rotation, linear for position."""
    from scipy.spatial.transform import Slerp
    t_new = np.clip(t_new, track.t[0], track.t[-1])
    sl = Slerp(track.t, track.rot)
    pos = np.stack([np.interp(t_new, track.t, track.pos[:, k]) for k in range(3)], 1)
    return Track(t_new, sl(t_new), pos)
