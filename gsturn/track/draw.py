"""Overlay: object mask (projected convex hull), object axes, tracked points and a status panel."""
from __future__ import annotations

import cv2
import numpy as np

from .tracker import State


def hull_mask(hull: np.ndarray, R, t, K, shape) -> tuple[np.ndarray, np.ndarray]:
    """Silhouette of the convex hull at this pose (includes parts hidden by the hand) and its 2D outline."""
    proj, _ = cv2.projectPoints(hull.astype(np.float64), cv2.Rodrigues(R)[0], t, K, None)
    poly = cv2.convexHull(proj.reshape(-1, 2).astype(np.float32)).reshape(-1, 2)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.round(poly).astype(np.int32), 255, cv2.LINE_AA)
    return m, poly


def euler_xyz(R: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw in degrees for R = Rz @ Ry @ Rx (object axes in camera coordinates)."""
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        x, y, z = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])
    else:
        x, y, z = np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0
    return np.degrees([x, y, z])


def quat(R: np.ndarray) -> list[float]:
    """(w, x, y, z)"""
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        q[:] = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return (q * np.sign(q[0] or 1)).tolist()


def text(img, s, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_8)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(frame: np.ndarray, st: State, model, K: np.ndarray, info: dict, show_mask=True, show_points=True,
         show_axes=True) -> np.ndarray:
    out = frame.copy()
    if st.R is not None:
        R, t = st.R, st.t
        m, poly = hull_mask(model.hull, R, t, K, out.shape)
        if show_mask:
            # blend only inside the polygon's bounding box (full-frame numpy masking costs ~30 ms at 720p)
            x, y, w, h = cv2.boundingRect(np.round(poly).astype(np.int32))
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + w, out.shape[1]), min(y + h, out.shape[0])
            if x1 > x0 and y1 > y0:
                roi = out[y0:y1, x0:x1]
                tinted = cv2.addWeighted(roi, 0.6, np.full_like(roi, (60, 200, 60)), 0.4, 0)
                cv2.copyTo(tinted, m[y0:y1, x0:x1], roi)
        cv2.polylines(out, [np.round(poly).astype(np.int32)], True, (60, 255, 60), 2, cv2.LINE_AA)
        if show_axes:
            L = float(model.extent[0]) * 0.6
            ax = np.float64([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]])
            p, _ = cv2.projectPoints(ax, cv2.Rodrigues(R)[0], t, K, None)
            p = np.round(p.reshape(-1, 2)).astype(int)
            for i, c in ((3, (255, 80, 0)), (2, (0, 220, 0)), (1, (0, 0, 255))):   # z blue, y green, x red
                cv2.arrowedLine(out, tuple(p[0]), tuple(p[i]), c, 3, cv2.LINE_AA, tipLength=0.15)
        if show_points:
            for x, y in st.pts.astype(int):
                cv2.circle(out, (x, y), 2, (0, 255, 255), -1)
    # status panel
    lines = []
    col = {"track": (80, 255, 80), "coast": (0, 200, 255)}.get(st.status, (80, 80, 255))
    lines.append((f"{st.status.upper()}  {info.get('fps', 0):5.1f} fps  inliers {st.n_inliers}", col))
    if st.R is not None:
        u = model.units if model.units == "m" else "u"
        k = 100 if u == "m" else 1
        uu = "cm" if u == "m" else "u"
        tx, ty, tz = st.t * k
        r, p, y = euler_xyz(st.R)
        lines.append((f"pos  x {tx:7.1f}  y {ty:7.1f}  z {tz:7.1f} {uu}", (255, 255, 255)))
        lines.append((f"rot  roll {r:6.1f}  pitch {p:6.1f}  yaw {y:6.1f} deg", (255, 255, 255)))
    lines.append((f"track {info.get('track_ms', 0):4.1f} ms   reloc {info.get('reloc_ms', 0):5.0f} ms "
                  f"({info.get('reloc_kp', 0)} kp, {info.get('reloc_matches', 0)} matches, "
                  f"{info.get('reloc_inliers', 0)} inl)", (200, 200, 200)))
    if info.get("learned"):
        L = info["learned"]
        lines.append((f"learning: {L['frames']} frames, {L['samples']} samples, {L['keyframes']} new views",
                      (255, 200, 120)))
    for i, (s, c) in enumerate(lines):
        text(out, s, (10, 24 + 22 * i), color=c)
    return out
