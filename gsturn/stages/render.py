"""Stage 7: turntable videos of the trained splat, one full 360 degree turn about each principal axis."""
from __future__ import annotations

import subprocess

import numpy as np

from ..config import Config
from ..events import Events
from ..pipeline import StageError
from ..util import contact_sheet, find_tool, imwrite_rgb

AXIS_NAMES = ["long", "middle", "short"]


def hex_rgb(s: str) -> tuple[float, float, float]:
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) / 255 for i in (0, 2, 4))


def orbit(ctr: np.ndarray, axis: np.ndarray, start_dir: np.ndarray, dist: float, theta: float):
    """Camera on a circle around `axis` (through ctr), looking at ctr, with `axis` as screen-up."""
    k = axis / np.linalg.norm(axis)
    v = start_dir - k * (k @ start_dir)
    v = v / np.linalg.norm(v)
    d = v * np.cos(theta) + np.cross(k, v) * np.sin(theta)      # Rodrigues, v is perpendicular to k
    return ctr + dist * d


def run(job, cfg: Config, ev: Events) -> dict:
    if not cfg.render_enabled:
        ev.log("render_enabled=false: no turntable videos (use the splat viewer in the web UI)")
        return {"_msg": "skipped (render_enabled=false)", "skipped": True}
    import torch
    from ..splat_render import load_ply, look_at, principal_frame, render

    ply = job.dir / "work" / "train" / "splat.ply"
    tstats = job.dir / "work" / "train" / "stats.json"
    if not ply.exists() or (tstats.exists() and '"skipped": true' in tstats.read_text()):
        ev.log("no trained splat (train_backend=none): nothing to render")
        return {"_msg": "skipped (no splat)", "skipped": True}
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise StageError("ffmpeg not found")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = job.work("render")
    s = load_ply(ply)
    ctr, axes, radius = principal_frame(s)
    size, n = cfg.render_size, cfg.render_frames
    fov = np.deg2rad(40.0)
    f = 0.5 * size / np.tan(fov / 2)
    dist = 1.15 * radius / np.sin(fov / 2)
    bg = hex_rgb(cfg.render_bg)
    total = 3 * n
    videos, firsts = [], []
    for ai in range(3):
        axis = axes[ai]
        start = axes[(ai + 1) % 3] + 0.35 * axes[(ai + 2) % 3]   # slightly oblique so faces are not edge-on
        path = out / f"turntable_{ai + 1}_{AXIS_NAMES[ai]}.mp4"
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{size}x{size}", "-r", str(cfg.render_fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
               "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(n):
                cam = orbit(ctr, axis, start, dist, 2 * np.pi * i / n)
                img = render(s, look_at(cam, ctr, axis), cam, f, size, size, bg)
                if i == 0:
                    firsts.append(img)
                assert proc.stdin is not None
                proc.stdin.write(img.tobytes())
                done = ai * n + i + 1
                ev.progress(done, total, f"video {ai + 1}/3 ({AXIS_NAMES[ai]} axis), frame {i + 1}/{n}")
        finally:
            if proc.stdin:
                proc.stdin.close()
            rc = proc.wait()
        if rc != 0:
            raise StageError(f"ffmpeg failed writing {path.name}")
        videos.append(path)
        ev.artifact(path, f"360 turn about the {AXIS_NAMES[ai]} principal axis", kind="video")
    imwrite_rgb(out / "render_preview.jpg", contact_sheet(firsts, cols=3, thumb_w=size // 2,
                                                          labels=[f"{a} axis" for a in AXIS_NAMES]))
    ev.artifact(out / "render_preview.jpg", "first frame of each turntable video")
    return {"videos": [str(v.relative_to(job.dir)) for v in videos], "frames_per_video": n, "size": size,
            "splats_rendered": int(len(s.xyz)), "_msg": f"3 videos, {n} frames each, {size}x{size}"}
