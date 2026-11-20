"""Stage 1: downscale the (4K) input video with ffmpeg."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

from ..config import Config
from ..events import Events
from ..util import find_tool, run_streaming


def probe(path: Path) -> dict:
    ffprobe = find_tool("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found (brew install ffmpeg)")
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames,codec_name:stream_side_data=rotation"
                          ":format=duration,size", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s.get("r_frame_rate", "30/1").split("/")
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    w, h = int(s["width"]), int(s["height"])
    if abs(rot) in (90, 270):
        w, h = h, w  # ffmpeg autorotates on decode
    return {"width": w, "height": h, "fps": float(num) / float(den or 1), "codec": s.get("codec_name"),
            "frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None,
            "duration": float(d["format"].get("duration", 0)), "size": int(d["format"].get("size", 0)),
            "rotation": rot}


def needs_transcode(info: dict, height: int = 0, fps: float = 0.0, trim_start: float = 0.0,
                    trim_end: float = 0.0) -> bool:
    """False when the original file can be read as is: no scaling, resampling or trimming asked for, and no
    rotation flag that only ffmpeg would bake into the pixels (OpenCV does not reliably apply it)."""
    shrink = bool(height) and height < info["height"]
    return shrink or bool(fps) or bool(trim_start) or bool(trim_end) or bool(info.get("rotation"))


def encoder_args(hwaccel: str, out_w: int = 1280, out_h: int = 720, fps: float = 30) -> tuple[list[str], list[str], str]:
    """Returns (input args, output codec args, encoder name)."""
    ffmpeg = find_tool("ffmpeg")
    encs = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    has_vt = "h264_videotoolbox" in encs
    if hwaccel in ("auto", "videotoolbox") and has_vt:
        return (["-hwaccel", "videotoolbox"],
                # constant-quality (-q:v) is Apple-Silicon only; use a generous bitrate (~0.3 bit/pixel)
                ["-c:v", "h264_videotoolbox", "-b:v", f"{max(2.0, 0.3 * out_w * out_h * fps / 1e6):.1f}M",
                 "-allow_sw", "1"], "h264_videotoolbox")
    if hwaccel == "videotoolbox":
        raise RuntimeError("h264_videotoolbox not in this ffmpeg build")
    return [], ["-c:v", "libx264", "-preset", "medium", "-crf", "17"], "libx264"


def downscale(src: Path, dst: Path, height: int = 720, fps: float = 0.0, hwaccel: str = "auto",
              trim_start: float = 0.0, trim_end: float = 0.0, ev: Optional[Events] = None) -> dict:
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found (brew install ffmpeg)")
    dst.unlink(missing_ok=True)   # never write through a link to the original (see run())
    info = probe(src)
    out_h = height if height and height < info["height"] else info["height"]
    out_w = int(round(info["width"] * out_h / info["height"]))
    in_args, codec, enc = encoder_args(hwaccel, out_w, out_h, fps or info["fps"])
    vf = []
    if height and height < info["height"]:
        vf.append(f"scale=-2:{height}:flags=lanczos")
    if fps:
        vf.append(f"fps={fps}")
    vf.append("format=yuv420p")
    trim = []
    if trim_start:
        trim += ["-ss", str(trim_start)]
    if trim_end:
        trim += ["-to", str(trim_end)]
    duration = (trim_end or info["duration"]) - trim_start
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-hide_banner", "-nostats", *in_args, *trim, "-i", str(src), "-an",
           "-vf", ",".join(vf), *codec, "-movflags", "+faststart", "-progress", "pipe:1", str(dst)]
    if ev:
        ev.log(" ".join(cmd))
    tail: list[str] = []

    def on_line(line: str) -> None:
        tail.append(line)
        del tail[:-30]
        m = re.match(r"out_time_us=(\d+)", line)
        if m and ev and duration > 0:
            t = int(m.group(1)) / 1e6
            ev.progress(min(t, duration), duration, f"{t:.1f}/{duration:.1f}s ({enc})")

    rc = run_streaming(cmd, on_line)
    if rc != 0 and enc == "h264_videotoolbox" and hwaccel == "auto":
        if ev:
            ev.warn("videotoolbox encode failed; retrying with libx264")
        return downscale(src, dst, height, fps, "none", trim_start, trim_end, ev)
    if rc != 0:
        raise RuntimeError("ffmpeg failed:\n" + "\n".join(tail))
    out = probe(dst)
    return {"input": info, "output": out, "encoder": enc,
            "size_ratio": round(out["size"] / max(info["size"], 1), 4)}


def convert_one(src: Path, dst: Path, cfg: Config, ev: Events, trim_start: float, trim_end: float) -> dict:
    """Downscale / trim one video to `dst`, or link the original when nothing needs to change."""
    info = probe(src)
    if not needs_transcode(info, cfg.target_height, cfg.target_fps, trim_start, trim_end):
        # nothing to change: link the original instead of re-encoding it (later stages read the small video)
        dst.unlink(missing_ok=True)
        dst.symlink_to(src.resolve())
        ev.log(f"using the original file, no re-encode: {src}")
        return {"input": info, "output": info, "encoder": "none (original)", "size_ratio": 1.0,
                "_msg": f"original {info['width']}x{info['height']}, {info['frames'] or '?'} frames, "
                        f"{info['size'] / 1e6:.0f} MB, not re-encoded"}
    r = downscale(src, dst, cfg.target_height, cfg.target_fps, cfg.hwaccel, trim_start, trim_end, ev)
    o, i = r["output"], r["input"]
    r["_msg"] = (f"{i['width']}x{i['height']} -> {o['width']}x{o['height']}, {o['frames']} frames, "
                 f"{o['size'] / 1e6:.1f} MB ({r['encoder']})")
    return r


def sync_second(job, cfg: Config, ev: Events) -> dict:
    """Time offset of video2 relative to video (time in video2 = time in video + offset) and the matching trim."""
    from ..sync import estimate_offset, plot_sync
    a, b = Path(cfg.video), Path(cfg.video2)
    if not b.exists():
        raise RuntimeError(f"video2 not found: {b}")
    if cfg.video2_sync == "manual":
        info = {"method": "manual", "offset": float(cfg.video2_offset), "confidence": None, "notes": []}
    else:
        ev.log(f"estimating the time offset between {a.name} and {b.name} ({cfg.video2_sync})")
        res = estimate_offset(a, b, "auto" if cfg.video2_sync == "auto" else cfg.video2_sync)
        out = job.work("downscale")
        plot_sync(res, out / "sync.png", f"{a.name} -> {b.name}")
        ev.artifact(out / "sync.png", "time offset estimate: audio and motion correlation")
        info = {"method": res.method, "offset": float(res.offset), "confidence": round(res.confidence, 2),
                "prior": res.prior, "audio": res.audio, "motion": res.motion, "notes": res.notes}
        for n in res.notes:
            ev.log("sync: " + n)
    off = info["offset"]
    ev.log(f"video2 = video + {off:+.3f} s ({info['method']})")
    t0 = cfg.trim_start + off
    info["trim_start2"] = max(0.0, t0)
    info["trim_end2"] = (cfg.trim_end + off) if cfg.trim_end else 0.0
    job.save_json("downscale", "sync.json", info)
    return info


def run(job, cfg: Config, ev: Events) -> dict:
    if not cfg.video:
        raise RuntimeError("no input video in config")
    src = Path(cfg.video)
    if not src.exists():
        raise RuntimeError(f"video not found: {src}")
    dst = job.work("downscale") / "video_small.mp4"
    r = convert_one(src, dst, cfg, ev, cfg.trim_start, cfg.trim_end)
    if cfg.video2:
        sync = sync_second(job, cfg, ev)
        r2 = convert_one(Path(cfg.video2), job.work("downscale") / "video2_small.mp4", cfg, ev, sync["trim_start2"],
                         sync["trim_end2"])
        r["video2"] = {"input": r2["input"], "output": r2["output"], "sync": sync}
        r["_msg"] = r["_msg"] + f"; video2 offset {sync['offset']:+.3f} s ({sync['method']})"
    return r
