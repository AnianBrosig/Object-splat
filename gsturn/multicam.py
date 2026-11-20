"""One or two rigidly mounted cameras: file naming and the per-frame camera bookkeeping shared by all stages.

Camera 0 is `video`, camera 1 is `video2`. Frames of camera 1 are stored next to the others with the prefix
`c1_` (images, masks, hands, quality maps all use the same stem), and frames.json says which camera a frame
belongs to and which camera-0 frame it was recorded together with:

  {"width", "height",                      # camera 0
   "cameras": [{"width", "height", "video", "calibration", "offset"}, ...],
   "frames": [{"name", "src_index", "sharpness", "cam": 0|1, "t": seconds, "partner": "<cam 0 name>"}, ...]}

Single-camera jobs have none of the new keys and behave exactly as before.
"""
from __future__ import annotations

PREFIX = "c{}_"


def stem_of(name: str) -> str:
    return name.rsplit(".", 1)[0]


def frame_name(cam: int, index: int, ext: str = "jpg") -> str:
    return f"{PREFIX.format(cam) if cam else ''}{index:05d}.{ext}"


def cam_of(frame: dict) -> int:
    return int(frame.get("cam", 0))


def n_cameras(meta: dict) -> int:
    return len(meta.get("cameras") or [None])


def cameras(meta: dict) -> list[dict]:
    """Per-camera info; a single-camera frames.json is described by its top-level width/height."""
    return meta.get("cameras") or [{"width": meta["width"], "height": meta["height"]}]


def cams_for(meta: dict, stems: list[str]) -> list[int]:
    by = {stem_of(f["name"]): cam_of(f) for f in meta["frames"]}
    return [by[s] for s in stems]


def partners_for(meta: dict, stems: list[str]) -> dict[int, int]:
    """{index of a secondary-camera frame: index of its camera-0 partner}, only where both are in `stems`."""
    pos = {s: i for i, s in enumerate(stems)}
    out = {}
    for f in meta["frames"]:
        if cam_of(f) and f.get("partner"):
            a, b = stem_of(f["name"]), stem_of(f["partner"])
            if a in pos and b in pos:
                out[pos[a]] = pos[b]
    return out


def size_for(meta: dict, cam: int) -> tuple[int, int]:
    c = cameras(meta)[cam]
    return int(c["width"]), int(c["height"])
