"""Job configuration. Every field maps 1:1 to a CLI flag (--field-name) and to config.json."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "jobs"
VIDEOS_DIR = ROOT / "videos"

# single source of truth for the stage order (CLI, server, web UI and export all read it)
STAGES = ["downscale", "frames", "segment", "quality", "features", "sfm", "refine", "train", "render", "export"]


@dataclass
class Config:
    # input
    video: str = ""
    trim_start: float = 0.0          # seconds
    trim_end: float = 0.0            # seconds, 0 = until end
    # stage 1
    target_height: int = 0           # 0 = keep. With nothing else to change (fps, trim, rotation) the original file is used as is
    target_fps: float = 0.0          # 0 = keep
    hwaccel: str = "auto"            # auto | videotoolbox | none
    # stage 2
    max_frames: int = 200
    min_motion: float = 0.012
    sharp_keep_ratio: float = 0.55
    sharp_window: int = 15
    # stage 3
    seg_backend: str = "cutie"       # cutie | grabcut
    object_box: Optional[list[float]] = None          # relative x0,y0,x1,y1 on first selected frame
    hand_boxes: list[list[float]] = field(default_factory=list)
    bg_boxes: list[list[float]] = field(default_factory=list)   # definitely-background regions on frame 0
    erode_px: int = 2
    hand_dilate_px: int = 9
    cutie_max_internal_size: int = 480
    fill_holes_frac: float = 0.005   # fill interior mask holes smaller than this share of the object (0 = off)
    close_px: int = 0                # morphological closing radius before hole filling (0 = off)
    skin_filter: bool = True         # move skin-coloured object pixels next to the hand into the hand mask (leaks)
    # stage 3b: quality (trust maps against blur, defocus, reflections)
    quality_enabled: bool = True
    quality_tile: int = 32           # tile size in pixels for relative sharpness
    quality_window: int = 2          # neighbouring frames on each side used as sharpness reference
    quality_texture_floor: float = 20.0   # tiles whose reference energy is below this are not judged
    blur_rel_threshold: float = 0.4  # relative sharpness at which a tile counts as fully sharp
    specular_v: int = 230
    specular_s: int = 40
    quality_min_frame: float = 0.5   # frames with a smaller trusted share are excluded from SfM
    # stage 4
    feature_backend: str = "superpoint"   # superpoint | aliked | disk | sift
    max_keypoints: int = 1024
    feature_max_side: int = 1024     # long side of the extractor input in pixels (0 = native size)
    feature_crop: bool = True        # extract inside a fixed-size window that follows the visible object
    feature_crop_margin: float = 0.10     # window = largest visible object extent + this share on each side
    kp_hand_margin_px: int = 12      # drop keypoints this close to the hand (their descriptors see the finger)
    match_window: int = 4
    match_stride_pairs: int = 20
    exhaustive_below: int = 40
    min_matches: int = 15
    match_device: str = "auto"       # auto = cpu (variable shapes are slow on MPS) | cpu | mps | cuda
    # stage 5
    sfm_backend: str = "pycolmap"    # pycolmap | colmap-cli
    colmap_bin: str = ""             # "" = search PATH and the conda env
    focal_prior: float = 0.0         # pixels at working resolution, 0 = 1.2 * max(w, h)
    calibration: str = ""            # calibration JSON from `gsturn calibrate`; fixes the intrinsics
    refine_focal: Optional[bool] = None   # None = auto: refine without calibration, keep fixed with one
    # optional second camera, rigidly mounted to the first and recording the same object at the same time
    video2: str = ""                 # path of the second video ("" = single camera)
    video2_offset: float = 0.0       # seconds: time in video2 = time in video + offset
    video2_sync: str = "manual"      # manual | motion (estimate the offset from the motion in both videos)
    calibration2: str = ""           # calibration JSON of camera 2 ("" = estimate the focal length)
    object_box2: Optional[list[float]] = None   # first-frame boxes of video2 (relative x0,y0,x1,y1)
    hand_boxes2: list[list[float]] = field(default_factory=list)
    bg_boxes2: list[list[float]] = field(default_factory=list)
    rig: bool = True                 # (kept for old configs) the rigid mounting is checked after SfM, see sfm stats "rig"
    cross_window: int = 1            # camera 1 frame i is also matched with camera 2 frames i-w..i+w
    continuity_check: bool = True    # Kalman-style test for orientation jumps no hand motion explains
    sfm_focal_retry: bool = True     # on a jump or an implausible free focal: re-map with fixed candidate focals
    sfm_focal_candidates: str = "0.85,0.95,0.75"   # tried in this order, as multiples of the longer image side
    min_registered_ratio: float = 0.7
    max_reproj_error: float = 1.5
    # stage 5b: pose-guided re-matching
    refine_enabled: bool = False         # +0.1-0.5 track length for 9-15 % of the runtime on hand-held videos
    refine_mode: str = "remap"           # remap: re-run the mapper on the extended database | triangulate: keep poses
    refine_max_angle: float = 60.0       # degrees of relative rotation between candidate views (60 gained +6.3 % track length on IMG_8902, 30 gained nothing)
    refine_min_gap: int = 0              # minimum frame distance of candidate pairs, 0 = match_window
    refine_pairs_per_image: int = 5
    refine_max_epipolar_px: float = 4.0  # Sampson distance limit for the pose-guided filter
    refine_iterations: int = 1
    # stage 6
    train_backend: str = "brush"     # brush | opensplat | none (none stops after SfM: look at the sparse cloud)
    train_steps: int = 7000
    train_cmd: str = ""              # template, "" = backend default
    train_mask_mode: str = "weighted"   # weighted: object + empty background trained, hand/uncertain pixels ignored
    #                                  alpha: background and hand forced transparent | ignore: masks/ folder, only
    #                                  object pixels count | ignore_quality: like ignore, minus untrusted pixels
    train_crop: bool = True          # train on a per-frame crop of the visible object (same pixel density, fewer pixels)
    train_crop_margin: float = 0.06  # extra crop size on each side, share of the object extent
    train_crop_pad_px: int = 24      # plus this many pixels of empty background for the black-background loss
    train_hand_margin_px: int = 12   # pixels this close to the hand (and the hand itself) are not trained
    train_edge_ramp_px: int = 6      # loss weight ramps up over this distance from ignored regions
    train_max_resolution: int = 0    # Brush --max-resolution (0 = Brush default, 1920)
    train_residual_pass: bool = False    # short first training, find pixels that disagree with the other views,
    train_residual_steps: int = 2000     # ignore them, then train again
    train_residual_thresh: float = 0.30  # colour error (0..1) above which a trained pixel counts as an outlier
    # stage 7
    render_enabled: bool = False     # turntable videos from the CPU rasteriser (10-23 % of the runtime); the web viewer replaces them
    render_frames: int = 120         # frames per 360 degree turn
    render_size: int = 720           # square output, pixels
    render_fps: int = 30
    render_bg: str = "#1a1a1a"
    # misc
    device: str = "auto"             # auto | cpu | mps | cuda
    scheduler: str = "resource"      # resource: heavy stages of different jobs take turns per resource (cpu, gpu) |
    #                                  job: one job at a time for the whole run | off (also GSTURN_SCHEDULER)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def update(self, data: dict[str, Any], strict: bool = True) -> "Config":
        known = {f.name: f for f in dataclasses.fields(self)}
        for k, v in data.items():
            k = k.replace("-", "_")
            if k not in known:
                if strict:
                    raise KeyError(f"unknown config field: {k}")
                continue  # e.g. a field from an older version in an existing config.json
            setattr(self, k, coerce(known[k], v))
        return self

    @classmethod
    def load(cls, path: Path) -> "Config":
        cfg = cls()
        if path.exists():
            cfg.update(json.loads(path.read_text()), strict=False)
        return cfg

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


def field_type(f: dataclasses.Field) -> str:
    return str(f.type)


def coerce(f: dataclasses.Field, v: Any) -> Any:
    t = field_type(f)
    if v is None:
        return None
    if t == "int":
        return int(v)
    if t == "float":
        return float(v)
    if t in ("bool", "Optional[bool]"):
        if isinstance(v, str):
            v = v.strip().lower()
            if t == "Optional[bool]" and v in ("", "auto", "none", "null"):
                return None
            return v in ("1", "true", "yes", "on")
        return bool(v)
    if t == "str":
        return str(v)
    if "list[list[float]]" in t:
        if isinstance(v, str):
            return [parse_box(s) for s in v.split(";") if s.strip()]
        return [parse_box(b) for b in v]
    if "list[float]" in t:
        return parse_box(v)
    return v


def parse_box(v: Any) -> list[float]:
    if isinstance(v, str):
        v = [float(x) for x in v.split(",")]
    box = [float(x) for x in v]
    if len(box) != 4:
        raise ValueError(f"box needs 4 numbers x0,y0,x1,y1, got {v}")
    x0, y0, x1, y1 = box
    box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    if not all(0.0 <= x <= 1.0 for x in box):
        raise ValueError(f"box must be relative (0..1): {v}")
    return box


def refine_focal(cfg: Config) -> bool:
    """Effective refine_focal: explicit value if set, else False with a calibration and True without."""
    if cfg.refine_focal is not None:
        return bool(cfg.refine_focal)
    return not cfg.calibration


def resolve_device(name: str, purpose: str = "") -> str:
    """auto -> cuda, else mps (Apple GPU / AMD via Metal), else cpu.

    purpose "segment": Cutie gives the same masks on MPS but runs about 2x slower there than on this Intel CPU, so
    `auto` keeps it on the CPU (an explicit device is respected). pycolmap and torch's MPS backend do not mix in one
    process (segfault, see memory/mps-on-intel-mac), so once pycolmap has been imported (sfm, refine) MPS falls
    back to the CPU."""
    import sys
    dev = name
    if name == "auto":
        dev = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                dev = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and purpose != "segment":
                dev = "mps"
        except Exception:  # noqa: BLE001
            pass
    if dev == "mps" and "pycolmap" in sys.modules:
        return "cpu"
    return dev
