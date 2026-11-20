# turntable2splat: hand-rotated object video -> 3D Gaussian splat

Turn a video of an object that you rotate in your hands in front of a **fixed camera** into a 3D Gaussian splat
(`.ply`). The pipeline separates the object (and your hand) from the background, finds transformer-based keypoints on
the object only, recovers camera poses with Structure-from-Motion, and trains a splat. A local web UI shows every
stage with progress bars and debug output.

Status (2026-11-17): **implemented and run** on a MacBook Pro 16" 2019 (i9-9880H, AMD Radeon Pro 5500M 4 GB, macOS 26.7,
Python 3.12): 19 jobs on 17 videos so far, 8 of them up to a trained splat. The former
*(verify)* items are settled; see **section 0** for what the project can do today, **section 14** for what was
verified and the first measured run (`IMG_8651`), and **section 16** for the results of every run and what they
say about where it works and where it breaks.

---

## 0. What it can do today

**Pipeline** (every stage is re-runnable on its own, writes artifacts under `jobs/<id>/work/<stage>/` and reports progress)

| Capability | Where | Notes |
|---|---|---|
| Video in, any size | `downscale`, `frames` | 4K30, 1080p30 and 1080p120 tested. Phone rotation flags are applied. With nothing to change (`target_height` 0, no trim/fps) the original file is read as is, no re-encode. |
| Sharp, well-spaced frames | `frames` | Laplacian sharpness against the rolling median plus a motion threshold; up to `max_frames` (200). Optional lens undistortion when the calibration has k1/k2. |
| Object / hand separation | `segment` | Cutie video segmentation seeded by boxes on the first frame (object, hands, certain-background boxes), GrabCut fallback. Hand pixels are removed from the object with a margin, small holes are filled, area-jump and empty-mask warnings. |
| Per-pixel trust maps | `quality` | Blur relative to neighbouring frames, clipped and specular highlights, filled holes; frames with too little trusted area are left out of SfM. |
| Keypoints and matching | `features` | SuperPoint / ALIKED / DISK + LightGlue, or RootSIFT; only inside the trusted object mask; sliding-window, long-range and exhaustive pairing. |
| Structure-from-Motion | `sfm` | pycolmap (or COLMAP CLI), one shared PINHOLE camera, fixed intrinsics from a calibration or a free focal; quality gate on registered share and reprojection error. |
| Pose-guided re-matching | `refine` | Loop-closure pairs picked from the first-pass poses, filtered by the epipolar geometry of the known poses, re-mapped or re-triangulated; accepted only if not worse. |
| Splat training | `train` | Brush (Rust, wgpu/Metal on the Radeon) with alpha, `ignore` or `ignore_quality` masks; held-out PSNR/SSIM every 1000 steps and a side-by-side sheet. |
| Preview | `render`, `export` | Three 360-degree turntable videos from a built-in CPU rasteriser, `splat.ply`, sparse cloud when no splat was trained, `report.json`. |

**Added on 2026-11-17** (details in sections 17-22)

| Capability | Where | Notes |
|---|---|---|
| Faster feature stage | `features` | SuperPoint runs on the GPU (MPS) on a fixed-size window around the visible object, 1024 keypoints, window 4: IMG_8924 took 639 s instead of 4610 s. |
| Hand-aware training | `train` (`train_mask_mode=weighted`) | Object and empty background are trained (background as black), the hand, a margin around it and the soft silhouette band are ignored, per-frame crops of the visible object (section 18). |
| Skin-leak filter | `segment` | Skin-coloured pixels the tracker called object, next to the hand, are moved to the hand mask. |
| Second camera | `video2` options | Rigidly mounted second camera: automatic time sync (file start times + audio + motion), per-camera masks, cross-camera matching with in-plane rotation probing, one SfM model with a rigid-mount consistency check (section 17). |
| Orientation-continuity check | `sfm` | Kalman-style test for pose jumps no hand motion explains; on a jump or an implausible free focal it re-maps with fixed candidate focals (section 19). |
| Job scheduler | all runs | Heavy stages of different jobs take turns per resource (cpu, gpu), earliest job first (section 22). |
| Splat viewer | web UI | WebGL viewer for the trained splat and the de-lit splat (section 21). |
| De-lighting | `gsturn delight` | Removes the baked-in room light: albedo splat + before/after demo (section 20). |
| Meshes and physics | `export-mesh`, `shelf` | Splat to STL/OBJ/PLY with colours, and a physics playground with a tiltable shelf (section 23). |

**Tools around it**

- `python -m gsturn run` for one video, `python -m gsturn batch` for many (one job per video, one after another, boxes
  from a JSON file, `--skip-done`, `--dry-run`, a report at the end), `python -m gsturn serve` for the web UI with
  upload, box drawing, a bulk queue, live progress, artifact gallery, per-stage re-run and cancel.
- `python -m gsturn calibrate`: checkerboard video to camera intrinsics (pinhole or k1/k2), with a check against the SfM
  focal of a job. `python -m gsturn doctor`: toolchain report.
- Live pose tracking of the finished object with a webcam (`export-model`, `track`, `camera`, `refine-model`,
  `track-eval`, see [docs/realtime-pose-tracking.md](docs/realtime-pose-tracking.md)): locates, masks and orients the object, can learn
  the webcam's look while tracking.
- 37 test functions (`python -m pytest tests`): units, API incl. path traversal, a synthetic rotating cube end to end, the
  refine stage on a seam that only loop pairs can close, the tracker.

**What the runs show it is good at** (section 16): objects with printed or textured surfaces held in one or two hands
and turned in front of a fixed camera, taking 25-30 % of the frame or more. All frames register, shapes are clean and
printed graphics are crisp. **Where it breaks**: small or untextured objects (61-87 keypoints per frame), round or
nearly symmetric objects seen edge-on (a pose flip of 149 degrees in `IMG_8925`), a free focal length that
converges to about twice the right value (3 of the 7 1080p runs that reached SfM), and the hand-covered parts of the object.

---

## 1. Why this is not a standard splat workflow

Splat trainers need one camera pose per image. With a fixed camera and a moving object, COLMAP sees a static
background and concludes the camera never moved. The fix used here:

1. Segment the object in every frame and **ignore everything else** (background, hand).
2. Detect and match keypoints **only inside the object mask**.
3. Run SfM on those matches. The object now behaves like a static scene seen by an orbiting camera.
4. Train the splat in the object's frame, with the masks applied.

Consequences you should plan for:

- Lighting is fixed in the room but *rotates relative to the object*. Use diffuse, even light and matte objects.
- Anywhere your fingers always cover the object becomes a hole. Re-grip during the video.
- Weak texture (a white egg) breaks feature matching. Put markers or a pattern on the object.
- Soft or deforming objects (plush, cloth) violate SfM's rigidity assumption. Keep them held in one shape.

---

## 2. Target machine and hard constraints

Target: **Intel MacBook Pro 2019** (x86_64) with a discrete **AMD Radeon** GPU (13" models have Intel Iris only).
Videos: 4K30 (`IMG_8645`-`8651`) and 1080p120 (`IMG_8900`-`8927`) iPhone recordings, in `videos/`.

| Constraint | Consequence |
|---|---|
| PyTorch stopped publishing macOS x86_64 wheels after **2.2.2** (Python <= 3.12) | Use **Python 3.11 or 3.12**, `torch==2.2.2`, `torchvision==0.17.2`, **`numpy<2`**. Newer torch means building from source. |
| No CUDA | All torch work runs on **CPU** by default. MPS (Metal, AMD GPU) is opt-in via `device=mps`; expect some ops to fail *(verify)*. |
| Cutie / LightGlue need torch | Both run on CPU. Budget minutes, not seconds, per stage (see section 11). |
| Gaussian splat trainers mostly assume CUDA | Use **Brush** (Rust, wgpu -> Metal), verified on the Radeon Pro 5500M. No Intel-mac release binary: build from source (binary is `brush_app`). OpenSplat needs a libtorch for Intel macOS, which is the same torch problem. |
| Homebrew has no Intel bottle for COLMAP | Default SfM backend is **pycolmap** (pip wheel for macOS x86_64 exists). The COLMAP CLI comes from conda-forge (`~/miniforge3/envs/colmap`), found automatically. |
| 4K input is heavy | The first stage downscales to 720p (configurable). |

**Toolchain discovery.** The Claude session this spec was written in is linked to a *Windows desktop*
(`desk-abr2`, with Rust/cargo, scoop, WSL), not the MacBook, and no folder was connected. So discovery is built
into the project: run `python -m gsturn doctor` on the Mac. It reports binaries (ffmpeg, colmap, cargo, brush,
opensplat), Python packages and versions, torch devices (cuda/mps), GPU model (`system_profiler`), ffmpeg hardware
encoders, and prints concrete warnings and which backends are usable. The web UI shows the same report.

---

## 3. Pipeline

```
video (4K30) --1 downscale--> video_small.mp4 (720p)
             --2 frames----->  ~100-250 sharp, well-spaced frames
             --3 segment---->  object mask + hand mask per frame           [Cutie]
             --3b quality--->  per-pixel trust maps (blur, defocus, highlights)
             --4 features--->  keypoints in trusted mask + pair matches    [SuperPoint/ALIKED + LightGlue]
             --5 sfm-------->  camera poses + sparse points                [COLMAP]
             --5b refine---->  pose-guided loop-closure matches, re-triangulate + BA
             --6 train------>  Gaussian splat                              [Brush]
             --7 render----->  3 turntable videos (360 deg per axis)        [built-in CPU rasterizer]
             --8 export----->  splat.ply + videos + previews + report
```

Each stage reads/writes a folder under `jobs/<id>/work/<stage>/`, can be re-run on its own, and emits progress
events (section 8).

### Stage 1: downscale
- `ffmpeg`, Lanczos scale to `target_height` (default 720, keeps aspect ratio), optional `target_fps`, optional
  trim. Audio dropped. Output H.264, CRF 17.
- On macOS, `hwaccel=auto` decodes with VideoToolbox and encodes with `h264_videotoolbox` at ~0.3 bit/pixel
  (constant-quality `-q:v` is Apple-Silicon only); if that fails it retries with `libx264`.
- Also a standalone command: `python -m gsturn downscale videos/a.mp4 --height 720`.
- Reason: 4K frames make Cutie, keypoint extraction and training several times slower with little gain for a
  single object.

### Stage 2: frame selection
- Read every frame, compute sharpness (variance of the Laplacian) and motion (mean absolute difference of a
  96-px grayscale thumbnail vs. the last kept frame).
- Keep a frame if it is sharp relative to its neighbours (`>= sharp_keep_ratio * rolling median`) and moved enough
  (`>= min_motion`). If more than `max_frames` remain, thin uniformly.
- Target: neighbouring kept frames rotate by a few degrees, so overlap stays high (~70-90 %).
- Debug output: sharpness/motion plot with kept frames marked, contact sheet.

### Stage 3: separation (Cutie)
- **Cutie** is a semi-supervised video object segmentation model (object-level memory with an object transformer).
  It needs a mask on the first frame and propagates it.
- First-frame mask: you draw a box around the object (and optionally boxes around your hand(s)) on frame 0 in the
  web UI. OpenCV GrabCut turns each box into a mask. Where the object and hand masks overlap, each pixel goes to
  whichever HSV colour histogram (hand pixels outside the object box vs. object pixels outside the hand boxes)
  fits better.
- Object = id 1, hand = id 2, tracked as separate objects. Hand pixels are removed from the object mask (dilated
  by `hand_dilate_px`). The object mask is eroded by `erode_px` to keep halos out of the silhouette.
- Fallback backend `grabcut`: per-frame GrabCut seeded from the previous mask. No torch needed, works on plain
  backgrounds only. Use it to test the rest of the pipeline.
- Cutie loads its model with a hard-coded `.cuda()` call; the implementation wraps it so it runs on CPU/MPS.
  Internal resolution is capped by `cutie_max_internal_size` (default 480) for speed.
- Mask clean-up per frame: remove the dilated hand, keep the large components, **fill small interior holes**
  (background regions not touching the border and smaller than `fill_holes_frac` of the object, default 0.5 %,
  unless mostly hand), remove the hand again, erode. Filled pixels go to `segment/filled/` (yellow in the
  overlays); the quality stage trusts them only half (a hole is usually a reflection or a dark shadow).
  `close_px` > 0 seals hairline cracks first (off by default: it can bridge object and fingers).
- Quality checks: mask-area-over-time plot (sudden jumps flag tracking failures), count of empty masks, overlay
  contact sheet. Stage status becomes `warn` when either trips.

### Stage 3b: quality (trust maps)
- Per frame a map Q (`quality/maps/<stem>.png`, 255 = trusted object pixel). Background:
  [docs/ignore-blur-defocus-reflections.md](docs/ignore-blur-defocus-reflections.md).
- **Sharpness relative to neighbouring frames**: Laplacian energy per 32 px tile (`quality_tile`), on a grid
  anchored at the mask centre so a tile covers the same part of the object in neighbouring frames. Each tile is
  compared with the sharpest version of it within +-`quality_window` frames; below `blur_rel_threshold` (0.4) it
  ramps to untrusted. Untextured tiles are not judged. A whole defocused frame drops at once.
- **Clipped and specular pixels**: any channel >= 250, or bright (`specular_v`), unsaturated (`specular_s`)
  pixels that stand out from their surroundings (top-hat), dilated by 2 px.
- `Q = mask * sharpness * (1 - highlights) * (1 - 0.5 * filled)`. Frames with fewer than `quality_min_frame`
  (50 %) trusted mask pixels are excluded from SfM (files are kept, names stay stable).
- Consumers: `features` drops excluded frames and keeps keypoints only where Q >= 128; `train_mask_mode=
  ignore_quality` ignores pixels with Q < 64 during training. `quality_enabled=false` turns it all off.
- Debug output: `overview.jpg` (red blur, magenta highlights, yellow filled holes), `trusted.png` (% trusted per
  frame), `worst.jpg`. `warn` if more than 25 % of frames are excluded.
- If tracking drifts: fix the first-frame box, or lower `max_frames` gaps, or re-run from a later frame (planned:
  click-to-correct on any frame).

### Stage 4: transformer keypoints and matching
- Extractors: **SuperPoint** (default), **ALIKED**, **DISK**, matched with **LightGlue** (attention-based matcher
  with early exit). Fallback: OpenCV **SIFT** (RootSIFT) with a ratio test and mutual check.
- Keypoints are detected on the full frame (3x the target count), then **only those inside the eroded object
  mask are kept**, best `max_keypoints` by score. The mask is never painted into the image, so no artificial edge
  features appear.
- Pairs: exhaustive when the frame count is <= `exhaustive_below` (100); otherwise a sliding window of
  `match_window` neighbours plus long-range pairs every `match_stride_pairs` frames (finds loop closure when the
  object comes back around).
- Pairs with fewer than `min_matches` matches are dropped.
- Debug output: keypoint overlays, drawn matches for sample pairs, a pair-by-pair match-count matrix.
- If matching fails on smooth objects: add texture/markers first. Next options are the dense matcher RoMa, or
  MASt3R/VGGT for initial poses (both heavy on CPU; not part of the first version).

### Stage 5: Structure-from-Motion (COLMAP)
- Keypoints and matches are written into a COLMAP database, geometric verification is run, and the incremental
  mapper reconstructs one model.
- **Single shared camera**, PINHOLE model. With `calibration` set (JSON from `python -m gsturn calibrate`, see
  section 6) fx, fy, cx, cy come from the file, scaled to the working resolution, and stay fixed
  (`refine_focal` = auto -> false). Without it the focal starts at `focal_prior` (default `1.2 * max(w, h)`) and
  is refined. If the calibration kept lens distortion (k1, k2), the `frames` stage undistorts the frames once and
  the pinhole matrix of the undistorted images is used, so COLMAP, Brush and the renderer stay on PINHOLE.
- Calibration guards (`warn`, the run continues): different aspect ratio, camera model tag or working
  resolution than the calibration video, or a distortion calibration on frames that were not undistorted.
- Backends: `pycolmap` (default; pip wheel 3.12.x for macOS x86_64) or `colmap-cli` (conda-forge build; uses
  `feature_importer` + `matches_importer --match_type raw` so it is independent of the database schema version).
- Quality gate: registered images >= 70 % of frames, mean reprojection error <= ~1 px, plausible camera path.
  Otherwise the stage reports `warn`/`error` and the pipeline stops before the expensive training.
- Debug output: top-down and side view of camera positions and sparse points, per-image registration table.
- Output dataset: `images/` as **RGBA PNG with the object mask in the alpha channel**, plus `sparse/0/`
  (cameras, images, points3D) with matching file names.

### Stage 5b: refine (pose-guided re-matching)
- After the first pass every frame has a pose. Pairs of frames more than `refine_min_gap` (default
  `match_window`) apart whose relative rotation is below `refine_max_angle` (30 deg) and whose distance to the
  object is similar (ratio 0.7-1.4) are candidates; the best `refine_pairs_per_image` (5) per frame are matched
  with the saved descriptors (`features/descriptors.npz`, no re-extraction).
- Matches are filtered with the fundamental matrix of the two known poses (Sampson distance <=
  `refine_max_epipolar_px`, 4 px), which removes wrong matches on repeated patterns better than RANSAC alone.
- The new pairs go into a copy of the COLMAP database (`refine/database.db`) and are verified. Default
  `refine_mode=remap` re-runs the incremental mapper on the extended database. `refine_mode=triangulate` keeps
  the poses, extends the existing tracks through the new pairs and runs a global bundle adjustment (faster; used
  only when every frame was registered). Measured on `IMG_8651`: remap track length 8.95 -> 9.92 at +0.05 px;
  triangulate 8.95 -> 9.17 at +0.10 px; rebuilding all tracks from scratch (the original plan) fragmented them,
  8.95 -> 7.45.
- The refined model replaces `sfm/dataset/sparse/0` only if no frame was lost and the reprojection error rose by
  at most 0.1 px. The first pass is kept in `sfm/sparse_first_pass/0` (outside `dataset/`, so trainers never see
  two models); re-running `refine` starts again from it, re-running `sfm` deletes it.
- Debug output: `pairs.png` (time gap vs. angle), `match_matrix_refined.png` (new pairs in cyan), drawn loop
  matches, `cameras_before_after.png`, and a before/after table in the stats.

### Stage 6: splat training (Brush)
- **Brush** (Rust, wgpu, runs on AMD GPUs on macOS) trains from COLMAP-format data and writes a `.ply`.
- Install: `cargo install --git https://github.com/ArthurBrussee/brush --tag v0.3.0 brush-app --locked`
  (binary `brush_app`). The command is a template (`train_cmd`, placeholders `{bin} {data} {out} {steps}`);
  default: `{bin} {data} --total-steps {steps} --export-every {steps} --export-path {out} --export-name splat.ply
  --eval-every 1000 --eval-split-every 10`.
- Masks (`train_mask_mode`): `alpha` (default) - the RGBA alpha is matched, so background becomes transparent;
  pixels under the hand are also forced transparent, which is why re-gripping matters. `ignore` - writes a
  `masks/` folder; Brush then ignores every non-object pixel (no carving by fingers, but nothing pushes the
  background transparent, so expect floaters to crop in SuperSplat). Brush uses one or the other, not both.
- Brush only draws its progress bar on a terminal, so it runs under a pseudo-terminal and `step/total` is
  parsed from that.
- Alternatives: `opensplat` backend (needs a libtorch build for Intel macOS), or `none` to stop after SfM and
  look at the sparse cloud (useful for testing pose quality quickly).
- Default `train_steps` 7000 for a first look; 30000 for a final result.
- Progress is parsed from the trainer's stdout (`step/total`).

### Stage 7: render (turntable videos)
- Brush's CLI cannot render camera paths and the CUDA rasterizers do not run here, so `gsturn/splat_render.py`
  is a small torch CPU rasterizer: EWA projection of the 3-D covariances, SH colour up to degree 3, 16-px tiles,
  front-to-back alpha compositing (~1.5 s per 720x720 frame for ~33k splats on the i9).
- The object's principal axes are found by PCA of the confident splat centres. One video per axis
  (`turntable_1_long.mp4`, `_2_middle`, `_3_short`): the camera circles that axis once (360 degrees), the axis is
  screen-up, and the view starts slightly oblique so no face begins edge-on.
- Config: `render_frames` (120 = 4 s at 30 fps), `render_size` (720, square), `render_fps`, `render_bg`.
- Skipped when no splat was trained.

### Stage 8: export
- Copies `splat.ply` and the turntable videos to `jobs/<id>/outputs/`, writes `report.json` (all stage stats,
  versions, config) and a preview image. Recommended viewer/editor for cleanup: **SuperSplat** (browser).

---

## 4. Project layout

```
turntable2splat/
  README.md
  requirements-core.txt        # numpy<2, opencv, fastapi, uvicorn, python-multipart, matplotlib
  requirements-ml.txt          # torch==2.2.2, torchvision==0.17.2, kornia, + Cutie/LightGlue (git)
  videos/                      # your example videos (input)
  calibration/                 # camera calibration JSON + inspection images (python -m gsturn calibrate)
  jobs/                        # created at runtime: one folder per job
  gsturn/
    __main__.py  cli.py        # python -m gsturn <doctor|downscale|calibrate|run|batch|serve|export-model|track|camera|refine-model|track-eval>
    config.py  events.py  doctor.py  pipeline.py  calibrate.py  batch.py  splat_render.py
    track/                     # live pose tracking (SuperPoint map, relocalisation, learning)
    stages/ downscale.py frames.py segment.py quality.py features.py sfm.py refine.py train.py render.py export.py
    server.py                  # FastAPI + SSE
    web/ index.html app.js style.css
  scripts/setup_mac_intel.sh
  tests/
```

Language: **Python** (Cutie, LightGlue, COLMAP bindings and OpenCV are Python-native). Rust is only used by the
external trainer (Brush).

---

## 5. Setup on the Intel MacBook

```bash
# 0. Command line tools and Homebrew
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 1. System tools
brew install ffmpeg colmap git cmake
brew install python@3.11            # PyTorch needs <= 3.12 on Intel macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust, for Brush

# 2. Python environment
cd turntable2splat
/usr/local/bin/python3.11 -m venv .venv      # path may be /opt/homebrew... check `brew --prefix`
source .venv/bin/activate
pip install -U pip
pip install -r requirements-core.txt
pip install "numpy<2" torch==2.2.2 torchvision==0.17.2
pip install kornia
pip install "git+https://github.com/cvg/LightGlue.git"
pip install "git+https://github.com/hkchengrex/Cutie.git"
# optional, only if a macOS x86_64 wheel exists (verify): pip install pycolmap

# 3. Splat trainer
cargo install --git https://github.com/ArthurBrussee/brush   # crate/binary name: verify in the repo README
brush --help                                                 # note the training flags

# 4. Check everything
python -m gsturn doctor
```

Notes:
- Homebrew's Intel support has been shrinking; if `brew install colmap` has no bottle for your macOS version, it
  compiles from source (slow but works) or use the `pycolmap` backend.
- Model weights (Cutie, SuperPoint, LightGlue) download from GitHub on first use. Run
  `python -m gsturn doctor --fetch-weights` (planned) or run the pipeline once with network access.
- If `pip install torch==2.2.2` fails, you are on Python 3.13+/3.12-incompatible interpreter or an arm64 shell
  (Rosetta). `python -c "import platform;print(platform.machine())"` must print `x86_64`.
- Disk: weights ~1 GB, a job with 200 frames at 720p ~1-2 GB.

---

## 6. Usage

```bash
# one-off helpers
python -m gsturn doctor
python -m gsturn downscale videos/a.mp4 --height 720        # writes videos/a_720p.mp4

# headless run (box in relative coords x0,y0,x1,y1 on the first selected frame)
python -m gsturn run videos/a.mp4 --box 0.25,0.2,0.75,0.8 \
    --hand-box 0.0,0.5,0.3,1.0 --train none

# many videos: one job per video, one after another (a folder = all videos in it, calibration* skipped)
python -m gsturn batch videos/ --boxes-file boxes.json --to-stage export --train brush --skip-done
#   boxes.json: {"*": {"object_box": [..]}, "IMG_8901.MOV": {"object_box": [..], "hand_boxes": [[..]], "bg_boxes": []}}
#   a video without an object box stops after frame selection; --dry-run lists what would run

# camera calibration (checkerboard video, square side in mm); use the result with --calibration
python -m gsturn calibrate videos/calibration.MOV --square-mm 22 [--pattern 6x6] [--height 720] \
    [--distortion auto|zero|fit] [--check-job jobs/<id>] [-o calibration/<name>.json]

# web UI
python -m gsturn serve --port 8000          # http://127.0.0.1:8000 (pick another port if 8000 is taken)

# live pose tracking with the built-in webcam (docs/realtime-pose-tracking.md)
python -m gsturn export-model --job jobs/<id> [--object-size <metres>]   # once per object
python -m gsturn camera [--calibrate --square-mm 22]   # preview, or webcam intrinsics -> calibration/
python -m gsturn camera --record [path.mp4] [--seconds 20]   # record the webcam (default videos/webcam0_<time>.mp4)
python -m gsturn track --model jobs/<id> [--source 0|video.mp4] [--poses p.jsonl] [--save-video o.mp4] \
    [--record-raw camera.mp4]                          # raw = plain camera feed, overlay = with pose drawn
python -m gsturn track --model jobs/<id> --learn       # adapt the map to the webcam while tracking (saved on exit)
python -m gsturn refine-model --model jobs/<id> --video videos/webcam0_<time>.mp4   # same, offline, + new points
python -m gsturn track-eval --job jobs/<id>            # accuracy against the SfM poses
```

Live tracking keys: q quit, space pause, m mask, a axes, p points, r relocalise, s snapshot (+ mask PNG).
macOS asks for camera access on first use; allow it for VS Code / Terminal under Privacy & Security > Camera.

CLI flags map 1:1 to config fields; `--from-stage segment --to-stage sfm` re-runs part of a job. `--bg-box` marks
table or background that lies inside the object's box (thin diagonal objects). The web UI has the same options plus a
bulk queue (`/api/batch`): drop videos, draw boxes, start; the queue runs the jobs one after another.

`calibrate` downscales the video with the same code as the pipeline (`--height` must match the jobs'
`target_height`; other resolutions are scaled), drops blurred frames, auto-detects the inner-corner count,
picks up to 40 views spread over board tilt and image position, and calibrates twice (k1/k2 free and fixed at
zero). With `--distortion auto` it keeps k1/k2 only if fixing them raises the RMS by >= 0.15 px and they move
pixels by >= 0.5 px in the central 80 % of the image. It fails if the RMS is > 1.5 px, fewer than 15 views
remain, or (with `--check-job`) the focal differs by more than 5 % from that job's SfM focal. Besides the JSON it
writes `<name>_views.jpg` and `<name>_residuals.png`. Videos whose name starts with `calibration` are not offered
as job videos in the UI.

### Config reference (`jobs/<id>/config.json`)

| Field | Default | Meaning |
|---|---|---|
| `target_height` | 0 | Downscale target (0 = keep the original; with nothing else to change it is read without re-encoding) |
| `target_fps` | 0 | Resample fps (0 = keep) |
| `hwaccel` | auto | `videotoolbox` on macOS if available |
| `max_frames` | 200 | Cap on selected frames |
| `min_motion`, `sharp_window` | 0.012, 15 | Min image change vs. last kept frame; frames of the rolling median |
| `sharp_keep_ratio` | 0.55 | Drop frames blurrier than ratio * median |
| `seg_backend` | cutie | `cutie` or `grabcut` |
| `object_box`, `hand_boxes` | none | Relative boxes on frame 0 |
| `erode_px`, `hand_dilate_px` | 2, 9 | Mask post-processing |
| `fill_holes_frac`, `close_px` | 0.005, 0 | Fill interior mask holes below this share of the object (0 = off); optional closing radius |
| `quality_enabled` | true | Run the quality stage and use its maps downstream |
| `quality_tile`, `quality_window`, `quality_texture_floor` | 32, 2, 20 | Tile size, neighbour frames per side, minimum texture to judge |
| `blur_rel_threshold`, `specular_v`, `specular_s` | 0.4, 230, 40 | Blur ramp; highlight brightness and max saturation |
| `quality_min_frame` | 0.5 | Exclude frames with less trusted mask area |
| `feature_backend` | superpoint | `superpoint`, `aliked`, `disk`, `sift` |
| `max_keypoints` | 2048 | Per frame, inside the mask |
| `match_window`, `match_stride_pairs`, `exhaustive_below` | 8, 20, 100 | Pairing strategy |
| `sfm_backend` | pycolmap | `pycolmap` or `colmap-cli` (`colmap_bin` to override the path) |
| `calibration` | "" | Calibration JSON; fixes the intrinsics (and undistorts frames if it has k1/k2) |
| `focal_prior`, `refine_focal` | 0, auto | Intrinsics handling; auto = refine only without calibration |
| `refine_enabled`, `refine_mode` | true, remap | Pose-guided re-matching; `remap` or `triangulate` |
| `refine_max_angle`, `refine_min_gap`, `refine_pairs_per_image` | 60, 0, 5 | Candidate pairs (gap 0 = `match_window`) |
| `refine_max_epipolar_px`, `refine_iterations` | 4, 1 | Pose-guided match filter; rounds |
| `train_backend`, `train_steps`, `train_cmd` | none, 7000, "" | Trainer and command template |
| `train_mask_mode` | alpha | `alpha`, `ignore` or `ignore_quality` (see stages 3b and 6) |
| `cutie_max_internal_size` | 480 | Cutie working resolution |
| `min_registered_ratio`, `max_reproj_error` | 0.7, 1.5 | SfM quality gate |
| `trim_start`, `trim_end` | 0, 0 | Seconds; 0 = whole video |
| `device` | auto | `auto` (CUDA else CPU), `cpu`, `mps`, `cuda` |

---

## 7. Capture guidelines (these decide most of the quality)

- Plain, uncluttered background; diffuse light (window light or two soft lamps); no flicker.
- Fixed exposure, focus and white balance; fast shutter to avoid motion blur.
- Rotate slowly; show every side, including top and bottom (tilt the object); **re-grip halfway**.
- Keep the object fully in frame and fairly large (it should fill 30-60 % of the image height).
- Matte objects. Glossy, transparent or very smooth objects need matte spray, markers or a pattern.
- Calibrate the camera once with a printed checkerboard (`python -m gsturn calibrate`). Record the calibration
  video with **the same lens (1x), resolution, frame rate and stabilisation setting** as the object videos;
  iPhone stabilisation crops and shifts the image per frame, so use "standard" or off for both. Re-calibrate if
  any of these change. Move the board through the whole image, including the corners, and tilt it up to ~45 deg.
- The board fixes the intrinsics only; reconstructions have no units unless a board or marker of known size is
  visible in the object video.

---

## 8. Web frontend

Single page, no build step (vanilla JS), served by FastAPI.

**Layout**
1. *Input*: drag-and-drop upload, or pick a file from `videos/`; parameter panel (resolution, max frames,
   backends, device, train steps) with the defaults above.
2. *First-frame box*: after stages 1-2 finish, the first selected frame is shown on a canvas. Drag a box around the
   object; optionally add hand boxes. "Run pipeline" starts stages 3-7.
3. *Pipeline panel*: one row per stage with a **progress bar**, status chip (`pending/running/done/warn/error`),
   elapsed time and a one-line message.
4. *Gallery* (per stage): the artifacts that stage produced: selection plot and contact sheet, mask overlays and
   area plot, keypoint and match images, match matrix, camera/point plots, training preview, final result.
5. *Debug*: live log tail, raw event table, the job's `config.json`, `doctor` report, per-stage stats, buttons to
   **re-run a single stage**, cancel, and download the job folder.

**API**
| Endpoint | Purpose |
|---|---|
| `GET /api/doctor` | toolchain report |
| `GET /api/videos` | files in `videos/` |
| `POST /api/jobs` | create job (upload or existing path) and start stages 1-2 |
| `POST /api/jobs/{id}/run` | start stages (body: config updates, `from`, `to`) |
| `POST /api/jobs/{id}/cancel` | kill the running stage process |
| `GET /api/jobs/{id}` | config, per-stage status and artifacts |
| `GET /api/jobs/{id}/events` | Server-Sent Events stream (resumable by byte offset) |
| `GET /api/jobs/{id}/files/{path}` | artifact files (path-checked inside the job folder) |
| `GET /api/jobs/{id}/log` | raw log |

**Process model.** Each run is a subprocess (`python -m gsturn run --job <dir> ...`) so heavy ML code cannot block
or crash the server, and cancelling is a process kill. The pipeline appends JSON lines to `events.jsonl`:

```json
{"type":"stage","stage":"segment","status":"running","pct":0,"t":1761300000.1}
{"type":"progress","stage":"segment","pct":42.0,"done":84,"total":200,"msg":"frame 84/200"}
{"type":"artifact","stage":"segment","path":"work/overlays/overview.jpg","caption":"mask overlays"}
{"type":"log","stage":"features","level":"warn","msg":"LightGlue unavailable; using SIFT"}
{"type":"stage","stage":"segment","status":"done","msg":"200 masks (cutie)","stats":{...}}
```

The server tails this file and relays it over SSE; reloading the page replays the history.

---

## 9. Debugging checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| Mask includes shadow/hand | weak first-frame box | redraw box, add hand box, raise `erode_px` |
| Area plot jumps | Cutie lost the object | better first frame, fewer low-quality frames |
| Very few keypoints | smooth object | texture/markers, `aliked`, higher `max_keypoints` |
| Match matrix has no bright diagonal | too much rotation between frames | lower `min_motion`, more frames |
| SfM registers < 70 % | weak matches or mask errors | check stage 3/4 outputs first, then try `sift` vs `superpoint` |
| Cameras form a clump | background features leaked in | verify masks, shrink mask |
| Doubled surfaces in splat | wrong poses or object deformed | check SfM report, keep the object rigid |
| Holes in splat | never seen / always covered | recapture with re-grip |
| Everything slow | CPU only | lower `target_height`, `max_frames`, `cutie_max_internal_size`; try `device=mps` |

---

## 10. Test plan on the videos in `videos/`

For each video, run the full pipeline once with `train_backend=none` (fast), then with Brush for the ones that pass.

| Check | Pass criterion |
|---|---|
| Downscale | output resolution = target, duration within 1 frame, file much smaller |
| Frames | 100-250 frames kept, no blurry frames in contact sheet |
| Segmentation | overlays show object only; no empty masks; no area jumps > 50 % |
| Features | median >= 300 keypoints/frame; weakest sequential pair >= 25 matches |
| SfM | >= 70 % frames registered; mean reprojection error <= 1.5 px; camera path wraps the object |
| Training | loss decreases; novel views look coherent from all sides |
| UI | progress bars advance per stage; artifacts appear; re-run of one stage works; cancel works |

Also test deliberately hard cases (textureless object, object with fingers over it, dim light) and record which
stage fails first. That ordering tells you where to invest.

---

## 11. Expected runtime (first estimates; measured values in section 16)

On an Intel laptop CPU, 720p, ~150 frames: downscale 4K30 of 20 s in ~1-3 min; frame selection ~1 min; Cutie
~1-3 s/frame (several minutes); SuperPoint+LightGlue exhaustive pairing 150 frames (~11k pairs) is the slowest
step, so use the sliding window for > 100 frames (~1.5k pairs, tens of minutes on CPU); COLMAP mapping a few
minutes; Brush 7k steps depends heavily on the AMD GPU (minutes to an hour). Use `target_height=480` and
`max_frames=100` for a quick first pass.

---

## 12. Known limitations and open questions

- Cutie needs a good first-frame mask; very low contrast or fur-like objects are hard.
- Small or untextured objects (under about 100 keypoints per frame) register only part of the frames; round or symmetric
  objects seen edge-on can flip in SfM (section 16).
- No dynamic/deformable reconstruction; one rigid object only.
- Lighting baked into the splat; no relighting.
- Brush CLI/crate names, mask-from-alpha support, `pycolmap` Intel-macOS wheels, and MPS behaviour need
  verification on the machine (see *(verify)* markers).
- Intel macOS is a shrinking target: torch is frozen at 2.2.2 there. A Linux/Windows box with an NVIDIA GPU would
  allow newer models (SAM 2/3, VGGT/MASt3R poses, gsplat) and be much faster.
- Questions for you: (1) exact MacBook model (13" vs 16", GPU model) and macOS version; (2) typical object
  size/texture; (3) camera model and whether it can be calibrated; (4) whether a CUDA machine is ever available
  for training.

---

## 13. Milestones

1. `doctor`, config, event stream, downscale, frame selection.
2. Segmentation with GrabCut, then Cutie; overlay artifacts.
3. SIFT matching -> COLMAP -> pose plots (end-to-end without torch).
4. SuperPoint/ALIKED + LightGlue.
5. Brush training + export.
6. Web UI (upload, box tool, progress bars, gallery, debug).
7. Run the test plan on `videos/`, tune defaults from the results.

---

## 14. Implementation notes and the first measured run (2026-10-24)

**Settled *(verify)* items**
- torch 2.2.2 / torchvision 0.17.2 install on Python 3.12 x86_64; `torch.backends.mps` is available.
- Installing Cutie and LightGlue from git upgrades numpy to 2.x, OpenCV to 5.x and scipy; `scripts/setup_mac_intel.sh`
  pins them back (`numpy<2`, `opencv-python-headless<4.12`, `scipy<1.14`, `pandas<2.3`).
- Cutie's loader calls `.cuda()`; `gsturn` loads the same config and weights itself (weights go to `weights/`).
- pycolmap 3.12.5 has a macOS x86_64 wheel; the COLMAP CLI (3.11.1) comes from conda-forge.
- Brush v0.3.0 builds with cargo, runs on the Radeon Pro 5500M via Metal, reads COLMAP data, and supports alpha
  and `masks/` (see stage 6).

**Changes from the original plan**
- Default SfM backend is `pycolmap` instead of `colmap-cli` (no Homebrew bottle on Intel).
- New `train_mask_mode` option (Brush cannot combine alpha and ignore-masks).
- Runs write `run.pid`, so the server also sees and can cancel runs started from the CLI.
- The page loads the summarised job state, then streams only new events (instead of replaying the whole log).

**Measured on `IMG_8651.MOV`** (13 s, 4K HEVC, pen box held in both hands, CPU for all torch work)

| Stage | Result | Time |
|---|---|---|
| downscale | 3840x2160 -> 1280x720, 393 frames | ~20 s |
| frames | 200 of 393 kept (capped by `max_frames`) | ~5 s |
| segment (Cutie, CPU) | 200 masks, 0 empty; 2 "area jumps" flagged where the box turns edge-on | ~11 min (3.5 s/frame) |
| features (SuperPoint + LightGlue, CPU) | 646 keypoints/frame, 1560/1664 pairs kept, weakest sequential pair 128 | ~17 min |
| sfm (pycolmap) | 200/200 registered, 11.3k points, 1.35 px, focal refined 1536 -> 1091 | ~2 min |
| train (Brush, Radeon 5500M) | 33.4k splats, 7.8 MB, held-out PSNR 22.7 dB / SSIM 0.932 | 11 min for 7k steps |
| render | 3 x 120 frames, 720x720 | ~9 min |

### Result quality (`IMG_8651.MOV`, 7k steps)

Judged from Brush's held-out views (every 10th frame, not used for training; side-by-side sheet in
`work/train/eval_comparison.jpg`), the three turntable videos in `outputs/`, and the splat geometry.

| Aspect | Assessment |
|---|---|
| Camera poses | Very good. All 200 frames registered, 1.35 px mean reprojection error, cameras wrap the whole box including top and bottom. The SfM step is not the bottleneck. |
| Shape | Good. The splat centres form a clean cuboid with sharp edges; proportions about 1 : 0.54 : 0.21. Almost no floaters, because the background was forced transparent. |
| Large print and graphics | Good. "uni-ball eye micro", "UB-150 BLUE", the pen illustration and the blue checker pattern are crisp and correctly coloured from all sides. |
| Fine detail | Fair. Small print (ingredients, "MITSUBISHI PENCIL CO.") and barcodes are blurred; the 720p input and 7k steps limit this. |
| Hand-covered regions | Weakest point. Where fingers covered the box in many frames, the splat shows dark or reddish patches (e.g. a bottom corner and one end face); in a few views a corner has a small bite. Areas that the hand covered in only some frames are filled in from other views. |
| Novel views (turntables) | Fair to poor. Views close to the recorded ones look like the held-out frames. Angles the video never showed, mainly looking straight onto the large white face and some corners, are blurry with dark smudges. The splat only knows what the camera saw. |
| Edges | Mostly clean silhouettes; some corners are dark or slightly ragged where the hand mask touched them. |
| Numbers | PSNR rose steadily 20.8 -> 22.7 dB (1k -> 7k steps), SSIM 0.905 -> 0.932, still climbing at 7k. PSNR is understated: ground truth has holes where the hand was (alpha 0), while the splat fills them, which counts as error. |

What would improve it most, in order: (1) re-grip more often, or try `train_mask_mode=ignore` so finger
areas are not forced transparent; (2) `train_steps=30000`; (3) a higher `target_height` (1080) for the small
print; (4) diffuse lighting, since the fixed room light moving across the rotating box is baked in as shading.

**Tests**: `python -m pytest tests` - unit tests, API tests (including path traversal), and an end-to-end run on a
synthetic rotating textured cube (GrabCut + SIFT + pycolmap): 90/90 frames registered, focal 514.5 vs. true 520,
0.6 px.

---

## 15. Further docs

- [docs/ignore-blur-defocus-reflections.md](docs/ignore-blur-defocus-reflections.md) - per-pixel trust masks so blurred, defocused and reflective areas are left out of SfM and training
- [docs/lighting-shadows-reflections.md](docs/lighting-shadows-reflections.md) - options for lighting, shadows and reflections (capture, segmentation, SfM, training)
- [docs/sfm-known-view-angles.md](docs/sfm-known-view-angles.md) - improving SfM by re-detecting views seen before (loop closure, pose-guided re-matching, multi-video registration)
- [docs/implementation-plan.md](docs/implementation-plan.md) - how to implement camera calibration, mask hole filling, the quality stage and pose-guided re-matching
- [docs/realtime-pose-tracking.md](docs/realtime-pose-tracking.md) - using the reconstruction to locate, mask and orient the object live from a camera

---

## 16. Results of all runs (2026-10-24 to 2026-11-17)

Every job in `jobs/` read from its `events.jsonl` and stage `stats.json`. "Wall" is the sum of the stage times; jobs
of one batch ran at the same time, so single stage times are inflated and noisy (e.g. Cutie 391 s on `IMG_8900` against
2116 s on `IMG_8901`). PSNR is Brush's held-out value on a black background, so it grows when the object is small
(`IMG_8902`, 5 % of the frame) and is not comparable between videos.

| Video (source) | Frames | Object / hand share of frame | Keypoints / frame | Registered | Points | Reproj. px | Focal px | Splats | PSNR | Wall | Reached |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `IMG_8651` 4K30 -> 720p, pen box (3 runs: 200, 200, 384 frames) | 200 | 19 % / 36 % | 646 | 100 % | 11.3k | 1.35 | 1091 | 33.4k | 22.7 | 53 min | export |
| `IMG_8645` 4K30, small object | 104 | 5 % / 29 % | 87 | 68 % | 0.4k | 1.35 | 1077 | - | - | 11 min | SfM gate failed |
| `IMG_8646` 4K30, small object | 200 | 2.6 % / 35 % | 61 | 75 % | 1.1k | 1.38 | 1077 | 2.1k | 29.8 | 30 min | export |
| `IMG_8647` 4K30 | 200 | 34 % / 22 % | - | - | - | - | - | - | - | 8 min | quality |
| `IMG_8648`-`8650` 4K30 | 200 | - | - | - | - | - | - | - | - | 0-1 min | frames (no box yet) |
| `IMG_8900` 1080p120, lighter-gas can | 188 | 14 % / 13 % | 1010 | 100 % | 8.8k | 1.33 | 1632 | 22.7k | 29.8 | 100 min | export |
| `IMG_8901` 1080p120 | 200 | 15 % / 16 % | 940 | 100 % | 17.9k | 1.31 | 1751 | 38.7k | 25.3 | 156 min | export |
| `IMG_8902` 1080p120, small object | 103 | 5 % / 39 % | 415 | 100 % | 5.1k | 1.27 | 1770 | 12.4k | 33.1 | 91 min | export |
| `IMG_8903` 1080p120 | 200 | 31 % / 0 % | 2048 | 100 % | 55.7k | 1.55 | 3130 | - | - | 129 min | SfM gate failed (1.55 > 1.5) |
| `IMG_8904`, `8905`, `8923` | - | 51 % / 0 % (8904) | - | - | - | - | - | - | - | 1-10 min | cancelled or stopped early |
| `IMG_8926` 1080p120 | 184 | 13 % / 23 % | 723 | 100 % | 9.5k | 1.53 | 2073 | - | - | 59 min | SfM gate failed (1.53) |
| `IMG_8924` 1080p120, salad bowl view | 200 | 30 % / 17 % | 2039 | 100 % | 29.3k | 1.48 | 3386 | 55.2k | 27.5 | 179 min | export |
| `IMG_8925` 1080p120, salad bowl | 200 | 43 % / 12 % | 1298 | 95 % | 29.8k | 1.48 | 3438 | 80.6k | 22.4 | 167 min | export |

**Where the time goes** (1080p jobs, 200 frames; shares of the wall time): keypoints and matching 22-67 % (extraction on
the CPU at full 1080p is about 8.7 s per frame, matching about 1.7 s per pair at 2048 keypoints and about 0.4 s at
1000), Brush training 20-45 % (7000 steps, 0.25-0.38 s per step at 1080p against 0.09 s at 720p), the turntable
renders 10-23 %, refine 9-15 %, segmentation 5-10 %, SfM 4-8 %.

**Findings**

- **Pose estimates are good on textured objects.** All frames registered in 9 of the 12 runs that reached SfM (95 % in
  one, 68-75 % in the two weak-texture ones); the sequential
  rotation step is 2-7 degrees per selected frame (median), with accelerations of 1-3 degrees. Reprojection error sits
  at 1.3-1.55 px at 1080p, which is mediocre and mostly set by the keypoint detector, not by the resolution.
- **Weak texture is the first failure.** `IMG_8645`/`8646` (object 2.6-5 % of the frame, 61-87 keypoints) register only
  68-75 % of the frames. `IMG_8646` still gives a splat (2.1k splats), `IMG_8645` stops at the SfM gate.
- **The free focal length is unreliable for some videos.** The 4K30 runs, scaled to the same width, give 1615-1637 px at
  1920 wide and the 1080p runs on `IMG_8900`-`8902` give 1632-1770 px, but `IMG_8903`, `8924`, `8925` converge to
  3130-3438 px and `IMG_8926` to 2073 px, with higher reprojection error (1.48-1.55 px) and two quality-gate failures.
  None of these jobs used a calibration. A near-planar or round object seen with a weak perspective cannot tell focal
  length from distance.
- **`IMG_8925`: pose flip at an edge-on view.** Frames 63 and 64 (the round bowl seen edge-on) are registered 149 degrees
  apart while every other step is below 14 degrees; the essential matrix of that pair gives 3.5 degrees with a focal of
  1650 px and 35 degrees with the 3438 px SfM chose. Frames 64 onward form a consistent but wrongly rotated block, which
  most likely explains the ghosted, smeared held-out renders and the low 22.4 dB PSNR. The object also leaves the frame at the end
  (frames 188-197 unregistered, 196-199 empty masks). A fixed, calibrated focal is the first thing to try.
- **The hand is the main quality limit of the splat.** Hands cover 12-39 % of the frame, more than the object on small
  objects. With the default `train_mask_mode=alpha` the hand zone is forced transparent, which cuts bites out of the
  object (see the bottom rows of `work/train/eval_comparison.jpg` of `IMG_8900` and `IMG_8925`), and skin that leaked
  into the object mask in the first frames of `IMG_8900` is learned as geometry.
- **Refine helps little on hand-held videos**: mean track length 19.7 -> 20.2 (`8900`), 10.2 -> 10.3 (`8901`), 7.8 -> 7.8
  (`8902`), 10.9 -> 11.2 (`8924`), 6.5 -> 6.6 (`8925`, +3 registered frames), 12.9 -> 13.9 on the 384-frame `8651` run
  where the loop closes; one result was rejected (reproj 1.35 -> 1.51). It costs 9-15 % of the time.
- **Training:** 7000 Brush steps take 11 min at 720p (33k splats) and 30-45 min at 1080p (13-81k splats). PSNR still
  climbed at 7000 steps on `IMG_8651` (20.8 -> 22.7 dB from 1000 to 7000).

- **A larger angle allowance in refine helps** (`IMG_8902`, 103 frames, same first-pass model for every row,
  `refine_pairs_per_image` 5; each pair is filtered by the epipolar geometry of the known poses):

  | `refine_max_angle` | pairs added | mean track length | reproj. px | min points per image |
  |---|---|---|---|---|
  | 30 (old default) | 21 | 7.777 -> 7.773 | 1.2713 -> 1.2728 | 209 -> 208 |
  | 45 | 120 | 7.777 -> 8.169 (+5.0 %) | 1.2713 -> 1.2921 | 209 -> 205 |
  | 60 (new default) | 149 | 7.777 -> 8.264 (+6.3 %) | 1.2713 -> 1.2900 | 209 -> 208 |
  | 90 | 149 | 7.777 -> 8.292 (+6.6 %) | 1.2713 -> 1.2999 | 209 -> 208 |

  The pairs come from the frames that have fewer than five partners within 30 degrees (the candidates nearest in angle
  are matched first), so the gain sits on the views that were weakly connected. 60 degrees captures nearly all of it;
  90 adds nothing but 0.01 px of error. One video only: confirm on another before relying on the size of the gain.

**Next steps these runs point to**, in order: a calibration made in the same camera mode (1080p120) with `calibration`
set; ignoring the hand instead of making it transparent; extraction and matching on a crop around the object with
fewer keypoints; checking the trajectory for orientation jumps before training.

---

---

## 17. Second camera (`video2`)

A second camera, rigidly mounted relative to the first and recording the same event, gives wider baselines, a second
view of what the fingers hide from the first one, and removes the focal-length/depth ambiguity of a single camera.
The two videos are not synchronised and may differ in resolution, frame rate, orientation and lens (the tested pairs:
iPhone 14 Pro Max 1080x1920 60 fps portrait + iPhone 17 Pro Max 1920x1080 30 fps).

```bash
python -m gsturn run videos/A/IMG_8934.MOV --video2 videos/A/IMG_7456.MOV --video2-sync auto \
    --trim-start 3.4 --box ... --hand-box ... --object-box2 ... --hand-boxes2 "x0,y0,x1,y1"
```

**Time sync** (`gsturn/sync.py`, `video2_sync=auto|audio|motion|manual`, `video2_offset` for manual; time in video2 =
time in video + offset). Searching the whole +-30 s range found false peaks in two of three pairs, so the search is
anchored:
1. *Metadata prior*: iPhones write the recording start (whole seconds, network-synced clocks) into the file. The offset
   is good to about +-1.5 s from that, and the correlations are only searched in prior +-4 s.
2. *Audio*: GCC-PHAT cross-correlation of the two soundtracks (150 Hz-5 kHz), 1 ms resolution when the peak is sharp.
3. *Motion*: cross-correlation of the frame-difference energy of both videos at 50 Hz, about one frame of resolution,
   works without sound.
4. *Joint score*: both curves as clipped robust z-scores plus a weak bump at the prior; a very sharp audio peak that
   belongs to another event (loud tap repeated) cannot outvote the motion evidence. The final value comes from the
   audio peak next to the joint maximum. The result carries a confidence, notes about disagreement, a plot
   (`work/downscale/sync.png`) and is checked visually with `sync.side_by_side`.

Results on the three pairs in `videos/A`, `B`, `C`: A +2.215 s (audio and motion agree within 2 ms), C -1.603 s (14 ms),
B +2.204 s (audio's strongest peak, +5.55 s, was a different event; its third peak and the motion peak agree on
+2.20 s; flagged low confidence, confirmed by eye: same hand and object configuration in both cameras).

**Pipeline with two cameras.** Frames of camera 2 are named `c1_<k>` and stored with the others; `frames.json` records
the camera of every frame, its time and its camera-1 partner. Camera-1 frames are chosen as before; for each, the
sharpest camera-2 frame within one frame of the same moment is taken. Segmentation runs once per camera (boxes
`object_box2`, `hand_boxes2`, `bg_boxes2` on camera 2's first frame), quality maps compare only frames of the same
camera, features use one extraction window per camera, and every camera-1 frame is matched with its partner and
`cross_window` neighbours. Camera 2 may look at the object rolled by 90 degrees or more: before extraction a few
synchronised pairs are matched with camera 2's window turned by 0, 45, ... 315 degrees and the angle with the most
epipolar inliers is used for that camera. SfM writes one PINHOLE camera per video into the COLMAP database (focal
length from `calibration2` or estimated). COLMAP's rig feature did not pair the images of the two cameras into
frames (tested with pycolmap 3.12.5), so the cameras are solved independently and the rigid mounting is *checked*
afterwards: the pose of camera 2 relative to camera 1 must be the same at every moment, and its spread over all
synchronised pairs (`rig` in the SfM stats: rotation and translation spread) measures registration and sync error. The
registration is checked per camera and warns when one camera registered less than half of its frames. Training uses the images of both cameras, each with
its own camera. Refine and the live-tracking export work on one camera only.

---

## 18. Keeping the hand out of the splat

`train_mask_mode=weighted` (default), implemented in `gsturn/stages/traindata.py` and `gsturn/handmask.py`. It uses
two facts read from Brush's source: the training background is black, and with a `masks/` folder the loss is
(L1 + SSIM) x mask, i.e. the mask is a soft per-pixel weight and the image is used as it is.

| Pixels | Image | Weight |
|---|---|---|
| object (trusted) | photograph | 1 |
| empty background | black = "nothing here" | 1 (this is what keeps floaters out) |
| hand, `train_hand_margin_px` around it, blurred or specular object pixels, the band just outside the silhouette | black | 0, rising to 1 over `train_edge_ramp_px` |

Old modes stay: `alpha` (hand and background forced transparent: carves the object where fingers were), `ignore`
(object only: nothing pushes the background to transparent), `ignore_quality`.

**Per-frame crops** (`train_crop`, `gsturn/crop.py`): the crop is the box of the *visible* object mask (percentile
extents of its large connected components, so a leaked speck or a hand does not stretch it) plus a margin and
`train_crop_pad_px` of background, rounded up to a multiple of 32; Brush receives a COLMAP camera per image with the
shifted principal point. The pixels it trains on are the same pixels as before, just without the empty rest of the frame
(IMG_8924: 50 % of the full-frame pixels; small objects save much more). The ignored hand zone does not need to be
inside the crop. For feature extraction the window has one fixed size per camera (the largest visible extent in the
video) so every frame is seen at the same scale, and only its centre follows the object (median over 5 frames).

**Skin leaks** (`skin_filter`): Cutie sometimes calls the thumb or a fingertip object. A colour model of the hand
(Gaussian on the Lab a/b chroma of the eroded hand mask) marks object pixels that look like skin, lie near the hand and
are connected to it; they join the hand mask. If a noticeable share of the object far from the hand also looks like
skin (a pink or wooden object) nothing is removed. On IMG_8900 it finds the thumb and finger tips that the first frames
had in the object mask. **Keypoints** near the hand are dropped (`kp_hand_margin_px`): a SuperPoint descriptor sees
16-32 px around its point, so near fingers it describes the finger.

**Residual pass** (`train_residual_pass`, off by default): train `train_residual_steps` steps, render every training
view with the CPU rasteriser, mark object pixels whose colour error exceeds `train_residual_thresh`, ignore them and
train again. Whatever the other views agree on is in the splat; a finger leak or a moved highlight is not.

---

## 19. Orientation continuity (Kalman) and the focal retry

IMG_8925 ended with a pose break: frames 63 and 64 (the round bowl seen edge-on) registered 149 degrees apart, and
everything after frame 64 sat in a consistent but wrongly rotated block. A Kalman filter or smoother on the finished
poses cannot repair that (the block satisfies the image evidence in its wrong orientation; smoothing only bends it
away from that), but the filter's innovation detects it reliably. `gsturn/kalman.py` runs a constant-velocity filter
over the camera orientations with the real frame timestamps, takes the angle between the predicted and the registered
orientation, and flags a frame when it is above `max(45 deg, 6 sigma)` with sigma from the median absolute innovation
(so one break does not inflate it); one jump gives one report. In the `sfm` stage a flagged jump, or a free focal
outside 0.55-1.25 x the long image side (phone main cameras), triggers `sfm_focal_retry`: the mapper is re-run with the
focal fixed at each of `sfm_focal_candidates` (x long side) and the model with the fewest jumps, then the lowest
reprojection error, is kept. `smooth` (local quadratic fit in each pose's own frame) and `interpolate` (slerp) are there
for jitter and for frames SfM could not register. Tests: `tests/test_kalman.py` (a clean hand-turn with re-grips gives
no break; a 150 degree block flip and a single bad frame are each found once).

---

## 20. De-lighting (`python -m gsturn delight --job jobs/<id> --demo`)

The camera and the room lights are fixed while the object turns, so the lighting a surface point receives changes from
frame to frame while its albedo does not. With n_c the surface normal in the camera frame of frame i,
`colour(p, i) = albedo(p) * E(n_c(p, i))`, where E (9 spherical-harmonic coefficients per colour channel) is the same
function for the whole video. `gsturn/delight.py`: (1) normals from the PCA of the neighbouring splats, outward; (2) for
every frame the splats that are visible (front-facing, inside the trusted mask, not behind a splat in front) and the
photographed colour at their projection; (3) alternating robust least squares for E and the per-splat albedo, with the
mean lighting fixed to 1 and white (the room light's colour cast goes into E, so the result is lit by a neutral white
light), Huber-like weights discounting highlights, hand shadows and leaks; (4) a splat with the albedo as colour, no
view-dependent terms. Outputs in `outputs/`: `splat_delit.ply` (shown in the viewer) and `delight/` with
`delight_turntable.mp4` (original | de-lit), `delight_relit.mp4` (the de-lit splat under a white light circling the
object: it only looks right if the albedo is clean), `delight_sheet.jpg`, `lighting.png` (the recovered room light on a
sphere) and `delight.json` (including the drop in colour variation between views per splat). Limits: shadows the object
casts on itself are not modelled, glossy reflections are discounted rather than explained, and colour is only recovered
where some frame saw the surface.

---

## 21. Splat viewer in the web UI

Section 4b of the page: pick a splat file of the job (original, de-lit, trainer output), "Open viewer", drag to orbit,
right-drag to pan, wheel to zoom. `gaussian-splats-3d` (MIT) and three.js are vendored in `gsturn/web/vendor/`, nothing
is loaded from a CDN, and no WebGL context is created until the button is pressed. `GET /api/jobs/<id>/splats` lists
the files with a suggested camera (centre and radius of the confident splats, cached next to each file). The
turntable render stage is off by default (`render_enabled=false`).

---

## 22. Job scheduler (`gsturn/scheduler.py`)

Two jobs that both extract and match features, run Cutie or COLMAP split the CPU between them and take twice as long.
Every stage now asks for the resources it keeps busy: `cpu` (torch on the CPU, COLMAP, x264, the software rasteriser)
and/or `gpu` (Brush, torch on MPS/CUDA). Stages that need different resources overlap (job A trains while job B
matches); stages that need the same one take turns, **the job that started first goes first on every resource**, and a
stage is never interrupted. A later job may use a resource when every job ahead of it is blocked on something else
(backfill). Mechanics: one `flock` file per resource (the kernel frees it when a process dies) plus a queue directory
of small JSON files, shared by the web UI, the CLI and batch runs in `~/.cache/gsturn/locks`
(`GSTURN_LOCK_DIR`); all resources of a stage are taken together or not at all. `scheduler=resource|job|off` (also
`GSTURN_SCHEDULER`); `job` runs one job at a time for the whole run. Waiting is logged and does not count as stage
time. Tests: `tests/test_scheduler.py`.

---

## 23. Meshes and the physics shelf

**Splat to mesh** (`python -m gsturn export-mesh --job jobs/<id> --size-cm 20`, `gsturn/mesh.py`): the opacity-weighted
density of the Gaussians on a voxel grid (in the object's principal frame), smoothed, thresholded, largest blob, gaps
closed with the smallest closing that makes the solid at least 80 % of its convex hull (a shell with unseen ends needs a
large one, a slice-wise hull fills tubes and cups), marching cubes, vertex colours from the nearest splats. The mesh is
scaled so its longest side is `--size-cm` (the reconstruction has no units), lies flat on y = 0, and is written as
`<name>.stl` (millimetres), `.obj` and `.ply` (metres, vertex colours) and `<name>.physics.json`. For physics the solid
is split into convex parts (k-means on the voxels when the solidity is low) and each part is fitted with the primitive
(box, cylinder, sphere) that overlaps it best. A splat only covers what the camera saw: unseen faces are filled by
the solid, not measured.

**Shelf** (`python -m gsturn shelf`, folder `physics/`): `physics/objects.json` lists the jobs and the real sizes of the
objects (estimates: set the real ones), `shelf --build` exports them, the page (three.js + Rapier WASM, all vendored)
shows a shelf with the objects on it and a **Tilt shelf** button that tips it forward about its front-bottom edge; the
objects slide off, fall to the floor and pile up against each other inside a walled area. Also: tilt angle and speed,
time scale, Shake, Drop another, Reset, downloads of every mesh. `node physics/test_sim.mjs` checks the simulation
headless (everything stays on the level shelf, all objects leave a tilted one, nothing overlaps, reset and spawn work).

---
