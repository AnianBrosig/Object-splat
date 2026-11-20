# Implementation plan: calibration, hole filling, quality stage, pose-guided re-matching

Four changes, in the order they should be built. Each section lists the files to touch, the algorithm, new
config fields, outputs, tests, and how to tell it worked. Background and rationale:
[ignore-blur-defocus-reflections.md](ignore-blur-defocus-reflections.md) and
[sfm-known-view-angles.md](sfm-known-view-angles.md).

| # | Change | Where | Effort |
|---|---|---|---|
| 1 | Camera calibration from `videos/calibration.MOV` | new `gsturn/calibrate.py`, CLI, `stages/sfm.py` | ~0.5 day |
| 2 | Fill small holes in object masks | `stages/segment.py` | ~2 h |
| 3 | New `quality` stage (trust maps) | new `stages/quality.py`, small hooks in `features.py`, `train.py` | ~1 day |
| 4 | Pose-guided re-matching (two-pass SfM) | new `stages/refine.py`, `features.py` (save descriptors) | ~1 day |

New stage order once all are in:

```
downscale -> frames -> segment -> quality -> features -> sfm -> refine -> train -> render -> export
```

---

## 0. Plumbing for new stages (do once, before 3 and 4)

The stage list is currently written out in four places. Adding a stage means updating all of them, so first
reduce them to one:

| Place | Today | Change to |
|---|---|---|
| `gsturn/config.py` `STAGES` | the list | keep: single source of truth |
| `gsturn/web/app.js` line 22 `const STAGES = [...]` | hard-coded copy | read `st.order` from `GET /api/jobs/{id}` (already returned by `server.job_state`) and from a new `GET /api/stages` for the "new job" view |
| `gsturn/stages/export.py` line 50 `for st in (...)` | hard-coded tuple | `for st in STAGES` (skip `export` itself) |
| `gsturn/web/app.js` run button "segment → export" | starts at `segment` | unchanged; `quality` follows `segment` automatically |

Optional stages (`quality`, `refine`) must also work on older jobs. If their inputs are missing they raise
`StageError` with a clear message. If disabled by config they return `{"skipped": True, "_msg": "disabled"}`, so
later stages run as before. Downstream stages read their outputs only **if present**.

---

## 1. Camera calibration (`videos/calibration.MOV`, 22 mm squares)

### What the video contains (checked)

- 3840x2160, 30 fps HEVC, 378 frames, iPhone 17 Pro Max, the same model tag as the object videos.
- Printed checkerboard, **7x7 squares = 6x6 inner corners**, square side **22 mm**. OpenCV
  `findChessboardCornersSB` finds the full 6x6 grid in 27 of 38 sampled frames. Misses are motion-blurred or
  partly out of frame.
- Trial calibration at 720p (54 views, every 5th frame): **f = 1082-1092 px**, principal point (652, 370),
  RMS 1.05-1.10 px, board 225-500 mm from the camera. The SfM self-calibration on `IMG_8651` gave **1091 px**,
  so both agree. The prior currently used (`1.2 * max(w, h)` = 1536 px) is 40 % too high.
- Distortion from the trial: k1 ≈ 0.24, k2 ≈ -0.77. That is large for an iPhone main camera, whose video is
  normally lens-corrected in camera, and it is poorly constrained: one flat board, mostly central, blurred
  frames. See "Distortion" below.

### New module `gsturn/calibrate.py` + CLI

```
python -m gsturn calibrate videos/calibration.MOV --square-mm 22 [--pattern 6x6] [--height 720] \
       [-o calibration/iphone17promax_4k30_1x.json]
```

Steps:

1. **Same pixel grid as the pipeline.** Downscale with `stages.downscale.downscale()` to the same
   `target_height` (720) the jobs use, into a temp folder. Do not resize frames with OpenCV separately: a
   different scaler or rounding shifts the principal point slightly.
2. **Sample and score frames.** Every 3rd frame. Variance of the Laplacian on the gray frame; drop frames below
   0.5 × median (blur kills corner accuracy).
3. **Detect corners.** `cv2.findChessboardCornersSB(gray, pattern, flags=NORMALIZE_IMAGE | EXHAUSTIVE |
   ACCURACY)`. With no `--pattern`, try sizes from large to small (e.g. 9x7 down to 4x3) on 15 frames and take
   the largest pattern found in ≥ 30 % of them. That gives 6x6 here. Disable OpenCL
   (`cv2.ocl.setUseOpenCL(False)`); on this Mac it prints cache warnings.
4. **Pick diverse views.** For each detection compute a quick board pose (`solvePnP` with the guess
   f = 1.2 × width). Bin views by board tilt (angle of the board normal: 0-15°, 15-30°, 30-45°, > 45°) and by
   where the corners fall in the image (3x3 grid). Take at most ~40 views, spread over the bins. Corners near
   the image edges are what constrain distortion, so prefer views that reach the borders.
5. **Calibrate.** `cv2.calibrateCamera` with `CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST |
   CALIB_FIX_K3` and an initial guess (`CALIB_USE_INTRINSIC_GUESS`, f = 1100, centre = image centre).
   Object points: `np.mgrid[0:6, 0:6] * 22.0` (mm, z = 0).
6. **Reject outliers.** Compute per-view RMS. Drop views above 2 × median RMS and calibrate again. Repeat up
   to 2 times.
7. **Check the result.** Fail with a clear message if RMS > 1.5 px, fewer than 15 views remain, or f differs
   from the SfM value of an existing job by > 5 %. That last check is optional, via `--check-job`.
8. **Write JSON.**

```json
{
  "camera": "iPhone 17 Pro Max", "source": "videos/calibration.MOV", "source_sha1": "...",
  "source_size": [3840, 2160], "width": 1280, "height": 720,
  "model": "PINHOLE", "fx": 1086.5, "fy": 1086.5, "cx": 645.8, "cy": 365.0,
  "dist": [0.0, 0.0, 0.0, 0.0], "dist_fitted": [0.24, -0.77, 0.0, 0.0],
  "rms_px": 1.05, "views": 40, "pattern": [6, 6], "square_mm": 22.0,
  "board_distance_mm": [225, 321, 498], "created": "2026-10-24T18:00:00"
}
```

Also save `calibration/<name>_views.jpg` (contact sheet of the used views with corners drawn) and
`<name>_residuals.png` (reprojection error per view, and as arrows over the image) for inspection.

Keep the calibration video out of the job picker: move it to `calibration/`, or filter names starting with
`calibration` in `server.videos()`.

### Distortion

Decide with data, not by default:

1. Calibrate twice: with k1, k2 free (`dist_fitted`) and with distortion fixed at zero.
2. If fixing distortion raises RMS by < 0.15 px, or the fitted distortion moves pixels by < 0.5 px inside the
   central 80 % of the image (evaluate `cv2.undistortPoints` on a grid), store `"dist": [0,0,0,0]` and keep the
   pipeline's PINHOLE model. This is the expected outcome for iPhone video.
3. Otherwise undistort the frames once in the `frames` stage (`cv2.initUndistortRectifyMap` + `remap`, with the
   new camera matrix from `getOptimalNewCameraMatrix(alpha=0)`) and write that matrix as the PINHOLE
   intrinsics. Undistorting the images keeps COLMAP, Brush and the renderer on the same simple pinhole model.
   Brush has no distortion support.

### Using it in the pipeline

- Config: `calibration: str = ""` (path to the JSON). When set:
  - `sfm.camera_params()` returns `fx, fy, cx, cy` from the file, scaled by `job_width / calib_width`.
    Change the return value to four numbers and pass `f"{fx},{fy},{cx},{cy}"` (PINHOLE allows fx ≠ fy).
  - `refine_focal` defaults to `False` when a calibration is given, unless set explicitly.
  - Also set `ba_refine_principal_point=False` (already the case).
- Guards (raise `StageWarn`, keep running): the source aspect ratio differs from the job video's; the camera
  model tag differs; the working resolution differs (scaling is exact only if both come from the same sensor
  crop).
- Capture rule for the docs: record the calibration with **the same lens (1x), resolution, frame rate and
  stabilisation setting** as the object videos. iPhone stabilisation crops and shifts the image per frame; use
  "standard" or off for both. Re-calibrate if any of these change.
- Report: `sfm` stats gain `"calibration": {"file": ..., "fx": ...}`, and the camera plot title shows "fixed
  intrinsics".

### Scale

The board fixes intrinsics only. The object videos contain no board, so their reconstructions still have no
units. Put the board (or an ArUco marker of known size) on the table in the object videos to get metric scale
later. Not part of this change.

### Tests

- `tests/test_calibrate.py`: render a synthetic checkerboard video with a known K (reuse the projection code from
  `tests/test_synthetic.py`), run `calibrate`, and assert f within 1 % and principal point within 2 px.
- Re-run `IMG_8651` from `sfm` with the calibration and `refine_focal=false`. Expect a similar registration
  (200/200) and a reprojection error ≤ the current 1.35 px. If it gets clearly worse, the calibration does not
  match the capture settings.

---

## 2. Fill small holes in the object mask

### Where

`segment.clean_object()` (`gsturn/stages/segment.py`, line 89). The order inside the function becomes:

1. remove dilated hand pixels (existing),
2. keep the large components (existing),
3. **fill small holes (new)**,
4. **re-apply the hand removal (new)**, so no filled pixel lies on the hand,
5. erode (existing).

### Algorithm

A hole is a background region that does not touch the image border. Fill it only if it is small and not
caused by the hand.

```python
def fill_small_holes(obj: np.ndarray, hand: np.ndarray, max_frac: float, close_px: int = 0):
    """obj/hand: uint8 0/1. Returns (filled_obj, filled_pixels)."""
    m = obj.copy()
    if close_px > 0:  # optional: seal hairline cracks first
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    area = int(m.sum())
    inv = (1 - m).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    h, w = m.shape
    filled = np.zeros_like(m)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches_border or a > max_frac * area:
            continue
        comp = cc == i
        if hand[comp].mean() > 0.2:      # a finger seen through/over the object: not a hole
            continue
        filled[comp] = 1
    out = np.maximum(m, filled)
    return out, filled
```

Notes:

- Use 4-connectivity for the background and 8 for the object (already used for components), so a diagonal
  one-pixel gap does not join a hole to the outside.
- `MORPH_CLOSE` can bridge the gap between object and fingers, so it is off by default (`close_px=0`). The
  re-applied hand removal in step 4 undoes any bridging onto the hand.
- Gaps between fingers that open to the border, and notches in the outline, are not holes. They stay as they are.

### Config, outputs, stats

- `fill_holes_frac: float = 0.005` (0 disables), `close_px: int = 0`.
- Write the filled pixels to `work/segment/filled/<stem>.png`. The `quality` stage marks them as "object, but
  untrusted" (a hole is usually a reflection or a dark shadow).
- Overlays: draw filled pixels in yellow.
- Stats: `filled_px_median`, `frames_with_filled`, `max_filled_frac`.

### Tests (`tests/test_units.py`)

A synthetic mask with: a small interior hole (filled), a large interior hole > 0.5 % (kept), an interior hole
under a hand blob (kept), and a notch open to the border (kept). Assert the filled pixels equal the small hole.

---

## 3. New stage `gsturn/stages/quality.py`

Builds a per-pixel **trust map** `Q` for every selected frame: 255 = trusted object pixel, 0 = untrusted or not
object. Runs after `segment`, before `features`.

### Inputs and outputs

| In | `work/frames/images/*.jpg`, `work/frames/frames.json`, `work/segment/masks/*.png`, `work/segment/hands/*.png`, `work/segment/filled/*.png` (if present) |
|---|---|
| Out | `work/quality/maps/<stem>.png` (uint8 Q, full frame size), `work/quality/quality.json`, `work/quality/overview.jpg`, `work/quality/trusted.png`, `stats.json` |

### Per-frame computation (720p, roughly 0.1-0.2 s per frame on the CPU)

Let `M` be the object mask (0/1), `I` the RGB frame, `G` its grayscale.

**a) Sharpness relative to neighbouring frames (motion blur, defocus)**

1. `L = cv2.Laplacian(cv2.GaussianBlur(G, (3,3), 0), cv2.CV_32F)`; energy per tile `E[t] = mean(L² over tile)`
   on a grid of `quality_tile` px (32), only for tiles with ≥ 50 % mask coverage.
2. **Align tiles to the object, not the image.** Build the grid relative to the mask's bounding-box centre, so
   tile (r, c) covers roughly the same part of the object in neighbouring frames. Kept frames differ by only a
   few degrees.
3. Keep a sliding window over frames `t-k … t+k` (`quality_window` = 2). For each tile take the reference
   `E_ref = max(E over the window)`.
4. Relative sharpness `r = E / E_ref`. Untextured tiles (`E_ref < quality_texture_floor`, e.g. 20) get `r = 1`:
   a blank area cannot be judged, and it produces no keypoints anyway.
5. Tile score `s = clip((r - 0.5·b) / (0.5·b), 0, 1)` with `b = blur_rel_threshold` (0.4): 0 below 0.2, ramping
   to 1 at 0.4. Upsample the tile grid to pixels with `cv2.resize(..., INTER_LINEAR)`.
6. Whole-frame defocus is covered as well: if every tile in the frame drops at once, the frame score falls (see
   per-frame score below).

**b) Clipped and specular pixels (reflections)**

```python
hsv = cv2.cvtColor(I, cv2.COLOR_RGB2HSV)
v, s = hsv[..., 2].astype(np.int16), hsv[..., 1]
clipped = (I.max(axis=2) >= 250)
tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
spec = (v >= specular_v) & (s <= specular_s) & (tophat >= 25)
bad_light = cv2.dilate((clipped | spec).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
```

Defaults: `specular_v = 230`, `specular_s = 40`. White print on the box is bright and unsaturated too, but it
is not brighter than its surroundings, so the top-hat test keeps it. If white areas still get flagged, raise the
top-hat threshold.

**c) Combine**

```
Q = M * s_pixel * (1 - bad_light) * (1 - 0.5 * filled)     # scaled to 0..255
```

Filled-hole pixels count as half trusted: used for training as part of the object, excluded from keypoints by
the 0.5 threshold below.

**Per-frame record** in `quality.json`:

```json
{"stem": "00099", "trusted_frac": 0.41, "blur_frac": 0.52, "specular_frac": 0.03,
 "filled_frac": 0.004, "median_rel_sharpness": 0.37, "excluded": true}
```

`trusted_frac` = share of mask pixels with `Q ≥ 128`. `excluded = trusted_frac < quality_min_frame` (0.5).

### Debug artifacts for the UI

All emitted through `ev.artifact(...)`, so the gallery shows them with no UI changes:

1. `overview.jpg`: contact sheet of 40 evenly spaced frames **plus the 8 worst**, each labelled with its
   `trusted_frac`. Colours: object trusted = untouched; blur = red tint; specular/clipped = magenta; filled =
   yellow; outside the mask = dimmed (the same style as `segment.overlay()`).
2. `trusted.png`: line plot of `% trusted` per frame, with stacked lines for blur, specular and filled, and a
   dashed line at `quality_min_frame`. Mark excluded frames in red. Same matplotlib style as `area.png`.
3. Optional: `worst.jpg` with the 4 worst frames at full size, map next to image.

Report progress with `ev.progress(i + 1, n, f"frame {i+1}/{n}")`. End with `StageWarn` if more than 25 % of
frames are excluded (`"x of n frames below quality_min_frame"`), otherwise `done` with
`"_msg": "median trusted 87 %, 14 frames excluded"`.

### Consumers (each guarded by "quality output exists and enabled")

| Stage | Change |
|---|---|
| `features.run` | drop frames with `excluded: true` from the `stems` list before extraction. `sfm` reads the stems from `features/index.json`, so the exclusion carries through. Do **not** delete or rename image files; the names stay stable for the UI. |
| `features.filter_in_mask` | pass `mask & (Q >= 128)` instead of `mask` |
| `train.prepare_masks` | new `train_mask_mode = "ignore_quality"`: `masks/<stem>.png = alpha AND Q ≥ 64`. Untrusted pixels are ignored and the background is ignored too. That is the trade-off described in [ignore-blur-defocus-reflections.md](ignore-blur-defocus-reflections.md); `alpha` stays the default. |
| `sfm.build_rgba_images` | unchanged (alpha stays the plain object mask) |

### Config

```python
quality_enabled: bool = True
quality_tile: int = 32
quality_window: int = 2
quality_texture_floor: float = 20.0
blur_rel_threshold: float = 0.4
specular_v: int = 230
specular_s: int = 40
quality_min_frame: float = 0.5
```

### Tests

- Unit: a textured frame with the right half Gaussian-blurred (σ = 3), plus a sharp neighbour frame. Assert
  that blur is flagged on the right half only. A blank grey tile must not be flagged. A small saturated white
  disk on a red surface must be flagged as specular; a large white printed area must not.
- Integration: re-run `IMG_8651` from `quality`. Expect frames ~99-110 to have the lowest `trusted_frac`
  (that is where the box is smeared). Compare `features` and `sfm` stats with the current run: keypoints per
  frame, points per image, reprojection error.

---

## 4. Pose-guided re-matching (two-pass SfM): new stage `refine`

After the first SfM pass every frame has a pose. Find frames that look at the object from a similar direction
but are far apart in time, match them, and re-triangulate and bundle-adjust with the extra matches.

### Prerequisite: keep descriptors

`features.run` currently saves only keypoint positions (`keypoints.npz`) and matches. Re-matching needs the
descriptors, and re-extracting them would cost ~1.6 s per frame on this CPU. Add:

```python
np.savez_compressed(out / "descriptors.npz", **{s: d.astype(np.float16) for s, d in zip(stems, descs)})
np.savez_compressed(out / "scores.npz", **{s: sc.astype(np.float16) for ...})   # keypoint_scores
```

Store them in the same order as `keypoints.npz` (after `filter_in_mask`). For ALIKED/DISK also store `scales`
and `oris` if present, because LightGlue uses them. Size: ~650 keypoints × 256 × 2 bytes ≈ 0.3 MB per frame
(SuperPoint), about 70 MB for 200 frames before compression. For SIFT the RootSIFT descriptors are 128-D.

Add a loader that rebuilds the LightGlue input dict (`keypoints`, `descriptors`, `keypoint_scores`,
`image_size`) from these files, and reuse `features.match_lightglue` / `match_sift` unchanged.

### Step 1: candidate pairs from poses

Load `work/sfm/dataset/sparse/0` with pycolmap. For every registered image `i`:

- rotation `R_i` (world→camera, `image.cam_from_world.rotation.matrix()`), centre
  `C_i = image.projection_center()`,
- object centre `O` = median of `points3D` xyz (after removing the 1 %/99 % percentiles per axis).

For each pair `(i, j)` with `|i - j| > refine_min_gap` (default `match_window`, 8):

- **relative rotation angle** `θ_ij = arccos((trace(R_i R_jᵀ) - 1) / 2)`. This covers both viewing
  direction and in-plane roll. SuperPoint/LightGlue tolerate roughly ±30-40° of in-plane rotation, so the limit
  matters for roll too.
- **distance ratio** `ρ_ij = |C_i - O| / |C_j - O|` must lie within 0.7-1.4 (similar apparent size).
- keep if `θ_ij < refine_max_angle` (30°) and the pair is not already in `matches.npz` with ≥ `min_matches`.

Rank each image's candidates by `θ` and keep the best `refine_pairs_per_image` (5). Deduplicate. Expected for
200 frames: ≤ 500 pairs. At the ~0.28 s per pair measured for LightGlue on `IMG_8651`, that is ≈ 2-3 min.

### Step 2: match and pre-filter with known geometry

For each candidate pair:

1. Match descriptors (LightGlue, or SIFT ratio test).
2. **Guided filter**: build the fundamental matrix from the two known poses and the shared intrinsics
   (`F = K⁻ᵀ [t]ₓ R K⁻¹` from the relative pose). Drop matches with Sampson error > `refine_max_epipolar_px`
   (4 px). This removes wrong matches between repeated patterns (the checker print) much better than
   RANSAC alone.
3. Keep pairs with ≥ `min_matches` remaining.

### Step 3: add to the model

Default mode `refine_mode = "triangulate"` (fast, keeps poses):

1. Copy `work/sfm/database.db` to `work/refine/database.db` (keep the original intact).
2. `db.write_matches(id_i, id_j, m)` for the new pairs. Write their names to `pairs_new.txt` and run
   `pycolmap.verify_matches(db, pairs_new.txt, TwoViewGeometryOptions())`, so only new pairs are verified.
3. `rec = pycolmap.Reconstruction(sparse/0)`, then
   `rec = pycolmap.triangulate_points(rec, db, image_dir, out_dir, clear_points=True, options=IncrementalPipelineOptions(...), refine_intrinsics=False)`.
   With `clear_points=True` all tracks are rebuilt from the full match graph, so old short tracks merge through the
   new loop pairs.
4. Global bundle adjustment: `pycolmap.bundle_adjustment(rec, opts)` with `opts.refine_focal_length =
   cfg.refine_focal` (False with a calibration), `refine_principal_point = False`, `refine_extra_params = False`.
   Poses and points are refined.

Alternative `refine_mode = "remap"`: run the full `incremental_mapping` again on the extended database. Slower,
but it can register frames the first pass missed. Use it automatically when the first pass registered < 100 %.

### Step 4: iterate (optional)

`refine_iterations` (default 1). With refined poses, recompute candidates (step 1) and repeat. A second round
rarely adds much. Stop early if fewer than 20 new pairs are found.

### Step 5: accept or reject, and write

Compare before and after:

| Metric | Source |
|---|---|
| registered images | `rec.num_reg_images()` |
| mean reprojection error | `rec.compute_mean_reprojection_error()` |
| mean track length | `rec.compute_mean_track_length()` |
| points3D, min/median points per image | per image `num_points3D` |

Accept if registered images did not drop and reprojection error did not rise by more than 0.1 px. Then:

- move the original to `work/sfm/dataset/sparse_first_pass/0` and write the refined model to
  `work/sfm/dataset/sparse/0`, so `train` and `render` pick it up with no changes. Rewrite `model_txt/`.
- Otherwise keep the original and end with `StageWarn("refinement rejected: ...")`.

Re-running `sfm` must also delete `sparse_first_pass/` (it belongs to the old pass).

### Artifacts and stats

- `cameras_before_after.png`: `sfm.plot_model()` for both models side by side.
- `pairs.png`: scatter of time gap `|i - j|` vs. angle `θ` for all candidates; accepted pairs coloured by match
  count.
- `match_matrix_refined.png`: the existing match matrix with the new pairs overlaid in a second colour, so
  the added loop closures are visible off the diagonal.
- One or two `draw_matches` images of the strongest new long-range pairs.
- Stats: `candidates`, `pairs_matched`, `pairs_added`, before/after table, `accepted`.
  `_msg`: `"+412 loop pairs, track length 10.3 -> 13.8, reproj 1.35 -> 1.12 px"`.

### Config

```python
refine_enabled: bool = True
refine_mode: str = "triangulate"        # triangulate | remap
refine_max_angle: float = 30.0          # degrees
refine_min_gap: int = 0                 # 0 = use match_window
refine_pairs_per_image: int = 5
refine_max_epipolar_px: float = 4.0
refine_iterations: int = 1
```

### Tests

- Synthetic cube (`tests/test_synthetic.py`): its 90 frames make one full turn, so frame 0 and frame 89 see the
  same side. Run `sfm` then `refine`. Assert that loop pairs across the seam are added, that mean track length
  increases, and that reprojection error does not get worse.
- Unit: the relative-rotation angle and Sampson filter on hand-built poses, e.g. two cameras 20° apart with
  correct and wrong matches.
- `IMG_8651` targets, compared with today (track length 10.3, reproj 1.35 px, min 227 points per image):
  track length > 12, reproj ≤ 1.2 px, min points per image > 300, and a higher eval PSNR after re-training
  (22.7 today).

---

## Suggested order and checkpoints

1. **Calibration** (independent). Checkpoint: JSON with f ≈ 1085 ± 10 px at 720p; `IMG_8651` SfM with fixed
   intrinsics still registers 200/200.
2. **Hole filling** (small, independent). Checkpoint: overlays show yellow fills only on small interior holes.
3. **Stage plumbing** (section 0), then **quality**. Checkpoint: `trusted.png` dips at frames ~99-110 on
   `IMG_8651`.
4. **Save descriptors**, then **refine**. Checkpoint: before/after stats meet the targets above.
5. Re-run `IMG_8651` end to end and update README section 14 with the new numbers. Then run the remaining
   videos with the same settings.
