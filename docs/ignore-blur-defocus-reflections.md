# Ignoring motion blur, defocus and bad reflections

Goal: pixels that are blurred, out of focus or show a reflection should not feed **keypoints/SfM** or **splat
training**. Today they do. The pipeline's only per-pixel exclusions are background and hand; image quality is
judged once per *frame*, on the *whole* image.

## Where the current pipeline falls short

| Stage | Today | Gap |
|---|---|---|
| `frames` | variance of the Laplacian over the whole frame, compared with a rolling median (`sharp_keep_ratio` 0.55) | the background (table, sharp) dominates the score, so a frame with a blurred object still passes; a frame is either all in or all out |
| `features` | keeps keypoints inside the eroded object mask (`filter_in_mask`) | keypoints on highlights or blurred areas are kept |
| `train` | `alpha` mode: object = opaque, everything else transparent; `ignore` mode: non-object pixels ignored | no way to say "this part of the object is bad in this frame" |

Evidence from `IMG_8651`: frames ~99-110 (box edge-on, turning fast) are visibly smeared, yet all were kept.
They have the fewest 3D points (min 227 at frame 99) and the weakest matched pair (109-110, 128 matches).

## Strategy: one quality map per frame, used three times

For every kept frame compute a **trust map** `Q(x, y)` in [0, 1], restricted to the object mask:

```
Q = sharp(x,y) * not_specular(x,y) * not_clipped(x,y)
```

Save it as `work/quality/<frame>.png` next to the masks, then use it in three places:

1. **Frame selection**: drop a frame if too little of the object is trusted (e.g. mean `Q` over the mask below
   0.5), instead of judging whole-frame sharpness.
2. **Keypoints**: in `filter_in_mask`, keep only keypoints where `Q` is above a threshold (e.g. 0.5).
3. **Training**: write `Q > threshold` into the masks Brush reads, so bad pixels are ignored (see "Training
   masks" below for the trade-off).

### A. Motion blur and defocus: local sharpness

Both remove high-frequency detail; they differ in that motion blur has a direction.

1. **Local sharpness map**: compute the Laplacian (or gradient magnitude) energy in small tiles (e.g. 32x32 px)
   inside the object mask.
2. **Normalise per tile, not globally.** A plain white area of the box is "unsharp" even when perfectly in focus.
   Compare each tile with the *same surface area in sharp neighbouring frames*. Two ways:
   - before SfM: compare the tile's energy with the maximum energy of the same tile position in the
     ±2 neighbouring frames (cheap; works because consecutive frames differ by a few degrees);
   - after SfM: project the tile into other frames with the known poses and compare with the sharpest view of
     that surface (exact; needs a second pass).
   A tile at below ~40 % of its best view is marked as blurred.
3. **Motion blur direction (optional)**: the ratio of gradient energy along x and y (or the structure tensor)
   shows directional smear. That separates motion blur from defocus, which helps diagnose captures but is not
   needed to mask them.
4. **Predicted blur from poses (after SfM)**: the object's rotation between two frames and the camera model give
   how far each surface point moves on screen per frame. Times the exposure fraction (shutter / frame interval),
   that is the expected blur in pixels. Above ~1.5 px at working resolution, mark the area as blurred. This needs
   no image statistics and is reliable on low-texture areas, but needs the shutter time (iPhone metadata or a
   manual camera app).

Defocus specifically: with a fixed-focus camera it appears when the object comes too close or goes too far.
It then affects the whole object at once, so the frame-level check (step 1 of the strategy) catches it; the
tile check covers depth of field across a tilted object.

### B. Reflections and clipped highlights

1. **Clipped**: any channel ≥ 250, dilated by a few pixels. Clipped pixels hold no texture and no correct colour.
2. **Specular candidates**: high brightness and low saturation (in HSV: V high, S low), clearly brighter than
   the surrounding area (a top-hat filter on V). Specular highlights are usually white even on coloured surfaces.
3. **Confirm by multi-view inconsistency (best signal)**: a reflection moves across the surface as the object
   turns; real texture does not. After SfM (or after a first training run), compare each frame with what the
   model predicts for it:
   - SfM: colour of each 3D point in each view vs. its median over all views,
   - splat: render the trained splat from that frame's pose (Brush eval renders, `--eval-save-to-disk`) and take
     the per-pixel difference.
   Pixels that are much brighter than the consensus in only a few views are reflections. Mask them and re-train
   (one extra iteration).
4. Pixels near the object edge that show the background or hand colour (halo) are already removed by
   `erode_px`; raise it if halos remain.

### C. Holes left by the masks (todo item 3)

Reflections and shadows punch small holes into the object mask. Fill holes smaller than ~0.5 % of the object
area (`cv2.morphologyEx` close, then fill enclosed holes), but **after** that combine with `Q`. A hole caused by a
reflection should become "object, but untrusted", not "background". Otherwise the splat in `alpha` mode is
forced to be transparent there.

## Training masks: the trade-off

Brush reads **either** alpha (object = opaque, rest transparent) **or** a `masks/` folder (masked pixels ignored),
not both (see README stage 6). Options:

| Option | Background | Bad pixels | Note |
|---|---|---|---|
| `alpha` (today) | forced transparent | used as normal | default |
| `ignore` with mask = object AND trusted | ignored (floaters possible, crop in SuperSplat) | ignored | simplest correct choice |
| alpha, with bad pixels replaced by the model's render (inpainted from other views) | forced transparent | replaced by consensus | best result, needs a first training pass |
| patch Brush to accept alpha + a separate loss weight map | forced transparent | ignored | small Rust change in `brush-dataset`; cleanest long-term |

Recommended: start with `ignore` + trust mask. If floaters are a problem, use the two-pass inpainting.

## Implementation plan

1. New module `gsturn/stages/quality.py` that writes `work/quality/*.png` plus a per-frame score, with an
   overlay contact sheet and a "% trusted per frame" plot for the UI. Runs after `segment`.
2. `frames`: optional second selection pass on the per-frame trusted fraction (re-uses the existing kept list).
3. `features.filter_in_mask`: also require `Q` above a threshold.
4. `train.prepare_masks`: new `train_mask_mode=ignore_quality`.
5. Second pass (optional `refine` stage, shared with `docs/sfm-known-view-angles.md`): after SfM, use poses for
   predicted blur and multi-view reflection checks, update `Q`, re-train.
6. Config fields: `quality_min_frame` (0.5), `quality_tile` (32), `blur_rel_threshold` (0.4),
   `specular_v`/`specular_s`, `fill_holes_frac` (0.005).

How to judge it: the same `report.json` metrics as the other docs (keypoints and 3D points per frame in the weak
stretch, reprojection error, eval PSNR). Also look at the eval renders around frames 99-110 and 157 for smearing
and baked-in highlights.
