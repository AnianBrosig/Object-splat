# Improving SfM by re-detecting known view angles

## The idea

A hand-held turn shows the same side of the object several times: the box goes around, comes back, gets tilted,
and gets re-gripped. Whenever a later frame shows an orientation that was seen before, matching it to that
earlier frame ties the two ends of the sequence together (**loop closure**). Each such link:

- turns two short 3D point tracks into one long track (more views per point means better triangulation),
- stops drift from building up along the sequence (without it, errors add up frame by frame), and
- lets frames that were blurred or half-covered register through a sharp earlier frame of the same view.

The current pipeline only finds these links by chance: it matches each frame to its 8 neighbours, plus fixed
long-range pairs every 20 frames (`match_window`, `match_stride_pairs`). It never looks at what a frame shows.

## Where `IMG_8651` stands

| Metric | Value | Comment |
|---|---|---|
| registered | 200/200 | good |
| mean reprojection error | 1.35 px | just inside the 1.5 px gate; 0.6 px on the synthetic test |
| mean track length | 10.3 views | fairly short for 200 frames that all show the same object |
| 3D points per image | median 565, min 227 (frame 99) | the edge-on, blurred stretch |
| weakest sequential pair | 128 matches (frames 109-110) | same stretch |
| focal length | prior 1536 px, refined to 1091 px | a large correction, see section 3 |

The match matrix (`work/features/match_matrix.png`) is a bright diagonal band plus a sparse grid of stride pairs.
The stride pairs that do match are exactly the loop closures; the grid samples only ~1 in 10 of the frame
combinations that could have them.

## Options

### 1. Pose-guided re-matching (two-pass SfM) - recommended first step

After the first SfM pass every frame has a pose, so its **viewing direction relative to the object** is known.

1. For every pair of registered frames, compute the angle between their viewing directions (object centre to
   camera centre) and the difference in roll.
2. Select pairs that look at the object from similar directions (for example < 30°) but are **far apart in time**
   (more than `match_window` frames). These are the "known view angle" re-detections.
3. Match only those pairs with LightGlue (the keypoints are already stored in `keypoints.npz`; only matching runs).
4. Add them to the database and re-run the mapper, or triangulate and bundle-adjust the existing model
   (`pycolmap.triangulate_points` + `bundle_adjustment`). The second option is faster and keeps the poses.
5. Optionally repeat once.

Cost: one extra matching pass over a targeted set of pairs. That is far fewer than exhaustive (19,900 pairs for
200 frames) and every pair is likely to match. Implementation: a new optional stage `refine` between `sfm` and
`train`, roughly 150 lines.

Expect: longer tracks, more points per image in weak stretches, lower reprojection error, and a closed camera
path in the `cameras.png` plots (no gap or doubled-up path where the sequence revisits a view).

### 2. Guided matching for weak frames

With known poses, a keypoint in frame A must lie on a known line (its epipolar line) in frame B. Searching only
along that line finds matches that the unguided matcher misses in blurred or low-texture frames. COLMAP has this
built in (`guided_matching` in the matching options). Apply it to the weakest stretch (frames with < 300
points). Effort: small.

### 3. Fix the intrinsics with a one-off calibration

The focal length moved from 1536 to 1091 px (29 %) during bundle adjustment. When focal length and object
distance are both free, part of the geometry error is absorbed into the wrong parameter. Calibrate the iPhone
once at the recording settings (checkerboard, OpenCV `calibrateCamera`, 4K main camera, 720p working resolution),
put it in `focal_prior`, and set `refine_focal=false`. This also makes 3D results comparable across videos.
Effort: small.

### 4. Retrieval-based loop pairs instead of fixed strides

Choose long-range pairs by appearance instead of a fixed grid. Compute one global descriptor per frame (for
example DINOv2, or SuperPoint descriptors pooled over the object mask) on the masked crop, and match each frame
to its top-k most similar frames that are not neighbours. This finds loop closures **before** SfM, which helps
when the first pass fails or splits into several models. Effort: medium (another model on CPU, but one forward
pass per frame).

### 5. Register additional videos into one model

The other videos (IMG_8645-8652) show the same kind of object from different grips. If they show the **same
object**, register their frames into the `IMG_8651` model:

- localise each new frame against the existing 3D points (2D-3D matches plus PnP), or
- add their images and features to the same database and continue the mapper from the existing model
  (`incremental_mapping(..., input_path=existing_model)`).

The views the first video missed (the bottom, areas always under the fingers) then fill in. This is the most
direct use of "known view angles": the first reconstruction becomes the reference that the other captures
register to. Effort: medium.

### 6. Rotation prior from frame order

Between consecutive kept frames the object turns only a few degrees. Adding this as a soft prior in bundle
adjustment (or as an initial pose for registration) prevents a frame from jumping to a mirror pose on symmetric
objects such as boxes. Low priority while all frames register correctly.

## Suggested order

1. Calibrate the camera (option 3): an hour of work, removes one free parameter.
2. Pose-guided re-matching (option 1) as an optional `refine` stage; report track length and points per image
   before and after.
3. Guided matching for weak frames (option 2).
4. Once the other videos have been run: multi-video registration (option 5).

Success criteria, all from `report.json`: mean track length clearly above 10, minimum points per image above
~400, reprojection error ≤ 1 px, and a higher training PSNR on the eval views than today's 22.7.
