# Using the reconstruction to locate, mask and orient the object live

## Implemented: approach A, real time on this MacBook (Intel i9, AMD GPU)

```
python -m gsturn export-model --job jobs/<job> [--object-size 0.145]   # once, ~1 min -> outputs/object_model.npz
python -m gsturn camera                                               # webcam preview (s snapshot, r record)
python -m gsturn camera --calibrate --square-mm 22                    # webcam intrinsics -> calibration/
python -m gsturn track --model jobs/<job>                              # live: built-in webcam
python -m gsturn track --model jobs/<job> --source video.mp4 --poses poses.jsonl --save-video out.mp4
python -m gsturn track-eval --job jobs/<job>                           # accuracy vs the SfM poses
```

Code in `gsturn/track/`. What differs from the plan below, and why:

- **SuperPoint runs on the GPU (MPS)**: 23 ms per 640x360 image instead of ~650 ms on the CPU, identical output.
  This makes relocalisation take ~60-70 ms, so no ORB/SIFT fallback was needed. MPS compiles kernels per tensor
  shape (~200 ms each), so crops are padded to a fixed canvas, candidate sets to fixed sizes, and all shapes
  are compiled once at start-up (~3 s).
- **No retrieval step**: keypoints are matched directly to the 3D points (descriptor = mean SuperPoint
  descriptor over the views, stored at 4 image scales: 1, 0.7, 0.5, 0.35, so a smaller/further object
  still matches). While tracking, only points seen from keyframes within 40 deg of the predicted view and the
  3 nearest scales are compared (the prior pose replaces retrieval); when lost, the whole map is searched.
- **Threads**: relocalisation runs in a background thread on a crop around the predicted box (every 0.3 s,
  or sooner when few points remain); its result is carried to the current frame with KLT and merged.
  KLT + PnP on every frame takes ~7 ms, so the loop runs at the camera rate (29-30 fps at 1280x720).
- **Mask**: the convex hull of the cleaned points, projected at the pose (0.2 ms). It includes the parts hidden by
  the hand. Fine for convex objects like the box; a mesh or splat render would be needed for concave ones.
- **Smoothing**: an adaptive filter (strong at rest, none during fast motion) instead of a Kalman filter.
  After losing the object, the last pose is held for up to 6 frames with status `coast`.
- **pycolmap is not imported** by the tracker: loading it in the same process makes torch MPS segfault. The
  SfM model is read from `work/sfm/model_txt`.

Measured on job `IMG_8651` (pen box), `track-eval`:

| test | result |
|---|---|
| held-out keyframes (map without the frame and its +-2 neighbours), crop + prior, image scale 1 / 0.5 / 0.33 | 20/20 each; median rotation error 0.28 / 0.29 / 0.48 deg, translation 0.1-0.3 % of distance; 58-70 ms |
| same, whole frame (lost) | 19-20/20; 0.2-1.2 deg; 140-215 ms |
| whole video, paced at 30 fps, vs the 200 SfM poses | 190-199/200 frames with a pose, median 0.4-0.6 deg, 0.2 % translation; tracking 7 ms/frame |
| same with darker, noisy, blurred, JPEG-compressed frames (webcam-like) | 199/200, median 0.4-0.5 deg |

### Orientation and the live camera (added after the first webcam tests)

- **In-plane rotation.** SuperPoint descriptors are not rotation invariant: relocalisation worked within +-30 deg
  of how the object was held in the capture video and failed beyond 45 deg (0/17 at 90 deg). Now:
  while tracking, the crop is rotated so the object looks as in the nearest keyframe; when lost, the whole
  frame is tried at 8 rotations (45 deg steps, 2 per attempt, starting from the last one that worked).
  Held-out keyframes rotated by 0-250 deg: 16-17/17 found in both modes, no false detections on 21 frames
  without the object.
- **Webcam focal length.** `calibration/webcam0_1280x720.json`, from `videos/webcam0_20261024-211950.mp4`
  (22 mm squares, 48 views, median tilt 31 deg): **f = 948 +-3 px** (68 deg horizontal FOV), principal point
  (632, 361), pinhole (k1/k2 lowered the RMS only from 1.77 to 1.70 px). The first board recording
  (`..._191736.mp4`) had only face-on views, which leave f undetermined (850-1000 px fit equally well;
  with a free principal point the solver collapses to 130-280 px). Estimating f from the tracked box gave
  997 px (+-0.9 %), 5 % high: it inherits the small shape error of the SfM model (SfM's iPhone focal was 1 %
  off the calibrated one). Use the board value; the box estimate is only a fallback without a board.
- **Learning the camera's view of the object.** The map only knew how the phone saw the box. Two ways to
  teach it the webcam's view; both sample descriptors at the projected 3D points from an upright crop, keep
  only samples that agree with the map (hands do not), and add new views as keyframes:
  - `track --learn`: live, in a background thread, from the moment the object is found; saved into the
    package (plus new triangulated points) on exit. On `webcam0_20261024-191614.mp4` (deterministic replay):
    900 -> 956 of 1078 frames tracked; the longest dropout while the box is visible fell from ~90 to ~16 frames.
  - `refine-model --video rec.mp4`: offline, two passes over a recording, also triangulates new 3D points
    (>= 3 webcam views each). The original package is kept as `object_model_base.npz`. Learning from the even
    2-second chunks of the box recording and testing on the odd ones: cold search 28 -> 34 of 60 frames,
    tracked 342 -> 400 of 538 frames (55 webcam keyframes and 265 new points added).

Limits seen: during one fast, motion-blurred half turn a few frames are off by 5-25 deg (mostly `coast` frames)
before tracking recovers. The webcam pose is only as good as its intrinsics: without a calibration a 60 deg field of view is
assumed, which mainly biases the distance. Positions are in SfM units unless `--object-size` is given.

## What a finished job gives you

| Data | Where | Use for live tracking |
|---|---|---|
| 200 frames with **known 6-DoF pose** and intrinsics | `work/sfm/dataset/sparse/0` (`images.bin`, `cameras.bin`) | reference views ("keyframes") with ground-truth orientation |
| object masks per frame | `work/segment/masks/` | labelled training data for a detector/segmenter |
| ~11k 3D points, each seen in ~10 frames | `points3D.bin` + `work/features/keypoints.npz` | 2D-3D correspondences: each point can carry a SuperPoint descriptor |
| Gaussian splat | `outputs/splat.ply` | renders the object from **any** pose: masks, synthetic training images, render-and-compare |

Two gaps to close before using it in the real world:

- **Scale.** SfM has no units. Measure one edge of the object (or put a marker of known size in one frame) and
  scale the model, so positions come out in metres.
- **Object frame.** The model's axes are arbitrary (first camera). Define an object frame, e.g. the centroid plus
  principal axes of the points (the export preview already computes these), or by hand: "x along the long edge".
  Orientation is then reported in that frame.

## Approach A: feature-based localisation (recommended first)

Uses only what the pipeline already produces; this is the classic "visual localisation against a map" approach
(as in hloc, or OnePose for objects).

**Offline (once per object)**
1. For each 3D point, store the mean of its SuperPoint descriptors over the frames that see it.
2. For each keyframe, store a global descriptor (e.g. DINOv2 or pooled SuperPoint on the masked crop) for fast
   lookup of "which reference view looks most like this".

**Online, per camera frame**
1. **Detect**: find the object's 2D box. Options: retrieval over a sliding window (slow), a small detector
   trained on the masks (approach C), or simply the box projected from the previous pose while tracking.
2. **Retrieve** the 3-5 most similar keyframes in the box.
3. **Match** SuperPoint keypoints of the live crop to those keyframes with LightGlue. Each match gives a 2D-3D pair.
4. **PnP + RANSAC** (`cv2.solvePnPRansac`) gives the **6-DoF pose**: position and orientation of the object
   relative to the camera. Hands and background are simply outliers.
5. **Mask**: render the splat (or the convex hull or a mesh of the points) at that pose. That gives a mask that
   includes the parts hidden by the hand; intersect with a skin detector or a hand segmenter if you need the
   visible part only.
6. **Track**: for the next frames, follow the inlier points with optical flow (KLT) and solve PnP again
   (milliseconds). Run the full match (steps 2-4) again only every N frames or when the inlier count drops.
7. Smooth the pose over time (e.g. a constant-velocity Kalman filter on position and rotation).

**Speed on this MacBook**: SuperPoint took ~1.6 s per 720p frame on the CPU here. That is too slow for full
re-localisation every frame, but fine for "re-localise every 1-2 s, track with KLT in between". Options to get
faster: run at 480p or on the object crop only; SuperPoint on MPS (unverified); ORB/SIFT descriptors in the
reference map instead (fast on the CPU, less robust); or any NVIDIA GPU, where SuperPoint + LightGlue run at
20-50 fps.

**Limits**: needs texture (fine for the pen box, poor for plain objects) and fails on views missing from the
capture (e.g. the bottom, if never shown).

## Approach B: render-and-compare refinement with the splat

Given a rough pose (from A, or from the previous frame), adjust the pose until the splat render matches the live
image (minimise the photometric and mask difference, as in iNeRF or 3DGS pose refinement). This gives very
precise orientation and works on weaker texture, but needs a differentiable renderer on the GPU. Use it as
the final refinement step on top of A, not on its own.

## Approach C: learned detector and pose network (most robust, most work)

The splat makes training data cheap: render the object from thousands of random poses, composite it over random
backgrounds and hand images, and you get perfect labels (box, mask, 6-DoF pose) for free. With the 200 real
masked frames added:

- **Detection + mask**: fine-tune a small instance-segmentation model (e.g. YOLO-seg) on these. It runs in real
  time even on a CPU at low resolution. For a mask that follows the object frame to frame, Cutie (already in the
  pipeline) can track from the detected first mask.
- **Orientation**:
  - *keypoint-based*: a network predicts 2D positions of ~8-20 chosen 3D points (PVNet-style), then PnP. Robust to
    occlusion by hands.
  - *direct*: regress the rotation (e.g. 6D rotation representation) from the crop. Simpler, less accurate.
  - *model-based foundation models*: FoundationPose and similar take a mesh or reference images of a novel object
    and estimate and track pose without per-object training. They need a mesh (extract one from the splat, e.g.
    2DGS/SuGaR, or Poisson on a dense cloud) and an NVIDIA GPU.

## Things to plan for

- **Symmetry**: a box looks the same rotated 180° except for its print. Texture-based methods (A) resolve this
  via the print; learned methods (C) need a symmetry-aware loss or textured training renders.
- **Hand occlusion**: expected and handled by RANSAC (A) or occlusion augmentation (C). Capture all sides, and
  re-grip, so every side has unoccluded reference views.
- **Lighting**: the live scene will be lit differently from the capture. Descriptors (A) tolerate this fairly
  well; for C, randomise brightness/colour in the synthetic renders. See `docs/lighting-shadows-reflections.md`.
- **Camera**: use the same camera calibration format; if the live camera differs, use its own intrinsics in PnP.
  The 3D model does not depend on the capture camera.

## Suggested path

1. Add an `export_model` step that writes a compact "object package": scaled 3D points with mean descriptors,
   keyframes with poses and global descriptors, intrinsics, the object frame, and the splat.
2. Prototype approach A as a script on a recorded video (`python -m gsturn track videos/new.mov --model ...`) that
   shows the pose axes and the rendered mask over the video. Measure accuracy on held-out frames of a captured
   video, where the SfM pose is ground truth.
3. Add KLT tracking between re-localisations to reach real-time rates on this machine.
4. If robustness is not enough (fast motion, textureless objects): approach C with synthetic data from the splat,
   trained on a GPU machine.
