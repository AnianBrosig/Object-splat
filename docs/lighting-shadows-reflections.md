# Dealing with lighting, shadows and reflections

The camera and the room light stay fixed while the object turns in your hands. So, **relative to the object**, the
light moves. Every effect below comes from that: the same surface point looks different from frame to frame. That
hurts three stages:

| Stage | What goes wrong |
|---|---|
| segment | shadows and bright highlights pull the mask edge in or out; dark faces blend into a dark background |
| features / sfm | highlights produce keypoints that slide over the surface (they belong to the light, not the object) and create wrong matches; exposure changes reduce match counts |
| train | the splat has to explain changing brightness with one appearance per Gaussian; this shows up as floaters, blotchy colour, or highlights "painted" onto the surface |

What we see in `IMG_8651` (pen box, diffuse window light): exposure is stable and there are no strong highlights,
so lighting is **not** the main problem in this video. The weakest stretch (frames ~99-110: fewest 3D points at
frame 99, weakest matched pair at 109, mask-area jump at 110) is motion blur plus fingers covering the narrow face
while the box turns edge-on. The glossy barcode label and the plastic window do show small moving highlights
(e.g. frame 157). Those are the cases the options below target.

Options are ordered from cheapest to most effort. **Effort**: S = config/capture only, M = a few hundred lines
in this repo, L = new model or a CUDA machine.

## 1. Capture (biggest effect, no code)

| Option | Helps with | Effort |
|---|---|---|
| Lock exposure, focus and white balance (iPhone: long-press AE/AF lock; better, an app like Blackmagic Camera with manual ISO/shutter/WB) | brightness drift between frames | S |
| Fast shutter (1/250 s or faster) with more light | motion blur, the main failure in `IMG_8651` | S |
| Large diffuse sources: a window with a sheer curtain, softboxes, or a cheap light tent | hard shadows, highlights | S |
| Light from the camera direction (ring light, or lamps right next to the lens) | shadows fall behind the object, so the camera barely sees them | S |
| **Cross-polarisation**: linear polariser film on the light, a polarising filter on the lens rotated 90° to it | removes almost all specular reflection, the standard photogrammetry trick | S-M (needs hardware) |
| Matte spray (e.g. AESUB, which evaporates) on glossy or transparent parts | reflections, transparency | S |
| Contrasting, uniform background (a colour that is neither on the object nor skin) | segmentation under shadows | S |
| Gloves in a colour unlike the object (or matching the background) | hand and object separation, hand shadows | S |

## 2. Frame selection and segmentation

| Option | How | Effort |
|---|---|---|
| Reject blurred frames more strictly | raise `sharp_keep_ratio` (0.55 to ~0.7); also score sharpness **inside the object mask only** (currently the whole frame, so a sharp background hides a blurred object). Needs masks before selection, or a second selection pass after `segment`. | M |
| Per-pixel "do not trust" mask | flag pixels that are clipped (V ≈ 255) or bright with low saturation (typical specular highlight). Exclude them from keypoints (stage 4) and give them zero weight in training (Brush `masks/`). This is also todo item 1. | M |
| Fill small mask holes | morphological closing plus filling holes below N px. Reflections and dark shadows often punch holes in the mask (todo item 3). | S-M |
| Shadow-tolerant first-frame mask | run GrabCut in a colour space that separates brightness from colour (Lab or HSV with V down-weighted), so a shadow on the table is not treated as object | M |

## 3. Features and SfM

| Option | How | Effort |
|---|---|---|
| Normalise contrast before extraction | CLAHE on the luminance channel, applied to the copy fed to the detector only. Makes SuperPoint and SIFT more stable under exposure changes. | S |
| Drop highlight keypoints | use the "do not trust" mask from section 2 in `filter_in_mask` | S |
| Stricter geometric verification | lower `TwoViewGeometry.max_error` (4 px to ~2 px at 720p). Matches on moving highlights fail the epipolar check more often. | S |
| Second pass using the reconstruction | after SfM, every 3D point has a colour in every image that sees it. Points whose brightness varies far more than the rest across views are view-dependent (highlights). Remove their observations, re-run bundle adjustment. | M |

## 4. Splat training

| Option | How | Effort |
|---|---|---|
| Per-frame colour correction | estimate a gain and offset (or a 3x3 colour matrix) per image from the colours of shared 3D points, then correct the images before training. Removes exposure and white-balance drift. Fits naturally between `sfm` and `train`. | M |
| SH degree | Brush already uses `--sh-degree 3` by default, so each Gaussian can change colour with viewing direction and mild highlights are modelled. This works here only because camera and light are both fixed in the room, so "view direction" also stands in for "light direction". Lowering it (`--sh-degree 0/1` via `train_cmd`) gives a flatter, more albedo-like result with fewer baked-in highlights. | S |
| Ignore-masks for highlights | `train_mask_mode=ignore` plus highlight pixels removed from `masks/`, so the splat is not forced to reproduce them | S-M |
| Per-image appearance embeddings | trainers like nerfstudio splatfacto (`use_appearance_embedding`) or gsplat learn one small vector per image that absorbs lighting changes. Handles shadows that move across the object best, but needs CUDA. | L |
| Relightable or intrinsic splats (separate albedo, normal, specular) | research-grade (e.g. GS-IR, Relightable 3DGS). Only worth it if you need to relight the object later. | L |

## 5. Recommendation

1. Capture: AE/AF/WB lock, fast shutter, diffuse light near the camera. Try cross-polarisation if glossy objects
   matter.
2. Code (M, highest value): sharpness measured inside the mask, plus the per-pixel "do not trust" mask (blur and
   highlights), used by both features and training. This covers todo items 1 and 3.
3. Code: per-frame colour correction before training.
4. Only if the above is not enough: move training to a CUDA machine for appearance embeddings.

To compare options, re-run from the affected stage and compare `report.json` values: mask-area jumps,
keypoints per frame, SfM reprojection error and track length, and training PSNR (22.7 for `IMG_8651` at 7k steps).
