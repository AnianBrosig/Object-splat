# Going back to the classic pipeline

The pipeline as it was before 2026-11-17 (full-frame SuperPoint on the CPU, 2048 keypoints, window 8, refine, alpha
masks, full-frame training) was slower, and for your footage it gave the better splats. Nothing was deleted: every
change of 2026-11-17 has an off switch, so the classic pipeline is a **configuration**, not a different code base.

The git repository has a single commit ("initial", 2026-11-20) and it already contains the new code, so the old
files cannot be checked out. The switches below are the way back, and they were checked against the old behaviour
(section 3).

---

## 1. The classic settings

Copy-paste JSON (the same keys as `jobs/<id>/config.json`):

```json
{
  "device": "cpu",
  "match_device": "cpu",
  "feature_crop": false,
  "feature_max_side": 0,
  "kp_hand_margin_px": 0,
  "max_keypoints": 2048,
  "match_window": 8,
  "exhaustive_below": 100,
  "skin_filter": false,
  "refine_enabled": true,
  "refine_max_angle": 30.0,
  "continuity_check": false,
  "sfm_focal_retry": false,
  "train_mask_mode": "alpha",
  "train_crop": false,
  "train_residual_pass": false,
  "train_steps": 7000,
  "render_enabled": true,
  "scheduler": "off"
}
```

`train_backend` stays `brush` (the old default was `none`, you asked for Brush). Everything not listed is unchanged
since the old pipeline.

| Setting | New default | Classic | What it does | Re-run from |
|---|---|---|---|---|
| `device` | `auto` (GPU for keypoints, CPU for Cutie) | `cpu` | **`auto` used to mean CPU**; the string did not change but the meaning did, so a copied old config still says `auto` | `features` |
| `match_device` | `auto` (= CPU) | `cpu` | LightGlue device | `features` |
| `feature_crop`, `feature_max_side` | `true`, 1024 | `false`, 0 | SuperPoint on a window around the object, resized, instead of the whole frame at native size | `features` |
| `kp_hand_margin_px` | 12 | 0 | drop keypoints within 12 px of the hand | `features` |
| `max_keypoints` | 1024 | 2048 | keypoints per frame | `features` |
| `match_window` | 4 | 8 | neighbouring frames matched with each frame | `features` |
| `exhaustive_below` | 40 | 100 | all-pairs matching up to this many frames | `features` |
| `skin_filter` | `true` | `false` | move skin-coloured object pixels next to the hand into the hand mask | `segment` |
| `refine_enabled`, `refine_max_angle` | `false`, 60 | `true`, 30 | pose-guided loop-closure re-matching | `refine` |
| `continuity_check`, `sfm_focal_retry` | `true`, `true` | `false`, `false` | orientation-jump test and fixed-focal re-mapping | `sfm` |
| `train_mask_mode` | `weighted` | `alpha` | hand and background forced transparent through the RGBA alpha | `train` |
| `train_crop` | `true` | `false` | train on per-frame crops | `train` |
| `train_steps` | 7000 | 7000 | the A/B/C runs of 2026-11-17 used 3000-5000 to save time | `train` |
| `render_enabled` | `false` | `true` | the three turntable videos | `render` |
| `scheduler` | `resource` | `off` | jobs take turns per resource; only affects concurrency, not results | any |

`--from-stage X` re-runs stage X and every stage after it. The first changed stage in the column decides how much
has to be redone; changing only training switches (`train_*`) reuses the SfM model.

---

## 2. How to apply it

**New job from the command line** (every config field is a flag, booleans as `true`/`false`):

```bash
python -m gsturn run videos/IMG_8924.MOV --box 0.15,0.07,0.70,1.0 --hand-box 0.70,0.11,1.0,1.0 \
  --device cpu --match-device cpu --feature-crop false --feature-max-side 0 --kp-hand-margin-px 0 \
  --max-keypoints 2048 --match-window 8 --exhaustive-below 100 --skin-filter false \
  --refine-enabled true --refine-max-angle 30 --continuity-check false --sfm-focal-retry false \
  --train-mask-mode alpha --train-crop false --render-enabled true --scheduler off
```

**Existing job** (change its `config.json`, then re-run from the first stage that matters):

```bash
python - <<'EOF'
import json
from pathlib import Path
from gsturn.config import Config
CLASSIC = {"device": "cpu", "match_device": "cpu", "feature_crop": False, "feature_max_side": 0,
           "kp_hand_margin_px": 0, "max_keypoints": 2048, "match_window": 8, "exhaustive_below": 100,
           "skin_filter": False, "refine_enabled": True, "refine_max_angle": 30.0, "continuity_check": False,
           "sfm_focal_retry": False, "train_mask_mode": "alpha", "train_crop": False,
           "train_residual_pass": False, "train_steps": 7000, "render_enabled": True, "scheduler": "off"}
job = Path("jobs/<job id>")
cfg = Config.load(job / "config.json")
cfg.update(CLASSIC, strict=True)        # strict: a typo in a key raises an error
cfg.save(job / "config.json")
EOF
python -m gsturn run --job jobs/<job id> --from-stage downscale
# for a job that still has its frames and masks, the stage from the table in section 1 is enough (features, sfm, train, ...)
```

**Same video again, in classic mode, with the boxes of an earlier run** (here the A close-up of 2026-11-17):

```bash
python - <<'EOF'
import json
from gsturn.config import Config
from gsturn.pipeline import new_job, run
CLASSIC = {...}                         # the dict from above
src = json.load(open("jobs/20261117-023513-IMG_8934-f46a/config.json"))
keep = ("object_box", "hand_boxes", "bg_boxes", "trim_start", "trim_end", "max_frames")
cfg = Config().update({k: src[k] for k in keep}, strict=True).update(CLASSIC, strict=True)
run(new_job(src["video"], cfg))
EOF
```

**Web UI**: open "All config fields" under the parameters and set the same values (`train_mask_mode` has `alpha` in
its list; the rest are plain fields).

**Two traps**
- A `config.json` written before 2026-11-17 does not contain the new fields, so loading it gives them their **new**
  defaults (crops on, GPU keypoints, ...). Apply the classic dict even when re-running an old job unchanged.
- Every job currently in `jobs/` (old ones and the 2026-11-17 ones) keeps its SfM model, features, splat and stats, but
  its frame images, masks, quality maps and SfM dataset images are gone, so none of them can be re-run from the middle;
  re-run from `downscale` (the source videos are still in `videos/`). Jobs you start from now on keep their
  intermediate data until it is cleaned up.

---

## 3. What was verified, and what was not

Checked on 2026-11-20:
- **Keypoints**: with `feature_crop=false`, `feature_max_side=0` the new extractor applies an identity transform and
  returns *bit-identical* keypoints and descriptors to the old code path (three frames of `IMG_8902`, 1920x1080,
  2048 keypoints, CPU).
- **Training**: with `train_crop=false` and a non-weighted mask mode the train stage skips the new dataset builder and
  trains on the SfM dataset exactly as before (RGBA PNG alpha, optional `masks/` folder for `ignore`).
- **Config**: every key above exists in `Config`; for the settings that already existed in the old pipeline the values
  equal what the old run `IMG_8900` stored in its `config.json` (it also used Brush); the command-line form parses to
  the same config as the dict.
- **Pairs, matching, SfM, refine**: single-camera code paths are unchanged apart from the switches above.

Not identical, by design: `features` also writes `affine.npz`, `frames.json` carries `cam`/`t` fields, `sfm` stats
may contain a `continuity` block. None of these change a result. COLMAP and Brush are not bit-reproducible, so a
re-run differs slightly from the original in any case.

---

## 4. Faster, same quality

Two things from the new code are free: they do not change what the pipeline computes.

- **Keypoints on the GPU.** On native-size frames the GPU returns the same keypoints as the CPU (100 % within 0.5 px,
  descriptor difference about 3e-6) in 0.2 s instead of 3-4 s per frame. Use `device: auto` instead of `cpu` in the
  classic dict. Do **not** use `device: mps`: an explicit device is also applied to Cutie, which is about 2x slower
  there (same masks). Extraction was 1734 s of the 4610 s of the old `IMG_8924` features stage; matching (2850 s at
  2048 keypoints, CPU) stays.
- **`render_enabled: false`**, **`scheduler: resource`**: no effect on results. The web viewer replaces the turntable
  videos.

Measured on `IMG_8924`, old run against new run (both while other jobs were using the CPU): features 4610 s vs 639 s,
training 3864 s vs 1942 s, whole job 3.6 h vs 55 min. Classic with GPU keypoints should be roughly 50 min for features
instead of 77; that number is an estimate.

---

## 5. If classic is better: finding out which switch costs quality

The classic setting is the safe answer. If you want some of the speed back, change **one switch at a time** and compare
on the same video. Cheapest first, because training-only switches reuse the SfM model:

| # | Switch to test (everything else classic) | Re-run from | Rough cost |
|---|---|---|---|
| 1 | `train_steps` 7000 vs 3000-5000 | `train` | the A/B/C runs used 3000-5000, so they are not a like-for-like test |
| 2 | `train_mask_mode=weighted` with `train_crop=false` | `train` | training only |
| 3 | `train_crop=true` with `alpha` | `train` | training only |
| 4 | `skin_filter=true` | `segment` | masks change; it can shave skin-coloured objects (beige cardboard) |
| 5 | `refine_enabled=true`, `refine_max_angle` 60 | `refine` | refine only |
| 6 | `max_keypoints` 1024, `match_window` 4 | `features` | features + SfM + training |
| 7 | `feature_crop=true`, `feature_max_side` 1024 | `features` | features + SfM + training |

Judge by the held-out views (`work/train/eval_comparison.jpg`) and the viewer, **not by PSNR**: the weighted/cropped
training measures PSNR on crops with less black background, so its numbers are lower than a full-frame classic run
of the same quality (`IMG_8924`: 22.1 new against 27.5 classic). The only like-for-like visual check done so far was
that video: the hand-covered regions were cleaner with weighted masks (no bite out of the lid), and the other areas
looked comparable. That is one video, so it does not contradict your experience with the others.

---

## 6. Where the classic results are

Splats of the old pipeline that are still on disk (`outputs/splat.ply`, open them in the web UI's splat viewer):

| Job | Video | Splats | PSNR (full frame) | Registered |
|---|---|---|---|---|
| `20261024-164201-IMG_8651-0d63` | `IMG_8651` | 33380 | 22.7 | 200 |
| `20261024-193331-IMG_8646-e301` | `IMG_8646` | 2086 | 29.8 | 149 |
| `20261113-215535-IMG_8900-5851` | `IMG_8900` | 22714 | 29.8 | 188 |
| `20261113-215535-IMG_8901-5c68` | `IMG_8901` | 38682 | 25.3 | 199 |
| `20261113-215535-IMG_8902-25f5` | `IMG_8902` | 12441 | 33.1 | 103 |

(`IMG_8924` and `IMG_8925` of 2026-11-16 are in `work/train/splat.ply` of their jobs.) PSNR is not comparable between
videos; compare the same video before and after.

The stage-by-stage description of the classic algorithm is README sections 3-6; sections 17-23 describe what was added.
