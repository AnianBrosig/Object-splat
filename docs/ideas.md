# Ideas for ObjectSplat

Two lists: real-world uses of the project, and extensions to the project itself. Both come from the README
(sections 0, 12 and 16) and the measured runs.

---

## Part 1: Real-world uses

The pipeline suits textured, matte, hand-sized objects turned in front of a fixed camera, on a cheap laptop and
without a turntable or a multi-camera rig.

### Sell, share and archive

- **Second-hand marketplace listings.** A seller films a 20-second spin of a jacket, a trainer, a camera or a toy.
  The buyer gets a 360-degree view that shows scratches and wear better than photos do.
- **Small-shop product pages.** Potters, jewellers and board-game makers get "spin it yourself" product viewers
  without a photo studio. Glossy or transparent objects need matte spray or markers, so matte ceramics and fabric
  items work best.
- **Museum and school collections.** Pupils or volunteers digitise fossils, coins, tools and artefacts with a phone
  and a laptop.
- **Heirloom archive.** Scan a teapot, a child's favourite toy or a trophy before it breaks or gets lost. The `.ply`
  stays on your own disk.

### Insurance, repair and evidence

- **Home inventory for insurance.** Scan valuables as 3D models. A checkerboard or marker of known size in the shot
  gives real units.
- **Damage documentation.** Compare a returned rental item, a dented part or a cracked casing before and after. Two
  scans in the object's own frame can be overlaid and diffed.
- **Repair help.** Scan a broken part or an appliance and send the model to a repair cafe or a 3D-printing
  volunteer, who can design a replacement without the object in front of them.

### Making and printing

- **Reverse-engineering spare parts.** Scan a knob, clip or bracket and use the splat as a reference for CAD. The
  splat is not a mesh, but it shows proportions and edges (the `IMG_8651` run gave a clean cuboid).
- **Custom-fit design.** Scan a tool handle or a phone and design a case or holder around it.
- **Miniature and prop makers.** Cosplayers and modellers capture a prop with its texture and printed graphics.
  Large print and graphics come out crisp.

### Games, AR and creative work

- **Objects into games and VR.** Scan a real mug, plant or shoe and use it as a prop in a game scene or VR room.
- **Stop-motion and animation reference.** A 360-degree reference of a puppet or prop for lighting and perspective.
- **Social "scan and share" objects.** Send a friend a small 3D souvenir from a trip.

### Live tracking (the `track` command)

The finished model locates, masks and orients the object from a webcam.

- **Tangible controller.** A scanned object (a wooden block, a figurine, a mug) becomes a 3D input device. Turning it
  in your hand rotates a model on screen, with no electronics inside.
- **Interactive board and toy games.** Track a physical piece and show its digital twin or its stats when you hold
  it up to the camera.
- **Teaching aid.** A student turns a physical molecule model, a geology sample or an anatomy model, and labels on
  screen follow it.
- **Accessible interfaces.** People who cannot use a mouse or keyboard could use a familiar object as a knob.
- **Part verification at a workbench.** The webcam checks that a part is the right one and is oriented correctly
  before assembly.
- **Hand-over detection.** Object and hand masks are already separated, so the system can tell when an item is
  picked up or put down (tool checkout, lab).

### Science and teaching

- **Citizen science.** Record insects, shells, minerals or seeds as 3D models and upload them to open datasets.
- **Photogrammetry teaching.** The UI shows every stage with debug images, so it shows how SfM works and fails.
- **Training data for object detection.** A scanned object gives many views, masks and poses automatically. One
  20-second video could generate labelled data for pose-estimation models.

### Limits to keep in mind

- Small or untextured objects (61-87 keypoints per frame) fail first. Stickers, markers or patterns help.
- Round or symmetric objects seen edge-on can flip in SfM (`IMG_8925`). Calibrate the focal length first.
- The hand is the main quality limit. Re-grip often, or try `train_mask_mode=ignore`.
- A run takes about an hour or more on the current laptop: fine for archiving or listings, too slow for anything
  instant.

Best mix of useful and achievable: the marketplace / product-page scanner (textured objects, no live speed needed)
and the tangible controller (it uses the finished tracker).

---

## Part 2: Extensions to the project

Ordered roughly by payoff.

### Fix what the runs say is weakest

1. **Ignore the hand instead of making it transparent.** Hands cover 12-39 % of the frame, and `alpha` mode cuts
   bites out of the object. Make `ignore_quality` the default, and add a Brush patch or fork that applies a
   per-pixel loss weight.
2. **Calibrate once per camera mode and warn when it is missing.** The free focal converged to about twice the right
   value in 3 of 7 1080p runs, causing two SfM gate failures and the `IMG_8925` pose flip. Add an automatic prior
   (phone EXIF or a calibration lookup by resolution and fps), a focal sanity check against the `1.2*max(w,h)`
   prior, and a hard stop when the focal jumps.
3. **Pose-jump detector before training.** Check the camera path for steps over about 30 degrees and fail early,
   which saves the 30-45 min Brush run. Then retry with a fixed focal or a different seed pair.
4. **Crop-based features.** Extraction at full 1080p costs about 8.7 s per frame. Cropping to the object bounding box
   (the mask is already there) with fewer keypoints could cut the 22-67 % feature share a lot.

### Speed

5. **Cache and resume more.** Add content-hash caching so a re-run with a changed calibration skips segmentation and
   features.
6. **Run SuperPoint and LightGlue on MPS selectively.** SuperPoint on the AMD GPU is fast, but the pycolmap import
   segfaults MPS. Run features in their own subprocess with MPS and keep SfM separate.
7. **Faster previews.** The turntable renders take 10-23 % of the wall time. Render at 360 px by default, full size
   on request, or show a WebGL splat viewer in the UI.
8. **Cloud or CUDA offload.** Add a "run on remote box" backend for features and training. The stage-per-folder
   design makes this a small step.

### Better reconstruction

9. **Dense or foundation-model initialisation.** Use RoMa, MASt3R or VGGT as a fallback when the keypoint count is
   low (small, untextured objects such as `IMG_8645` / `8646`). Needs CUDA or heavy CPU time.
10. **Click-to-correct segmentation.** Planned in the README. When the area plot jumps, the user clicks on one frame
    and Cutie re-propagates from there.
11. **Two-video fusion.** Record the object twice (once per side, or re-gripped) and register both into one model
    (see `docs/sfm-known-view-angles.md`). This also fills hand-covered holes.
12. **Meshing and export.** Add Poisson or TSDF meshing plus GLB / USDZ export for game engines, AR Quick Look and
    3D printing. The output today is only `.ply`.
13. **Scale from a marker.** Detect a printed board or ArUco marker in the video and write metric scale into the
    model. This enables measurements and 3D printing, and removes the manual object size for the tracker.
14. **Relighting or de-lighting.** Fixed room light is baked into the splat. Even a per-frame exposure model or a
    simple albedo estimate would help.

### Live tracking

15. **Webcam calibration and the live test.** The open items from the tracking status. Finishing them is the
    quickest route to a demo.
16. **Multi-object tracking.** Track several scanned objects in one scene from a model library.
17. **Pose output over OSC, WebSocket or MIDI.** Stream the pose to Unity, Blender, TouchDesigner or a browser. This
    makes the tangible controller real.
18. **Splat-based pose refinement.** Render the splat at the current pose and align it to the live frame
    (photometric). This should be more accurate than keypoint matching alone on smooth objects.

### Product and workflow

19. **Capture assistant.** While recording, show coverage ("top not seen", "hand covering this face too long") and
    exposure or blur warnings, so bad captures are caught before a 90-minute run.
20. **Automatic quality score and report.** One number plus reasons per job (keypoints, registration, focal sanity,
    hand coverage), shown in the batch report and the UI.
21. **Browser viewer and sharing.** Serve a SuperSplat-style viewer from the UI with a share link, to support the
    marketplace and archive uses.
22. **Dataset generator.** Export masks, poses and frames in COCO / BOP format for training pose-estimation or
    detection models, using the tracker's eval tooling.

### Where to start

- **Quality:** items 1, 2 and 3. They are small changes aimed at failures the runs actually show.
- **Demo:** items 15, 13, then 17. The tracker is done; only calibration and a live test are open.
