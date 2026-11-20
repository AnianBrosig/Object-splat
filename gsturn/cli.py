"""python -m gsturn <doctor|downscale|calibrate|run|batch|serve|export-model|track|camera|track-eval>"""
from __future__ import annotations

import argparse
import dataclasses
import json
import signal
import sys
from pathlib import Path

from .config import STAGES, Config


def _add_config_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("config (maps 1:1 to config.json fields)")
    for f in dataclasses.fields(Config):
        if f.name in ("video", "object_box", "hand_boxes", "bg_boxes"):
            continue
        g.add_argument("--" + f.name.replace("_", "-"), dest="cfg_" + f.name, default=None,
                       help=f"default {f.default!r}" if f.default is not dataclasses.MISSING else None)


def _config_updates(args: argparse.Namespace) -> dict:
    upd = {k[4:]: v for k, v in vars(args).items() if k.startswith("cfg_") and v is not None}
    if getattr(args, "box", None):
        upd["object_box"] = args.box
    if getattr(args, "hand_box", None):
        upd["hand_boxes"] = args.hand_box
    if getattr(args, "bg_box", None):
        upd["bg_boxes"] = args.bg_box
    if getattr(args, "train", None):
        upd["train_backend"] = args.train
    return upd


def _sigterm(*_):
    raise KeyboardInterrupt


def _track_cmd(args: argparse.Namespace) -> int:
    if args.cmd == "export-model":
        from .track.model import build
        build(Path(args.job), args.object_size, device=args.device)
        return 0
    if args.cmd == "camera":
        from .track.camera import calibrate_camera, preview
        if args.calibrate:
            from .calibrate import parse_pattern
            calibrate_camera(args.index, args.width, args.height, parse_pattern(args.pattern), args.views,
                             args.square_mm, Path(args.output) if args.output else None)
        else:
            from .track.camera import default_recording
            rec = None
            if args.record is not None:
                rec = default_recording(args.index) if args.record == "auto" else Path(args.record)
            preview(args.index, args.width, args.height, record=rec, seconds=args.seconds, fps=args.fps)
        return 0
    if args.cmd == "track":
        from .track.live import run as live
        from .track.tracker import TrackerConfig
        cfg = TrackerConfig(device=args.device, reloc_interval=args.reloc_interval, smooth=args.smooth)
        live(args.model, args.source, args.width, args.height, args.calib, args.hfov, not args.no_display,
             args.save_video, args.poses, args.seconds, realtime=not args.fast, cfg=cfg,
             record_raw=args.record_raw, learn=args.learn, learn_save=not args.no_save_learning)
        return 0
    if args.cmd == "refine-model":
        import cv2
        from .track.camera import calib_path, guess_K, intrinsics
        from .track.model import find_model
        from .track.refine import refine
        path = find_model(args.model)
        for v in args.video:
            cap = cv2.VideoCapture(v)
            W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            cp = Path(args.calib) if args.calib else calib_path(W, H, 0)
            calib_out = None
            if cp.exists():
                K, undist, _ = intrinsics(cp, W, H)
                if undist is not None:
                    print("note: the calibration has lens distortion; refine-model uses the pinhole part only")
                print(f"intrinsics from {cp} (fx {K[0, 0]:.0f})")
                focal = False
            else:
                K = guess_K(W, H, args.hfov)
                calib_out = cp            # no calibration yet: save the focal estimated from the object
                focal = not args.no_focal
                print(f"no calibration at {cp}; starting from a {args.hfov:.0f} deg field of view"
                      + ("; the focal length will be estimated from the object" if focal else ""))
            res = refine(path, Path(v), K, passes=args.passes, calibrate=focal, calib_out=calib_out,
                         max_keyframes=args.max_keyframes)
            print(json.dumps(res["passes"], indent=1))
        return 0
    if args.cmd == "track-eval":
        from .track.evaluate import held_out, sequence
        from .track.model import ObjectModel, find_model
        job = Path(args.job)
        model = ObjectModel.load(find_model(str(job)))
        print("held-out keyframes (map without the frame and its neighbours):")
        ho = held_out(model, job, every=args.every)
        print(json.dumps(ho["summary"], indent=1))
        video = Path(args.video) if args.video else job / "work" / "downscale" / "video_small.mp4"
        print(f"sequence test on {video} (paced at the video frame rate):")
        sq = sequence(model, video, out_video=Path(args.save_video) if args.save_video else None)
        print(json.dumps(sq["summary"], indent=1))
        out = job / "outputs" / "track_eval.json"
        out.write_text(json.dumps({"held_out": ho, "sequence": sq}, indent=1, default=float))
        print(f"wrote {out}")
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gsturn", description="hand-rotated object video -> 3D Gaussian splat")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="report toolchain")
    d.add_argument("--json", action="store_true")
    d.add_argument("--fetch-weights", action="store_true", help="download model weights now")

    ds = sub.add_parser("downscale", help="downscale a video")
    ds.add_argument("video")
    ds.add_argument("--height", type=int, default=720)
    ds.add_argument("--fps", type=float, default=0.0)
    ds.add_argument("--hwaccel", default="auto")
    ds.add_argument("-o", "--output", default="")

    c = sub.add_parser("calibrate", help="camera intrinsics from a checkerboard video")
    c.add_argument("video")
    c.add_argument("--square-mm", type=float, required=True, help="side length of one board square")
    c.add_argument("--pattern", default="", help="inner corners, e.g. 6x6 (default: auto-detect)")
    c.add_argument("--height", type=int, default=0, help="working height (0 = original), must match the jobs' target_height")
    c.add_argument("--step", type=int, default=3, help="use every n-th frame")
    c.add_argument("--max-views", type=int, default=40)
    c.add_argument("--hwaccel", default="auto")
    c.add_argument("--distortion", choices=["auto", "zero", "fit"], default="auto",
                   help="auto: keep k1/k2 only if they clearly help; zero: pinhole; fit: always keep k1/k2")
    c.add_argument("--check-job", default="", help="job folder whose SfM focal the result must match (5 %%)")
    c.add_argument("-o", "--output", default="", help="JSON path (default calibration/<camera>_<res>p<fps>.json)")

    r = sub.add_parser("run", help="run the pipeline (new job from a video, or an existing --job)")
    r.add_argument("video", nargs="?", default="")
    r.add_argument("--job", default="", help="existing job folder")
    r.add_argument("--from-stage", choices=STAGES)
    r.add_argument("--to-stage", choices=STAGES)
    r.add_argument("--box", help="object box x0,y0,x1,y1 (relative, first selected frame)")
    r.add_argument("--hand-box", action="append", help="hand box (repeatable)")
    r.add_argument("--bg-box", action="append",
                   help="background box (repeatable): region on frame 0 that is certainly not the object, e.g. "
                        "table inside the object box of a thin diagonal object")
    r.add_argument("--train", help="shortcut for --train-backend (default brush; none = stop after SfM)")
    r.add_argument("--quiet", action="store_true")
    _add_config_flags(r)

    b = sub.add_parser("batch", help="bulk run: one job per video, one after another",
                       description="Each video becomes its own job. Boxes: --boxes-file (per video), else --box/"
                                   "--hand-box/--bg-box for all videos. A video without an object box stops after "
                                   "frame selection (draw its box later in the web UI).")
    b.add_argument("videos", nargs="+", help="video files and/or folders (a folder = all videos in it)")
    b.add_argument("--boxes-file", default="", help="JSON {video name or '*': {object_box, hand_boxes, bg_boxes}}")
    b.add_argument("--box", help="object box x0,y0,x1,y1 for every video without an entry in --boxes-file")
    b.add_argument("--hand-box", action="append", help="hand box (repeatable)")
    b.add_argument("--bg-box", action="append", help="background box (repeatable)")
    b.add_argument("--to-stage", choices=STAGES, default="export")
    b.add_argument("--train", help="shortcut for --train-backend")
    b.add_argument("--stop-on-error", action="store_true", help="stop at the first failed video (default: go on)")
    b.add_argument("--skip-done", action="store_true", help="skip videos that already have a job finished up to --to-stage")
    b.add_argument("--dry-run", action="store_true", help="list what would run, create nothing")
    _add_config_flags(b)

    s = sub.add_parser("serve", help="web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    # ---- live pose tracking (docs/realtime-pose-tracking.md)
    em = sub.add_parser("export-model", help="object package for live tracking (outputs/object_model.npz)")
    em.add_argument("--job", required=True, help="finished job folder (needs sfm)")
    em.add_argument("--object-size", type=float, default=0.0,
                    help="object length along its longest axis in metres (0 = keep SfM units)")
    em.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")

    xm = sub.add_parser("export-mesh", help="splat -> mesh (STL / OBJ / PLY with colours) + convex parts for physics")
    xm.add_argument("--job", required=True, help="job folder with a trained splat")
    xm.add_argument("--size-cm", type=float, required=True, help="real length of the object's longest side")
    xm.add_argument("--name", default="", help="file name stem (default: the video name)")
    xm.add_argument("-o", "--output", default="", help="folder (default: <job>/outputs/mesh)")
    xm.add_argument("--res", type=int, default=112, help="voxels along the longest side")
    xm.add_argument("--density", type=float, default=350.0, help="kg per m3 of solid, for the mass")
    xm.add_argument("--solid", choices=["fill", "shell", "hull"], default="fill")

    dl = sub.add_parser("delight", help="remove the baked-in room light from a splat (neutral light) + demo renders")
    dl.add_argument("--job", required=True)
    dl.add_argument("--demo", action="store_true", help="also render before/after and re-lit videos")
    dl.add_argument("--frames", type=int, default=48, help="demo frames per video")
    dl.add_argument("--size", type=int, default=480, help="demo frame size")

    cl = sub.add_parser("clean", help="delete intermediate images/videos under jobs/*/work (outputs/ is kept)")
    cl.add_argument("jobs", nargs="*", help="job ids (default: all jobs)")
    cl.add_argument("--dry-run", action="store_true", help="only report what would be deleted")

    sh = sub.add_parser("shelf",help="physics playground: objects on a shelf you can tilt (physics/)")
    sh.add_argument("--build", action="store_true", help="export the meshes listed in physics/objects.json first")
    sh.add_argument("--rebuild", action="store_true", help="like --build, even if the meshes are up to date")
    sh.add_argument("--only", action="append", help="object name (repeatable): only build this one")
    sh.add_argument("--port", type=int, default=8765)
    sh.add_argument("--no-open", action="store_true")
    sh.add_argument("--no-serve", action="store_true", help="only build")

    tk = sub.add_parser("track", help="live object pose from the webcam or a video")
    tk.add_argument("--model", required=True, help="job folder or object_model.npz")
    tk.add_argument("--source", default="0", help="camera index (0 = built-in webcam) or a video file")
    tk.add_argument("--width", type=int, default=1280)
    tk.add_argument("--height", type=int, default=720)
    tk.add_argument("--calib", default="", help="calibration JSON (default calibration/webcam<i>_<w>x<h>.json)")
    tk.add_argument("--hfov", type=float, default=0.0, help="horizontal field of view guess if not calibrated")
    tk.add_argument("--save-video", default="", help="record the overlay video here (.mp4)")
    tk.add_argument("--record-raw", default="", help="record the plain camera video here (.mp4)")
    tk.add_argument("--poses", default="", help="write one JSON line per frame (pose, status) here")
    tk.add_argument("--no-display", action="store_true")
    tk.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until q)")
    tk.add_argument("--fast", action="store_true", help="video files: process as fast as possible")
    tk.add_argument("--device", default="auto")
    tk.add_argument("--reloc-interval", type=float, default=0.3, help="seconds between relocalisations")
    tk.add_argument("--smooth", type=float, default=0.25, help="pose filter alpha at rest (1 = no smoothing)")
    tk.add_argument("--learn", action="store_true",
                    help="adapt the map to this camera while tracking; saved to the package on exit")
    tk.add_argument("--no-save-learning", action="store_true", help="with --learn: keep the changes in memory only")

    cm = sub.add_parser("camera", help="webcam preview / recording (r record, s snapshot) or chessboard calibration")
    cm.add_argument("--index", type=int, default=0)
    cm.add_argument("--width", type=int, default=1280)
    cm.add_argument("--height", type=int, default=720)
    cm.add_argument("--calibrate", action="store_true")
    cm.add_argument("--square-mm", type=float, default=22.0, help="chessboard square size (calibration)")
    cm.add_argument("--pattern", default="", help="inner corners, e.g. 6x6 (default: auto-detect)")
    cm.add_argument("--views", type=int, default=25)
    cm.add_argument("--record", nargs="?", const="auto", default=None, metavar="PATH",
                    help="start recording right away (default path videos/webcam<i>_<time>.mp4)")
    cm.add_argument("--seconds", type=float, default=0.0, help="stop recording after this long (0 = on q / r)")
    cm.add_argument("--fps", type=float, default=30.0, help="frame rate of the recorded video")
    cm.add_argument("-o", "--output", default="", help="calibration JSON path")

    rm = sub.add_parser("refine-model", help="teach the object package how the object looks to the webcam")
    rm.add_argument("--model", required=True, help="job folder or object_model.npz")
    rm.add_argument("--video", required=True, action="append", help="webcam recording of the object (repeatable)")
    rm.add_argument("--passes", type=int, default=2)
    rm.add_argument("--calib", default="", help="camera calibration JSON (default calibration/webcam0_<w>x<h>.json)")
    rm.add_argument("--hfov", type=float, default=60.0, help="field of view guess when there is no calibration")
    rm.add_argument("--no-focal", action="store_true", help="do not estimate the focal length from the object")
    rm.add_argument("--max-keyframes", type=int, default=80, help="webcam keyframes added per pass")

    te = sub.add_parser("track-eval", help="tracking accuracy against the SfM poses of the job")
    te.add_argument("--job", required=True)
    te.add_argument("--every", type=int, default=10, help="held-out test: every n-th keyframe")
    te.add_argument("--video", default="", help="sequence test video (default work/downscale/video_small.mp4)")
    te.add_argument("--save-video", default="", help="overlay video of the sequence test")

    args = ap.parse_args(argv)

    if args.cmd in ("export-model", "track", "camera", "track-eval", "refine-model"):
        return _track_cmd(args)

    if args.cmd == "export-mesh":
        from .mesh import export_all
        job = Path(args.job)
        ply = next((job / r for r in ("outputs/splat.ply", "work/train/splat.ply") if (job / r).exists()), None)
        if ply is None:
            print(f"error: no splat.ply in {job}", file=sys.stderr)
            return 1
        name = args.name or (Path(json.loads((job / "config.json").read_text()).get("video", "object")).stem)
        out = Path(args.output) if args.output else job / "outputs" / "mesh"
        info = export_all(ply, out, name, args.size_cm / 100.0, args.density, args.res, solid=args.solid)
        print(json.dumps({k: info[k] for k in ("size_m", "volume_cm3", "mass_kg", "solidity", "convex_parts", "faces_written")}))
        print(f"wrote {', '.join(info['files'])} to {out}")
        return 0

    if args.cmd == "delight":
        from .delight import delight, demo
        from .pipeline import Job
        job = Job(Path(args.job).resolve())
        src = next((job.dir / r for r in ("outputs/splat.ply", "work/train/splat.ply") if (job.dir / r).exists()), None)
        if src is None:
            print(f"error: no splat.ply in {job.dir}", file=sys.stderr)
            return 1
        out = job.outputs / "splat_delit.ply"
        info = delight(job, src, out)
        if args.demo:
            files = demo(job, src, out, job.outputs / "delight", info, args.frames, args.size)
            print("wrote " + ", ".join(str(f.relative_to(job.dir)) for f in files))
        print(f"wrote {out}")
        return 0

    if args.cmd == "clean":
        from .cleanup import clean_jobs
        clean_jobs(only=[Path(j).name for j in args.jobs] or None, dry_run=args.dry_run)
        return 0

    if args.cmd == "shelf":
        from . import shelf
        if args.build or args.rebuild or not (shelf.PHYSICS / "objects" / "manifest.json").exists():
            shelf.build(force=args.rebuild, only=args.only)
        if not args.no_serve:
            shelf.serve(args.port, not args.no_open)
        return 0

    if args.cmd == "doctor":
        from .doctor import print_report, report
        rep = report(fetch_weights=args.fetch_weights)
        if args.json:
            print(json.dumps(rep, indent=2))
        else:
            print_report(rep)
        return 0

    if args.cmd == "downscale":
        from .events import Events
        from .stages.downscale import downscale
        src = Path(args.video)
        dst = Path(args.output) if args.output else src.with_name(f"{src.stem}_{args.height}p.mp4")
        res = downscale(src, dst, args.height, args.fps, args.hwaccel, ev=Events(None))
        o = res["output"]
        print(f"wrote {dst}  {o['width']}x{o['height']}  {o['frames']} frames  {o['size'] / 1e6:.1f} MB "
              f"({res['encoder']})")
        return 0

    if args.cmd == "calibrate":
        from .calibrate import CalibrationError, calibrate, parse_pattern
        from .events import Events
        try:
            res = calibrate(Path(args.video), args.square_mm, parse_pattern(args.pattern), args.height,
                            Path(args.output) if args.output else None,
                            Path(args.check_job) if args.check_job else None, step=args.step,
                            max_views=args.max_views, hwaccel=args.hwaccel, distortion=args.distortion,
                            ev=Events(None))
        except CalibrationError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"wrote {res['output']}\n  {res['width']}x{res['height']}  f={res['fx']:.1f}px  "
              f"c=({res['cx']:.1f}, {res['cy']:.1f})  RMS {res['rms_px']:.2f}px  {res['views']} views  "
              f"dist={res['dist']} (fitted {res['dist_fitted'][:2]})")
        return 0

    if args.cmd == "run":
        from .pipeline import Job, new_job, run
        signal.signal(signal.SIGTERM, _sigterm)
        upd = _config_updates(args)
        if args.job:
            job = Job(Path(args.job).resolve())
            cfg = job.load_config()
            if args.video:
                upd["video"] = str(Path(args.video).resolve())
        else:
            if not args.video:
                ap.error("give a video or --job")
            cfg = Config()
            job = new_job(args.video, cfg)
            cfg = job.load_config()
        cfg.update(upd)
        cfg.save(job.config_path)
        print(f"job {job.dir}", file=sys.stderr)
        ok = run(job, args.from_stage, args.to_stage, echo=not args.quiet)
        return 0 if ok else 1

    if args.cmd == "batch":
        from . import batch
        from .config import parse_box
        signal.signal(signal.SIGTERM, _sigterm)
        try:
            videos = batch.find_videos(args.videos)
            boxes = batch.load_boxes(args.boxes_file)
            upd = _config_updates(args)
            default = {k: upd.pop(k) for k in batch.BOX_KEYS if k in upd}
            if "object_box" in default:
                parse_box(default["object_box"])
        except (OSError, ValueError, KeyError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if not videos:
            print("error: no videos found", file=sys.stderr)
            return 2
        if args.skip_done:
            keep = []
            for v in videos:
                done = batch.finished_job(v, args.to_stage)
                print(f"skip {v.name}: job {done} already finished" if done else f"queue {v.name}")
                if not done:
                    keep.append(v)
            videos = keep
        if args.dry_run:
            for v in videos:
                has = bool(batch.boxes_for(v, boxes, default).get("object_box"))
                print(f"{v.name}: {'full run to ' + args.to_stage if has else 'no object box -> stops after frames'}")
            return 0
        try:
            items = batch.create_jobs(videos, upd, boxes, default, args.to_stage)
        except (ValueError, KeyError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        try:
            batch.run_batch(items, keep_going=not args.stop_on_error)
        except KeyboardInterrupt:
            print("interrupted", file=sys.stderr)
        report = batch.write_report(items)
        print("\n" + "\n".join(f"{i.state:9s} {Path(i.video).name:28s} {i.job}  {i.msg}" for i in items))
        print(f"report {report}")
        return 0 if all(i.state in ("done", "skipped") for i in items) else 1

    if args.cmd == "serve":
        import uvicorn
        from .server import app
        print(f"http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return 0
    return 1
