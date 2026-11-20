"""Stage 6: train a Gaussian splat (Brush by default) from the COLMAP dataset."""
from __future__ import annotations

import os
import pty
import re
import select
import shlex
import shutil
import signal
import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..events import Events
from ..pipeline import StageError
from ..util import find_tool

DEFAULT_CMDS = {
    # {bin} {data} {out} {steps} are substituted; the rest is passed through.
    "brush": "{bin} {data} --total-steps {steps} --export-every {steps} --export-path {out} "
             "--export-name splat.ply --eval-every 1000 --eval-split-every 10 --eval-save-to-disk",
    "opensplat": "{bin} {data} -n {steps} -o {out}/splat.ply",
}
BINARIES = {"brush": ["brush", "brush_app"], "opensplat": ["opensplat"]}
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def run_pty(cmd: list[str], on_line, cwd: Path | None = None, env: dict | None = None) -> int:
    """Run under a pseudo-terminal so tools that only draw progress bars on a TTY still report progress."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave, cwd=cwd, env=env,
                            close_fds=True)
    os.close(slave)
    buf = b""
    try:
        while True:
            r, _, _ = select.select([master], [], [], 0.5)
            if master in r:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                buf += chunk
                parts = re.split(rb"[\r\n]", buf)
                buf = parts.pop()
                for p in parts:
                    s = ANSI.sub("", p.decode("utf-8", "replace")).strip()
                    if s:
                        on_line(s)
            elif proc.poll() is not None:
                break
        return proc.wait()
    except BaseException:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    finally:
        os.close(master)


def prepare_masks(dataset: Path, mode: str, quality_dir: Path | None = None) -> int:
    """mode 'ignore': masks/ next to images/ -> Brush ignores non-object pixels instead of forcing transparency.
    mode 'ignore_quality': like 'ignore', and untrusted pixels (quality map Q < 64) are ignored too.
    Returns the number of masks that used a quality map."""
    from .quality import TRUST_TRAIN
    md = dataset / "masks"
    if md.exists():
        shutil.rmtree(md)
    if mode not in ("ignore", "ignore_quality"):
        return 0
    md.mkdir()
    used = 0
    for p in sorted((dataset / "images").glob("*.png")):
        rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        m = rgba[..., 3]
        if mode == "ignore_quality" and quality_dir is not None:
            q = cv2.imread(str(quality_dir / p.name), cv2.IMREAD_GRAYSCALE)
            if q is not None and q.shape == m.shape:
                m = np.where((m > 0) & (q >= TRUST_TRAIN), 255, 0).astype(np.uint8)
                used += 1
        cv2.imwrite(str(md / p.name), m)
    return used


def train_once(binary: str, backend: str, template: str, dataset: Path, out: Path, steps: int, ev: Events,
               label: str = "", extra_args: list[str] | None = None) -> dict:
    """One trainer run. Returns {"plys", "evals", "splats", "tail", "rc"}."""
    for old in out.glob("*.ply"):
        old.unlink()
    cmd = [a.format(bin=binary, data=str(dataset), out=str(out), steps=steps) for a in shlex.split(template)]
    cmd += extra_args or []
    ev.log(" ".join(shlex.quote(c) for c in cmd))
    tail: list[str] = []
    evals: list[dict] = []
    splats = [0]
    warned: set[str] = set()
    prefix = f"{label}: " if label else ""

    def on_line(line: str) -> None:
        tail.append(line)
        del tail[:-60]
        m = re.search(r"(\d+)\s*/\s*(\d+)", line)
        if m and int(m.group(2)) == steps:
            ev.progress(int(m.group(1)), steps, f"{prefix}step {m.group(1)}/{steps}"
                        + (f", {splats[0]} splats" if splats[0] else ""))
        m = re.search(r"Eval iter (\d+): PSNR ([\d.]+), ssim ([\d.]+)", line)
        if m and not any(e["iter"] == int(m.group(1)) for e in evals):  # the TTY redraw repeats lines
            evals.append({"iter": int(m.group(1)), "psnr": float(m.group(2)), "ssim": float(m.group(3))})
            ev.log(f"{prefix}eval @ {m.group(1)}: PSNR {m.group(2)}, SSIM {m.group(3)}")
        m = re.search(r"(?:splat count|iter \d+,)\s*(\d+)\s*splats?", line) or re.search(r"splat count (\d+)", line)
        if m:
            splats[0] = int(m.group(1))
        if (re.search(r"\b(ERROR|WARN)\b", line) or "\u26a0" in line or "\u274c" in line) and line not in warned:
            warned.add(line)
            ev.log(line[:500], "warn")

    env = dict(os.environ, RUST_LOG=os.environ.get("RUST_LOG", "info"))
    rc = run_pty(cmd, on_line, cwd=out, env=env)
    plys = sorted(out.glob("*.ply"), key=lambda p: p.stat().st_mtime)
    return {"rc": rc, "plys": plys, "evals": evals, "splats": splats[0], "tail": tail}


def run(job, cfg: Config, ev: Events) -> dict:
    backend = cfg.train_backend
    out = job.work("train")
    if backend == "none":
        ev.log("train_backend=none: skipping training (inspect the sparse model instead)")
        return {"_msg": "skipped (train_backend=none)", "skipped": True}
    sfm_dataset = job.dir / "work" / "sfm" / "dataset"
    if not (sfm_dataset / "sparse" / "0").exists():
        raise StageError("no SfM model: run stage 'sfm' first")
    binary = next((b for b in (find_tool(n) for n in BINARIES.get(backend, [backend])) if b), None)
    if not binary:
        raise StageError(f"{backend} binary not found (see README section 5)")
    template = cfg.train_cmd or DEFAULT_CMDS.get(backend)
    if not template:
        raise StageError(f"no command template for backend {backend}; set train_cmd")
    extra = []
    if backend == "brush" and cfg.train_max_resolution > 0 and "--max-resolution" not in template:
        extra = ["--max-resolution", str(cfg.train_max_resolution)]

    # the training set: per-frame crops and/or the weighted masks need a dataset of their own; the plain legacy
    # modes (alpha / ignore without crops) train on the SfM dataset as it is
    own = cfg.train_crop or cfg.train_mask_mode == "weighted"
    bstats: dict = {}
    if own:
        from . import traindata
        dataset = out / "dataset"
        ev.log(f"building the training set: mask mode {cfg.train_mask_mode}, crops {'on' if cfg.train_crop else 'off'}")
        bstats = traindata.build(job, cfg, ev, dataset)
        ev.log(f"training set: {bstats['images']} images, {100 * bstats['crop_pixels_share']:.0f} % of the full-frame "
               f"pixels" + (f", mean loss weight {bstats['mean_loss_weight']}" if bstats["mean_loss_weight"] is not None else ""))
    else:
        dataset = sfm_dataset
        qdir = None
        if cfg.train_mask_mode == "ignore_quality":
            from .quality import load_quality
            if load_quality(job, cfg) is None:
                ev.warn("train_mask_mode=ignore_quality but no (current) quality maps: using plain 'ignore'")
            else:
                qdir = job.dir / "work" / "quality" / "maps"
        n_q = prepare_masks(dataset, cfg.train_mask_mode, qdir)
        if qdir is not None:
            ev.log(f"ignore_quality: {n_q} masks combined with trust maps")

    residual = {}
    if cfg.train_residual_pass and backend == "brush" and own and cfg.train_mask_mode != "alpha":
        p1 = out / "pass1"
        p1.mkdir(exist_ok=True)
        ev.log(f"residual pass: {cfg.train_residual_steps} steps first, then the pixels that disagree are ignored")
        r1 = train_once(binary, backend, template, dataset, p1, cfg.train_residual_steps, ev, "pass 1", extra)
        if r1["rc"] != 0 or not r1["plys"]:
            raise StageError(f"{backend} failed in the residual pass (exit {r1['rc']}):\n" + "\n".join(r1["tail"][-15:]))
        from . import traindata
        extra_ignore = traindata.residual_ignore(job, cfg, dataset, r1["plys"][-1], bstats["crops"], ev)
        bstats = traindata.build(job, cfg, ev, dataset, extra_ignore)
        residual = {"flagged_share": round(float(np.mean([m.mean() for m in extra_ignore.values()])), 4)
                    if extra_ignore else 0.0}

    res = train_once(binary, backend, template, dataset, out, cfg.train_steps, ev,
                     "pass 2" if residual else "", extra)
    (out / "trainer_log.txt").write_text("\n".join(res["tail"]))
    if res["rc"] != 0 or not res["plys"]:
        raise StageError(f"{backend} failed (exit {res['rc']}):\n" + "\n".join(res["tail"][-15:]))
    ply = res["plys"][-1]
    if ply.name != "splat.ply":
        shutil.copy(ply, out / "splat.ply")
    total, evals = cfg.train_steps, res["evals"]
    ev.progress(total, total, "done", force=True)
    sheet = eval_sheet(out, dataset)
    if sheet:
        ev.artifact(sheet, "held-out views: ground truth (left) vs. splat render (right), final step")
    stats = {"backend": backend, "steps": total, "ply": "work/train/splat.ply",
             "ply_mb": round((out / "splat.ply").stat().st_size / 1e6, 1), "evals": evals,
             "splats": res["splats"] or ply_count(out / "splat.ply"), "mask_mode": cfg.train_mask_mode}
    if bstats:
        stats["dataset"] = {k: v for k, v in bstats.items() if k != "crops"}
    if residual:
        stats["residual"] = residual
    if evals:
        stats["final_psnr"] = evals[-1]["psnr"]
    stats["_msg"] = f"{stats['splats']} splats, {stats['ply_mb']} MB" + (
        f", PSNR {evals[-1]['psnr']:.2f}" if evals else "")
    return stats


def eval_sheet(out: Path, dataset: Path, n: int = 6):
    """Side-by-side crops of held-out ground truth vs. Brush's render from the last eval."""
    from ..util import contact_sheet, imwrite_rgb, pick_evenly
    evals = sorted(out.glob("eval_*"), key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else -1)
    if not evals:
        return None
    pairs = []
    for r in pick_evenly(sorted(evals[-1].glob("*.png")), n):
        gt = cv2.imread(str(dataset / "images" / r.name), cv2.IMREAD_UNCHANGED)
        rd = cv2.imread(str(r))
        if gt is None or rd is None or gt.shape[:2] != rd.shape[:2]:
            continue
        if gt.ndim == 3 and gt.shape[2] == 4:
            a = gt[..., 3:4].astype(np.float32) / 255
            gt_rgb = (gt[..., :3] * a).astype(np.uint8)      # premultiplied, like the render's black background
            ys, xs = np.nonzero(gt[..., 3] > 0)
        else:                                               # weighted/ignore datasets: black background already
            gt_rgb = gt[..., :3]
            ys, xs = np.nonzero(gt_rgb.max(2) > 0)
        if len(xs) == 0:
            continue
        pad = 20
        y0, y1 = max(0, ys.min() - pad), min(gt.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(gt.shape[1], xs.max() + pad)
        side = np.concatenate([gt_rgb[y0:y1, x0:x1], rd[y0:y1, x0:x1]], 1)
        side = cv2.resize(side, (720, int(720 * side.shape[0] / side.shape[1])))
        pairs.append(cv2.cvtColor(side, cv2.COLOR_BGR2RGB))
    if not pairs:
        return None
    h = max(p.shape[0] for p in pairs)
    pairs = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in pairs]
    path = out / "eval_comparison.jpg"
    imwrite_rgb(path, contact_sheet(pairs, cols=2, thumb_w=720, labels=[""] * len(pairs)))
    return path


def ply_count(path: Path) -> int:
    with open(path, "rb") as f:
        for _ in range(50):
            line = f.readline().decode("ascii", "replace")
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line.startswith("end_header"):
                break
    return 0


def read_ply_gaussians(path: Path, max_n: int = 200000):
    """Return xyz (N,3), rgb (N,3 0..1), opacity (N) from a 3DGS-style binary ply."""
    with open(path, "rb") as f:
        props, n, fmt = [], 0, ""
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("element") and props:
                break
            elif line.startswith("property"):
                t, name = line.split()[1], line.split()[2]
                props.append((name, {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1",
                                     "uint8": "u1", "int": "i4", "uint": "u4", "short": "i2",
                                     "ushort": "u2", "half": "f2"}.get(t, "f4")))
            if line == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported ply format {fmt}")
        dt = np.dtype([(nm, "<" + t) for nm, t in props])
        data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    if n > max_n:
        data = data[np.random.default_rng(0).choice(n, max_n, replace=False)]
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float32)
    names = data.dtype.names
    if "f_dc_0" in names:
        c0 = 0.28209479177387814
        rgb = np.stack([data[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32) * c0 + 0.5
    elif "red" in names:
        rgb = np.stack([data["red"], data["green"], data["blue"]], 1) / 255.0
    else:
        rgb = np.full_like(xyz, 0.6)
    op = 1 / (1 + np.exp(-data["opacity"].astype(np.float32))) if "opacity" in names else np.ones(len(xyz))
    return xyz, np.clip(rgb, 0, 1), op
