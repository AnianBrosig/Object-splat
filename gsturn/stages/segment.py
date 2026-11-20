"""Stage 3: separate object (id 1) and hand (id 2) in every selected frame. Backends: cutie, grabcut."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from ..config import ROOT, Config, resolve_device
from ..events import Events
from ..pipeline import StageError, StageWarn
from ..util import contact_sheet, imread_rgb, imwrite_rgb, pick_evenly

WEIGHTS_DIR = ROOT / "weights"
CUTIE_URL = "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth"
CUTIE_MD5 = "a6071de6136982e396851903ab4c083a"


# --------------------------------------------------------------------------- first-frame masks
def box_px(box: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (int(round(x0 * w)), int(round(y0 * h)), int(round(x1 * w)), int(round(y1 * h)))


def boxes_mask(boxes: list[list[float]], w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    for b in boxes or []:
        x0, y0, x1, y1 = box_px(b, w, h)
        m[max(0, y0):y1, max(0, x0):x1] = True
    return m


def grabcut_box(img: np.ndarray, box: list[float], extra_bg: np.ndarray | None = None, iters: int = 5,
                sure_bg: np.ndarray | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box_px(box, w, h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise StageError(f"box too small: {box}")
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    if extra_bg is not None:
        mask[(extra_bg > 0) & (mask == cv2.GC_PR_FGD)] = cv2.GC_PR_BGD
    if sure_bg is not None:
        mask[sure_bg] = cv2.GC_BGD
        if not (mask == cv2.GC_PR_FGD).any():
            raise StageError("background boxes cover the whole object box")
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.grabCut(bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _hs_hist(img_hsv: np.ndarray, sel: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([img_hsv], [0, 1], sel.astype(np.uint8) * 255, [30, 32], [0, 180, 0, 256])
    hist = cv2.GaussianBlur(hist, (5, 5), 0) + 1e-3
    return hist / hist.sum()


def first_frame_labels(img: np.ndarray, obj_box: list[float], hand_boxes: list[list[float]],
                       bg_boxes: list[list[float]] | None = None) -> np.ndarray:
    """Label map: 0 bg, 1 object, 2 hand. Overlaps between object and hand are decided by colour.
    bg_boxes are certain background for both object and hands."""
    h, w = img.shape[:2]
    bg = boxes_mask(bg_boxes, w, h) if bg_boxes else None
    obj = grabcut_box(img, obj_box, sure_bg=bg)
    labels = np.zeros((h, w), np.uint8)
    labels[obj > 0] = 1
    if not hand_boxes:
        return labels
    hand = np.zeros((h, w), np.uint8)
    for hb in hand_boxes:
        hand |= grabcut_box(img, hb, sure_bg=bg)
    in_obj_box = np.zeros((h, w), bool)
    x0, y0, x1, y1 = box_px(obj_box, w, h)
    in_obj_box[y0:y1, x0:x1] = True
    in_hand_box = np.zeros((h, w), bool)
    for hb in hand_boxes:
        a0, b0, a1, b1 = box_px(hb, w, h)
        in_hand_box[b0:b1, a0:a1] = True
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hand_ref = (hand > 0) & ~in_obj_box          # hand pixels outside the object box
    obj_ref = (obj > 0) & ~in_hand_box           # object pixels outside any hand box
    contested = (obj > 0) & (hand > 0)
    labels[(hand > 0) & ~(obj > 0)] = 2
    if hand_ref.sum() > 200 and obj_ref.sum() > 200 and contested.any():
        ph = _hs_hist(hsv, hand_ref)
        po = _hs_hist(hsv, obj_ref)
        hb_ = (hsv[..., 0].astype(int) * 30 // 180).clip(0, 29)
        sb_ = (hsv[..., 1].astype(int) * 32 // 256).clip(0, 31)
        is_hand = ph[hb_, sb_] > po[hb_, sb_]
        is_hand = cv2.morphologyEx(is_hand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        labels[contested & is_hand] = 2
        labels[contested & ~is_hand] = 1
    elif contested.any():
        labels[contested] = 2
    return labels


# --------------------------------------------------------------------------- post-processing
def fill_small_holes(obj: np.ndarray, hand: np.ndarray, max_frac: float, close_px: int = 0):
    """Fill background regions enclosed by the object (not touching the border) that are smaller than
    max_frac of the object area and not mostly hand. obj/hand: 0/1. Returns (filled_obj, filled_pixels)."""
    m = (obj > 0).astype(np.uint8)
    if close_px > 0:  # optional: seal hairline cracks first
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    area = int(m.sum())
    filled = np.zeros_like(m)
    if area == 0 or max_frac <= 0:
        return m, filled
    inv = (1 - m).astype(np.uint8)
    # 4-connectivity for the background (8 for the object), so a diagonal one-pixel gap does not
    # connect a hole to the outside
    n, cc, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = m.shape
    hand01 = hand > 0
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches_border or a > max_frac * area:
            continue
        comp = cc[y:y + bh, x:x + bw] == i
        if hand01[y:y + bh, x:x + bw][comp].mean() > 0.2:   # a finger over the object: not a hole
            continue
        filled[y:y + bh, x:x + bw][comp] = 1
    return np.maximum(m, filled), filled


def clean_object(obj: np.ndarray, hand: np.ndarray, erode_px: int, hand_dilate_px: int,
                 fill_holes_frac: float = 0.0, close_px: int = 0, return_filled: bool = False):
    """Object mask (0/255) without hand pixels and specks, small holes filled, slightly eroded.
    With return_filled=True also returns the filled-hole pixels (0/255, after erosion clipping)."""
    obj = (obj > 0).astype(np.uint8)
    hand_zone = None
    if hand_dilate_px > 0 and hand.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hand_dilate_px + 1,) * 2)
        hand_zone = cv2.dilate((hand > 0).astype(np.uint8), k) > 0
        obj[hand_zone] = 0
    # keep the largest component and any component at least 10 % of it
    n, cc, st, _ = cv2.connectedComponentsWithStats(obj, 8)
    if n > 2:
        areas = st[1:, cv2.CC_STAT_AREA]
        keep = np.flatnonzero(areas >= 0.1 * areas.max()) + 1
        obj = np.isin(cc, keep).astype(np.uint8)
    filled = np.zeros_like(obj)
    if fill_holes_frac > 0:
        obj, filled = fill_small_holes(obj, (hand > 0).astype(np.uint8), fill_holes_frac, close_px)
        if hand_zone is not None:   # re-apply the hand removal: no filled (or closed) pixel on the hand
            obj[hand_zone] = 0
            filled[hand_zone] = 0
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)
        obj = cv2.erode(obj, k)
    filled &= obj
    if return_filled:
        return obj * 255, filled * 255
    return obj * 255


def overlay(img: np.ndarray, obj: np.ndarray, hand: np.ndarray, filled: np.ndarray | None = None) -> np.ndarray:
    out = img.astype(np.float32)
    dim = (obj == 0) & (hand == 0)
    out[dim] *= 0.35
    out[hand > 0] = 0.5 * out[hand > 0] + 0.5 * np.array([230, 60, 60])
    if filled is not None and filled.any():
        out[filled > 0] = 0.4 * out[filled > 0] + 0.6 * np.array([250, 220, 40])
    o = out.astype(np.uint8)
    cnts, _ = cv2.findContours((obj > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cnts, -1, (60, 230, 90), 2)
    return o


# --------------------------------------------------------------------------- cutie
def _download(url: str, dst: Path, md5: str) -> None:
    if dst.exists() and hashlib.md5(dst.read_bytes()).hexdigest() == md5:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if hashlib.md5(tmp.read_bytes()).hexdigest() != md5:
        raise RuntimeError(f"checksum mismatch for {url}")
    tmp.rename(dst)


def load_cutie(device: str, max_internal_size: int):
    """Cutie's get_default_model() calls .cuda(); this loads the same config and weights on any device."""
    import torch
    import cutie
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict
    from cutie.inference.inference_core import InferenceCore
    from cutie.inference.utils.args_utils import get_dataset_cfg
    from cutie.model.cutie import CUTIE

    weights = WEIGHTS_DIR / "cutie-base-mega.pth"
    _download(CUTIE_URL, weights, CUTIE_MD5)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3.2", config_dir=str(Path(list(cutie.__path__)[0]) / "config"),
                               job_name="eval_config"):
        cfg = compose(config_name="eval_config")
    with open_dict(cfg):
        cfg["weights"] = str(weights)
        cfg["max_internal_size"] = max_internal_size
        cfg["amp"] = False
    get_dataset_cfg(cfg)
    net = CUTIE(cfg).to(device).eval()
    net.load_weights(torch.load(str(weights), map_location=device))
    proc = InferenceCore(net, cfg=cfg)
    proc.max_internal_size = max_internal_size
    return proc


def segment_cutie(paths: list[Path], labels0: np.ndarray, cfg: Config, ev: Events):
    import warnings

    import torch
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
    device = resolve_device(cfg.device, "segment")
    ev.log(f"loading Cutie on {device} (max_internal_size {cfg.cutie_max_internal_size})")
    proc = load_cutie(device, cfg.cutie_max_internal_size)
    objects = [int(i) for i in np.unique(labels0) if i > 0]
    with torch.inference_mode():
        for i, p in enumerate(paths):
            img = imread_rgb(p)
            t = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device)
            if i == 0:
                prob = proc.step(t, torch.from_numpy(labels0).long().to(device), objects=objects)
            else:
                prob = proc.step(t, end=(i == len(paths) - 1))
            lab = proc.output_prob_to_mask(prob).cpu().numpy().astype(np.uint8)
            yield i, img, lab


def segment_grabcut(paths: list[Path], labels0: np.ndarray, cfg: Config, ev: Events):
    prev = labels0
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    for i, p in enumerate(paths):
        img = imread_rgb(p)
        if i == 0:
            yield i, img, labels0
            continue
        h, w = img.shape[:2]
        s = 0.5
        small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        lab = np.zeros((h, w), np.uint8)
        for obj_id in (1, 2):
            pm = cv2.resize((prev == obj_id).astype(np.uint8), (small.shape[1], small.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            if pm.sum() < 20:
                continue
            m = np.full(pm.shape, cv2.GC_BGD, np.uint8)
            m[cv2.dilate(pm, k_out) > 0] = cv2.GC_PR_BGD
            m[pm > 0] = cv2.GC_PR_FGD
            m[cv2.erode(pm, k_in) > 0] = cv2.GC_FGD if obj_id == 1 else cv2.GC_PR_FGD
            bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
            try:
                cv2.grabCut(bgr, m, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
            except cv2.error:
                continue
            fg = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
            fg = cv2.resize(fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            lab[fg & (lab == 0)] = obj_id
        prev = lab
        yield i, img, lab


def run(job, cfg: Config, ev: Events) -> dict:
    from ..handmask import skin_leak
    from ..multicam import cam_of, n_cameras
    meta = job.load_json("frames", "frames.json")
    names = [f["name"] for f in meta["frames"]]
    cams = [cam_of(f) for f in meta["frames"]]
    img_dir = job.dir / "work" / "frames" / "images"
    paths = [img_dir / n for n in names]
    # one sequence per camera: Cutie follows the object through that camera's frames, seeded by that camera's boxes
    seqs = []
    for c in range(n_cameras(meta)):
        idx = [i for i, cc in enumerate(cams) if cc == c]
        box, hboxes, bboxes = (cfg.object_box, cfg.hand_boxes, cfg.bg_boxes) if c == 0 else \
            (cfg.object_box2, cfg.hand_boxes2, cfg.bg_boxes2)
        if not box:
            raise StageError("no object box: draw a box around the object on the first frame (object_box)" if c == 0
                             else "no object box for camera 2: set object_box2 on the first frame of video2")
        seqs.append((c, idx, box, hboxes, bboxes))
    out = job.work("segment")
    for sub in ("masks", "hands", "overlays", "filled"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        (out / sub).mkdir()

    backend = cfg.seg_backend
    if backend == "cutie":
        try:
            import cutie  # noqa: F401
        except Exception as e:  # noqa: BLE001
            ev.warn(f"Cutie unavailable ({e}); using grabcut")
            backend = "grabcut"
    gen = segment_cutie if backend == "cutie" else segment_grabcut

    n_all = len(paths)
    areas = [0.0] * n_all
    hand_areas = [0.0] * n_all
    filled_px = [0] * n_all
    leak_frames, leak_px = 0, 0
    overlays_small = []
    sheet_idx = set(pick_evenly(list(range(n_all)), 48))
    done = 0
    img = None
    for c, idx, box, hboxes, bboxes in seqs:
        img0 = imread_rgb(paths[idx[0]])
        labels0 = first_frame_labels(img0, box, hboxes, bboxes)
        tag = "" if c == 0 else f"_c{c}"
        imwrite_rgb(out / f"first_frame_init{tag}.jpg", overlay(img0, (labels0 == 1) * 255, (labels0 == 2) * 255))
        ev.artifact(out / f"first_frame_init{tag}.jpg", f"first-frame mask from boxes (green object, red hand)"
                    + (f", camera {c + 1}" if len(seqs) > 1 else ""))
        if (labels0 == 1).sum() < 100:
            raise StageError("first-frame object mask is empty; redraw the object box"
                             + (f" (camera {c + 1})" if len(seqs) > 1 else ""))
        for k, img, lab in gen([paths[i] for i in idx], labels0, cfg, ev):
            i = idx[k]
            hand = ((lab == 2) * 255).astype(np.uint8)
            if cfg.skin_filter and hand.any():
                leak = skin_leak(img, ((lab == 1) * 255).astype(np.uint8), hand)
                if leak.any():                     # skin the tracker called object: it belongs to the hand
                    hand = np.where(leak, 255, hand).astype(np.uint8)
                    lab = np.where(leak, 2, lab).astype(np.uint8)
                    leak_frames += 1
                    leak_px += int(leak.sum())
            obj, filled = clean_object(lab == 1, hand, cfg.erode_px, cfg.hand_dilate_px, cfg.fill_holes_frac,
                                       cfg.close_px, return_filled=True)
            stem = names[i].rsplit(".", 1)[0]
            cv2.imwrite(str(out / "masks" / f"{stem}.png"), obj)
            cv2.imwrite(str(out / "hands" / f"{stem}.png"), hand)
            cv2.imwrite(str(out / "filled" / f"{stem}.png"), filled)
            filled_px[i] = int((filled > 0).sum())
            ov = overlay(img, obj, hand, filled)
            small = cv2.resize(ov, (640, int(640 * ov.shape[0] / ov.shape[1])), interpolation=cv2.INTER_AREA)
            imwrite_rgb(out / "overlays" / f"{stem}.jpg", small, 85)
            if i in sheet_idx:
                overlays_small.append((stem, small))
            areas[i] = float((obj > 0).mean())
            hand_areas[i] = float((hand > 0).mean())
            done += 1
            ev.progress(done, n_all, f"frame {done}/{n_all} ({backend}"
                        + (f", camera {c + 1}" if len(seqs) > 1 else "") + ")")

    a = np.array(areas)
    empty = int((a < 1e-4).sum())
    jumps = []
    for c, idx, *_ in seqs:                         # area jumps between neighbours of the same camera
        ac = a[idx]
        ratio = ac[1:] / np.maximum(ac[:-1], 1e-6) if len(ac) > 1 else np.array([1.0])
        jumps += [int(idx[j + 1]) for j in np.flatnonzero((ratio > 1.5) | (ratio < 1 / 1.5))]
    jumps.sort()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(100 * a, label="object area %")
    ax.plot(100 * np.array(hand_areas), label="hand area %", alpha=0.7)
    for j in jumps:
        ax.axvline(j, c="red", lw=0.8, alpha=0.6)
    ax.set_xlabel("selected frame")
    ax.set_ylabel("% of image")
    ax.legend(fontsize=8)
    ax.set_title(f"mask area ({len(jumps)} jumps > 50 %, {empty} empty)")
    fig.tight_layout()
    fig.savefig(out / "area.png", dpi=110)
    plt.close(fig)
    ev.artifact(out / "area.png", "mask area over time (red lines: jumps > 50 %)")
    overlays_small.sort()
    sheet = contact_sheet([s_ for _, s_ in overlays_small], cols=8, thumb_w=200,
                          labels=[n for n, _ in overlays_small])
    imwrite_rgb(out / "overview.jpg", sheet)
    ev.artifact(out / "overview.jpg", "mask overlays (green outline object, red hand, yellow filled holes)")

    stats = {"backend": backend, "frames": len(a), "cameras": len(seqs), "empty_masks": empty,
             "area_jumps": jumps[:50], "n_area_jumps": len(jumps), "median_object_area": round(float(np.median(a)), 4),
             "median_hand_area": round(float(np.median(hand_areas)), 4),
             "skin_leak_frames": leak_frames, "skin_leak_px_per_frame": int(leak_px / max(1, leak_frames)),
             "filled_px_median": int(np.median(filled_px)), "frames_with_filled": int(np.count_nonzero(filled_px)),
             "max_filled_frac": round(float(max((f / max(1.0, ar * img.shape[0] * img.shape[1])
                                                 for f, ar in zip(filled_px, areas)), default=0.0)), 5)}
    msg = f"{len(a)} masks ({backend}), median area {100 * np.median(a):.1f} %" + (
        f", skin removed from {leak_frames} frames" if leak_frames else "")
    if empty or jumps:
        raise StageWarn(f"{msg}; {empty} empty, {len(jumps)} area jumps", stats)
    stats["_msg"] = msg
    return stats
