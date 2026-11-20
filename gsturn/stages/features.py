"""Stage 4: keypoints inside the object mask + pairwise matching (LightGlue or SIFT)."""
from __future__ import annotations

import cv2
import numpy as np

from ..config import Config, resolve_device
from ..events import Events
from ..pipeline import StageError, StageWarn
from ..crop import feature_windows, robust_bbox
from ..handmask import keypoint_mask
from ..multicam import cams_for, partners_for, size_for
from ..util import contact_sheet, imread_rgb, imwrite_rgb, pick_evenly


# --------------------------------------------------------------------------- pairs
def make_pairs(n: int, window: int, stride: int, exhaustive_below: int) -> list[tuple[int, int]]:
    if n <= exhaustive_below:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs = set()
    for i in range(n):
        for d in range(1, window + 1):
            if i + d < n:
                pairs.add((i, i + d))
    # long-range pairs: every `stride`-th frame against every other `stride`-th frame (loop closure)
    if stride > 0:
        anchors = list(range(0, n, stride))
        offs = max(1, stride // 2)
        for a in anchors:
            for b in range(0, n, offs):
                if b - a > window:
                    pairs.add((a, b))
    return sorted(pairs)


def make_pairs_multi(cams: list[int], partner: dict[int, int], window: int, stride: int, exhaustive_below: int,
                     cross_window: int) -> list[tuple[int, int]]:
    """Pairs for one or several cameras. cams[i] is the camera of frame i (frames are ordered camera by camera);
    partner[i] is the frame of camera 0 recorded at the same moment as frame i of another camera. Within a camera
    the usual sliding window plus long-range pairs; across cameras every frame is matched with the partner and
    `cross_window` neighbours on each side (the neighbours add other relative rotations)."""
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(cams):
        groups.setdefault(c, []).append(i)
    pairs: set[tuple[int, int]] = set()
    for idx in groups.values():
        for a, b in make_pairs(len(idx), window, stride, exhaustive_below):
            pairs.add((idx[a], idx[b]))
    if len(groups) > 1:
        pos = {i: k for idx in groups.values() for k, i in enumerate(idx)}
        for j, i0 in partner.items():                       # j: frame of a secondary camera, i0: its partner in camera 0
            idx = groups[cams[j]]
            for d in range(-cross_window, cross_window + 1):
                k = pos[j] + d
                if 0 <= k < len(idx):
                    pairs.add((min(i0, idx[k]), max(i0, idx[k])))
    return sorted(pairs)


# --------------------------------------------------------------------------- extraction
def load_extractor_and_matcher(name: str, max_kp: int, device: str):
    import torch  # noqa: F401
    from lightglue import ALIKED, DISK, LightGlue, SuperPoint
    cls = {"superpoint": SuperPoint, "aliked": ALIKED, "disk": DISK}[name]
    extractor = cls(max_num_keypoints=max_kp).eval().to(device)
    matcher = LightGlue(features=name).eval().to(device)
    return extractor, matcher


def filter_in_mask(kpts: np.ndarray, scores: np.ndarray, mask: np.ndarray, keep: int) -> np.ndarray:
    h, w = mask.shape
    xi = np.clip(np.round(kpts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(kpts[:, 1]).astype(int), 0, h - 1)
    inside = np.flatnonzero(mask[yi, xi] > 0)
    order = inside[np.argsort(-scores[inside])][:keep]
    return np.sort(order)


class WindowExtractor:
    """SuperPoint / ALIKED / DISK on a window of the frame, resized to a fixed shape per camera (so the GPU compiles
    its kernels once) and optionally turned in the image plane (cameras that look at the object from a very
    different roll). Keypoints are reported in full-frame pixels for SfM, and in window coordinates for matching,
    where the object fills the image the way LightGlue expects."""

    def __init__(self, backend: str, max_kp: int, device: str, match_device: str | None = None):
        self.extractor, _ = load_extractor_and_matcher(backend, 3 * max_kp, device)
        self.matcher = load_matcher(backend, match_device or device)
        self.device = device
        self.match_device = match_device or device
        self.max_kp = max_kp

    @staticmethod
    def transform(win, scale: float, rot_deg: float = 0.0):
        """2x3 affine full frame -> processed image and the processed size (multiples of 8)."""
        x0, y0, x1, y1 = win
        w, h = x1 - x0, y1 - y0
        pw, ph = max(8, int(round(w * scale / 8)) * 8), max(8, int(round(h * scale / 8)) * 8)
        sx, sy = pw / w, ph / h
        A = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]], np.float64)
        if rot_deg:
            R = cv2.getRotationMatrix2D((pw / 2, ph / 2), rot_deg, 1.0)
            corners = np.array([[0, 0, 1], [pw, 0, 1], [pw, ph, 1], [0, ph, 1]], np.float64) @ R.T
            lo, hi = corners.min(0), corners.max(0)
            R[:, 2] -= lo
            pw, ph = int(np.ceil((hi[0] - lo[0]) / 8) * 8), int(np.ceil((hi[1] - lo[1]) / 8) * 8)
            A = R @ np.vstack([A, [0, 0, 1]])
        return A, (pw, ph)

    def extract(self, img_rgb: np.ndarray, A: np.ndarray, size: tuple[int, int], kp_mask: np.ndarray):
        """Returns (feats dict in processed coordinates, keypoints in full-frame pixels)."""
        import torch
        pw, ph = size
        proc = self._warp(img_rgb, A, pw, ph)
        t = torch.from_numpy(proc).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.inference_mode():
            f = self.extractor.extract(t, resize=None)
        kp = f["keypoints"][0].cpu().numpy()
        sc = f["keypoint_scores"][0].cpu().numpy() if "keypoint_scores" in f else np.ones(len(kp))
        Ainv = np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2]
        full = kp @ Ainv[:, :2].T + Ainv[:, 2]
        sel = filter_in_mask(full, sc, kp_mask, self.max_kp)
        st = torch.from_numpy(sel).to(self.device)
        md = self.match_device
        g = {"keypoints": f["keypoints"][:, st].to(md), "descriptors": f["descriptors"][:, st].to(md),
             "image_size": torch.tensor([[pw, ph]], dtype=torch.float32, device=md)}
        for k in ("keypoint_scores", "scales", "oris"):
            if k in f:
                g[k] = f[k][:, st].to(md)
        return g, full[sel].astype(np.float32)

    @staticmethod
    def _warp(img: np.ndarray, A: np.ndarray, pw: int, ph: int) -> np.ndarray:
        """Window -> processed image. Shrinking uses area averaging (no aliasing), then the optional turn."""
        sx = float(np.hypot(A[0, 0], A[1, 0]))
        if sx < 1:
            # crop first (cheap), scale with INTER_AREA, then apply the remaining rotation/translation
            Ainv = np.linalg.inv(np.vstack([A, [0, 0, 1]]))
            c = Ainv @ np.array([[0, pw, pw, 0], [0, 0, ph, ph], [1, 1, 1, 1]], np.float64)
            x0, y0 = max(0, int(np.floor(c[0].min()))), max(0, int(np.floor(c[1].min())))
            x1, y1 = min(img.shape[1], int(np.ceil(c[0].max()))), min(img.shape[0], int(np.ceil(c[1].max())))
            crop = img[y0:y1, x0:x1]
            small = cv2.resize(crop, (max(1, int(round(crop.shape[1] * sx))), max(1, int(round(crop.shape[0] * sx)))),
                               interpolation=cv2.INTER_AREA)
            f = small.shape[1] / max(1, crop.shape[1])
            M = np.vstack([A, [0, 0, 1]]) @ np.array([[1 / f, 0, x0], [0, 1 / f, y0], [0, 0, 1]])
            return cv2.warpAffine(small, M[:2], (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.warpAffine(img, A, (pw, ph), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def extract_lightglue(paths, kp_masks, wins, scales, rots, cfg: Config, ev: Events, device: str,
                      match_device: str, backend: str | None = None):
    """Features for every frame. wins[i]: window in full-frame pixels, scales[i] / rots[i]: resize factor and
    in-plane turn (degrees). Returns (feats, matcher, keypoints in full pixels, affines, processed sizes)."""
    ex = WindowExtractor(backend or cfg.feature_backend, cfg.max_keypoints, device, match_device)
    feats, kpts, affs, sizes = [], [], [], []
    for i, p in enumerate(paths):
        img = imread_rgb(p)
        A, size = ex.transform(wins[i], scales[i], rots[i])
        g, kp = ex.extract(img, A, size, kp_masks[i])
        feats.append(g)
        kpts.append(kp)
        affs.append(A)
        sizes.append(size)
        ev.progress(15 * (i + 1) / len(paths), 100, f"keypoints {i + 1}/{len(paths)} ({len(kp)} in mask)")
    return feats, ex.matcher, kpts, affs, sizes


def probe_rotations(paths, kp_masks, wins, scales, cams, partner, cfg: Config, ev: Events, device: str,
                    match_device: str, backend: str, angles=tuple(range(0, 360, 45)), n_pairs: int = 6) -> dict:
    """In-plane turn (degrees) for each secondary camera that gives the most cross-camera matches.

    SuperPoint/ALIKED descriptors tolerate about +-30 degrees of roll, and two phones pointing at the same table
    are often rolled by 90 degrees (portrait vs landscape) or more. A few synchronised pairs are matched with
    the secondary window turned by each angle; the angle with the most epipolar inliers is used for the whole
    camera. 0 for camera 0, and for a camera where no angle gives a usable number of matches."""
    ex = WindowExtractor(backend, cfg.max_keypoints, device, match_device)
    imgs: dict[int, np.ndarray] = {}

    def img(i):
        if i not in imgs:
            imgs[i] = imread_rgb(paths[i])
        return imgs[i]
    rot = {0: 0.0}
    for c in range(1, max(cams) + 1):
        js = pick_evenly([j for j in sorted(partner) if cams[j] == c], n_pairs)
        if not js:
            rot[c] = 0.0
            continue
        base = {}
        for j in js:
            i0 = partner[j]
            A, size = ex.transform(wins[i0], scales[i0], 0.0)
            base[j] = ex.extract(img(i0), A, size, kp_masks[i0])
        score = {}
        for k, ang in enumerate(angles):
            tot = 0
            for j in js:
                A, size = ex.transform(wins[j], scales[j], float(ang))
                gj, kj = ex.extract(img(j), A, size, kp_masks[j])
                g0, k0 = base[j]
                m, _ = match_lightglue({0: g0, 1: gj}, ex.matcher, 0, 1)
                if len(m) >= 8:
                    _, inl = cv2.findFundamentalMat(k0[m[:, 0]], kj[m[:, 1]], cv2.FM_RANSAC, 3.0, 0.99)
                    tot += int(inl.sum()) if inl is not None else 0
            score[ang] = tot
            ev.progress(k + 1, len(angles), f"camera {c}: probing in-plane rotation {ang} deg -> {tot} inliers")
        best = max(score, key=score.get)
        ev.log(f"camera {c} cross-camera inliers over {len(js)} synchronised pairs by in-plane rotation: "
               + ", ".join(f"{a}: {v}" for a, v in score.items()))
        if score[best] < 8 * len(js):
            ev.warn(f"camera {c}: at most {score[best]} cross-camera inliers over {len(js)} pairs; the views may "
                    f"be too different to link by matching (the cameras may end up as separate models)")
        rot[c] = float(best) if score[best] >= 8 * len(js) else 0.0
        ev.log(f"camera {c}: using an in-plane rotation of {rot[c]:.0f} deg for feature extraction")
    return rot


def match_lightglue(feats, matcher, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    with torch.inference_mode():
        out = matcher({"image0": feats[i], "image1": feats[j]})
    m = out["matches"][0].cpu().numpy().astype(np.int32)
    s = out["scores"][0].cpu().numpy().astype(np.float32)
    return m, s


def extract_sift(paths, masks, cfg: Config, ev: Events):
    sift = cv2.SIFT_create(nfeatures=3 * cfg.max_keypoints)
    feats = []
    for i, (p, m) in enumerate(zip(paths, masks)):
        gray = cv2.cvtColor(imread_rgb(p), cv2.COLOR_RGB2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None or not kps:
            feats.append({"kpts": np.zeros((0, 2), np.float32), "desc": np.zeros((0, 128), np.float32)})
            continue
        kp = np.array([k.pt for k in kps], np.float32)
        sc = np.array([k.response for k in kps], np.float32)
        sel = filter_in_mask(kp, sc, m, cfg.max_keypoints)
        d = desc[sel].astype(np.float32)
        d /= np.maximum(d.sum(1, keepdims=True), 1e-8)   # RootSIFT
        d = np.sqrt(d)
        feats.append({"kpts": kp[sel], "desc": d, "scores": sc[sel]})
        ev.progress(15 * (i + 1) / len(paths), 100, f"SIFT {i + 1}/{len(paths)} ({len(sel)} in mask)")
    return feats


_flann = None


def match_sift(feats, i: int, j: int, ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    global _flann
    a, b = feats[i]["desc"], feats[j]["desc"]
    if len(a) < 2 or len(b) < 2:
        return np.zeros((0, 2), np.int32), np.zeros(0, np.float32)
    if _flann is None:
        _flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})
    ab = _flann.knnMatch(a, b, k=2)
    ba = _flann.knnMatch(b, a, k=1)
    back = {m[0].queryIdx: m[0].trainIdx for m in ba if m}
    out, sc = [], []
    for pair in ab:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance and back.get(m.trainIdx) == m.queryIdx:
            out.append((m.queryIdx, m.trainIdx))
            sc.append(1.0 - m.distance / max(n.distance, 1e-8))
    return np.array(out, np.int32).reshape(-1, 2), np.array(sc, np.float32)


# --------------------------------------------------------------------------- saved features (for refine)
def save_features(out, stems, feats, backend: str) -> None:
    """descriptors.npz / scores.npz (+ scales/oris for ALIKED/DISK), same order as keypoints.npz."""
    descs, scores, extra = {}, {}, {}
    for s, f in zip(stems, feats):
        if backend == "sift":
            descs[s] = f["desc"].astype(np.float16)
            scores[s] = f.get("scores", np.ones(len(f["kpts"]))).astype(np.float16)
        else:
            descs[s] = f["descriptors"][0].cpu().numpy().astype(np.float16)
            sc = f.get("keypoint_scores")
            scores[s] = (sc[0].cpu().numpy() if sc is not None else np.ones(len(descs[s]))).astype(np.float16)
            for k in ("scales", "oris"):
                if k in f:
                    extra[f"{k}/{s}"] = f[k][0].cpu().numpy().astype(np.float32)
    np.savez_compressed(out / "descriptors.npz", **descs)
    np.savez_compressed(out / "scores.npz", **scores)
    if extra:
        np.savez_compressed(out / "extras.npz", **extra)


def load_features(job, stems: list[str], backend: str, device: str = "cpu", sizes=None):
    """Rebuild matcher inputs from the saved files: LightGlue dicts or SIFT dicts, in `stems` order."""
    d = job.dir / "work" / "features"
    if not (d / "descriptors.npz").exists():
        raise StageError("features/descriptors.npz missing: re-run stage 'features' (older jobs did not save "
                         "descriptors)")
    kp, de, sc = np.load(d / "keypoints.npz"), np.load(d / "descriptors.npz"), np.load(d / "scores.npz")
    ex = np.load(d / "extras.npz") if (d / "extras.npz").exists() else None
    feats = []
    if backend == "sift":
        for s in stems:
            feats.append({"kpts": kp[s], "desc": de[s].astype(np.float32), "scores": sc[s].astype(np.float32)})
        return feats
    import torch
    aff = np.load(d / "affine.npz") if (d / "affine.npz").exists() else None
    for i, s in enumerate(stems):
        w, h = sizes[i] if sizes is not None else (0, 0)
        kps = kp[s].astype(np.float32)
        if aff is not None and s in aff.files:             # matcher coordinates = window, as in the features stage
            a = aff[s]
            A = a[:6].reshape(2, 3)
            kps = (kps @ A[:, :2].T + A[:, 2]).astype(np.float32)
            w, h = int(a[6]), int(a[7])
        g = {"keypoints": torch.from_numpy(kps)[None].to(device),
             "descriptors": torch.from_numpy(de[s].astype(np.float32))[None].to(device),
             "keypoint_scores": torch.from_numpy(sc[s].astype(np.float32))[None].to(device),
             "image_size": torch.tensor([[w, h]], dtype=torch.float32, device=device)}
        if ex is not None:
            for k in ("scales", "oris"):
                if f"{k}/{s}" in ex.files:
                    g[k] = torch.from_numpy(ex[f"{k}/{s}"])[None].to(device)
        feats.append(g)
    return feats


def load_matcher(backend: str, device: str):
    from lightglue import LightGlue
    return LightGlue(features=backend).eval().to(device)


# --------------------------------------------------------------------------- debug drawings
def draw_keypoints(img: np.ndarray, kp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[mask == 0] = (out[mask == 0] * 0.4).astype(np.uint8)
    for x, y in kp:
        cv2.circle(out, (int(x), int(y)), 2, (255, 230, 0), -1, cv2.LINE_AA)
    return out


def draw_matches(a: np.ndarray, b: np.ndarray, ka: np.ndarray, kb: np.ndarray, m: np.ndarray, title: str):
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:] = b
    rng = np.random.default_rng(0)
    show = m if len(m) <= 300 else m[rng.choice(len(m), 300, replace=False)]
    for i0, i1 in show:
        c = tuple(int(v) for v in rng.integers(80, 255, 3))
        p0 = (int(ka[i0, 0]), int(ka[i0, 1]))
        p1 = (int(kb[i1, 0] + a.shape[1]), int(kb[i1, 1]))
        cv2.line(canvas, p0, p1, c, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, c, -1)
        cv2.circle(canvas, p1, 2, c, -1)
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- stage
def run(job, cfg: Config, ev: Events) -> dict:
    from .quality import TRUST_KP, load_map, load_quality
    meta = job.load_json("frames", "frames.json")
    names = [f["name"] for f in meta["frames"]]
    stems = [n.rsplit(".", 1)[0] for n in names]
    img_dir = job.dir / "work" / "frames" / "images"
    mask_dir = job.dir / "work" / "segment" / "masks"

    # quality stage (optional): drop excluded frames, keep keypoints on trusted pixels only.
    # Image files are not touched, so names stay stable for the UI.
    quality = load_quality(job, cfg)
    excluded: list[str] = []
    if quality is not None:
        excluded = [s for s in quality.get("excluded", []) if s in stems]
        if len(stems) - len(excluded) < 10:
            ev.warn(f"quality would leave only {len(stems) - len(excluded)} frames; ignoring the exclusions")
            excluded = []
        keep = [i for i, s in enumerate(stems) if s not in set(excluded)]
        names, stems = [names[i] for i in keep], [stems[i] for i in keep]
        if excluded:
            ev.log(f"quality: excluding {len(excluded)} frames ({', '.join(excluded[:12])}"
                   f"{' ...' if len(excluded) > 12 else ''})")
    paths = [img_dir / n for n in names]
    hand_dir = job.dir / "work" / "segment" / "hands"
    cams, partner = cams_for(meta, stems), partners_for(meta, stems)
    ncam = max(cams) + 1
    masks, boxes = [], []
    n_trust_maps = 0
    for s in stems:
        m = cv2.imread(str(mask_dir / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise StageError(f"missing mask {s}.png: run stage 'segment' first")
        if quality is not None:
            q = load_map(job, s)
            if q is not None and q.shape == m.shape:
                m = np.where((m > 0) & (q >= TRUST_KP), 255, 0).astype(np.uint8)
                n_trust_maps += 1
        boxes.append(robust_bbox(m))                       # visible, trusted object: sizes the extraction window
        hd = cv2.imread(str(hand_dir / f"{s}.png"), cv2.IMREAD_GRAYSCALE) if cfg.kp_hand_margin_px > 0 else None
        masks.append(keypoint_mask(m, hd, cfg.kp_hand_margin_px))
    out = job.work("features")
    n = len(paths)

    backend = cfg.feature_backend
    device = resolve_device(cfg.device)
    if backend != "sift":
        try:
            import lightglue  # noqa: F401
        except Exception as e:  # noqa: BLE001
            ev.warn(f"LightGlue unavailable ({e}); using SIFT")
            backend = "sift"
    if cfg.match_device != "auto":
        match_device = cfg.match_device
    else:
        match_device = "cuda" if device == "cuda" else "cpu"     # variable keypoint counts recompile MPS kernels

    wins, scales, rots, win_info = [None] * n, [1.0] * n, [0.0] * n, []
    for c in range(ncam):
        idx = [i for i in range(n) if cams[i] == c]
        W, H = size_for(meta, c)
        if cfg.feature_crop and backend != "sift":
            ws, (ww, wh) = feature_windows([boxes[i] for i in idx], W, H, cfg.feature_crop_margin)
        else:
            ws, (ww, wh) = [(0, 0, W, H)] * len(idx), (W, H)
        sc = float(np.clip(cfg.feature_max_side / max(ww, wh), 0.1, 2.0)) if cfg.feature_max_side > 0 else 1.0
        for i, w_ in zip(idx, ws):
            wins[i], scales[i] = w_, sc
        win_info.append({"camera": c, "window": [ww, wh], "frame": [W, H], "scale": round(sc, 3)})
        ev.log(f"camera {c}: extraction window {ww}x{wh} of {W}x{H} px, scale {sc:.2f} "
               f"-> {int(round(ww * sc))}x{int(round(wh * sc))}")

    ev.log(f"extracting {backend} keypoints on {device if backend != 'sift' else 'cpu'}, matching on "
           f"{match_device if backend != 'sift' else 'cpu'}")
    affs = sizes_proc = None
    if backend == "sift":
        feats = extract_sift(paths, masks, cfg, ev)
        kpts = [f["kpts"] for f in feats]
        matcher = None
    else:
        if ncam > 1:
            cam_rot = probe_rotations(paths, masks, wins, scales, cams, partner, cfg, ev, device, match_device,
                                      backend)
            rots = [cam_rot[c] for c in cams]
            for c in range(ncam):
                win_info[c]["rotation_deg"] = cam_rot[c]
        try:
            feats, matcher, kpts, affs, sizes_proc = extract_lightglue(paths, masks, wins, scales, rots, cfg, ev,
                                                                       device, match_device, backend)
        except Exception as e:  # noqa: BLE001
            if device == "cpu":
                raise
            ev.warn(f"extraction on {device} failed ({type(e).__name__}: {e}); retrying on the CPU")
            device, match_device = "cpu", "cpu"
            feats, matcher, kpts, affs, sizes_proc = extract_lightglue(paths, masks, wins, scales, rots, cfg, ev,
                                                                       "cpu", "cpu", backend)

    pairs = make_pairs_multi(cams, partner, cfg.match_window, cfg.match_stride_pairs, cfg.exhaustive_below,
                             cfg.cross_window)
    ev.log(f"matching {len(pairs)} pairs ({backend})")
    matches: dict[str, np.ndarray] = {}
    counts = np.zeros((n, n), np.int32)
    import time
    t0 = time.time()
    for k, (i, j) in enumerate(pairs):
        if len(kpts[i]) < 8 or len(kpts[j]) < 8:
            continue
        m, _ = match_sift(feats, i, j) if backend == "sift" else match_lightglue(feats, matcher, i, j)
        counts[i, j] = counts[j, i] = len(m)
        if len(m) >= cfg.min_matches:
            matches[f"{i}_{j}"] = m
        if k % 5 == 0 or k == len(pairs) - 1:
            el = time.time() - t0
            eta = el / (k + 1) * (len(pairs) - k - 1)
            ev.progress(15 + 85 * (k + 1) / len(pairs), 100, f"pair {k + 1}/{len(pairs)}, eta {eta / 60:.1f} min")

    np.savez_compressed(out / "keypoints.npz", **{s: kp.astype(np.float32) for s, kp in zip(stems, kpts)})
    np.savez_compressed(out / "matches.npz", **matches)
    save_features(out, stems, feats, backend)
    if affs is not None:       # full frame -> matcher coordinates, so refine can rebuild the matcher input
        np.savez_compressed(out / "affine.npz", **{s_: np.concatenate([a_.reshape(-1), np.array(z, float)])
                                                   for s_, a_, z in zip(stems, affs, sizes_proc)})
    job.save_json("features", "index.json", {"stems": stems, "backend": backend, "pairs": len(pairs),
                                             "excluded": excluded, "trust_maps": n_trust_maps,
                                             "cams": cams, "windows": win_info, "device": device,
                                             "match_device": match_device})

    # debug artifacts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(np.log1p(counts), cmap="magma")
    ax.set_title(f"matches per pair (log), {len(matches)} kept of {len(pairs)} pairs")
    ax.set_xlabel("frame")
    ax.set_ylabel("frame")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "match_matrix.png", dpi=110)
    plt.close(fig)
    ev.artifact(out / "match_matrix.png", "match-count matrix (bright diagonal = good sequential overlap)")

    sel = pick_evenly(list(range(n)), 16)
    kp_imgs = [draw_keypoints(imread_rgb(paths[i]), kpts[i], masks[i]) for i in sel]
    imwrite_rgb(out / "keypoints.jpg", contact_sheet(kp_imgs, cols=4, thumb_w=400,
                                                     labels=[f"{stems[i]}: {len(kpts[i])}" for i in sel]))
    ev.artifact(out / "keypoints.jpg", "keypoints kept inside the object mask")

    samples = []
    if n > 1:
        samples += [(i, i + 1) for i in pick_evenly(list(range(n - 1)), 3)]
    long_pairs = sorted((k for k in matches if abs(int(k.split("_")[1]) - int(k.split("_")[0])) > cfg.match_window),
                        key=lambda k: -len(matches[k]))[:2]
    samples += [tuple(int(x) for x in k.split("_")) for k in long_pairs]
    for i, j in samples:
        m = matches.get(f"{i}_{j}", np.zeros((0, 2), np.int32))
        canvas = draw_matches(imread_rgb(paths[i]), imread_rgb(paths[j]), kpts[i], kpts[j], m,
                              f"{stems[i]} - {stems[j]}: {len(m)} matches")
        p = out / f"matches_{stems[i]}_{stems[j]}.jpg"
        imwrite_rgb(p, cv2.resize(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2)))
        ev.artifact(p, f"matches {stems[i]} - {stems[j]}")

    nk = np.array([len(k) for k in kpts])
    seq_ids = [i for i in range(n - 1) if cams[i] == cams[i + 1]]
    seq = np.array([counts[i, i + 1] for i in seq_ids]) if seq_ids else np.array([0])
    stats = {"backend": backend, "frames": n, "pairs": len(pairs), "pairs_kept": len(matches),
             "median_keypoints": int(np.median(nk)), "min_keypoints": int(nk.min()),
             "median_seq_matches": int(np.median(seq)), "weakest_seq_pair": int(seq.min()),
             "weakest_seq_pair_at": int(seq_ids[int(np.argmin(seq))]) if seq_ids else 0,
             "excluded_frames": len(excluded), "windows": win_info, "cameras": ncam,
             "trust_maps_used": n_trust_maps}
    msg = (f"{int(np.median(nk))} kp/frame (median), {len(matches)}/{len(pairs)} pairs kept, "
           f"weakest sequential pair {int(seq.min())}" + (f", {len(excluded)} frames excluded" if excluded else ""))
    cross_warn = ""
    if ncam > 1:
        cross = [(i, j) for i, j in pairs if cams[i] != cams[j]]
        kept = [len(matches[f"{i}_{j}"]) for i, j in cross if f"{i}_{j}" in matches]
        stats.update({"cross_pairs": len(cross), "cross_pairs_kept": len(kept),
                      "median_cross_matches": int(np.median(kept)) if kept else 0})
        msg += f"; cross-camera {len(kept)}/{len(cross)} pairs, median {stats['median_cross_matches']} matches"
        if len(kept) < 0.3 * max(1, len(cross)):
            cross_warn = "few cross-camera matches: the cameras will probably not be linked"
            ev.warn(cross_warn)
    if np.median(nk) < 300 or seq.min() < 25 or cross_warn:
        raise StageWarn(msg + ("; " + cross_warn if cross_warn else ""), stats)
    stats["_msg"] = msg
    return stats
