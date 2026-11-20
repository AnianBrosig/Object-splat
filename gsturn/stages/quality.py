"""Stage 3b: per-pixel trust maps against motion blur, defocus, clipped highlights and specular reflections.

Q = 255 for a trusted object pixel, 0 for untrusted or not object. Consumers (features, train) use the maps only
if they exist and quality_enabled is set, so older jobs keep working.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..events import Events
from ..pipeline import StageError, StageWarn
from ..util import contact_sheet, imread_rgb, imwrite_rgb, pick_evenly

TRUST_KP = 128      # Q threshold for keypoints (filled holes at 50 % trust fall below it)
TRUST_TRAIN = 64    # Q threshold for train_mask_mode=ignore_quality


# --------------------------------------------------------------------------- sharpness
def laplacian_energy(gray: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F)
    return lap * lap


class TileGrid:
    """Tile grid anchored at the mask's bounding-box centre, so tile (r, c) covers roughly the same part of the
    object in neighbouring frames. Keys are tile offsets relative to that centre."""

    def __init__(self, mask: np.ndarray, tile: int):
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        if len(xs):
            cx = int(round((xs.min() + xs.max()) / 2))
            cy = int(round((ys.min() + ys.max()) / 2))
        else:
            cx, cy = w // 2, h // 2
        self.tile, self.h, self.w = tile, h, w
        self.px = (tile - cx % tile) % tile          # padding so tile borders fall on cx + k * tile
        self.py = (tile - cy % tile) % tile
        self.nc = -(-(w + self.px) // tile)
        self.nr = -(-(h + self.py) // tile)
        self.c0 = (-self.px - cx) // tile           # key of padded column 0 (exact division)
        self.r0 = (-self.py - cy) // tile

    def reduce(self, img: np.ndarray) -> np.ndarray:
        """Per-tile sum of img (zero padding)."""
        t = self.tile
        pad = np.zeros((self.nr * t, self.nc * t), np.float32)
        pad[self.py:self.py + self.h, self.px:self.px + self.w] = img
        return pad.reshape(self.nr, t, self.nc, t).sum((1, 3))

    def upsample(self, grid: np.ndarray) -> np.ndarray:
        t = self.tile
        big = cv2.resize(grid.astype(np.float32), (self.nc * t, self.nr * t), interpolation=cv2.INTER_LINEAR)
        return big[self.py:self.py + self.h, self.px:self.px + self.w]

    def keys(self) -> tuple[np.ndarray, np.ndarray]:
        return np.arange(self.nr) + self.r0, np.arange(self.nc) + self.c0


def tile_energies(gray: np.ndarray, mask: np.ndarray, tile: int) -> dict:
    """Laplacian energy per tile with >= 50 % mask coverage: mean over the object pixels of the 3x3 tile
    neighbourhood. The smoothing matters on real video: as the object turns, print slides from one tile into the
    next, and single-tile energies then change without any blur (on IMG_8651 that flagged 22 % of each frame)."""
    m = (mask > 0).astype(np.float32)
    grid = TileGrid(mask > 0, tile)
    e = grid.reduce(laplacian_energy(gray) * m)
    cov = grid.reduce(m)
    e3 = cv2.boxFilter(e, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    c3 = cv2.boxFilter(cov, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    valid = cov >= 0.5 * tile * tile
    rk, ck = grid.keys()
    out = {}
    for i, j in zip(*np.nonzero(valid)):
        out[(int(rk[i]), int(ck[j]))] = float(e3[i, j] / c3[i, j])
    return {"grid": grid, "E": out}


def fill_invalid(vals: np.ndarray, valid: np.ndarray, default: float = 1.0) -> np.ndarray:
    """Give tiles without a value the value of the nearest valid tile."""
    if valid.all():
        return vals
    if not valid.any():
        return np.full_like(vals, default)
    _, lab = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 3,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    src = np.zeros(lab.max() + 1, np.float32)
    ys, xs = np.nonzero(valid)
    src[lab[ys, xs]] = vals[ys, xs]
    return src[lab]


def sharpness_score(cur: dict, neighbours: list[dict], cfg: Config) -> tuple[np.ndarray, np.ndarray, float]:
    """Pixel map s in 0..1 (1 = as sharp as the sharpest neighbour), relative-sharpness tile grid, median r."""
    grid: TileGrid = cur["grid"]
    rk, ck = grid.keys()
    r = np.ones((grid.nr, grid.nc), np.float32)
    valid = np.zeros((grid.nr, grid.nc), bool)
    rel = []
    for i, rr in enumerate(rk):
        for j, cc in enumerate(ck):
            e = cur["E"].get((int(rr), int(cc)))
            if e is None:
                continue
            ref = max([e] + [n["E"][(int(rr), int(cc))] for n in neighbours if (int(rr), int(cc)) in n["E"]])
            valid[i, j] = True
            if ref < cfg.quality_texture_floor:
                r[i, j] = 1.0          # blank area: cannot be judged, and gives no keypoints anyway
            else:
                r[i, j] = e / ref
                rel.append(r[i, j])
    b = cfg.blur_rel_threshold
    s = np.clip((r - 0.5 * b) / (0.5 * b), 0, 1)
    s = fill_invalid(s, valid)
    return np.clip(grid.upsample(s), 0, 1), r, float(np.median(rel)) if rel else 1.0


# --------------------------------------------------------------------------- reflections
def bad_light_mask(img: np.ndarray, specular_v: int, specular_s: int, tophat_min: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """(bad, specular) boolean maps: clipped or small bright unsaturated highlights, dilated by 2 px.

    Clipped = at least two channels >= 250 (a single saturated channel, e.g. bright red print, keeps its texture).
    Highlights must be brighter than their surroundings. The reference is a closing (fills dark print up to
    15 px) followed by an opening (removes small bright spots), so white paper between dark letters is not
    flagged, while a small highlight on a darker surface is."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clipped = (img >= 250).sum(axis=2) >= 2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ref = cv2.morphologyEx(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    tophat = cv2.subtract(gray, ref)
    spec = (v >= specular_v) & (s <= specular_s) & (tophat >= tophat_min)
    bad = cv2.dilate((clipped | spec).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return bad, spec


def trust_map(mask: np.ndarray, s_pix: np.ndarray, bad: np.ndarray, filled: np.ndarray | None) -> np.ndarray:
    q = (mask > 0).astype(np.float32) * s_pix * (1.0 - bad.astype(np.float32))
    if filled is not None:
        q *= 1.0 - 0.5 * (filled > 0)
    return np.clip(q * 255.0 + 0.5, 0, 255).astype(np.uint8)


def visualize(img: np.ndarray, mask: np.ndarray, s_pix: np.ndarray, bad: np.ndarray,
              filled: np.ndarray | None) -> np.ndarray:
    """Object trusted: untouched; blur: red; clipped/specular: magenta; filled: yellow; outside mask: dimmed."""
    out = img.astype(np.float32)
    m = mask > 0
    out[~m] *= 0.35
    blur = m & (s_pix < 0.5)
    out[blur] = 0.45 * out[blur] + 0.55 * np.array([235, 40, 40])
    if filled is not None:
        f = m & (filled > 0)
        out[f] = 0.4 * out[f] + 0.6 * np.array([250, 220, 40])
    sp = m & bad
    out[sp] = 0.35 * out[sp] + 0.65 * np.array([235, 40, 235])
    return out.astype(np.uint8)


# --------------------------------------------------------------------------- helpers for consumers
def load_quality(job, cfg: Config):
    """quality.json if the stage ran, is enabled and is newer than the segment masks, else None."""
    if not cfg.quality_enabled:
        return None
    qdir = job.dir / "work" / "quality"
    p = qdir / "quality.json"
    if not p.exists():
        return None
    seg = job.dir / "work" / "segment" / "stats.json"
    if seg.exists() and seg.stat().st_mtime > p.stat().st_mtime:
        return None          # masks were rebuilt after the quality maps: stale
    data = json.loads(p.read_text())
    data["dir"] = qdir
    return data


def load_map(job, stem: str) -> np.ndarray | None:
    m = cv2.imread(str(job.dir / "work" / "quality" / "maps" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    return m


# --------------------------------------------------------------------------- stage
def run(job, cfg: Config, ev: Events) -> dict:
    out = job.work("quality")
    if not cfg.quality_enabled:
        (out / "quality.json").unlink(missing_ok=True)
        return {"skipped": True, "_msg": "disabled"}
    meta = job.load_json("frames", "frames.json")
    names = [f["name"] for f in meta["frames"]]
    stems = [n.rsplit(".", 1)[0] for n in names]
    img_dir = job.dir / "work" / "frames" / "images"
    seg = job.dir / "work" / "segment"
    if not (seg / "masks").is_dir():
        raise StageError("missing segment/masks: run stage 'segment' first")
    has_filled = (seg / "filled").is_dir()
    for sub in ("maps",):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        (out / sub).mkdir()

    def read_mask(kind: str, stem: str):
        m = cv2.imread(str(seg / kind / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None and kind == "masks":
            raise StageError(f"missing mask {stem}.png: run stage 'segment' first")
        return m

    n = len(stems)
    # pass 1: tile energies (needed for the sliding-window reference)
    tiles = []
    for i, (nm, st) in enumerate(zip(names, stems)):
        gray = cv2.cvtColor(imread_rgb(img_dir / nm), cv2.COLOR_RGB2GRAY)
        tiles.append(tile_energies(gray, read_mask("masks", st), cfg.quality_tile))
        ev.progress(i + 1, 2 * n, f"sharpness {i + 1}/{n}")

    # pass 2: scores, maps, records
    k = max(0, cfg.quality_window)
    from ..multicam import cams_for
    cams = cams_for(meta, stems)
    records, smalls = [], []
    for i, (nm, st) in enumerate(zip(names, stems)):
        img = imread_rgb(img_dir / nm)
        mask = read_mask("masks", st)
        filled = read_mask("filled", st) if has_filled else None
        nb = [tiles[j] for j in range(max(0, i - k), min(n, i + k + 1)) if j != i and cams[j] == cams[i]]
        s_pix, _, med_r = sharpness_score(tiles[i], nb, cfg)
        bad, _ = bad_light_mask(img, cfg.specular_v, cfg.specular_s)
        q = trust_map(mask, s_pix, bad, filled)
        cv2.imwrite(str(out / "maps" / f"{st}.png"), q)
        m = mask > 0
        area = max(1, int(m.sum()))
        rec = {"stem": st,
               "trusted_frac": round(float((q[m] >= TRUST_KP).sum() / area), 4),
               "blur_frac": round(float((s_pix[m] < 0.5).sum() / area), 4),
               "specular_frac": round(float(bad[m].sum() / area), 4),
               "filled_frac": round(float(((filled[m] > 0).sum() if filled is not None else 0) / area), 4),
               "median_rel_sharpness": round(med_r, 3)}
        rec["excluded"] = bool(rec["trusted_frac"] < cfg.quality_min_frame or not m.any())
        records.append(rec)
        vis = visualize(img, mask, s_pix, bad, filled)
        smalls.append(cv2.resize(vis, (320, int(320 * vis.shape[0] / vis.shape[1])), interpolation=cv2.INTER_AREA))
        ev.progress(n + i + 1, 2 * n, f"frame {i + 1}/{n}")

    excluded = [r["stem"] for r in records if r["excluded"]]
    job.save_json("quality", "quality.json", {
        "frames": records, "excluded": excluded,
        "params": {k_: getattr(cfg, k_) for k_ in ("quality_tile", "quality_window", "quality_texture_floor",
                                                   "blur_rel_threshold", "specular_v", "specular_s",
                                                   "quality_min_frame")}})

    # debug artifacts
    tf = np.array([r["trusted_frac"] for r in records])
    worst = [int(i) for i in np.argsort(tf)[:8]]
    sel = sorted(set(pick_evenly(list(range(n)), 40)) - set(worst)) + worst
    labels = [f"{stems[i]} {100 * tf[i]:.0f}%" + (" X" if records[i]["excluded"] else "")
              + (" (worst)" if i in worst else "") for i in sel]
    imwrite_rgb(out / "overview.jpg", contact_sheet([smalls[i] for i in sel], cols=8, thumb_w=240, labels=labels), 85)
    ev.artifact(out / "overview.jpg", "trust maps: red blur, magenta specular/clipped, yellow filled holes "
                                      "(40 evenly spaced + 8 worst, label = % trusted)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(100 * tf, label="trusted %", lw=1.4)
    ax.plot([100 * r["blur_frac"] for r in records], label="blur %", lw=1, alpha=0.8)
    ax.plot([100 * r["specular_frac"] for r in records], label="specular/clipped %", lw=1, alpha=0.8)
    ax.plot([100 * r["filled_frac"] for r in records], label="filled %", lw=1, alpha=0.8)
    ax.axhline(100 * cfg.quality_min_frame, ls="--", c="gray", lw=1, label="quality_min_frame")
    ex = [i for i, r in enumerate(records) if r["excluded"]]
    if ex:
        ax.scatter(ex, 100 * tf[ex], c="red", s=14, zorder=3, label="excluded")
    ax.set_xlabel("selected frame")
    ax.set_ylabel("% of mask pixels")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title(f"trusted object pixels per frame ({len(ex)} of {n} excluded)")
    fig.tight_layout()
    fig.savefig(out / "trusted.png", dpi=110)
    plt.close(fig)
    ev.artifact(out / "trusted.png", "% trusted per frame (red: excluded from SfM)")

    rows = []
    for i in worst[:4]:
        img = imread_rgb(img_dir / names[i])
        q = cv2.imread(str(out / "maps" / f"{stems[i]}.png"), cv2.IMREAD_GRAYSCALE)
        qc = cv2.applyColorMap(q, cv2.COLORMAP_VIRIDIS)[..., ::-1]
        row = np.concatenate([img, qc], 1)
        cv2.putText(row, f"{stems[i]}: {100 * tf[i]:.0f}% trusted", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA))
    if rows:
        wmax = max(r.shape[1] for r in rows)         # cameras may differ in size (landscape / portrait)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for r in rows]
        imwrite_rgb(out / "worst.jpg", np.concatenate(rows, 0), 85)
        ev.artifact(out / "worst.jpg", "4 worst frames: image and trust map (bright = trusted)")

    stats = {"frames": n, "excluded": len(excluded), "excluded_stems": excluded[:100],
             "median_trusted": round(float(np.median(tf)), 4), "min_trusted": round(float(tf.min()), 4),
             "worst": [stems[i] for i in worst],
             "median_blur": round(float(np.median([r["blur_frac"] for r in records])), 4),
             "median_specular": round(float(np.median([r["specular_frac"] for r in records])), 4)}
    msg = f"median trusted {100 * np.median(tf):.0f} %, {len(excluded)} frames excluded"
    if len(excluded) > 0.25 * n:
        raise StageWarn(f"{len(excluded)} of {n} frames below quality_min_frame; {msg}", stats)
    stats["_msg"] = msg
    return stats
