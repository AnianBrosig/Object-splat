"""Time offset between two unsynchronised recordings of the same event.

Convention (same as Config.video2_offset): time in video B = time in video A + offset.

A coarse prior comes first: iPhones write the recording start time (whole seconds, phone clocks are network
synced) into the file. It is good to about +-2 s, which is enough to rule out every false peak further away. Then
two independent estimates are searched only in that window and cross-checked:
  1. audio   - GCC-PHAT cross-correlation of the two soundtracks (both phones hear the same room: hands, taps,
               speech). Resolution about 1 ms, so it is the preferred result when its peak is sharp.
  2. motion  - cross-correlation of the "how much is moving in the picture" signals (frame-difference energy at
               50 Hz). Both cameras see the same hands and object move at the same moments. Resolution about one
               frame, works without any sound.
The two should agree within a few frames. If they do, the audio value is used. If they do not, the one with the
clearer correlation peak wins and the disagreement is reported, so a wrong sync is never silent.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .util import find_tool

AUDIO_SR = 16000
MOTION_HZ = 50.0


@dataclass
class SyncResult:
    offset: float                      # seconds, time in B = time in A + offset
    method: str                        # audio | motion | manual
    confidence: float                  # peak-to-second-peak ratio of the winning correlation (>= 2 is clear)
    audio: Optional[float] = None
    audio_conf: float = 0.0
    motion: Optional[float] = None
    motion_conf: float = 0.0
    agree: Optional[bool] = None
    notes: list[str] = field(default_factory=list)
    curves: dict = field(default_factory=dict)      # lags + correlation values, for the plot
    prior: Optional[float] = None      # offset suggested by the start times in the file metadata


# --------------------------------------------------------------------------- helpers
def _ffmpeg() -> str:
    f = find_tool("ffmpeg")
    if not f:
        raise RuntimeError("ffmpeg not found")
    return f


def read_audio(path: Path, sr: int = AUDIO_SR) -> Optional[np.ndarray]:
    """Mono float32 samples, or None when the file has no audio stream."""
    cmd = [_ffmpeg(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0 or len(out.stdout) < sr * 0.5 * 4:
        return None
    return np.frombuffer(out.stdout, np.float32).copy()


def read_motion(path: Path, hz: float = MOTION_HZ, size: int = 96) -> np.ndarray:
    """Mean absolute frame difference (grayscale, size px wide) at `hz`, one value per 1/hz s from t = 0.
    ffmpeg's fps filter resamples to a constant rate, so VFR phone videos line up in time."""
    h = int(round(size * 9 / 16))
    cmd = [_ffmpeg(), "-v", "error", "-i", str(path), "-an", "-vf", f"fps={hz},scale={size}:{h}:flags=area,format=gray",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    # the video may have been rotated by ffmpeg (portrait): shape is whatever the scaler produced
    n = len(raw) // (size * h)
    fr = np.frombuffer(raw[:n * size * h], np.uint8).reshape(n, h, size).astype(np.float32) / 255.0
    d = np.abs(np.diff(fr, axis=0)).mean((1, 2))
    return np.concatenate([[0.0], d])


def _peak_conf(c: np.ndarray, i: int, guard: int) -> float:
    """Height of the peak over the best value outside +-guard samples around it (robust to a broad main lobe)."""
    lo, hi = max(0, i - guard), min(len(c), i + guard + 1)
    rest = np.concatenate([c[:lo], c[hi:]])
    second = float(rest.max()) if len(rest) else 1e-9
    return float(c[i] / max(second, 1e-9))


def _parabola(c: np.ndarray, i: int) -> float:
    """Sub-sample position of the peak at index i."""
    if i <= 0 or i >= len(c) - 1:
        return float(i)
    a, b, g = c[i - 1], c[i], c[i + 1]
    den = a - 2 * b + g
    return float(i) if abs(den) < 1e-12 else float(i + 0.5 * (a - g) / den)


# --------------------------------------------------------------------------- metadata prior
def start_time(path: Path) -> Optional[float]:
    """Recording start as a unix time (s) from the file's metadata, or None."""
    import json
    from datetime import datetime
    ffprobe = find_tool("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format_tags", "-of", "json", str(path)],
                             capture_output=True, text=True, check=True).stdout
        tags = json.loads(out).get("format", {}).get("tags", {})
    except Exception:  # noqa: BLE001
        return None
    for key in ("com.apple.quicktime.creationdate", "creation_time"):
        v = tags.get(key)
        if not v:
            continue
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00").replace("+0100", "+01:00")).timestamp() \
                if "+" in v[10:] or v.endswith("Z") else datetime.fromisoformat(v).timestamp()
        except ValueError:
            continue
    return None


def metadata_offset(path_a: Path, path_b: Path) -> Optional[float]:
    """Prior for the offset (time in B = time in A + offset) from the two start times, resolution about 1 s."""
    sa, sb = start_time(path_a), start_time(path_b)
    return None if sa is None or sb is None else float(sa - sb)


# --------------------------------------------------------------------------- audio
def gcc_phat(a: np.ndarray, b: np.ndarray, sr: int, max_lag: float, band=(150.0, 5000.0),
             window: Optional[tuple[float, float]] = None):
    """Lag of b relative to a (seconds, positive = the sound happens later in b) from the PHAT-weighted
    cross-correlation. Returns (offset, confidence, lags, correlation)."""
    n = 1
    while n < len(a) + len(b):
        n *= 2
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    G = B * np.conj(A)
    G /= np.maximum(np.abs(G), 1e-12)                 # PHAT: keep only the phase, sharp peaks even with reverb
    f = np.fft.rfftfreq(n, 1.0 / sr)
    G[(f < band[0]) | (f > band[1])] = 0
    cc = np.fft.irfft(G, n)
    m = int(max_lag * sr)
    cc = np.concatenate([cc[-m:], cc[:m + 1]])        # lags -m .. +m
    lags = np.arange(-m, m + 1) / sr
    c = np.abs(cc)
    if window is not None:                         # only lags inside [lo, hi] may win
        keep = (lags >= window[0]) & (lags <= window[1])
        c, lags = c[keep], lags[keep]
        m = int(round(-lags[0] * sr))
    i = int(np.argmax(c))
    conf = _peak_conf(c, i, guard=int(0.02 * sr))
    j = _parabola(c, i)
    return float((j - m) / sr), conf, lags, c


def audio_offset(path_a: Path, path_b: Path, max_lag: float = 30.0,
                 window: Optional[tuple[float, float]] = None) -> tuple[Optional[float], float, tuple]:
    a, b = read_audio(path_a), read_audio(path_b)
    if a is None or b is None:
        return None, 0.0, ()
    # normalise: phones differ a lot in gain and one of them may have wind/handling noise
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    off, conf, lags, c = gcc_phat(a, b, AUDIO_SR, max_lag, window=window)
    return off, conf, (lags, c)


# --------------------------------------------------------------------------- motion
def motion_offset(path_a: Path, path_b: Path, max_lag: float = 30.0, hz: float = MOTION_HZ,
                  window: Optional[tuple[float, float]] = None):
    ea, eb = read_motion(path_a, hz), read_motion(path_b, hz)

    def prep(e):
        e = np.sqrt(e)                                            # compress the huge hand-in/hand-out jumps
        k = int(hz * 1.5)                                         # remove slow drift (lighting, exposure)
        e = e - np.convolve(e, np.ones(k) / k, mode="same")
        s = np.convolve(e, np.ones(int(hz * 0.1) or 1) / int(hz * 0.1 or 1), mode="same")   # 0.1 s smoothing
        return (s - s.mean()) / (s.std() + 1e-9)
    a, b = prep(ea), prep(eb)
    n = 1
    while n < len(a) + len(b):
        n *= 2
    cc = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    m = min(int(max_lag * hz), len(a) + len(b) - 2)
    cc = np.concatenate([cc[-m:], cc[:m + 1]])
    # normalise by the overlap length so that big lags (short overlap) are not favoured
    lags_i = np.arange(-m, m + 1)
    overlap = np.maximum(1, np.minimum(len(a), len(b) - lags_i) - np.maximum(0, -lags_i))
    cc = cc / overlap
    if window is not None:
        keep = (lags_i / hz >= window[0]) & (lags_i / hz <= window[1])
        cc, lags_i = cc[keep], lags_i[keep]
        m = int(-lags_i[0])
    i = int(np.argmax(cc))
    conf = _peak_conf(cc, i, guard=int(0.5 * hz))
    j = _parabola(cc, i)
    return float((j - m) / hz), conf, (lags_i / hz, cc)


# --------------------------------------------------------------------------- combination
def estimate_offset(path_a: Path, path_b: Path, mode: str = "auto", max_lag: float = 30.0,
                    tol: float = 0.2, search: float = 4.0) -> SyncResult:
    """mode: auto (metadata prior, audio + motion cross-checked) | audio | motion.
    With a metadata prior the search is limited to prior +- `search` seconds."""
    res = SyncResult(offset=0.0, method="none", confidence=0.0)
    prior = metadata_offset(path_a, path_b)
    win = None
    if prior is not None:
        win = (prior - search, prior + search)
        res.prior = prior
        res.notes.append(f"start times in the files suggest {prior:+.0f} s (+-1.5 s); searching {win[0]:+.1f}..{win[1]:+.1f} s")
    else:
        res.notes.append("no start time in the metadata: searching the whole +-%.0f s range" % max_lag)
    if mode in ("auto", "audio"):
        try:
            off, conf, curve = audio_offset(path_a, path_b, max_lag, win)
        except Exception as e:  # noqa: BLE001
            off, conf, curve = None, 0.0, ()
            res.notes.append(f"audio failed: {e}")
        if off is None:
            res.notes.append("no usable audio track in one of the videos")
        else:
            res.audio, res.audio_conf = off, conf
            res.curves["audio"] = curve
    if mode in ("auto", "motion"):
        off, conf, curve = motion_offset(path_a, path_b, max_lag, window=win)
        res.motion, res.motion_conf = off, conf
        res.curves["motion"] = curve
    ca, cm = res.curves.get("audio"), res.curves.get("motion")
    if ca and cm and len(ca) and len(cm):
        off, conf = _joint(ca, cm, prior=res.prior)
        res.offset, res.method, res.confidence = off, "audio+motion", conf
        res.agree = (abs(res.audio - off) <= tol) and (abs(res.motion - off) <= tol)
        if res.agree:
            res.notes.append("audio and motion peaks agree within %.0f ms" % (1000 * max(abs(res.audio - off), abs(res.motion - off))))
        else:
            res.notes.append(f"individual peaks differ (audio {res.audio:+.2f} s, motion {res.motion:+.2f} s); the "
                             f"joint score picks {off:+.3f} s")
    elif ca and len(ca):
        res.offset, res.method, res.confidence = res.audio, "audio", res.audio_conf
    elif cm and len(cm):
        res.offset, res.method, res.confidence = res.motion, "motion", res.motion_conf
    if res.confidence < 3.0 and res.method != "none":
        res.notes.append("low confidence: check the side-by-side video before using this offset")
    if res.prior is not None and abs(res.offset - res.prior) > 2.0:
        res.notes.append(f"result is {abs(res.offset - res.prior):.1f} s from the metadata prior")
    return res


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / max(mad, 1e-9)


def _joint(ca: tuple, cm: tuple, tol: float = 0.03, guard: float = 0.3, zmax: float = 6.0,
           prior: Optional[float] = None) -> tuple[float, float]:
    """Offset that both correlation curves support. Each curve is turned into robust z-scores on the coarse
    (motion) lag grid, the audio curve max-pooled over +-tol first so its sharp 1 ms peak survives the coarser
    grid, clipped at `zmax` so that one very sharp PHAT peak (a loud tap that happens to repeat) cannot outvote the
    other evidence, plus a weak bump around the metadata prior. The sum is maximised; the audio peak near that point then sets the final value (about 1 ms).
    Returns (offset, confidence = z-score margin of the best lag over the best lag more than `guard` s away)."""
    la, a = ca
    lm, m = cm
    pool = np.empty(len(lm))
    j0 = 0
    for k, t in enumerate(lm):                                  # audio max-pooled onto the motion grid
        lo = np.searchsorted(la, t - tol, "left")
        hi = np.searchsorted(la, t + tol, "right")
        pool[k] = a[lo:hi].max() if hi > lo else 0.0
        j0 = hi
    z = np.minimum(_robust_z(pool), zmax) + np.minimum(_robust_z(m), zmax)
    if prior is not None:
        z = z + 3.0 * np.exp(-0.5 * ((lm - prior) / 1.5) ** 2)
    i = int(np.argmax(z))
    far = np.abs(lm - lm[i]) > guard
    conf = float(z[i] - (z[far].max() if far.any() else 0.0))
    t0 = float(lm[i])
    sel = (la >= t0 - 0.06) & (la <= t0 + 0.06)
    if sel.any() and a[sel].max() > np.percentile(a, 99):
        k = np.flatnonzero(sel)[int(np.argmax(a[sel]))]
        t0 = float(la[0] + _parabola(a, k) * (la[1] - la[0]))
    else:                                                       # no sharp audio peak: refine on the motion curve
        t0 = float(lm[0] + _parabola(m, int(np.argmin(np.abs(lm - t0)))) * (lm[1] - lm[0]))
    del j0
    return t0, conf


def plot_sync(res: SyncResult, path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in ("audio", "motion") if k in res.curves and len(res.curves[k])]
    fig, axes = plt.subplots(1, max(1, len(keys)), figsize=(6 * max(1, len(keys)), 3.2), squeeze=False)
    for ax, k in zip(axes[0], keys):
        lags, c = res.curves[k]
        ax.plot(lags, c, lw=0.8)
        v = getattr(res, k)
        ax.axvline(v, c="red", lw=0.8, ls="--")
        ax.set_title(f"{k}: {v:+.3f} s (peak ratio {getattr(res, k + '_conf'):.1f})", fontsize=9)
        ax.set_xlabel("offset (s)")
    fig.suptitle(title or f"sync: {res.offset:+.3f} s via {res.method}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def side_by_side(path_a: Path, path_b: Path, offset: float, out: Path, height: int = 540, fps: float = 25.0,
                 seconds: float = 0.0) -> None:
    """Both videos next to each other at the same moment, with audio from A. A visual check of the sync."""
    ff = _ffmpeg()
    ss_a, ss_b = max(0.0, -offset), max(0.0, offset)       # seek the one that starts earlier
    dur = f"-t {seconds}" if seconds else ""
    cmd = [ff, "-y", "-v", "error", "-ss", f"{ss_a:.3f}", "-i", str(path_a), "-ss", f"{ss_b:.3f}", "-i", str(path_b),
           "-filter_complex",
           f"[0:v]fps={fps},scale=-2:{height}[a];[1:v]fps={fps},scale=-2:{height}[b];[a][b]hstack=inputs=2:shortest=1[v]",
           "-map", "[v]", "-an", *([] if not seconds else ["-t", str(seconds)]), "-c:v", "libx264", "-crf", "20",
           "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    del dur
