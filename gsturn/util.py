"""Small shared helpers: tool discovery, subprocess with live output, image IO."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Callable, Optional

HOME = Path.home()
EXTRA_BIN_DIRS = [
    HOME / ".cargo" / "bin",
    HOME / "miniforge3" / "envs" / "colmap" / "bin",
    HOME / "mambaforge" / "envs" / "colmap" / "bin",
    HOME / "miniconda3" / "envs" / "colmap" / "bin",
    Path("/usr/local/bin"),
    Path("/opt/homebrew/bin"),
]


def find_tool(name: str, override: str = "") -> Optional[str]:
    if override:
        return override if Path(override).exists() or shutil.which(override) else None
    p = shutil.which(name)
    if p:
        return p
    for d in EXTRA_BIN_DIRS:
        c = d / name
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return None


class Cancelled(Exception):
    pass


def run_streaming(cmd: list[str], on_line: Optional[Callable[[str], None]] = None,
                  cwd: Optional[Path] = None, env: Optional[dict] = None) -> int:
    """Run a command, feeding merged stdout/stderr lines (split on \\n and \\r) to on_line."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=env,
                            bufsize=0)
    buf = b""
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                idx = min([i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0], default=-1)
                if idx < 0:
                    break
                line, buf = buf[:idx], buf[idx + 1:]
                if on_line and line.strip():
                    on_line(line.decode("utf-8", "replace"))
        if buf.strip() and on_line:
            on_line(buf.decode("utf-8", "replace"))
        return proc.wait()
    except BaseException:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


def imread_rgb(path: Path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: Path, img, quality: int = 92) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if str(path).lower().endswith((".jpg", ".jpeg")) else []
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), params)


def contact_sheet(images: list, cols: int = 8, thumb_w: int = 240, labels: Optional[list[str]] = None):
    """Grid of RGB uint8 images."""
    import cv2
    import numpy as np
    if not images:
        return np.zeros((10, 10, 3), np.uint8)
    h0, w0 = images[0].shape[:2]
    th = max(1, int(round(thumb_w * h0 / w0)))
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * th, cols * thumb_w, 3), 30, np.uint8)
    for i, im in enumerate(images):
        t = cv2.resize(im, (thumb_w, th), interpolation=cv2.INTER_AREA)
        if labels:
            cv2.putText(t, labels[i], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(t, labels[i], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * thumb_w:(c + 1) * thumb_w] = t
    return sheet


def pick_evenly(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    import numpy as np
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]
