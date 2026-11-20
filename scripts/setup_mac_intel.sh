#!/usr/bin/env bash
# Setup for Intel macOS (x86_64). Re-runnable. See README section 5.
set -euo pipefail
cd "$(dirname "$0")/.."

[ "$(uname -m)" = "x86_64" ] || { echo "expected x86_64, got $(uname -m) (Rosetta/arm64 shell?)"; exit 1; }
command -v brew >/dev/null || { echo "install Homebrew first"; exit 1; }
command -v ffmpeg >/dev/null || brew install ffmpeg

PY=""
for c in python3.12 python3.11 /usr/local/bin/python3.12 /usr/local/bin/python3.11; do
  if command -v "$c" >/dev/null; then PY="$(command -v "$c")"; break; fi
done
[ -n "$PY" ] || { brew install python@3.12; PY=/usr/local/bin/python3.12; }
echo "python: $PY"

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements-ml.txt
# Cutie/LightGlue pull numpy 2 / opencv 5 / new scipy; pin back to what torch 2.2.2 needs
.venv/bin/pip uninstall -y opencv-python || true
.venv/bin/pip install -r requirements-core.txt
.venv/bin/pip install --force-reinstall --no-deps "opencv-python-headless<4.12" "numpy<2"

# COLMAP CLI (optional; pycolmap is the default backend). Homebrew has no Intel bottle, conda-forge does.
if ! command -v colmap >/dev/null && [ ! -x "$HOME/miniforge3/envs/colmap/bin/colmap" ]; then
  if command -v conda >/dev/null; then conda create -y -p "$HOME/miniforge3/envs/colmap" -c conda-forge colmap || true
  else echo "skip colmap CLI (no conda); pycolmap backend will be used"; fi
fi

# Brush (no Intel-mac release binary; build from source, ~10-20 min)
if ! command -v brush_app >/dev/null && [ ! -x "$HOME/.cargo/bin/brush_app" ]; then
  if command -v cargo >/dev/null; then
    cargo install --git https://github.com/ArthurBrussee/brush --tag v0.3.0 brush-app --locked
  else echo "install Rust (https://rustup.rs) to build Brush"; fi
fi

.venv/bin/python -m gsturn doctor --fetch-weights
