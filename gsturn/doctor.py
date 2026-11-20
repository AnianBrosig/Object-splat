"""Toolchain report: binaries, Python packages, torch devices, GPU, ffmpeg encoders, warnings."""
from __future__ import annotations

import importlib
import importlib.metadata as md
import platform
import subprocess
import sys
from typing import Any

from .util import find_tool

BINARIES = ["ffmpeg", "ffprobe", "colmap", "cargo", "brush", "opensplat"]
PACKAGES = ["numpy", "opencv-python-headless", "opencv-python", "fastapi", "uvicorn", "matplotlib",
            "torch", "torchvision", "kornia", "lightglue", "cutie", "pycolmap"]


def _run(cmd: list[str], timeout: float = 15) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def _version(name: str, path: str) -> str:
    if name in ("ffmpeg", "ffprobe"):
        out = _run([path, "-version"])
    elif name == "colmap":
        out = _run([path, "help"])
    else:
        out = _run([path, "--version"])
    return out.strip().splitlines()[0] if out.strip() else ""


def report(fetch_weights: bool = False) -> dict[str, Any]:
    r: dict[str, Any] = {
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "mac_ver": platform.mac_ver()[0], "python": sys.version.split()[0],
                     "executable": sys.executable},
        "binaries": {}, "packages": {}, "torch": {}, "gpu": [], "ffmpeg_encoders": [], "warnings": [],
        "backends": {},
    }
    w = r["warnings"]
    for b in BINARIES:
        p = find_tool(b) or (find_tool("brush_app") if b == "brush" else None)
        r["binaries"][b] = {"path": p, "version": _version(b, p) if p else ""}
    for pkg in PACKAGES:
        try:
            r["packages"][pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            r["packages"][pkg] = None

    # torch
    try:
        import torch
        t = {"version": torch.__version__, "cuda": torch.cuda.is_available(),
             "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
             "threads": torch.get_num_threads()}
        r["torch"] = t
    except Exception as e:  # noqa: BLE001
        r["torch"] = {"error": str(e)}

    # import checks for ML packages (installed != importable)
    imports = {}
    for mod in ["lightglue", "cutie", "pycolmap", "kornia"]:
        try:
            importlib.import_module(mod)
            imports[mod] = True
        except Exception as e:  # noqa: BLE001
            imports[mod] = f"{type(e).__name__}: {e}"
    r["imports"] = imports

    # GPU
    if platform.system() == "Darwin":
        out = _run(["system_profiler", "SPDisplaysDataType"], timeout=30)
        cur: dict[str, str] = {}
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("Chipset Model:"):
                cur = {"model": s.split(":", 1)[1].strip()}
                r["gpu"].append(cur)
            elif ":" in s and cur is not None and any(s.startswith(k) for k in ("VRAM", "Metal", "Vendor")):
                k, v = s.split(":", 1)
                cur[k.strip()] = v.strip()

    ff = r["binaries"]["ffmpeg"]["path"]
    if ff:
        enc = _run([ff, "-hide_banner", "-encoders"])
        r["ffmpeg_encoders"] = [ln.split()[1] for ln in enc.splitlines()
                                if any(k in ln for k in ("videotoolbox", "libx264", "nvenc", "qsv"))
                                and len(ln.split()) > 1]

    # warnings
    plat = r["platform"]
    if plat["system"] == "Darwin" and plat["machine"] != "x86_64":
        w.append(f"machine is {plat['machine']}: this setup targets Intel macOS (x86_64)")
    pyv = tuple(int(x) for x in plat["python"].split(".")[:2])
    if plat["system"] == "Darwin" and plat["machine"] == "x86_64" and pyv > (3, 12):
        w.append("Python > 3.12: no torch wheels for Intel macOS; use Python 3.11/3.12")
    np_v = r["packages"].get("numpy")
    if np_v and np_v.startswith("2") and (r["torch"].get("version", "").startswith("2.2")):
        w.append("numpy 2.x with torch 2.2: install numpy<2")
    if not ff:
        w.append("ffmpeg missing: brew install ffmpeg")
    if not r["binaries"]["colmap"]["path"] and imports.get("pycolmap") is not True:
        w.append("no SfM backend: pip install pycolmap, or install colmap")
    if imports.get("cutie") is not True:
        w.append("Cutie not importable: segmentation falls back to grabcut")
    if imports.get("lightglue") is not True:
        w.append("LightGlue not importable: features fall back to SIFT")
    if not r["binaries"]["brush"]["path"]:
        w.append("brush missing: training backend 'brush' unavailable (see README section 5)")
    if r["torch"].get("mps") is False and plat["system"] == "Darwin":
        w.append("torch reports no MPS; device=mps unavailable")

    r["backends"] = {
        "seg": (["cutie"] if imports.get("cutie") is True else []) + ["grabcut"],
        "features": (["superpoint", "aliked", "disk"] if imports.get("lightglue") is True else []) + ["sift"],
        "sfm": (["pycolmap"] if imports.get("pycolmap") is True else [])
               + (["colmap-cli"] if r["binaries"]["colmap"]["path"] else []),
        "train": (["brush"] if r["binaries"]["brush"]["path"] else [])
                 + (["opensplat"] if r["binaries"]["opensplat"]["path"] else []) + ["none"],
        "device": ["cpu"] + (["mps"] if r["torch"].get("mps") else []) + (["cuda"] if r["torch"].get("cuda") else []),
    }

    if fetch_weights:
        r["weights"] = fetch_all_weights()
    return r


def fetch_all_weights() -> dict[str, str]:
    out = {}
    try:
        from .stages.features import load_extractor_and_matcher
        for name in ("superpoint", "aliked", "disk"):
            try:
                load_extractor_and_matcher(name, 512, "cpu")
                out[name] = "ok"
            except Exception as e:  # noqa: BLE001
                out[name] = f"error: {e}"
    except Exception as e:  # noqa: BLE001
        out["lightglue"] = f"error: {e}"
    try:
        from .stages.segment import load_cutie
        load_cutie("cpu", 480)
        out["cutie"] = "ok"
    except Exception as e:  # noqa: BLE001
        out["cutie"] = f"error: {e}"
    return out


def print_report(r: dict[str, Any]) -> None:
    p = r["platform"]
    print(f"platform  {p['system']} {p['mac_ver']} {p['machine']}  python {p['python']} ({p['executable']})")
    for g in r["gpu"]:
        print(f"gpu       {g.get('model')}  {g.get('VRAM (Total)', g.get('VRAM (Dynamic, Max)', ''))}  "
              f"{g.get('Metal Support', '')}")
    print("binaries")
    for k, v in r["binaries"].items():
        print(f"  {k:<10} {v['path'] or 'MISSING':<55} {v['version'][:40]}")
    print("packages")
    for k, v in r["packages"].items():
        if v or k != "opencv-python":
            print(f"  {k:<24} {v or '-'}")
    print(f"torch     {r['torch']}")
    print(f"imports   {r['imports']}")
    print(f"encoders  {', '.join(r['ffmpeg_encoders'])}")
    print("backends")
    for k, v in r["backends"].items():
        print(f"  {k:<9} {', '.join(v)}")
    if r.get("weights"):
        print(f"weights   {r['weights']}")
    if r["warnings"]:
        print("warnings")
        for x in r["warnings"]:
            print(f"  ! {x}")
    else:
        print("no warnings")
