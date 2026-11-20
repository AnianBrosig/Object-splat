"""The shelf playground: export the finished objects as meshes and serve the physics page (physics/).

physics/objects.json lists which jobs to use and how big the real objects are. `build` writes, for each object,
STL / OBJ / PLY (with vertex colours) and a *.physics.json for the browser into physics/objects/, plus the
manifest.json the page reads. `serve` hosts the folder (browsers do not load local JSON from file://).
"""
from __future__ import annotations

import json
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import JOBS_DIR, ROOT

PHYSICS = ROOT / "physics"


def find_splat(job_id: str) -> Path | None:
    d = JOBS_DIR / job_id
    for rel in ("outputs/splat.ply", "work/train/splat.ply"):
        if (d / rel).exists():
            return d / rel
    return None


def build(force: bool = False, log=print, only: list[str] | None = None) -> list[dict]:
    from .mesh import export_all
    cfg = json.loads((PHYSICS / "objects.json").read_text())
    out = PHYSICS / "objects"
    out.mkdir(exist_ok=True)
    manifest = []
    for e in cfg["objects"]:
        slug = "".join(c if c.isalnum() else "-" for c in e["name"].lower()).strip("-")
        if only and slug not in only and e["name"] not in only:
            continue
        ply = find_splat(e["job"])
        if ply is None:
            log(f"skip {e['name']}: no splat in job {e['job']}")
            continue
        target = out / f"{slug}.physics.json"
        stamp = out / f"{slug}.info.json"
        same = stamp.exists() and json.loads(stamp.read_text()).get("size_cm") == e["size_cm"] and \
            json.loads(stamp.read_text()).get("density") == e.get("density", 350)
        if not force and target.exists() and target.stat().st_mtime > ply.stat().st_mtime and same:
            info = json.loads(stamp.read_text())
            log(f"{e['name']}: up to date")
        else:
            log(f"{e['name']}: {ply}")
            info = export_all(ply, out, slug, e["size_cm"] / 100.0, density=e.get("density", 350), log=log)
            info.update({"size_cm": e["size_cm"], "density": e.get("density", 350), "job": e["job"]})
            stamp.write_text(json.dumps(info, indent=1))
        manifest.append({"name": e["name"], "file": f"objects/{slug}.physics.json", "mass_kg": info["mass_kg"],
                         "size_m": info["size_m"], "solidity": info["solidity"], "parts": info["convex_parts"],
                         "downloads": [f"objects/{slug}{x}" for x in (".stl", ".obj", ".ply")]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *a):   # quiet
        pass


def serve(port: int = 8765, open_browser: bool = True) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(PHYSICS)))
    url = f"http://127.0.0.1:{port}/"
    print(f"shelf playground at {url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
