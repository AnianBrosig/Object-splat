// Browser viewer for trained Gaussian splats (WebGL, gaussian-splats-3d vendored in /static/vendor).
// Loaded on demand by app.js so that nothing heavy runs until the viewer is opened.
import * as GS from "/static/vendor/gaussian-splats-3d.module.js";

let viewer = null;

export function dispose() {
  if (viewer) { try { viewer.dispose(); } catch (e) { /* already gone */ } viewer = null; }
}

export function webglOk() {
  try { return !!document.createElement("canvas").getContext("webgl2"); } catch (e) { return false; }
}

/** Show `url` in `box`. view = {centre, radius, up} from /api/jobs/<id>/splats. */
export async function show(box, url, view, setMsg) {
  dispose();
  box.replaceChildren();
  if (!webglOk()) { setMsg("WebGL 2 is not available in this browser"); return false; }
  const root = document.createElement("div");
  root.style.cssText = "position:absolute;inset:0";
  box.appendChild(root);
  const c = view.centre, r = Math.max(view.radius, 1e-3), up = view.up || [0, -1, 0];
  // start in front of the object, slightly above, at about 3 radii
  const dir = [0.35, -0.45, -1.0];
  const n = Math.hypot(...dir);
  const pos = [c[0] + (dir[0] / n) * r * 3.0, c[1] + (dir[1] / n) * r * 3.0, c[2] + (dir[2] / n) * r * 3.0];
  viewer = new GS.Viewer({
    rootElement: root, cameraUp: up, initialCameraPosition: pos, initialCameraLookAt: c,
    sharedMemoryForWorkers: false,           // no COOP/COEP headers needed
    gpuAcceleratedSort: false, selfDrivenMode: true, useBuiltInControls: true,
  });
  await viewer.addSplatScene(url, { splatAlphaRemovalThreshold: 5, showLoadingUI: true, progressiveLoad: false });
  viewer.start();
  return true;
}
