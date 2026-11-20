// Browser front end for the shelf: three.js draws what sim.js (Rapier) simulates.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { ShelfSim } from './sim.js';

const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);

async function main() {
  const manifest = await (await fetch('objects/manifest.json')).json();
  if (!manifest.length) throw new Error('no objects: run  python -m gsturn shelf --build');
  const datas = await Promise.all(manifest.map(async (m) => (await fetch(m.file)).json()));
  const sim = await ShelfSim.create(datas, { seed: +(q.get('seed') || 3) });

  // ---------------------------------------------------------------- scene
  const renderer = new THREE.WebGLRenderer({ canvas: $('view'), antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x15171b);
  scene.fog = new THREE.Fog(0x15171b, 4, 9);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 30);
  camera.position.set(1.05, 0.95, 2.1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.32, 0.55);
  controls.update();
  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x3a3326, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(1.6, 3.0, 2.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: 0.5, far: 9 });
  scene.add(sun);

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshStandardMaterial({ color: 0x8a8f99, roughness: 0.95 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);
  scene.add(new THREE.GridHelper(30, 120, 0x555a66, 0x3b3f49));

  // pen walls
  const { x0, x1, z1 } = sim.pen;
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x6b7280, roughness: 0.8, transparent: true, opacity: 0.55 });
  const addWall = (cx, cz, sx, sz) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(sx, 0.5, sz), wallMat);
    m.position.set(cx, 0.25, cz); m.castShadow = m.receiveShadow = true; scene.add(m);
  };
  const D = sim.layout.D;
  addWall(x0 - 0.02, (z1 - D / 2) / 2, 0.04, z1 + D / 2 + 0.2);
  addWall(x1 + 0.02, (z1 - D / 2) / 2, 0.04, z1 + D / 2 + 0.2);
  addWall(0, z1 + 0.02, x1 - x0 + 0.08, 0.04);

  // shelf parts follow their kinematic bodies
  const woodMat = new THREE.MeshStandardMaterial({ color: 0xb98d5a, roughness: 0.7 });
  const shelfMeshes = sim.frame.map((f) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(...f.rest.size), woodMat);
    m.castShadow = m.receiveShadow = true;
    scene.add(m);
    return { m, f };
  });

  // objects: one geometry per object type, coloured by the splat
  const geoms = datas.map((d) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(d.visual.positions, 3));
    const c = d.visual.colors, lin = new Float32Array(c.length), col = new THREE.Color();
    for (let i = 0; i < c.length; i += 3) {
      col.setRGB(c[i], c[i + 1], c[i + 2], THREE.SRGBColorSpace);
      lin[i] = col.r; lin[i + 1] = col.g; lin[i + 2] = col.b;
    }
    g.setAttribute('color', new THREE.BufferAttribute(lin, 3));
    g.setIndex(d.visual.indices);
    g.computeVertexNormals();
    return g;
  });
  const objMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75, metalness: 0.0, side: THREE.DoubleSide });
  const meshes = new Map();
  const sync = () => {
    for (const it of sim.items) {
      let m = meshes.get(it);
      if (!m) {
        m = new THREE.Mesh(geoms[datas.indexOf(it.data)], objMat);
        m.castShadow = m.receiveShadow = true;
        scene.add(m); meshes.set(it, m);
      }
      const p = it.body.translation(), r = it.body.rotation();
      m.position.set(p.x, p.y, p.z); m.quaternion.set(r.x, r.y, r.z, r.w);
    }
    for (const [it, m] of [...meshes]) if (!sim.items.includes(it)) { scene.remove(m); meshes.delete(it); }
    for (const { m, f } of shelfMeshes) {
      const p = f.body.translation(), r = f.body.rotation();
      m.position.set(p.x, p.y, p.z); m.quaternion.set(r.x, r.y, r.z, r.w);
    }
  };

  // ---------------------------------------------------------------- ui
  let tilted = false;
  const setTilt = (on) => {
    tilted = on;
    sim.tiltTo(on ? +$('max').value : 0, +$('spd').value);
    $('tilt').textContent = on ? 'Level shelf' : 'Tilt shelf';
    $('tilt').classList.toggle('on', on);
  };
  $('tilt').onclick = () => setTilt(!tilted);
  $('reset').onclick = () => { sim.reset(); setTilt(false); };
  $('shake').onclick = () => sim.shake(0.7);
  $('drop').onclick = () => {
    const i = Math.floor(Math.random() * datas.length);
    sim.spawn(i, [(Math.random() - 0.5) * 0.4, 0.9 + Math.random() * 0.3, sim.layout.D / 2 + 0.25 + Math.random() * 0.3],
      Math.random() * 6);
  };
  for (const [id, out] of [['max', 'maxv'], ['spd', 'spdv'], ['ts', 'tsv']]) {
    $(id).oninput = () => { $(out).textContent = (+$(id).value).toFixed(id === 'ts' ? 1 : 0); if (tilted) setTilt(true); };
  }
  $('objs').innerHTML = manifest.map((m) => `<div><b>${m.name}</b> ${(m.size_m[0] * 100).toFixed(0)} cm, ${m.mass_kg} kg<br>` +
    m.downloads.map((d) => `<a href="${d}" download>${d.split('.').pop().toUpperCase()}</a>`).join('') + '</div>').join('');

  // ---------------------------------------------------------------- loop
  const resize = () => {
    const w = innerWidth, h = innerHeight;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  addEventListener('resize', resize); resize();

  // test / screenshot helpers: ?tilt=50&ff=8 fast-forwards the simulation before the first frame
  if (q.get('tilt')) { $('max').value = q.get('tilt'); $('maxv').textContent = q.get('tilt'); setTilt(true); }
  const ff = +(q.get('ff') || 0);
  if (ff) { for (let i = 0; i < ff * 60; i++) sim.step(1 / 60); }
  document.title = 'Object shelf (ready)';

  let last = performance.now();
  const frame = (now) => {
    const dt = Math.min(0.05, (now - last) / 1000) * +$('ts').value;
    last = now;
    sim.step(dt);
    sync();
    $('angle').textContent = sim.angle.toFixed(1) + '°';
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  };
  sync();
  requestAnimationFrame(frame);
}

main().catch((e) => { $('err').textContent = String((e && e.stack) || e); document.title = 'Object shelf (error)'; });
