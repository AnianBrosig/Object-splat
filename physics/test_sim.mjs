// Headless check of the shelf physics:  node physics/test_sim.mjs
import { readFileSync } from 'node:fs';
import { ShelfSim } from './sim.js';

const manifest = JSON.parse(readFileSync(new URL('./objects/manifest.json', import.meta.url)));
const objs = manifest.map((m) => JSON.parse(readFileSync(new URL('./' + m.file, import.meta.url))));
const sim = await ShelfSim.create(objs, { seed: 3 });
let fails = 0;
const check = (ok, msg) => { console.log((ok ? 'ok   ' : 'FAIL ') + msg); if (!ok) fails++; };
const say = (label) => { for (const it of sim.items) { const p = sim.position(it); console.log(`  ${label} ${it.name}: y ${p[1].toFixed(2)} z ${p[2].toFixed(2)} ${sim.onShelf(it) ? 'on shelf' : 'off'}`); } };

console.log(`shelf ${sim.layout.W.toFixed(2)} x ${sim.layout.H.toFixed(2)} x ${sim.layout.D.toFixed(2)} m, ${sim.layout.nb} boards, ${objs.length} objects`);
const t0 = Date.now();
// 1. settle on the level shelf
for (let i = 0; i < 180; i++) sim.step(1 / 60);          // 3 s
const speed = Math.max(...sim.items.map((it) => { const v = it.body.linvel(); return Math.hypot(v.x, v.y, v.z); }));
check(sim.items.every((it) => sim.onShelf(it)), 'all objects stay on the level shelf');
check(speed < 0.15, `objects come to rest (max speed ${speed.toFixed(3)} m/s)`);
check(Math.min(...sim.items.map((it) => sim.inShelf(it)[1])) > 0.0, 'nothing fell through a board');

// 2. tilt forward: friction 0.5 means sliding starts at about 27 degrees
sim.tiltTo(50, 25);
for (let i = 0; i < 60 * 14; i++) sim.step(1 / 60);      // 14 s
say('after tilt');
const off = sim.items.filter((it) => !sim.onShelf(it)).length;
check(Math.abs(sim.angle - 50) < 0.1, `shelf reached ${sim.angle.toFixed(1)} deg`);
check(off === sim.items.length, `all ${sim.items.length} objects left the shelf (${off})`);
check(sim.items.every((it) => sim.position(it)[1] < 0.15), 'all objects reached the floor');
check(sim.items.every((it) => sim.position(it)[2] > -0.05), 'objects landed in front of the shelf');
check(sim.items.every((it) => sim.position(it)[2] < sim.pen.z1 && Math.abs(sim.position(it)[0]) < sim.pen.x1), 'objects stay inside the walled floor area');
let minGap = 1e9;
for (let i = 0; i < sim.items.length; i++) for (let j = i + 1; j < sim.items.length; j++) {
  const a = sim.position(sim.items[i]), b = sim.position(sim.items[j]);
  minGap = Math.min(minGap, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
}
check(minGap > 0.03 && minGap < 0.6, `no two objects sit inside each other (min centre distance ${minGap.toFixed(3)} m)`);
check(sim.items.every((it) => sim.position(it).every(Number.isFinite)), 'positions are finite');

// 3. level again, reset, and spawn an extra object
sim.tiltTo(0, 40);
for (let i = 0; i < 60 * 5; i++) sim.step(1 / 60);
check(Math.abs(sim.angle) < 0.1, 'shelf levels again');
sim.reset();
for (let i = 0; i < 120; i++) sim.step(1 / 60);
check(sim.items.every((it) => sim.onShelf(it)), 'reset puts every object back on the shelf');
const extra = sim.spawn(0, [0, 1.0, 0.1]);
for (let i = 0; i < 120; i++) sim.step(1 / 60);
check(sim.position(extra)[1] < 0.8, 'a spawned object falls');
const sec = (Date.now() - t0) / 1000;
console.log(`simulated ${(3 + 14 + 5 + 2 + 2)} s of physics in ${sec.toFixed(1)} s of wall time`);
process.exit(fails ? 1 : 0);
