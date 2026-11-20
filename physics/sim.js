// Physics of the shelf (Rapier, WASM): no DOM, no rendering, so the same code runs in the browser and in a Node test.
// Units: metres, kilograms, seconds. y is up, z points to the viewer (the open front of the shelf).
import RAPIER from './vendor/rapier.es.js';

const T = 0.018;            // board thickness
const FRAME = 0.02;         // side panel thickness
const STEP = 1 / 120;

let ready = null;
export function initPhysics() {
  if (!ready) ready = RAPIER.init();
  return ready;
}

const qx = (th) => ({ x: Math.sin(th / 2), y: 0, z: 0, w: Math.cos(th / 2) });
const rotX = (th, v) => {                       // rotate a vector about +x
  const c = Math.cos(th), s = Math.sin(th);
  return [v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c];
};
const qyaw = (a) => ({ x: 0, y: Math.sin(a / 2), z: 0, w: Math.cos(a / 2) });

/** Solid volume of a primitive description (for the density that gives the object its mass). */
function primVolume(p) {
  if (p.type === 'box') return 8 * p.half[0] * p.half[1] * p.half[2];
  if (p.type === 'sphere') return (4 / 3) * Math.PI * p.radius ** 3;
  return Math.PI * p.radius ** 2 * p.height;
}

export class ShelfSim {
  static async create(objects, opts = {}) {
    await initPhysics();
    return new ShelfSim(objects, opts);
  }

  /**
   * @param objects parsed *.physics.json files: {name, mass_kg, size_m:[x,y,z], primitives:[...], parts:[...]}
   * @param opts {boards, friction, restitution, seed, gap}
   */
  constructor(objects, opts = {}) {
    this.objects = objects;
    this.opts = Object.assign({ boards: 0, friction: 0.5, restitution: 0.12, seed: 1, gap: 0.035 }, opts);
    this.world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    this.world.timestep = STEP;
    this.angle = 0;           // current tilt, degrees (positive = the top leans toward the viewer)
    this.target = 0;
    this.speed = 20;          // degrees per second
    this.time = 0;
    this.items = [];          // {name, body, data}
    this.frame = [];          // kinematic boxes of the shelf: {name, body, rest:{p, size}}
    this._rng = mulberry32(this.opts.seed);
    this._buildShelf();
    this._addFloor();
    this.reset();
  }

  // ------------------------------------------------------------------ layout
  _layout() {
    const objs = this.objects;
    const nb = this.opts.boards || Math.max(2, Math.min(4, Math.ceil(objs.length / 2)));
    const rows = Array.from({ length: nb }, () => []);
    const order = [...objs.keys()].sort((a, b) => objs[b].size_m[0] - objs[a].size_m[0]);
    const used = new Array(nb).fill(0);
    for (const i of order) {
      const k = used.indexOf(Math.min(...used));
      rows[k].push(i);
      used[k] += objs[i].size_m[0] + this.opts.gap;
    }
    const W = Math.max(0.5, Math.max(...used) + 0.12);
    const D = Math.max(0.32, Math.max(...objs.map((o) => o.size_m[2])) + 0.14);
    const rowH = rows.map((r) => Math.max(0.1, ...r.map((i) => objs[i].size_m[1])) + 0.10);
    return { nb, rows, W, D, rowH };
  }

  _buildShelf() {
    const L = (this.layout = this._layout());
    const { W, D, rowH, nb } = L;
    const boardY = [];                       // top surface of each board
    let y = T;
    for (let k = 0; k < nb; k++) { boardY.push(y); y += rowH[k] + T; }
    const H = y;
    L.boardY = boardY; L.H = H;
    this.pivot = [0, 0, D / 2];              // the front-bottom edge the shelf tips over
    const add = (name, cx, cy, cz, sx, sy, sz) => {
      const body = this.world.createRigidBody(RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(cx, cy, cz));
      this.world.createCollider(RAPIER.ColliderDesc.cuboid(sx / 2, sy / 2, sz / 2)
        .setFriction(this.opts.friction).setRestitution(this.opts.restitution), body);
      this.frame.push({ name, body, rest: { p: [cx, cy, cz], size: [sx, sy, sz] } });
    };
    for (let k = 0; k <= nb; k++) {          // boards: bottom, between rows, top
      const top = k < nb ? boardY[k] : H;
      add(`board${k}`, 0, top - T / 2, 0, W, T, D);
    }
    add('left', -W / 2 - FRAME / 2, H / 2, 0, FRAME, H + T, D);
    add('right', W / 2 + FRAME / 2, H / 2, 0, FRAME, H + T, D);
    add('back', 0, H / 2, -D / 2 - FRAME / 2, W + 2 * FRAME, H + T, FRAME);
  }

  _addFloor() {
    const { W, D } = this.layout;
    const fb = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0, -0.5, 0));
    this.world.createCollider(RAPIER.ColliderDesc.cuboid(30, 0.5, 30).setFriction(0.6).setRestitution(0.1), fb);
    // low walls around the floor in front of the shelf: things pile up and bump into each other instead of
    // rolling away. The pen is W + 0.5 m wide and reaches 0.9 m in front of the shelf.
    const hw = W / 2 + 0.25, front = D / 2 + 0.9;
    const wall = (x, z, hx, hz) => this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(hx, 0.25, hz).setTranslation(x, 0.25, z).setFriction(0.3).setRestitution(0.2),
      this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed()));
    wall(-hw - 0.02, (front - D / 2) / 2, 0.02, (front + D / 2) / 2 + 0.1);
    wall(hw + 0.02, (front - D / 2) / 2, 0.02, (front + D / 2) / 2 + 0.1);
    wall(0, front + 0.02, hw + 0.04, 0.02);
    this.pen = { x0: -hw, x1: hw, z1: front };
  }

  // ------------------------------------------------------------------ shelf motion
  /** Pose of every shelf part for a tilt of `deg` degrees about the pivot (instant, for reset). */
  _poseFrame(deg, instant) {
    const th = (deg * Math.PI) / 180;
    const q = qx(th);
    for (const f of this.frame) {
      const r = [f.rest.p[0] - this.pivot[0], f.rest.p[1] - this.pivot[1], f.rest.p[2] - this.pivot[2]];
      const rr = rotX(th, r);
      const p = { x: rr[0] + this.pivot[0], y: rr[1] + this.pivot[1], z: rr[2] + this.pivot[2] };
      if (instant) { f.body.setTranslation(p, true); f.body.setRotation(q, true); }
      else { f.body.setNextKinematicTranslation(p); f.body.setNextKinematicRotation(q); }
    }
  }

  /** Aim the tilt at `deg` (0 = level). */
  tiltTo(deg, speed) {
    this.target = deg;
    if (speed) this.speed = speed;
  }

  // ------------------------------------------------------------------ objects
  reset() {
    for (const it of this.items) this.world.removeRigidBody(it.body);
    this.items = [];
    this.angle = 0; this.target = 0; this.time = 0;
    this._poseFrame(0, true);
    const { rows, boardY } = this.layout;
    rows.forEach((row, k) => {
      const widths = row.map((i) => this.objects[i].size_m[0]);
      const total = widths.reduce((a, b) => a + b, 0) + this.opts.gap * (row.length - 1);
      let x = -total / 2;
      row.forEach((i, j) => {
        const yaw = (this._rng() - 0.5) * 0.35;
        this._addObject(this.objects[i], [x + widths[j] / 2, boardY[k] + 0.003, (this._rng() - 0.5) * 0.03], yaw);
        x += widths[j] + this.opts.gap;
      });
    });
    return this.items;
  }

  /** Add a copy of objects[index] at a world position; returns the item. */
  spawn(index, pos, yaw = 0) {
    return this._addObject(this.objects[index], pos, yaw);
  }

  _addObject(o, pos, yaw) {
    const body = this.world.createRigidBody(RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(pos[0], pos[1], pos[2]).setRotation(qyaw(yaw))
      .setLinearDamping(0.05).setAngularDamping(0.4).setCcdEnabled(true));
    const prims = o.primitives && o.primitives.length ? o.primitives : null;
    const vol = prims ? prims.reduce((a, p) => a + primVolume(p), 0) : 0;
    const density = prims ? o.mass_kg / Math.max(vol, 1e-9) : 300;
    const make = (desc) => desc.setFriction(this.opts.friction).setRestitution(this.opts.restitution).setDensity(density);
    if (prims) {
      for (const p of prims) {
        let d;
        if (p.type === 'box') d = RAPIER.ColliderDesc.cuboid(p.half[0], p.half[1], p.half[2]);
        else if (p.type === 'sphere') d = RAPIER.ColliderDesc.ball(p.radius);
        else d = RAPIER.ColliderDesc.cylinder(p.height / 2, p.radius);     // Rapier cylinders run along Y
        d.setTranslation(p.center[0], p.center[1], p.center[2]);
        if (p.quat) d.setRotation({ x: p.quat[0], y: p.quat[1], z: p.quat[2], w: p.quat[3] });
        this.world.createCollider(make(d), body);
      }
    } else {
      for (const part of o.parts) {                     // general convex hulls
        const d = RAPIER.ColliderDesc.convexHull(new Float32Array(part.vertices));
        if (d) this.world.createCollider(make(d), body);
      }
    }
    const it = { name: o.name, body, data: o };
    this.items.push(it);
    return it;
  }

  // ------------------------------------------------------------------ stepping
  step(dt) {
    let t = 0;
    while (t < dt - 1e-9) {
      const d = Math.min(STEP, dt - t);
      const err = this.target - this.angle;
      let next = this.angle;
      if (Math.abs(err) > 1e-6) next = this.angle + Math.sign(err) * Math.min(Math.abs(err), this.speed * d);
      this._poseFrame(next, false);       // where the kinematic parts must be at the end of this step
      this.world.timestep = d;
      this.world.step();
      this.angle = next;
      this.time += d;
      t += d;
    }
  }

  /** A small random kick to every object (shake the shelf). */
  shake(strength = 0.6) {
    for (const it of this.items) {
      const m = it.body.mass();
      it.body.applyImpulse({ x: (this._rng() - 0.5) * strength * m, y: this._rng() * strength * 0.8 * m,
                             z: (this._rng() - 0.5) * strength * m }, true);
    }
  }

  // ------------------------------------------------------------------ queries (for tests and the UI)
  position(it) { const p = it.body.translation(); return [p.x, p.y, p.z]; }

  /** Position relative to the (tilted) shelf. */
  inShelf(it) {
    const p = this.position(it);
    const r = [p[0] - this.pivot[0], p[1] - this.pivot[1], p[2] - this.pivot[2]];
    const l = rotX((-this.angle * Math.PI) / 180, r);
    return [l[0] + this.pivot[0], l[1] + this.pivot[1], l[2] + this.pivot[2]];
  }

  onShelf(it) {
    const p = this.inShelf(it);
    const { W, D, H } = this.layout;
    return Math.abs(p[0]) < W / 2 + 0.01 && Math.abs(p[2]) < D / 2 + 0.02 && p[1] > 0 && p[1] < H;
  }
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
