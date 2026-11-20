"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) e.append(k.nodeType ? k : String(k));
  return e;
};
const api = async (url, opts = {}) => {
  const r = await fetch(url, opts);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch {} throw new Error(m); }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
};
const fmtT = s => s == null ? "" : s < 60 ? `${s.toFixed(0)}s` : `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, "0")}`;

// stage order comes from the server (GET /api/stages, and "order" in the job state)
let STAGES = [];
const MAIN_PARAMS = ["trim_start", "target_height", "max_frames", "seg_backend", "feature_backend", "max_keypoints",
  "sfm_backend", "train_backend", "train_steps", "device"];
const CHOICES = {
  seg_backend: ["cutie", "grabcut"], feature_backend: ["superpoint", "aliked", "disk", "sift"],
  sfm_backend: ["pycolmap", "colmap-cli"], train_backend: ["brush", "none", "opensplat"],
  device: ["auto", "cpu", "mps", "cuda"], hwaccel: ["auto", "videotoolbox", "none"],
  train_mask_mode: ["weighted", "alpha", "ignore", "ignore_quality"], scheduler: ["resource", "job", "off"], refine_mode: ["remap", "triangulate"],
  refine_focal: ["auto", "true", "false"],
};
const HIDDEN = new Set(["video", "object_box", "hand_boxes", "bg_boxes"]);

const S = {
  jobId: "", defaults: {}, config: {}, stages: {}, events: [], running: false, es: null,
  boxes: { object: null, hands: [], bg: [] }, hist: [], boxMode: "object", img: null, galleryTab: "", debugTab: "log",
  selectedVideo: "", doctor: null, queue: [], batch: { items: [], running: false },
};

// ---------------------------------------------------------------- state from events (mirrors events.summarize)
function emptyStages() {
  const o = {};
  for (const s of STAGES) o[s] = { status: "pending", pct: 0, msg: "", artifacts: [], stats: {}, started: null, ended: null };
  return o;
}
function applyEvent(ev) {
  S.events.push(ev);
  if (S.events.length > 3000) S.events.splice(0, 1000);
  const s = S.stages[ev.stage];
  if (ev.type === "run_end") { S.running = false; return; }
  if (!s) return;
  if (ev.type === "stage") {
    if (ev.status === "running") {
      Object.assign(s, { status: "running", pct: 0, msg: ev.msg || "", started: ev.t, ended: null });
      s.artifacts = [];
      s.stats = {};
      S.running = true;
    } else {
      s.status = ev.status; s.ended = ev.t;
      if (["done", "warn", "skipped"].includes(ev.status)) s.pct = 100;
      if (ev.msg !== undefined) s.msg = ev.msg;
      if (ev.stats) s.stats = ev.stats;
    }
  } else if (ev.type === "progress") {
    s.pct = ev.pct; if (ev.msg) s.msg = ev.msg;
  } else if (ev.type === "artifact") {
    s.artifacts = s.artifacts.filter(a => a.path !== ev.path);
    s.artifacts.push({ path: ev.path, caption: ev.caption || "", kind: ev.kind || "image", t: ev.t });
  }
}

// ---------------------------------------------------------------- params form
function paramInput(name, value) {
  const def = S.defaults[name];
  let input;
  if (name === "refine_focal") {   // tri-state: null = auto
    const cur = value === null || value === undefined ? "auto" : String(value);
    input = h("select", { "data-k": name, "data-type": "tribool" }, CHOICES[name].map(c => h("option", { value: c, selected: c === cur }, c)));
  } else if (CHOICES[name]) {
    input = h("select", { "data-k": name }, CHOICES[name].map(c => h("option", { value: c, selected: c === value }, c)));
  } else if (typeof def === "boolean") {
    input = h("select", { "data-k": name, "data-type": "bool" },
      h("option", { value: "true", selected: value === true }, "true"), h("option", { value: "false", selected: value === false }, "false"));
  } else if (typeof def === "number") {
    input = h("input", { type: "number", step: "any", value, "data-k": name, "data-type": "num" });
  } else {
    input = h("input", { type: "text", value: value ?? "", "data-k": name });
  }
  return h("label", {}, name, input);
}
function renderParams() {
  const cfg = { ...S.defaults, ...S.config };
  $("#params").replaceChildren(...MAIN_PARAMS.map(k => paramInput(k, cfg[k])));
  $("#paramsAll").replaceChildren(...Object.keys(S.defaults).filter(k => !HIDDEN.has(k) && !MAIN_PARAMS.includes(k))
    .map(k => paramInput(k, cfg[k])));
}
function readParams() {
  const out = {};
  for (const el of $$("[data-k]", $("#inputSec"))) {
    const k = el.dataset.k; let v = el.value;
    if (el.dataset.type === "num") v = v === "" ? S.defaults[k] : Number(v);
    else if (el.dataset.type === "bool") v = v === "true";
    else if (el.dataset.type === "tribool") v = v === "auto" ? null : v === "true";
    if (JSON.stringify(v) !== JSON.stringify(S.defaults[k])) out[k] = v;
  }
  return out;
}

// ---------------------------------------------------------------- input / job creation
async function loadVideos() {
  const vids = await api("/api/videos");
  const list = $("#videoList");
  list.replaceChildren(...vids.map(v => h("label", {},
    h("input", { type: "radio", name: "video", value: v.name, checked: v.path === S.selectedVideo,
      onchange: () => selectVideo(v.path) }),
    h("span", {}, v.name), h("span", { class: "muted" }, `${v.mb} MB`))));
  if (!vids.length) list.append(h("span", { class: "muted" }, "no videos in videos/"));
}
// choosing a video (or changing trim_start) loads its first frame, so the boxes can be drawn before processing
function selectVideo(path) {
  // picking a video while a job is open (the page reopens the last job from the URL hash) starts a new job
  const keep = !S.jobId && $("#keepBoxes").checked;
  if (S.jobId) openJob("");
  if (path !== S.selectedVideo && !keep) clearBoxes();
  S.selectedVideo = path;
  $$("input[name=video]").forEach(r => r.checked = r.value === path.split("/").pop());
  S.img = null;
  loadPreview();
  updateStart();
}
function loadPreview() {
  if (S.jobId || !S.selectedVideo) { updateBoxSec(); return; }
  const trim = Number(($("[data-k=trim_start]") || {}).value) || 0;
  const url = `/api/preview-frame?path=${encodeURIComponent(S.selectedVideo)}&t=${trim}`;
  updateBoxSec();
  $("#boxInfo").textContent = "loading first frame...";
  const img = new Image();
  S.img = "loading";
  img.onload = () => {
    if (S.img !== "loading" || S.jobId) return;
    setBoxImage(img);
    $("#boxSec").scrollIntoView({ behavior: "smooth", block: "nearest" });
  };
  img.onerror = async () => {
    S.img = null;
    let m = "could not load the first frame";
    try { m = (await api(url)).detail || m; } catch (e) { m = e.message; }
    $("#boxInfo").textContent = m;
  };
  img.src = url;
}
function updateStart() {
  const b = $("#startBtn");
  b.disabled = !S.selectedVideo || S.uploading;
  b.textContent = S.boxes.object ? "Start full pipeline (downscale → segment → …)" : "Start (downscale + frame selection only)";
  b.title = S.boxes.object ? "boxes are drawn: run everything up to the stage chosen on the right"
    : "no object box yet: runs downscale + frames, then you draw the box on the selected frame";
  $("#inputMsg").textContent = S.uploading ? "uploading..." : S.selectedVideo ? S.selectedVideo.split("/").pop() : "";
}
function setupDrop() {
  const d = $("#drop");
  ["dragenter", "dragover"].forEach(t => d.addEventListener(t, e => { e.preventDefault(); d.classList.add("over"); }));
  ["dragleave", "drop"].forEach(t => d.addEventListener(t, e => { e.preventDefault(); d.classList.remove("over"); }));
  d.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) pickFile(f); });
  $("#fileInput").addEventListener("change", e => { const f = e.target.files[0]; if (f) pickFile(f); });
  $("#inputSec").addEventListener("change", e => { if (!S.jobId && e.target.dataset && e.target.dataset.k === "trim_start") { S.img = null; loadPreview(); } });
}
// uploads go to videos/ right away, so the first frame is available for the box tool before the job starts
function pickFile(f) {
  const fd = new FormData();
  fd.append("file", f);
  const bar = $("#uploadBar");
  S.uploading = true; updateStart(); bar.classList.remove("hidden");
  const x = new XMLHttpRequest();
  x.open("POST", "/api/upload");
  x.upload.onprogress = e => { if (e.lengthComputable) bar.firstElementChild.style.width = `${100 * e.loaded / e.total}%`; };
  x.onload = async () => {
    S.uploading = false; bar.classList.add("hidden");
    if (x.status >= 300) { alert(`Upload failed: ${x.responseText}`); updateStart(); return; }
    const r = JSON.parse(x.responseText);
    S.selectedVideo = r.path;   // set before the list reloads so the new radio comes up checked
    clearBoxes();
    await loadVideos();
    selectVideo(r.path);
  };
  x.onerror = () => { S.uploading = false; bar.classList.add("hidden"); alert("Upload failed: network error"); updateStart(); };
  x.send(fd);
}
async function startJob() {
  const config = readParams();
  const withBoxes = !!S.boxes.object;
  if (withBoxes) Object.assign(config, { object_box: S.boxes.object, hand_boxes: S.boxes.hands, bg_boxes: S.boxes.bg });
  const fd = new FormData();
  fd.append("config", JSON.stringify(config));
  fd.append("path", S.selectedVideo);
  if (withBoxes) fd.append("to", $("#runTo").value || "export");
  $("#startBtn").disabled = true;
  try {
    const res = await api("/api/jobs", { method: "POST", body: fd });
    await refreshJobList();
    openJob(res.id);
  } catch (e) { alert(`Could not start: ${e.message}`); updateStart(); }
}

// ---------------------------------------------------------------- job open + SSE
async function refreshJobList() {
  const jobs = await api("/api/jobs");
  const sel = $("#jobSelect");
  sel.replaceChildren(h("option", { value: "" }, "+ new job"),
    ...jobs.map(j => h("option", { value: j.id, selected: j.id === S.jobId }, `${j.id}${j.running ? " ●" : ""}`)));
}
async function openJob(id) {
  if (S.es) { S.es.close(); S.es = null; }
  S.jobId = id; S.events = []; S.stages = emptyStages(); S.img = null; S.galleryTab = "";
  location.hash = id ? `job=${id}` : "";
  $("#jobSelect").value = id;
  const show = !!id;
  for (const sec of ["#pipeSec", "#gallerySec", "#debugSec"]) $(sec).classList.toggle("hidden", !show);
  if (V.mod) V.mod.dispose();
  V.key = ""; $("#viewerSec").classList.add("hidden");
  if (!id) {
    S.config = {}; clearBoxes(); S.selectedVideo = ""; renderParams(); $("#inputSec").classList.remove("hidden");
    $$("input[name=video]").forEach(r => r.checked = false);
    updateBoxSec(); updateStart(); return;
  }
  const st = await api(`/api/jobs/${id}`);
  if (st.order && st.order.join() !== STAGES.join()) { STAGES = st.order; S.stages = emptyStages(); $("#stages tbody").replaceChildren(); }
  S.config = st.config; S.running = st.running;
  S.boxes.object = st.config.object_box; S.boxes.hands = st.config.hand_boxes || []; S.boxes.bg = st.config.bg_boxes || [];
  S.hist = [...(S.boxes.object ? ["object"] : []), ...S.boxes.hands.map(() => "hand"), ...S.boxes.bg.map(() => "bg")];
  renderParams();
  $("#dlBtn").href = `/api/jobs/${id}/download`;
  // seed from the server-side summary, then stream only new events
  for (const k of STAGES) if (st.stages[k]) Object.assign(S.stages[k], st.stages[k]);
  S.events = st.recent_events || [];
  S.es = new EventSource(`/api/jobs/${id}/events?offset=${st.events_offset}`);
  let pending = false;
  S.es.onmessage = m => {
    try { applyEvent(JSON.parse(m.data)); } catch { return; }
    if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); }
  };
  render();
}
async function pollRunning() {
  if (!S.jobId) return;
  try {
    const st = await api(`/api/jobs/${S.jobId}`);
    const was = S.running;
    S.running = st.running; S.config = st.config;
    // the server marks stages of crashed runs as error; adopt that
    for (const s of STAGES) if (!st.running && S.stages[s].status === "running") Object.assign(S.stages[s], st.stages[s]);
    if (was !== S.running) refreshJobList();
    render();
  } catch {}
}

// ---------------------------------------------------------------- render
// ---------------------------------------------------------------- splat viewer (section 4b)
const V = { job: "", key: "", list: [], mod: null };
async function updateViewer() {
  const sec = $("#viewerSec");
  const st = S.stages.train;
  const ready = S.jobId && st && ["done", "warn"].includes(st.status);
  if (!ready) { sec.classList.add("hidden"); return; }
  const key = S.jobId + "|" + (st.ended || "");
  if (V.key === key) return;
  V.key = key;
  let list = [];
  try { list = await api(`/api/jobs/${S.jobId}/splats`); } catch {}
  V.job = S.jobId; V.list = list;
  sec.classList.toggle("hidden", !list.length);
  $("#viewerSel").replaceChildren(...list.map((x, i) => h("option", { value: i }, `${x.label} (${x.mb} MB)`)));
  $("#viewerReset").classList.add("hidden");
  if (V.mod) V.mod.dispose();
  $("#viewerBox").replaceChildren(h("div", { class: "muted", id: "viewerMsg" },
    "Drag to orbit, right-drag to pan, wheel to zoom. Opens a WebGL view of the trained splat; the de-lit version appears here when it exists."));
}
async function openViewer() {
  const x = V.list[+$("#viewerSel").value || 0];
  if (!x) return;
  const box = $("#viewerBox");
  const msg = t => box.replaceChildren(h("div", { class: "muted", id: "viewerMsg" }, t));
  msg("loading viewer…");
  try {
    V.mod = V.mod || await import("/static/viewer.js");
    const ok = await V.mod.show(box, `/api/jobs/${V.job}/files/${x.path}`, x.view, msg);
    $("#viewerReset").classList.toggle("hidden", !ok);
  } catch (e) { msg("viewer failed: " + e.message); }
}
$("#viewerOpen").onclick = openViewer;
$("#viewerSel").onchange = () => { if ($("#viewerReset") && !$("#viewerReset").classList.contains("hidden")) openViewer(); };
$("#viewerReset").onclick = openViewer;

function render() {
  if (!S.jobId) return;
  updateViewer();
  const st = S.stages;
  const framesReady = ["done", "warn"].includes(st.frames.status);
  updateBoxSec();
  if (framesReady && !S.img) loadBoxImage();
  $("#runBtn").disabled = S.running || !S.boxes.object;
  $("#cancelBtn").disabled = !S.running;
  const rs = $("#runState");
  rs.textContent = S.running ? "running" : "idle"; rs.className = `pill ${S.running ? "run" : ""}`;
  renderStages(); renderGallery(); renderDebug(false);
}
function renderStages() {
  const tb = $("#stages tbody");
  if (!tb.children.length) {
    for (const s of STAGES) {
      tb.append(h("tr", { "data-s": s },
        h("td", { class: "name" }, s), h("td", {}, h("span", { class: "chip" })),
        h("td", { class: "barc" }, h("div", { class: "bar" }, h("div"))), h("td", { class: "t" }),
        h("td", { class: "msg" }),
        h("td", { class: "act" },
          h("button", { class: "small", title: "re-run only this stage", onclick: () => runStages(s, s) }, "re-run"), " ",
          h("button", { class: "small", title: "run from this stage to the end", onclick: () => runStages(s, $("#runTo").value || null) }, "from here"))));
    }
  }
  const now = Date.now() / 1000;
  for (const s of STAGES) {
    const x = S.stages[s], tr = $(`tr[data-s=${s}]`, tb);
    const chip = $(".chip", tr); chip.textContent = x.status; chip.className = `chip ${x.status}`;
    const bar = $(".bar", tr); bar.className = `bar ${x.status}`; bar.firstElementChild.style.width = `${x.pct || 0}%`;
    const el = x.started ? ((x.status === "running" ? now : (x.ended || now)) - x.started) : null;
    $(".t", tr).textContent = fmtT(el) + (x.status === "running" && x.pct ? ` ${x.pct.toFixed(0)}%` : "");
    $(".msg", tr).textContent = x.msg || "";
    $$("button", tr).forEach(b => b.disabled = S.running);
  }
}
function renderGallery() {
  const withArt = STAGES.filter(s => S.stages[s].artifacts.length);
  if (!S.galleryTab || !withArt.includes(S.galleryTab)) S.galleryTab = withArt[withArt.length - 1] || "";
  const tabs = $("#galleryTabs");
  const key = withArt.join(",") + "|" + S.galleryTab;
  if (tabs.dataset.key !== key) {
    tabs.dataset.key = key;
    tabs.replaceChildren(...withArt.map(s => h("button", { class: s === S.galleryTab ? "on" : "",
      onclick: () => { S.galleryTab = s; renderGallery(); } }, `${s} (${S.stages[s].artifacts.length})`)));
  }
  const g = $("#gallery");
  const arts = S.galleryTab ? S.stages[S.galleryTab].artifacts : [];
  const gkey = S.galleryTab + arts.map(a => a.path + a.t).join("|");
  if (g.dataset.key === gkey) return;
  g.dataset.key = gkey;
  if (!arts.length) { g.replaceChildren(h("p", { class: "muted" }, "No artifacts yet.")); return; }
  g.replaceChildren(...arts.map(a => {
    const url = `/api/jobs/${S.jobId}/files/${a.path}?v=${Math.round(a.t || 0)}`;
    if (a.kind === "video" || /\.(mp4|webm)$/i.test(a.path)) {
      return h("figure", {}, h("video", { src: url, controls: true, loop: true, muted: true, playsinline: true, preload: "metadata" }),
        h("figcaption", {}, a.caption || a.path));
    }
    if (a.kind === "image" || /\.(png|jpe?g|webp)$/i.test(a.path)) {
      return h("figure", {}, h("img", { src: url, loading: "lazy", alt: a.caption, onclick: () => lightbox(url, a.caption) }),
        h("figcaption", {}, a.caption || a.path));
    }
    return h("figure", {}, h("div", { class: "file" }, h("a", { href: url, target: "_blank", download: a.kind === "file" ? "" : undefined }, a.path.split("/").pop())),
      h("figcaption", {}, a.caption));
  }));
}
function lightbox(url, cap) {
  const lb = $("#lightbox"); $("img", lb).src = url; $(".cap", lb).textContent = cap; lb.classList.remove("hidden");
}

let lastLogFetch = 0;
async function renderDebug(force) {
  const body = $("#debugBody"), tab = S.debugTab;
  if (tab === "log") {
    if (!force && Date.now() - lastLogFetch < 2500) return;
    lastLogFetch = Date.now();
    const pre = body.querySelector("pre[data-tab=log]") || h("pre", { "data-tab": "log" });
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
    try { pre.textContent = await api(`/api/jobs/${S.jobId}/log?tail=400`); } catch (e) { pre.textContent = e.message; }
    if (!pre.isConnected) body.replaceChildren(pre);
    if (atBottom || force) pre.scrollTop = pre.scrollHeight;
  } else if (tab === "events") {
    if (!force && body.dataset.n == S.events.length) return;
    body.dataset.n = S.events.length;
    const rows = S.events.slice(-300).reverse().map(e => h("tr", {},
      h("td", {}, new Date(e.t * 1000).toLocaleTimeString()), h("td", {}, e.type), h("td", {}, e.stage || ""),
      h("td", {}, e.status || e.level || (e.pct != null ? `${e.pct}%` : "")), h("td", {}, e.msg || e.path || e.caption || "")));
    body.replaceChildren(h("div", { class: "evwrap" }, h("table", { class: "evtable" }, h("tbody", {}, rows))));
  } else if (force) {
    let data;
    if (tab === "stats") data = Object.fromEntries(STAGES.map(s => [s, S.stages[s].stats]));
    else if (tab === "config") data = S.config;
    else if (tab === "doctor") data = S.doctor || await api("/api/doctor");
    body.replaceChildren(h("pre", {}, JSON.stringify(data, null, 2)));
  }
}

// ---------------------------------------------------------------- box tool
// Two modes: no job yet (draw on a preview frame of the selected video) or an open job (draw on its first selected frame).
function updateBoxSec() {
  const ready = S.jobId ? ["done", "warn"].includes((S.stages.frames || {}).status) : !!S.selectedVideo;
  $("#boxSec").classList.toggle("hidden", !ready);
  $("#startBtn").classList.toggle("hidden", !!S.jobId);
  $("#runBtn").classList.toggle("hidden", !S.jobId);
  $("#inputMsg").classList.toggle("hidden", !!S.jobId);
  $("#queueAdd").classList.toggle("hidden", !!S.jobId);
  renderQueue();
}
function clearBoxes() {
  S.boxes = { object: null, hands: [], bg: [] }; S.hist = [];
  if (S.img instanceof Image) drawBoxes();
}
function setBoxImage(img) {
  S.img = img;
  const c = $("#boxCanvas"); c.width = img.naturalWidth; c.height = img.naturalHeight;
  drawBoxes();
}
function loadBoxImage() {
  const frames = S.stages.frames.artifacts.find(a => a.path.includes("/images/"));
  const path = frames ? frames.path : "work/frames/images/00000.jpg";
  const img = new Image();
  img.onload = () => setBoxImage(img);
  img.src = `/api/jobs/${S.jobId}/files/${path}?v=${Date.now()}`;
  S.img = "loading";
}
const BOX_COL = { object: "#28c76f", hand: "#ff5a5a", bg: "#4a90ff" };
function drawBoxes(temp) {
  const c = $("#boxCanvas"), ctx = c.getContext("2d");
  if (!(S.img instanceof Image)) return;
  ctx.drawImage(S.img, 0, 0);
  const W = c.width, H = c.height, lw = Math.max(2, W / 400);
  const box = (b, col, label) => {
    if (!b) return;
    const [x0, y0, x1, y1] = b;
    ctx.fillStyle = col + "22"; ctx.fillRect(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H);
    ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.strokeRect(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H);
    ctx.font = `${Math.round(W / 60)}px system-ui`; ctx.fillStyle = col; ctx.fillText(label, x0 * W + 6, y0 * H + W / 50);
  };
  box(S.boxes.object, BOX_COL.object, "object");
  S.boxes.hands.forEach((b, i) => box(b, BOX_COL.hand, `hand ${i + 1}`));
  S.boxes.bg.forEach((b, i) => box(b, BOX_COL.bg, `bg ${i + 1}`));
  if (temp) box(temp, BOX_COL[S.boxMode], "");
  const f = b => b ? b.map(v => v.toFixed(3)).join(",") : "-";
  $("#boxInfo").textContent = `object ${f(S.boxes.object)}  hands ${S.boxes.hands.map(f).join(" ; ") || "-"}  bg ${S.boxes.bg.map(f).join(" ; ") || "-"}`;
  $("#runBtn").disabled = S.running || !S.boxes.object;
  updateStart();
}
function setBoxMode(mode) {
  S.boxMode = mode;
  $$("#boxSec .seg button").forEach(x => x.classList.toggle("on", x.dataset.mode === mode));
}
function undoBox() {
  const k = S.hist.pop();
  if (k === "object") S.boxes.object = null; else if (k === "hand") S.boxes.hands.pop(); else if (k === "bg") S.boxes.bg.pop();
  drawBoxes();
}
function setupBoxTool() {
  const c = $("#boxCanvas");
  let start = null;
  const rel = e => { const r = c.getBoundingClientRect(); return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))]; };
  const norm = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  const inside = (b, p) => b && p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3];
  c.addEventListener("pointerdown", e => { if (e.button !== 0) return; start = rel(e); c.setPointerCapture(e.pointerId); });
  c.addEventListener("pointermove", e => { if (start) drawBoxes(norm(start, rel(e))); });
  c.addEventListener("pointercancel", () => { start = null; drawBoxes(); });
  c.addEventListener("pointerup", e => {
    if (!start) return;
    const b = norm(start, rel(e)); start = null;
    if ((b[2] - b[0]) < 0.01 || (b[3] - b[1]) < 0.01) { drawBoxes(); return; }
    if (S.boxMode === "object") {
      if (S.boxes.object) S.hist.splice(S.hist.indexOf("object"), 1);   // replaced, keep a single history entry
      S.boxes.object = b; S.hist.push("object");
    } else if (S.boxMode === "bg") { S.boxes.bg.push(b); S.hist.push("bg"); }
    else { S.boxes.hands.push(b); S.hist.push("hand"); }
    drawBoxes();
  });
  // right-click deletes the smallest box under the cursor
  c.addEventListener("contextmenu", e => {
    e.preventDefault();
    const p = rel(e), area = b => (b[2] - b[0]) * (b[3] - b[1]);
    const cand = [];
    if (inside(S.boxes.object, p)) cand.push({ k: "object", a: area(S.boxes.object) });
    S.boxes.hands.forEach((b, i) => inside(b, p) && cand.push({ k: "hand", i, a: area(b) }));
    S.boxes.bg.forEach((b, i) => inside(b, p) && cand.push({ k: "bg", i, a: area(b) }));
    if (!cand.length) return;
    const t = cand.sort((x, y) => x.a - y.a)[0];
    if (t.k === "object") S.boxes.object = null; else if (t.k === "hand") S.boxes.hands.splice(t.i, 1); else S.boxes.bg.splice(t.i, 1);
    const hi = S.hist.map((k, j) => [k, j]).filter(([k]) => k === t.k).map(([, j]) => j);
    S.hist.splice(hi[t.k === "object" ? 0 : t.i], 1);
    drawBoxes();
  });
  $$("#boxSec .seg button").forEach(b => b.addEventListener("click", () => setBoxMode(b.dataset.mode)));
  $("#clearBoxes").addEventListener("click", clearBoxes);
  $("#undoBox").addEventListener("click", undoBox);
  document.addEventListener("keydown", e => {
    if ($("#boxSec").classList.contains("hidden") || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (e.key === "Escape" && start) { start = null; drawBoxes(); }
    else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undoBox(); }
    else if (!e.metaKey && !e.ctrlKey && !e.altKey && { 1: "object", 2: "hand", 3: "bg" }[e.key]) setBoxMode({ 1: "object", 2: "hand", 3: "bg" }[e.key]);
  });
}

// ---------------------------------------------------------------- bulk queue
// Client side the queue is a list of {path, name, boxes}; "Start queue" sends it to the server, which creates one job per
// video and runs them one after another (GET /api/batch reports the progress).
const baseName = p => p.split("/").pop();
const cloneBoxes = () => JSON.parse(JSON.stringify(S.boxes));
function queueItem(path, boxes) {
  const i = S.queue.findIndex(q => q.path === path);
  const q = { path, name: baseName(path), boxes };
  if (i >= 0) S.queue[i] = q; else S.queue.push(q);   // adding a video again replaces its boxes
}
function addToQueue() {
  if (!S.selectedVideo) return;
  queueItem(S.selectedVideo, cloneBoxes());
  // move on to the next video that is not queued yet
  const radios = $$("input[name=video]"), cur = radios.findIndex(r => r.checked);
  const next = radios.slice(cur + 1).concat(radios.slice(0, Math.max(cur, 0)))
    .find(r => !S.queue.some(q => baseName(q.path) === r.value));
  if (next) { next.checked = true; next.dispatchEvent(new Event("change")); }
  renderQueue();
}
async function addAllToQueue() {
  const vids = await api("/api/videos");
  for (const v of vids) if (!S.queue.some(q => q.path === v.path)) queueItem(v.path, cloneBoxes());
  renderQueue();
}
function editQueued(q) {
  const r = $$("input[name=video]").find(x => x.value === q.name);
  if (r) r.checked = true;
  S.selectedVideo = q.path; S.boxes = JSON.parse(JSON.stringify(q.boxes)); S.img = null;
  S.hist = [...(S.boxes.object ? ["object"] : []), ...S.boxes.hands.map(() => "hand"), ...S.boxes.bg.map(() => "bg")];
  loadPreview(); updateStart();
}
function renderQueue() {
  const withBoxes = S.queue.filter(q => q.boxes.object).length;
  $("#queueSec").classList.toggle("hidden", !!S.jobId || (!S.queue.length && !S.batch.items.length));
  $("#queueList").replaceChildren(...S.queue.map((q, i) => h("div", { class: "qrow" },
    h("span", { class: "grow" }, q.name),
    h("span", { class: "muted" }, q.boxes.object ? `box${q.boxes.hands.length ? ` + ${q.boxes.hands.length} hand` : ""}` : "no box: frames only"),
    h("button", { class: "small", title: "show this video and its boxes", onclick: () => editQueued(q) }, "edit"),
    h("button", { class: "small", title: "remove from the queue", onclick: () => { S.queue.splice(i, 1); renderQueue(); } }, "x"))));
  $("#queueStart").disabled = !S.queue.length;
  $("#queueMsg").textContent = S.queue.length ? `${S.queue.length} video(s), ${withBoxes} with an object box` : "";
  $("#queueCancel").classList.toggle("hidden", !S.batch.running);
  $("#batchList").replaceChildren(...S.batch.items.map(i => h("div", { class: "qrow" },
    h("span", { class: "grow" }, i.video),
    h("span", { class: `st-${i.state}` }, i.state),
    h("span", { class: "muted" }, i.msg || (i.seconds ? fmtT(i.seconds) : "")),
    h("a", { onclick: () => { refreshJobList().then(() => openJob(i.job)); } }, i.job))));
}
async function startQueue() {
  const body = {
    items: S.queue.map(q => ({ path: q.path, config: q.boxes.object
      ? { object_box: q.boxes.object, hand_boxes: q.boxes.hands, bg_boxes: q.boxes.bg } : {} })),
    config: readParams(), to: $("#runTo").value || "export",
  };
  $("#queueStart").disabled = true;
  try {
    await api("/api/batch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    S.queue = []; await pollBatch(); refreshJobList();
  } catch (e) { alert(`Could not start the queue: ${e.message}`); }
  renderQueue();
}
async function pollBatch() {
  try {
    const was = S.batch.running;
    S.batch = await api("/api/batch");
    renderQueue();
    if (was && !S.batch.running) refreshJobList();
  } catch {}
}

// ---------------------------------------------------------------- actions
async function runStages(from, to) {
  const config = { ...readParams() };
  if (S.boxes.object) { config.object_box = S.boxes.object; config.hand_boxes = S.boxes.hands; config.bg_boxes = S.boxes.bg; }
  try {
    await api(`/api/jobs/${S.jobId}/run`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, from, to }) });
    S.running = true; render(); refreshJobList();
  } catch (e) { alert(e.message); }
}

async function initDoctor() {
  try {
    const d = S.doctor = await api("/api/doctor");
    const p = $("#doctorPill");
    p.textContent = d.warnings.length ? `doctor: ${d.warnings.length} warning(s)` : "doctor: ok";
    p.className = `pill ${d.warnings.length ? "warn" : "ok"}`;
    p.title = d.warnings.join("\n") + "\n\nbackends: " + Object.entries(d.backends).map(([k, v]) => `${k}=${v.join("/")}`).join("  ");
  } catch (e) { $("#doctorPill").textContent = "doctor failed"; }
}

async function main() {
  S.defaults = await api("/api/defaults");
  STAGES = await api("/api/stages");
  renderParams(); setupDrop(); setupBoxTool();
  $("#runTo").replaceChildren(...STAGES.slice(STAGES.indexOf("segment")).map(s => h("option", { value: s, selected: s === "export" }, s)));
  await loadVideos(); await refreshJobList();
  $("#startBtn").addEventListener("click", startJob);
  $("#queueAddBtn").addEventListener("click", addToQueue);
  $("#queueAllBtn").addEventListener("click", addAllToQueue);
  $("#queueClear").addEventListener("click", () => { S.queue = []; renderQueue(); });
  $("#queueStart").addEventListener("click", startQueue);
  $("#queueCancel").addEventListener("click", async () => { await api("/api/batch/cancel", { method: "POST" }); pollBatch(); });
  $("#runBtn").addEventListener("click", () => runStages("segment", $("#runTo").value));
  $("#cancelBtn").addEventListener("click", async () => { await api(`/api/jobs/${S.jobId}/cancel`, { method: "POST" }); pollRunning(); });
  $("#jobSelect").addEventListener("change", e => openJob(e.target.value));
  $("#lightbox").addEventListener("click", () => $("#lightbox").classList.add("hidden"));
  $$("#debugTabs button").forEach(b => b.addEventListener("click", () => {
    S.debugTab = b.dataset.tab; $$("#debugTabs button").forEach(x => x.classList.toggle("on", x === b));
    $("#debugBody").dataset.n = ""; renderDebug(true);
  }));
  initDoctor();
  const m = location.hash.match(/job=([^&]+)/);
  if (m) openJob(decodeURIComponent(m[1]));
  setInterval(pollRunning, 3000);
  pollBatch(); setInterval(pollBatch, 3000);
  setInterval(() => { if (S.running) renderStages(); }, 1000);
}
main();
