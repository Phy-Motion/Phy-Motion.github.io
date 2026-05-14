import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Base URL for our data/scenes — resolved relative to this module's location
// so the viewer works whether it's hosted from /index.html or from
// /4d_viewer/index.html.
const BASE = new URL('.', import.meta.url).href;
function dataUrl(p) { return new URL(p, BASE).href; }

// ---------- DOM ----------
const canvas = document.getElementById('three-canvas');
const video = document.getElementById('vid');
const scrub = document.getElementById('scrub');
const playBtn = document.getElementById('play-btn');
const picker = document.getElementById('scene-picker');
if (!canvas || !video || !scrub || !playBtn || !picker) {
  // Viewer markup isn't on this page; abort silently.
  throw new Error('viewer DOM not present');
}

// ---------- Three.js ----------
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();

// SMPL in-camera coords are OpenCV-style (X right, Y down, Z forward).
// Three.js is OpenGL (X right, Y up, Z back), so we mirror Y and Z when copying
// into the BufferAttribute. After that mirror, the body sits at z ≈ -4.5 in
// front of a camera placed near the origin.
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
camera.position.set(0, 0, 3);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.5;
controls.maxDistance = 15;

// Lighting
scene.add(new THREE.HemisphereLight(0xfaf5e8, 0x303642, 1.0));
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(2, 3, 4);
scene.add(dir);
const back = new THREE.DirectionalLight(0xb6c5d9, 0.6);
back.position.set(-3, 2, -2);
scene.add(back);


// Ground grid at origin (the body is normalized to origin in setFrame())
const grid = new THREE.GridHelper(4, 8, 0x88a5b8, 0xc8d2da);
grid.position.set(0, -1.0, 0);
grid.material.transparent = true;
grid.material.opacity = 0.35;
scene.add(grid);

// Mesh (built once after we load faces + first scene)
let smplMesh = null;
let facesIndex = null;            // shared Int32Array
const positions = { typed: null, attr: null };  // current frame buffer
const colors = { typed: null, attr: null };     // per-vertex color (for dynamic-accel heat)
let frameStride = 0;              // verts per frame

// ---------- Resize ----------
function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

// ---------- Load faces once ----------
async function loadFaces() {
  const resp = await fetch(dataUrl('data/faces.bin'));
  const buf = await resp.arrayBuffer();
  // WebGL drawElements only accepts UNSIGNED_SHORT or UNSIGNED_INT, NOT signed int32.
  // SMPL-X has 10475 verts so 16-bit unsigned would fit, but our exporter writes
  // int32. Reinterpret the buffer as Uint32Array (safe: faces are positive indices).
  facesIndex = new Uint32Array(buf);
}

// Kinematic axis: shared wrist/hand/hip vertex mask (true if vertex should be
// highlighted red on flagged frames).
let kinHighlightMask = null;
async function loadKinHighlightMask() {
  try {
    const r = await fetch(dataUrl('data/kin_highlight_mask.bin'));
    if (r.ok) kinHighlightMask = new Uint8Array(await r.arrayBuffer());
  } catch (e) { /* optional */ }
}

// ---------- Per-scene loading ----------
let currentScene = null;
let currentVertsF16 = null;       // Uint16Array (raw float16 bits)
let currentVertsF32 = null;       // Float32Array view per frame (reused)
let currentViolations = null;     // Uint8Array [T] — 1 if frame violates axis
let currentAccel = null;          // Uint8Array [T*V] or null — per-vertex accel heat (dynamic)
let currentVelocity = null;       // Uint8Array [T*V] or null — per-vertex velocity heat (kinematic)
let sceneOffset = { x: 0, y: 0, z: 0 };  // subtract from each vertex to anchor body near origin
let playheadFrame = 0;            // float frame index; advanced by animate(t) when playing

// Default body color
const COLOR_NORMAL = new THREE.Color(0xf2efe9);     // very light off-white
const COLOR_VIOLATE = new THREE.Color(0xc94432);    // red

// Per-axis highlights
const KIN_RED = new THREE.Color(0xd24330);          // kinematic highlight (red)
const KIN_FRAME_RANGE = [30, 41];                   // inclusive
const CON_MAGENTA = new THREE.Color(0xa83bd9);      // contact highlight (red+blue)
const CON_FRAME_START = 28;                         // from this frame onwards

// Per-vertex accel color: white → deep blue (dynamic axis)
const ACCEL_BASE = new THREE.Color(0xf5f3ee);       // near-white skin
const ACCEL_HIGH = new THREE.Color(0x1a5f8a);       // saturated blue

// Per-vertex velocity color: white → deep red (kinematic axis)
const VEL_BASE = new THREE.Color(0xf5f3ee);         // near-white skin
const VEL_HIGH = new THREE.Color(0xc94432);         // saturated red

function f16toF32(u16) {
  // Half -> float (IEEE 754 half precision)
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    let val;
    if (e === 0) {
      val = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    } else if (e === 0x1f) {
      val = f ? NaN : ((s ? -1 : 1) * Infinity);
    } else {
      val = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    out[i] = val;
  }
  return out;
}

async function loadScene(meta) {
  currentScene = meta;
  // Load verts
  const resp = await fetch(dataUrl(`data/${meta.key}/verts.bin`));
  const buf = await resp.arrayBuffer();
  currentVertsF16 = new Uint16Array(buf);
  // Convert all frames to float32 ONCE (10475×3×45 ≈ 5.6MB float32 -> fine)
  currentVertsF32 = f16toF32(currentVertsF16);
  frameStride = meta.num_verts * 3;

  // Load per-frame violation flags
  try {
    const vr = await fetch(dataUrl(`data/${meta.key}/violations.bin`));
    if (vr.ok) {
      currentViolations = new Uint8Array(await vr.arrayBuffer());
    } else {
      currentViolations = new Uint8Array(meta.num_frames);
    }
  } catch (e) {
    currentViolations = new Uint8Array(meta.num_frames);
  }

  // For the dynamic axis, load per-vertex acceleration heat-map
  currentAccel = null;
  if (meta.has_accel) {
    try {
      const ar = await fetch(dataUrl(`data/${meta.key}/accel.bin`));
      if (ar.ok) currentAccel = new Uint8Array(await ar.arrayBuffer());
    } catch (e) { /* ignore */ }
  }
  // For the kinematic axis, load per-vertex velocity heat-map
  currentVelocity = null;
  if (meta.has_velocity) {
    try {
      const vr2 = await fetch(dataUrl(`data/${meta.key}/velocity.bin`));
      if (vr2.ok) currentVelocity = new Uint8Array(await vr2.arrayBuffer());
    } catch (e) { /* ignore */ }
  }

  // Compute frame-0 centroid (OpenCV coords) so we can recenter the body at origin.
  let cx = 0, cy = 0, cz = 0;
  const V = meta.num_verts;
  for (let i = 0; i < frameStride; i += 3) {
    cx += currentVertsF32[i];
    cy += currentVertsF32[i + 1];
    cz += currentVertsF32[i + 2];
  }
  // After we mirror Y and Z, the centroid in OpenGL coords is (cx/V, -cy/V, -cz/V).
  sceneOffset = { x: cx / V, y: -cy / V, z: -cz / V };

  // (Re)build mesh
  if (smplMesh) {
    scene.remove(smplMesh);
    smplMesh.geometry.dispose();
    smplMesh.material.dispose();
  }
  const geom = new THREE.BufferGeometry();
  const positionAttr = new THREE.BufferAttribute(new Float32Array(frameStride), 3);
  positionAttr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('position', positionAttr);
  // Per-vertex color buffer (only filled meaningfully for the dynamic axis).
  const colorAttr = new THREE.BufferAttribute(new Float32Array(frameStride), 3);
  colorAttr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('color', colorAttr);
  geom.setIndex(new THREE.BufferAttribute(facesIndex, 1));

  positions.attr = positionAttr;
  positions.typed = positionAttr.array;
  colors.attr = colorAttr;
  colors.typed = colorAttr.array;

  const mat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    vertexColors: true,  // we always paint per-vertex; base color stays white
    metalness: 0.05,
    roughness: 0.6,
    flatShading: true,
    side: THREE.DoubleSide,
  });
  smplMesh = new THREE.Mesh(geom, mat);
  // We mutate vertices every frame; skip frustum culling so the mesh is always drawn
  // regardless of its (recomputed) bounding sphere.
  smplMesh.frustumCulled = false;
  scene.add(smplMesh);

  // Set up scrubber
  scrub.max = String(meta.num_frames - 1);
  scrub.value = '0';
  playheadFrame = 0;
  updateSliderTrack();

  // Set up video
  video.src = dataUrl(meta.video);
  video.load();

  setFrame(0);
  // Auto-fit camera to the first frame after a tick
  setTimeout(() => fitCamera(), 30);
}

function setFrame(frameIdx) {
  const intIdx = Math.max(0, Math.min(currentScene.num_frames - 1, Math.round(frameIdx)));
  const start = intIdx * frameStride;
  const src = currentVertsF32.subarray(start, start + frameStride);
  const dst = positions.typed;
  // OpenCV camera (X right, Y down, Z forward) -> OpenGL (X right, Y up, Z back),
  // then subtract sceneOffset so the body is centered at the world origin.
  const ox = sceneOffset.x, oy = sceneOffset.y, oz = sceneOffset.z;
  for (let i = 0; i < src.length; i += 3) {
    dst[i]     =  src[i]     - ox;
    dst[i + 1] = -src[i + 1] - oy;
    dst[i + 2] = -src[i + 2] - oz;
  }
  positions.attr.needsUpdate = true;

  paintVertexColors(intIdx);
  smplMesh.geometry.computeVertexNormals();
}

function paintVertexColors(frameIdx) {
  const V = currentScene.num_verts;
  const cArr = colors.typed;
  const axis = currentScene.axis;

  // Default: white body
  const rN = COLOR_NORMAL.r, gN = COLOR_NORMAL.g, bN = COLOR_NORMAL.b;
  for (let v = 0; v < V; v++) {
    const off = v * 3;
    cArr[off]     = rN;
    cArr[off + 1] = gN;
    cArr[off + 2] = bN;
  }

  // Dynamic: per-vertex accel heat
  if (axis === 'dynamic' && currentAccel) {
    const accStart = frameIdx * V;
    const rB = ACCEL_BASE.r, gB = ACCEL_BASE.g, bB = ACCEL_BASE.b;
    const rH = ACCEL_HIGH.r, gH = ACCEL_HIGH.g, bH = ACCEL_HIGH.b;
    for (let v = 0; v < V; v++) {
      const t = currentAccel[accStart + v] / 255.0;
      const off = v * 3;
      cArr[off]     = rB + (rH - rB) * t;
      cArr[off + 1] = gB + (gH - gB) * t;
      cArr[off + 2] = bB + (bH - bB) * t;
    }
  }

  // Kinematic: per-vertex velocity tint (white → red), applied to the whole body
  if (axis === 'kinematic' && currentVelocity) {
    const velStart = frameIdx * V;
    const rB = VEL_BASE.r, gB = VEL_BASE.g, bB = VEL_BASE.b;
    const rH = VEL_HIGH.r, gH = VEL_HIGH.g, bH = VEL_HIGH.b;
    for (let v = 0; v < V; v++) {
      const t = currentVelocity[velStart + v] / 255.0;
      const off = v * 3;
      cArr[off]     = rB + (rH - rB) * t;
      cArr[off + 1] = gB + (gH - gB) * t;
      cArr[off + 2] = bB + (bH - bB) * t;
    }
  }

  // Kinematic: mark wrist/hand/hip red on frames [30, 41] (on top of velocity heat)
  if (axis === 'kinematic' && kinHighlightMask &&
      frameIdx >= KIN_FRAME_RANGE[0] && frameIdx <= KIN_FRAME_RANGE[1]) {
    const r = KIN_RED.r, g = KIN_RED.g, b = KIN_RED.b;
    for (let v = 0; v < V; v++) {
      if (kinHighlightMask[v]) {
        const off = v * 3;
        cArr[off] = r; cArr[off + 1] = g; cArr[off + 2] = b;
      }
    }
  }

  // Contact: whole body magenta from frame CON_FRAME_START onwards
  if (axis === 'contact' && frameIdx >= CON_FRAME_START) {
    const r = CON_MAGENTA.r, g = CON_MAGENTA.g, b = CON_MAGENTA.b;
    for (let v = 0; v < V; v++) {
      const off = v * 3;
      cArr[off] = r; cArr[off + 1] = g; cArr[off + 2] = b;
    }
  }

  colors.attr.needsUpdate = true;
}

const DEFAULT_VIEW = {
  cameraPos: new THREE.Vector3(),
  target: new THREE.Vector3(),
};

function fitCamera() {
  if (!smplMesh) return;
  smplMesh.geometry.computeBoundingBox();
  const box = smplMesh.geometry.boundingBox;
  if (!box) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  controls.target.copy(center);
  // Body lives at z ≈ -4.5; pull the camera back along +Z so it's farther from the body
  camera.position.set(center.x, center.y + size * 0.1, center.z + size * 1.6);
  controls.update();
  // Remember this as the "default view" — clicking the reset button returns here
  DEFAULT_VIEW.cameraPos.copy(camera.position);
  DEFAULT_VIEW.target.copy(controls.target);
}

function resetView() {
  camera.position.copy(DEFAULT_VIEW.cameraPos);
  controls.target.copy(DEFAULT_VIEW.target);
  controls.update();
}

const resetBtn = document.getElementById('reset-view-btn');
if (resetBtn) resetBtn.addEventListener('click', resetView);

// ---------- Scrub <-> video sync ----------
let playing = false;
let lastTick = 0;

function setPlayButton(isPlaying) {
  playBtn.innerHTML = isPlaying
    ? '<span class="icon"><i class="fas fa-pause"></i></span><span>Pause</span>'
    : '<span class="icon"><i class="fas fa-play"></i></span><span>Play</span>';
}

function updateSliderTrack() {
  const v = Number(scrub.value);
  const m = Number(scrub.max) || 1;
  scrub.style.setProperty('--pct', `${(v / m) * 100}%`);
}

function syncFromScrub() {
  const f = parseInt(scrub.value, 10);
  playheadFrame = f;
  setFrame(f);
  // Seek the video to match. The async seek won't fight playback because the
  // user's scrub action also pauses (`playing=false`), so the animation loop
  // stops reading `video.currentTime` and won't overwrite playheadFrame.
  if (currentScene) {
    const t = f / currentScene.fps;
    // Avoid redundant seeks (each one triggers an async event chain).
    if (Math.abs(video.currentTime - t) > 1e-3) {
      video.currentTime = t;
    }
  }
}
scrub.addEventListener('input', () => {
  if (playing) {
    playing = false;
    setPlayButton(false);
    video.pause();
  }
  syncFromScrub();
  updateSliderTrack();
});

playBtn.addEventListener('click', () => {
  playing = !playing;
  setPlayButton(playing);
  if (playing) {
    // If we're at (or near) the end, restart from frame 0.
    if (currentScene && playheadFrame >= currentScene.num_frames - 1) {
      playheadFrame = 0;
      scrub.value = '0';
      setFrame(0);
      video.currentTime = 0;
      updateSliderTrack();
    }
    video.play().catch(() => {});
    lastTick = performance.now();
  } else {
    video.pause();
  }
});

function animate(t) {
  controls.update();
  if (playing && currentScene) {
    // Drive everything off the <video>'s clock so the SMPL playback stays
    // exactly in sync with the rendered video frames (and pauses if the video
    // stalls while buffering).
    playheadFrame = video.currentTime * currentScene.fps;
    if (playheadFrame >= currentScene.num_frames - 1 || video.ended) {
      playheadFrame = currentScene.num_frames - 1;
      playing = false;
      video.pause();
      setPlayButton(false);
    }
    scrub.value = String(Math.floor(playheadFrame));
    setFrame(playheadFrame);
    updateSliderTrack();
  }
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

// ---------- Axis picker ----------
const sceneLabel = document.getElementById('scene-label');
const axisFigure = document.getElementById('axis-figure');
const AXIS_FIGURES = {
  kinematic: './static/figures/smpl_kin.jpg',
  contact:   './static/figures/smpl_con.jpg',
  dynamic:   './static/figures/smpl_dyn.jpg',
};

// Definitions of each feasibility axis (shown above the viewer card).
const AXIS_DEFINITIONS = {
  kinematic:
    '<strong>Kinematic feasibility</strong> measures whether the recovered body motion is smooth and ' +
    'anatomically valid. We combine three normalized violations: angular-velocity violation ' +
    '(joint speeds against a clean-motion tolerance), self-penetration violation (fraction of frames ' +
    'with intersecting non-adjacent mesh triangles), and joint-limit violation (fraction of joints ' +
    'whose angles fall outside the valid MuJoCo range). ' +
    '<em>F<sub>kin</sub> = 1 &minus; ⅓(v<sub>vel</sub> + v<sub>spen</sub> + v<sub>lim</sub>)</em>; ' +
    'higher is better.',
  contact:
    '<strong>Contact feasibility</strong> measures whether the body interacts with the ground ' +
    'plausibly. We infer binary foot–ground contacts from foot height and velocity, then combine four ' +
    'normalized violations: foot sliding (displacement of a contacted foot), ground penetration ' +
    '(foot below the floor), foot floating (both feet airborne without a ballistic trajectory), and ' +
    'balance violation (projected center of mass outside the support polygon). ' +
    '<em>F<sub>con</sub> = 1 &minus; ¼(v<sub>slip</sub> + v<sub>gpen</sub> + v<sub>float</sub> + v<sub>bal</sub>)</em>.',
  dynamic:
    '<strong>Dynamic feasibility</strong> measures whether the recovered motion can be replayed by a ' +
    'physically plausible human body. Using inverse dynamics in MuJoCo, we estimate the forces required ' +
    'to reproduce the trajectory and compute three sub-scores: s<sub>τ</sub> penalizes unrealistic joint ' +
    'torques, s<sub>GRF</sub> penalizes excessive ground reaction forces, and s<sub>met</sub> penalizes ' +
    'unusually high joint effort (torque–velocity work proxy). ' +
    '<em>F<sub>dyn</sub> = ⅓(s<sub>τ</sub> + s<sub>GRF</sub> + s<sub>met</sub>)</em>; ' +
    'higher is more physically realizable.',
};

async function init() {
  setPlayButton(false);
  resize();
  await loadFaces();
  await loadKinHighlightMask();
  const scenes = await fetch(dataUrl('scenes.json')).then(r => r.json());
  if (!scenes.length) {
    if (sceneLabel) sceneLabel.textContent = 'No scenes available. Run scripts/export_smpl_viewer.py.';
    return;
  }

  // Map axis -> scene (the export gave each axis its own scene)
  const byAxis = {};
  for (const s of scenes) {
    if (s.axis) byAxis[s.axis] = s;
  }

  // Wire axis buttons (already in HTML)
  const buttons = picker.querySelectorAll('button[data-axis]');
  buttons.forEach((btn) => {
    const axis = btn.dataset.axis;
    const scene = byAxis[axis];
    if (!scene) {
      btn.disabled = true;
      btn.style.opacity = '0.4';
      return;
    }
    btn.addEventListener('click', () => {
      buttons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      if (sceneLabel) sceneLabel.innerHTML = AXIS_DEFINITIONS[axis] || '';
      if (axisFigure && AXIS_FIGURES[axis]) axisFigure.src = AXIS_FIGURES[axis];
      loadScene(scene);
    });
  });

  // Auto-load whichever axis is available, preferring kinematic
  const priority = ['kinematic', 'contact', 'dynamic'];
  for (const axis of priority) {
    if (byAxis[axis]) {
      const btn = picker.querySelector(`button[data-axis="${axis}"]`);
      if (btn) {
        btn.classList.add('active');
        if (sceneLabel) sceneLabel.innerHTML = AXIS_DEFINITIONS[axis] || '';
        if (axisFigure && AXIS_FIGURES[axis]) axisFigure.src = AXIS_FIGURES[axis];
      }
      await loadScene(byAxis[axis]);
      break;
    }
  }
  requestAnimationFrame(animate);
}
init();
