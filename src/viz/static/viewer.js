// gsplat-rt live viewer. Two render paths:
//   - anisotropic Gaussian splats (oriented ellipses via EWA covariance
//     projection, instanced quads, back-to-front sorted) when the scene carries
//     per-splat scale3 + quat;
//   - round isotropic discs (points) as a fallback for a raw point cloud.
// Three.js from the CDN importmap in index.html.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const POLL_MS = 500;

const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
camera.position.set(3, 2, 4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.zoomSpeed = 10.0;

const grid = new THREE.GridHelper(10, 20, 0x1c2530, 0x141a20);
scene.add(grid);

// Shared uniforms for the ellipse shader (focal length + viewport, in
// drawing-buffer pixels) and the disc shader (size scale).
const uFocal = { value: new THREE.Vector2(1000, 1000) };
const uViewport = { value: new THREE.Vector2(1, 1) };
const uSizeScale = { value: 800.0 };

// --- anisotropic splat material: instanced oriented ellipses (EWA splatting).
const ellipseMaterial = new THREE.ShaderMaterial({
  uniforms: { uFocal, uViewport },
  transparent: true, depthWrite: false, depthTest: false,
  blending: THREE.NormalBlending,
  vertexShader: /* glsl */`
    attribute vec3 iCenter;
    attribute vec3 iColor;
    attribute float iOpacity;
    attribute vec3 iScale;
    attribute vec4 iQuat;          // (w, x, y, z)
    uniform vec2 uFocal;
    uniform vec2 uViewport;
    varying vec3 vColor;
    varying float vOpacity;
    varying vec2 vQuad;

    mat3 quatToMat(vec4 q) {
      float w = q.x, x = q.y, y = q.z, z = q.w;
      return mat3(
        1.0 - 2.0*(y*y+z*z), 2.0*(x*y+w*z),       2.0*(x*z-w*y),
        2.0*(x*y-w*z),       1.0 - 2.0*(x*x+z*z), 2.0*(y*z+w*x),
        2.0*(x*z+w*y),       2.0*(y*z-w*x),       1.0 - 2.0*(x*x+y*y));
    }

    void main() {
      vColor = iColor;
      vOpacity = iOpacity;

      // 3D covariance Σ = (R S)(R S)ᵀ, with a Y-flip reflection on R to match the
      // Y-flipped centres (pipeline is +Y-down, Three.js +Y-up).
      mat3 R = quatToMat(iQuat);
      R[0].y = -R[0].y; R[1].y = -R[1].y; R[2].y = -R[2].y;
      mat3 S = mat3(iScale.x, 0.0, 0.0, 0.0, iScale.y, 0.0, 0.0, 0.0, iScale.z);
      mat3 M = R * S;
      mat3 Sigma = M * transpose(M);

      // Project Σ to a 2D screen-space covariance via the perspective Jacobian.
      vec4 t = modelViewMatrix * vec4(iCenter, 1.0);   // camera space
      float tz = t.z;
      mat3 J = mat3(
        uFocal.x / tz, 0.0,           0.0,
        0.0,           uFocal.y / tz, 0.0,
        -uFocal.x * t.x / (tz*tz), -uFocal.y * t.y / (tz*tz), 0.0);
      mat3 W = mat3(modelViewMatrix);
      mat3 T = J * W;
      mat3 cov = T * Sigma * transpose(T);

      float a = cov[0][0] + 0.3;      // low-pass dilation (matches CPU BLUR)
      float b = cov[0][1];
      float c = cov[1][1] + 0.3;
      float mid = 0.5 * (a + c);
      float det = a * c - b * b;
      float rad = sqrt(max(mid*mid - det, 0.01));
      float l1 = mid + rad;                 // eigenvalues (pixel²)
      float l2 = max(mid - rad, 0.01);
      vec2 e1 = normalize(vec2(b, l1 - a)); // principal axes
      vec2 e2 = vec2(-e1.y, e1.x);
      float r1 = min(3.0 * sqrt(l1), 1024.0);   // 3σ extents (pixels)
      float r2 = min(3.0 * sqrt(l2), 1024.0);

      vQuad = position.xy;                  // quad corner in [-1,1]
      vec2 offset = position.x * r1 * e1 + position.y * r2 * e2;   // pixels
      vec4 clip = projectionMatrix * t;
      clip.xy += (2.0 * offset / uViewport) * clip.w;   // pixel → clip offset
      gl_Position = clip;
    }`,
  fragmentShader: /* glsl */`
    varying vec3 vColor;
    varying float vOpacity;
    varying vec2 vQuad;
    void main() {
      float d2 = dot(vQuad, vQuad);
      if (d2 > 1.0) discard;
      float alpha = vOpacity * exp(-4.5 * d2);   // 3σ Gaussian across the quad
      if (alpha < 0.004) discard;
      gl_FragColor = vec4(vColor, alpha);
    }`,
});

// --- isotropic fallback: soft round discs (points), for a raw point cloud.
const discMaterial = new THREE.ShaderMaterial({
  uniforms: { uSizeScale },
  transparent: true, depthWrite: false, blending: THREE.NormalBlending,
  vertexShader: /* glsl */`
    attribute vec3 aColor;
    attribute float aScale;
    attribute float aOpacity;
    uniform float uSizeScale;
    varying vec3 vColor;
    varying float vOpacity;
    void main() {
      vColor = aColor; vOpacity = aOpacity;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = clamp(uSizeScale * aScale / max(-mv.z, 0.001), 1.0, 64.0);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: /* glsl */`
    varying vec3 vColor;
    varying float vOpacity;
    void main() {
      vec2 d = gl_PointCoord - vec2(0.5);
      float r2 = dot(d, d);
      if (r2 > 0.25) discard;
      gl_FragColor = vec4(vColor, exp(-r2 * 8.0) * vOpacity);
    }`,
});

let splat = null;          // current THREE.Points | THREE.Mesh
let splatKind = null;      // 'disc' | 'ellipse'
let sortAttrs = null;      // instance attrs to depth-sort (ellipse mode)
let framedOnce = false, lastCount = 0, lastBBox = null, lastCentroid = null;
let lastSortKey = null;
let lastScn = null;        // most recent scene, for re-render on mode toggle
let forceDiscs = false;    // 'A' key: force isotropic round points (A/B compare)
const _dir = new THREE.Vector3();

function flipY(means) {                  // pipeline +Y-down → Three.js +Y-up
  const m = Float32Array.from(means);
  for (let i = 1; i < m.length; i += 3) m[i] = -m[i];
  return m;
}

function centroid(m) {
  let x = 0, y = 0, z = 0; const n = m.length / 3;
  for (let i = 0; i < m.length; i += 3) { x += m[i]; y += m[i + 1]; z += m[i + 2]; }
  return n ? new THREE.Vector3(x / n, y / n, z / n) : new THREE.Vector3();
}

function clearSplat() {
  if (splat) { scene.remove(splat); splat.geometry.dispose(); splat = null; }
  sortAttrs = null;
}

function rebuildPoints(scn) {
  lastScn = scn;
  const n = scn.count | 0;
  const means = flipY(scn.means);
  const anisotropic = n > 0 && scn.scales3 && scn.quats && !forceDiscs;
  if (anisotropic) buildEllipses(scn, means, n);
  else buildDiscs(scn, means, n);

  const hasAniso = !!(scn.scales3 && scn.quats);
  const el = document.getElementById('s-render');
  if (el) {
    el.textContent = anisotropic ? 'splats (aniso)'
      : (hasAniso ? 'points (forced)' : 'points');
    el.style.color = anisotropic ? '#7ee3c7' : '#f0b072';
  }

  lastBBox = scn.bbox;
  lastCentroid = centroid(means);
  if (n > 0 && (!framedOnce || lastCount === 0)) {
    frameTo(lastBBox, lastCentroid); framedOnce = true;
  }
  lastCount = n;
}

function buildDiscs(scn, means, n) {
  clearSplat();
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(means, 3));
  geo.setAttribute('aColor', new THREE.Float32BufferAttribute(scn.colors, 3));
  geo.setAttribute('aScale', new THREE.Float32BufferAttribute(scn.scales, 1));
  geo.setAttribute('aOpacity', new THREE.Float32BufferAttribute(scn.opacities, 1));
  splat = new THREE.Points(geo, discMaterial);
  splat.frustumCulled = false;
  scene.add(splat);
  splatKind = 'disc';
}

function buildEllipses(scn, means, n) {
  clearSplat();
  const geo = new THREE.InstancedBufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(
    [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
  geo.setIndex([0, 1, 2, 0, 2, 3]);
  geo.instanceCount = n;
  const A = (arr, size) => new THREE.InstancedBufferAttribute(Float32Array.from(arr), size);
  geo.setAttribute('iCenter', new THREE.InstancedBufferAttribute(means, 3));
  geo.setAttribute('iColor', A(scn.colors, 3));
  geo.setAttribute('iOpacity', A(scn.opacities, 1));
  geo.setAttribute('iScale', A(scn.scales3, 3));
  geo.setAttribute('iQuat', A(scn.quats, 4));
  splat = new THREE.Mesh(geo, ellipseMaterial);
  splat.frustumCulled = false;
  scene.add(splat);
  splatKind = 'ellipse';
  sortAttrs = ['iCenter', 'iColor', 'iOpacity', 'iScale', 'iQuat']
    .map((k) => geo.getAttribute(k));
  lastSortKey = null;
}

// Back-to-front depth sort of the instances along the view direction (needed for
// correct alpha compositing). Gated on camera movement so it isn't every frame.
function maybeSort() {
  if (!sortAttrs) return;
  camera.getWorldDirection(_dir);
  const key = `${_dir.x.toFixed(2)},${_dir.y.toFixed(2)},${_dir.z.toFixed(2)}`;
  if (key === lastSortKey) return;
  lastSortKey = key;

  const centers = sortAttrs[0].array;
  const n = centers.length / 3;
  const order = Array.from({ length: n }, (_, i) => i);
  const depth = (i) => centers[i*3]*_dir.x + centers[i*3+1]*_dir.y + centers[i*3+2]*_dir.z;
  order.sort((a, b) => depth(b) - depth(a));   // farthest (largest along view) first

  for (const attr of sortAttrs) {
    const size = attr.itemSize, src = attr.array, out = new Float32Array(src.length);
    for (let i = 0; i < n; i++) {
      const s = order[i] * size, o = i * size;
      for (let k = 0; k < size; k++) out[o + k] = src[s + k];
    }
    attr.array.set(out);
    attr.needsUpdate = true;
  }
}

function frameTo(bbox, target) {
  const mn = new THREE.Vector3().fromArray(bbox.min);
  const mx = new THREE.Vector3().fromArray(bbox.max);
  const radius = Math.max(mx.distanceTo(mn) * 0.5, 0.5);
  controls.target.copy(target);
  camera.position.copy(target).add(new THREE.Vector3(1, 0.7, 1)
    .normalize().multiplyScalar(radius * 2.2));
  camera.near = radius / 100; camera.far = radius * 100;
  camera.updateProjectionMatrix();
  controls.update();
}

window.addEventListener('keydown', (e) => {
  if ((e.key === 'f' || e.key === 'F') && lastBBox) frameTo(lastBBox, lastCentroid);
  if (e.key === 'a' || e.key === 'A') {          // toggle anisotropic ↔ round points
    forceDiscs = !forceDiscs;
    if (lastScn) rebuildPoints(lastScn);
  }
});

// --- occupancy panel (top-down floor plan) -------------------------------
const occCanvas = document.getElementById('occCanvas');
const occCtx = occCanvas.getContext('2d');
function drawOccupancy(occ) {
  if (!occ || !occ.w) return;
  const { w, h, data } = occ;
  occCanvas.width = w; occCanvas.height = h;
  const img = occCtx.createImageData(w, h);
  for (let x = 0; x < w; x++) {
    for (let z = 0; z < h; z++) {
      const v = data[x * h + z];
      const px = ((h - 1 - z) * w + x) * 4;
      let r, g, b;
      if (v > 0) { r = 220; g = 40; b = 40; }
      else if (v === 0) { r = 235; g = 235; b = 235; }
      else { r = 26; g = 30; b = 36; }
      img.data[px] = r; img.data[px + 1] = g; img.data[px + 2] = b; img.data[px + 3] = 255;
    }
  }
  occCtx.putImageData(img, 0, 0);
}

// --- polling -------------------------------------------------------------
const $ = (id) => document.getElementById(id);
function setStatus(text, ok) { const s = $('status'); s.textContent = text; s.style.color = ok ? '#7ee3c7' : '#f0b072'; }

async function poll() {
  try {
    const scn = await (await fetch('api/scene')).json();
    rebuildPoints(scn);
    updateStats(scn.stats || {}, scn.count);
    setStatus('live', true);
  } catch (e) {
    setStatus('waiting for pipeline…', false);
  }
  try {
    const occ = await (await fetch('api/occupancy')).json();
    drawOccupancy(occ);
  } catch (e) { /* occupancy is optional */ }
}

function updateStats(stats, count) {
  $('s-source').textContent = stats.source ?? 'pipeline';
  $('s-count').textContent = (count ?? stats.count ?? 0).toLocaleString();
  $('s-fps').textContent = fmt(stats.fps);
  $('s-depth').textContent = fmt(stats.depth_ms, 1);
  $('s-backend').textContent = stats.depth_backend ?? '—';
  $('s-frames').textContent = stats.frames ?? '—';
  $('s-scale').textContent = fmt(stats.metric_scale, 3);
}
function fmt(v, d = 1) { return (v === undefined || v === null) ? '—' : Number(v).toFixed(d); }

// --- render loop + resize ------------------------------------------------
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const fovY = camera.fov * Math.PI / 180.0;
  const bh = renderer.domElement.height;          // drawing-buffer pixels
  uSizeScale.value = bh / (2.0 * Math.tan(fovY / 2.0));
  uFocal.value.set(uSizeScale.value, uSizeScale.value);
  uViewport.value.set(renderer.domElement.width, bh);
}
window.addEventListener('resize', resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  maybeSort();
  renderer.render(scene, camera);
}
animate();

poll();
setInterval(poll, POLL_MS);
