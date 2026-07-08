// gsplat-rt live viewer — renders the pipeline's splats as soft Gaussian discs
// and polls the server for scene / occupancy / stats. Three.js from the CDN
// importmap in index.html.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const POLL_MS = 500;                     // scene/stats refresh period

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
controls.zoomSpeed = 10.0;        // very snappy scroll/trackpad zoom

// A subtle ground grid gives the scene a sense of scale/orientation.
const grid = new THREE.GridHelper(10, 20, 0x1c2530, 0x141a20);
scene.add(grid);

// --- splat material: camera-facing discs with a radial Gaussian alpha falloff.
const splatMaterial = new THREE.ShaderMaterial({
  uniforms: { uSizeScale: { value: 800.0 } },
  transparent: true,
  depthWrite: false,
  blending: THREE.NormalBlending,
  vertexShader: /* glsl */`
    attribute vec3 aColor;
    attribute float aScale;
    attribute float aOpacity;
    uniform float uSizeScale;
    varying vec3 vColor;
    varying float vOpacity;
    void main() {
      vColor = aColor;
      vOpacity = aOpacity;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      float px = uSizeScale * aScale / max(-mv.z, 0.001);
      gl_PointSize = clamp(px, 1.0, 64.0);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: /* glsl */`
    varying vec3 vColor;
    varying float vOpacity;
    void main() {
      vec2 d = gl_PointCoord - vec2(0.5);
      float r2 = dot(d, d);
      if (r2 > 0.25) discard;
      float a = exp(-r2 * 8.0) * vOpacity;   // gaussian dab
      gl_FragColor = vec4(vColor, a);
    }`,
});

let points = null;
let framedOnce = false;
let lastCount = 0;
let lastBBox = null, lastCentroid = null;

function rebuildPoints(scn) {
  const n = scn.count | 0;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(scn.means, 3));
  geo.setAttribute('aColor', new THREE.Float32BufferAttribute(scn.colors, 3));
  geo.setAttribute('aScale', new THREE.Float32BufferAttribute(scn.scales, 1));
  geo.setAttribute('aOpacity', new THREE.Float32BufferAttribute(scn.opacities, 1));
  if (points) { scene.remove(points); points.geometry.dispose(); }
  points = new THREE.Points(geo, splatMaterial);
  points.frustumCulled = false;
  scene.add(points);

  lastBBox = scn.bbox;
  lastCentroid = centroid(scn.means);
  // Auto-frame on first data, and again whenever the scene goes empty→populated
  // (e.g. a live pipeline's first splats, or switching source without reload).
  if (n > 0 && (!framedOnce || lastCount === 0)) {
    frameTo(lastBBox, lastCentroid);
    framedOnce = true;
  }
  lastCount = n;
}

// Press F to re-fit the camera to the current cloud (handy if you orbit away).
window.addEventListener('keydown', (e) => {
  if ((e.key === 'f' || e.key === 'F') && lastBBox) frameTo(lastBBox, lastCentroid);
});

function centroid(means) {
  let x = 0, y = 0, z = 0;
  const n = means.length / 3;
  for (let i = 0; i < means.length; i += 3) { x += means[i]; y += means[i + 1]; z += means[i + 2]; }
  return n ? new THREE.Vector3(x / n, y / n, z / n) : new THREE.Vector3();
}

function frameTo(bbox, target) {
  const mn = new THREE.Vector3().fromArray(bbox.min);
  const mx = new THREE.Vector3().fromArray(bbox.max);
  const radius = Math.max(mx.distanceTo(mn) * 0.5, 0.5);
  // Look at the cloud's centroid (≈ origin for the demo sphere) and place the
  // camera so it sits centred on screen.
  controls.target.copy(target);
  camera.position.copy(target).add(new THREE.Vector3(1, 0.7, 1)
    .normalize().multiplyScalar(radius * 2.2));
  camera.near = radius / 100; camera.far = radius * 100;
  camera.updateProjectionMatrix();
  controls.update();               // apply the new target/position immediately
}

// --- occupancy panel (top-down floor plan) -------------------------------
const occCanvas = document.getElementById('occCanvas');
const occCtx = occCanvas.getContext('2d');
function drawOccupancy(occ) {
  if (!occ || !occ.w) return;
  const { w, h, data } = occ;                 // grid (X, Z) row-major
  occCanvas.width = w; occCanvas.height = h;
  const img = occCtx.createImageData(w, h);
  for (let x = 0; x < w; x++) {
    for (let z = 0; z < h; z++) {
      const v = data[x * h + z];
      // draw with depth (z) up: flip the row so it matches the PNG/ascii map
      const px = ((h - 1 - z) * w + x) * 4;
      let r, g, b;
      if (v > 0) { r = 220; g = 40; b = 40; }        // occupied
      else if (v === 0) { r = 235; g = 235; b = 235; } // free
      else { r = 26; g = 30; b = 36; }               // unknown
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
  renderer.setSize(w, h);          // also sets the canvas CSS size to fill the viewport
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  // Match on-screen disc size to projected world size (drawing-buffer pixels).
  const fovY = camera.fov * Math.PI / 180.0;
  splatMaterial.uniforms.uSizeScale.value =
    renderer.domElement.height / (2.0 * Math.tan(fovY / 2.0));
}
window.addEventListener('resize', resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

poll();
setInterval(poll, POLL_MS);
