"""Scene snapshots for the web viewer — from a pipeline, a .ply, or synthetic.

A :class:`SceneSnapshot` is the viewer's wire format: Gaussian centres + per-splat
colour / size / opacity, plus the top-down occupancy grid and the live stats. The
three sources produce one:

  PipelineSceneSource   — reads a running PipelineManager (optimized Gaussians if
                          the finalize stage ran, else the raw accumulating cloud
                          coloured by height).
  PlySceneSource        — a static INRIA 3DGS .ply (what the finalize stage writes).
  SyntheticSceneSource  — a procedural scene, for tests / a viewer smoke-run with
                          no GPU and no pipeline.

Pure numpy — no torch/cv2/pxr — so it runs and unit-tests anywhere.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# INRIA 3DGS .ply DC colour normalisation (matches gaussian.finalize).
_SH_C0 = 0.28209479177387814


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def height_colormap(values: np.ndarray) -> np.ndarray:
    """Map a 1-D array to RGB in [0,1] via a blue→cyan→yellow→red hue sweep.

    Used to colour the raw point cloud (which has no per-splat colour yet) so the
    live scene is legible — height reads as warmth.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return np.zeros((0, 3))
    lo, hi = float(np.min(v)), float(np.max(v))
    t = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    # Hue 240° (blue) → 0° (red) as t goes 0→1; full saturation/value.
    h = (1.0 - t) * (240.0 / 360.0)
    return _hsv_to_rgb(h, np.ones_like(h), np.ones_like(h))


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV→RGB, all inputs in [0,1]. Returns (N,3)."""
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


@dataclass
class SceneSnapshot:
    """One frame of scene state for the viewer."""

    means: np.ndarray                       # (N, 3) float
    colors: np.ndarray                      # (N, 3) float in [0, 1]
    scales: np.ndarray                      # (N,) world-space splat size
    opacities: np.ndarray                   # (N,) in [0, 1]
    occupancy: Optional[np.ndarray] = None  # (X, Z) int {-1,0,1} or None
    stats: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    def bbox(self):
        """(min[3], max[3]) of the centres, or unit cube when empty."""
        if self.count == 0:
            return [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
        return (self.means.min(axis=0).tolist(), self.means.max(axis=0).tolist())

    def decimated(self, max_points: int, rng=None) -> "SceneSnapshot":
        """A uniformly sub-sampled copy with at most ``max_points`` splats."""
        n = self.count
        if max_points <= 0 or n <= max_points:
            return self
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        return SceneSnapshot(
            self.means[idx], self.colors[idx], self.scales[idx],
            self.opacities[idx], self.occupancy, self.stats)


def _normalise(means, colors, scales, opacities, n) -> SceneSnapshot:
    """Coerce raw arrays into a well-formed SceneSnapshot (fills sane defaults)."""
    means = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    if colors is None:
        colors = height_colormap(means[:, 1]) if n else np.zeros((0, 3))
    colors = np.clip(np.asarray(colors, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    if scales is None:
        scales = np.full(n, 0.05)
    scales = np.asarray(scales, dtype=np.float64).ravel()
    if scales.ndim > 1 or scales.shape[0] != n:              # (N,3) → mean axis
        scales = np.asarray(scales, dtype=np.float64).reshape(n, -1).mean(axis=1)
    if opacities is None:
        opacities = np.full(n, 0.9)
    opacities = np.clip(np.asarray(opacities, dtype=np.float64).ravel(), 0.0, 1.0)
    return SceneSnapshot(means, colors, scales, opacities)


# ---------------------------------------------------------------------------
# .ply reader (INRIA 3DGS layout, matches gaussian.finalize.write_ply)
# ---------------------------------------------------------------------------

def read_ply(path: str) -> SceneSnapshot:
    """Read an INRIA 3DGS binary .ply into a SceneSnapshot.

    Understands the field layout our finalize stage writes (x y z, f_dc_0..2,
    opacity, scale_0..2, rot_0..3) as well as plain ``x y z [red green blue]``
    point clouds. SH DC → RGB, sigmoid(opacity), exp(scale). Little-endian
    float32 (or uchar colours) as declared in the header.
    """
    with open(path, "rb") as fh:
        # --- header ---
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a .ply file")
        fmt = fh.readline().strip()
        if b"binary_little_endian" not in fmt:
            raise ValueError(f"{path}: only binary_little_endian is supported ({fmt!r})")
        n = 0
        props = []            # list of (name, struct_char, nbytes)
        _ply_t = {b"float": ("f", 4), b"float32": ("f", 4), b"double": ("d", 8),
                  b"uchar": ("B", 1), b"uint8": ("B", 1), b"int": ("i", 4)}
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF in header")
            tok = line.split()
            if tok[0] == b"element" and tok[1] == b"vertex":
                n = int(tok[2])
            elif tok[0] == b"property":
                ch, nb = _ply_t.get(tok[1], ("f", 4))
                props.append((tok[2].decode(), ch, nb))
            elif tok[0] == b"end_header":
                break

        names = [p[0] for p in props]
        stride = sum(p[2] for p in props)
        buf = fh.read(n * stride)

    raw = np.frombuffer(buf, dtype=np.uint8).reshape(n, stride) if n else \
        np.zeros((0, stride), np.uint8)

    def col(name):
        """Extract one named property column as float64 (N,)."""
        off = 0
        for pname, ch, nb in props:
            if pname == name:
                dt = {"f": "<f4", "d": "<f8", "B": "u1", "i": "<i4"}[ch]
                return raw[:, off:off + nb].copy().view(dt).ravel().astype(np.float64)
            off += nb
        return None

    means = np.stack([col("x"), col("y"), col("z")], axis=-1) if n else np.zeros((0, 3))

    if "f_dc_0" in names:                       # 3DGS splat file
        f_dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=-1)
        colors = _SH_C0 * f_dc + 0.5
        opacities = _sigmoid(col("opacity")) if "opacity" in names else None
        if "scale_0" in names:
            scales = np.exp(np.stack([col("scale_0"), col("scale_1"),
                                      col("scale_2")], axis=-1)).mean(axis=1)
        else:
            scales = None
    elif "red" in names:                        # plain coloured point cloud
        colors = np.stack([col("red"), col("green"), col("blue")], axis=-1) / 255.0
        scales = opacities = None
    else:                                        # bare xyz
        colors = scales = opacities = None

    return _normalise(means, colors, scales, opacities, n)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class PlySceneSource:
    """Static source: one .ply, re-read from disk on each snapshot (cheap enough)
    so a viewer picks up a finalize-stage rewrite live."""

    def __init__(self, path: str):
        self.path = path

    def snapshot(self) -> SceneSnapshot:
        snap = read_ply(self.path)
        snap.stats = {"source": "ply", "count": snap.count}
        return snap


class SyntheticSceneSource:
    """Procedural scene (a coloured sphere shell) — GPU-free viewer smoke tests."""

    def __init__(self, n: int = 3000, seed: int = 0):
        rng = np.random.default_rng(seed)
        d = rng.standard_normal((n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        self._means = d * (1.0 + 0.05 * rng.standard_normal((n, 1)))
        self._colors = 0.5 + 0.5 * d           # direction → colour
        self._scales = np.full(n, 0.03)
        self._opac = np.full(n, 0.85)
        self._tick = 0

    def snapshot(self) -> SceneSnapshot:
        self._tick += 1
        return SceneSnapshot(
            self._means.copy(), np.clip(self._colors, 0, 1).copy(),
            self._scales.copy(), self._opac.copy(),
            occupancy=None, stats={"source": "synthetic", "tick": self._tick,
                                   "count": int(self._means.shape[0])})


class PipelineSceneSource:
    """Live source: reads a running PipelineManager without touching its hot path.

    Prefers the optimized Gaussians (per-splat colour/opacity/scale) once the
    finalize stage has run; otherwise the raw accumulating point cloud, coloured
    by height. Duck-typed: any object exposing ``latest_gaussians()``,
    ``latest_occupancy()`` and ``stats()`` works (handy for tests)."""

    def __init__(self, manager):
        self.manager = manager

    def snapshot(self) -> SceneSnapshot:
        m = self.manager
        model = getattr(m, "optimized_gaussians", None)
        if model is not None:
            snap = _normalise(model.means, model.rgb, model.scales,
                              model.alphas, model.num_gaussians)
        else:
            pts = m.latest_gaussians()
            pts = np.zeros((0, 3)) if pts is None else np.asarray(pts, np.float64)
            # Real per-point source-frame colour if the pipeline sampled it, else
            # None → _normalise falls back to the height ramp. Truncate to the
            # common length (the writer may append between the two snapshots).
            cols = (m.latest_gaussian_colors()
                    if hasattr(m, "latest_gaussian_colors") else None)
            if cols is not None and len(cols):
                k = min(len(pts), len(cols))
                pts, cols = pts[:k], np.asarray(cols)[:k]
            else:
                cols = None
            snap = _normalise(pts, cols, None, None, pts.shape[0])

        occ = m.latest_occupancy() if hasattr(m, "latest_occupancy") else None
        snap.occupancy = None if occ is None else np.asarray(occ)
        snap.stats = dict(m.stats()) if hasattr(m, "stats") else {}
        snap.stats["count"] = snap.count
        return snap
