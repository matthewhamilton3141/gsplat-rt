"""Source-camera intrinsics override in PipelineManager (run_live --tum-intrinsics
/ --camera-*). Real intrinsics must be rescaled from their native resolution into
the depth-input space the pipeline back-projects in, preserving fx != fy — which
the generic single-FOV fallback cannot express, and which is why the generic path
warps the live TUM map."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_manager import PipelineConfig, PipelineManager  # noqa: E402

# TUM freiburg1 pinhole model @ 640x480 (matches slam/tum_dataset.py).
_TUM = (517.306408, 516.469215, 318.643040, 255.313989)


def test_intrinsics_override_rescaled_to_depth_space():
    cfg = PipelineConfig(depth_input_w=518, depth_input_h=518,
                         camera_intrinsics=_TUM, camera_intrinsics_hw=(480, 640))
    pm = PipelineManager(cfg)
    sx, sy = 518 / 640.0, 518 / 480.0
    assert abs(pm._fx - _TUM[0] * sx) < 1e-6
    assert abs(pm._fy - _TUM[1] * sy) < 1e-6
    assert abs(pm._cx - _TUM[2] * sx) < 1e-6
    assert abs(pm._cy - _TUM[3] * sy) < 1e-6
    # Non-uniform 640x480 -> 518x518 resize must NOT collapse to fx == fy.
    assert abs(pm._fx - pm._fy) > 50.0


def test_intrinsics_hw_defaults_to_depth_size():
    # Omitting camera_intrinsics_hw means the values are already in depth space.
    cfg = PipelineConfig(depth_input_w=518, depth_input_h=518,
                         camera_intrinsics=(400.0, 410.0, 259.0, 250.0))
    pm = PipelineManager(cfg)
    assert (pm._fx, pm._fy, pm._cx, pm._cy) == (400.0, 410.0, 259.0, 250.0)


def test_fov_fallback_unchanged_without_override():
    cfg = PipelineConfig(depth_input_w=518, depth_input_h=518, camera_fov_deg=70.0)
    pm = PipelineManager(cfg)
    expect = (518 / 2.0) / math.tan(math.radians(70.0) / 2.0)
    assert abs(pm._fx - expect) < 1e-6
    assert pm._fx == pm._fy               # single FOV → square
    assert pm._cx == 259.0 and pm._cy == 259.0
