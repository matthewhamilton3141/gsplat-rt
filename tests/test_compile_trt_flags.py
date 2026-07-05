"""Unit tests for the TensorRT network-flag selection in compile_trt.

`make_network_flags` is the crux of the true-FP16 switch: it decides whether the
network is built EXPLICIT_BATCH and/or STRONGLY_TYPED. It's pure and takes the
`trt` module as an argument, so we exercise it here with fake modules that mimic
different TensorRT versions — no TensorRT install (or GPU) required.

Run:
    pytest tests/test_compile_trt_flags.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from depth.compile_trt import make_network_flags


class _Flags:
    """Stand-in for trt.NetworkDefinitionCreationFlag (enum values are bit indices)."""
    def __init__(self, **kw):
        for name, idx in kw.items():
            setattr(self, name, idx)


class _FakeTRT:
    def __init__(self, **kw):
        self.NetworkDefinitionCreationFlag = _Flags(**kw)


# TRT 8/9: EXPLICIT_BATCH present, no STRONGLY_TYPED.
_TRT9 = _FakeTRT(EXPLICIT_BATCH=0)
# TRT 10/11: STRONGLY_TYPED present, EXPLICIT_BATCH removed (implicit).
_TRT11 = _FakeTRT(STRONGLY_TYPED=1)


def test_weakly_typed_trt9_sets_only_explicit_batch():
    assert make_network_flags(_TRT9, strongly_typed=False) == (1 << 0)


def test_weakly_typed_trt11_sets_no_flags():
    # No EXPLICIT_BATCH attr on TRT 11 → implicit default → 0.
    assert make_network_flags(_TRT11, strongly_typed=False) == 0


def test_strongly_typed_trt11_sets_strongly_typed_bit():
    assert make_network_flags(_TRT11, strongly_typed=True) == (1 << 1)


def test_strongly_typed_on_old_trt_raises():
    # Asking for STRONGLY_TYPED on a TRT that lacks it must fail loudly, not
    # silently fall back to fp32.
    with pytest.raises(ValueError, match="TensorRT 10"):
        make_network_flags(_TRT9, strongly_typed=True)
