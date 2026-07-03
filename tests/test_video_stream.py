"""Benchmark: pull TARGET_FRAMES frames and assert throughput >= 60 FPS.

Synthetic mode (default): generates a temp video file so the test runs
without camera hardware or macOS camera-permission grants.

Webcam mode: set VIDEO_SOURCE=0 (or another integer index) in the environment
AND ensure the terminal has been granted Camera access in
System Settings > Privacy & Security > Camera.
"""
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingestion.video_stream import VideoStream

TARGET_FRAMES = 1000
MIN_FPS = 60.0
_USE_CAMERA = os.environ.get('VIDEO_SOURCE', '').strip() != ''


def _make_synthetic_video(path: str, n_frames: int = TARGET_FRAMES + 200):
    """Write a small synthetic video file at 120 FPS, 1280x720."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 120.0, (1280, 720))
    rng = np.random.default_rng(42)
    for _ in range(n_frames):
        frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_stream_fps():
    if _USE_CAMERA:
        source = int(os.environ['VIDEO_SOURCE'])
        stream = VideoStream(source=source)
        try:
            stream.start()
        except RuntimeError as exc:
            pytest.skip(f"Camera unavailable: {exc}")
    else:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        try:
            _make_synthetic_video(video_path)
            stream = VideoStream(source=video_path)
            stream.start()
            _run_benchmark(stream)
        finally:
            os.unlink(video_path)
        return

    _run_benchmark(stream)


def _run_benchmark(stream: VideoStream):
    frames_received = 0
    start = time.perf_counter()

    while frames_received < TARGET_FRAMES:
        frame = stream.get_frame(timeout=2.0)
        if frame is None:
            break
        assert frame.shape == (480, 640, 3), f"Unexpected frame shape: {frame.shape}"
        frames_received += 1

    elapsed = time.perf_counter() - start
    stream.stop()

    if frames_received == 0:
        pytest.skip("Video source produced no frames.")

    fps = frames_received / elapsed
    dropped_pct = 100.0 * stream.frames_dropped / max(stream.frames_captured, 1)

    print(f"\n--- VideoStream Benchmark ---")
    print(f"Frames received : {frames_received}")
    print(f"Frames captured : {stream.frames_captured}")
    print(f"Frames dropped  : {stream.frames_dropped} ({dropped_pct:.1f}%)")
    print(f"Elapsed time    : {elapsed:.3f}s")
    print(f"Average FPS     : {fps:.1f}")
    print(f"Requirement     : >= {MIN_FPS:.0f} FPS  {'PASS' if fps >= MIN_FPS else 'FAIL'}")
    print(f"----------------------------")

    assert fps >= MIN_FPS, f"FPS {fps:.1f} is below minimum {MIN_FPS}"


def _make_short_video(path: str, n_frames: int, fps: float = 30.0):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 240))
    rng = np.random.default_rng(1)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    writer.release()


def test_loop_reads_past_end_of_file():
    """A looped file source must keep producing frames beyond its length."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        path = tmp.name
    try:
        _make_short_video(path, n_frames=10)
        stream = VideoStream(source=path, loop=True)
        stream.start()
        time.sleep(0.3)
        stream.stop()
        # 10-frame clip, looped for 0.3s at disk speed → far more than 10 reads
        assert stream.frames_captured > 10
        assert stream.loops_completed >= 1
    finally:
        os.unlink(path)


def test_realtime_paces_to_frame_rate():
    """Real-time pacing must NOT consume a 30-frame/1s clip instantly."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        path = tmp.name
    try:
        _make_short_video(path, n_frames=30, fps=30.0)   # ~1 s of footage
        stream = VideoStream(source=path, realtime=True)
        stream.start()
        time.sleep(0.3)                                   # ~0.3 s in
        captured = stream.frames_captured
        stream.stop()
        # At 30 fps, ~0.3 s should yield well under the full 30 frames
        assert 2 <= captured < 25, f"pacing off: {captured} frames in ~0.3s"
    finally:
        os.unlink(path)


if __name__ == '__main__':
    test_video_stream_fps()
