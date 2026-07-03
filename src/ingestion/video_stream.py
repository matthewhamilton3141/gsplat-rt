import cv2
import queue
import threading
import time
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
QUEUE_MAXSIZE = 4


class VideoStream:
    """Thread-safe video capture with a dedicated background reader thread.

    Frames are resized to (width x height) and placed into a bounded queue.
    When the queue is full the oldest frame is evicted to keep consumer latency low.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = TARGET_WIDTH,
        height: int = TARGET_HEIGHT,
        queue_size: int = QUEUE_MAXSIZE,
        loop: bool = False,
        realtime: bool = False,
        fps: Optional[float] = None,
    ):
        self.source = source
        self.width = width
        self.height = height
        # loop: rewind a finished file source instead of stopping (no-op for a
        #   live webcam, which never ends).
        # realtime: pace reads to the source frame rate instead of reading at
        #   disk speed — so a file plays like a live camera. Without this a file
        #   is consumed in ~1s regardless of its duration.
        self.loop = loop
        self.realtime = realtime
        self._fps_override = fps
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self.frames_captured = 0
        self.frames_dropped = 0
        self.loops_completed = 0

    def start(self) -> 'VideoStream':
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        # Minimize OpenCV's internal ring buffer to cut capture latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name='VideoCapture'
        )
        self._thread.start()
        logger.info(
            "VideoStream started: source=%s output=%dx%d", self.source, self.width, self.height
        )
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                # Releasing a capture while another thread is inside read() is
                # undefined in OpenCV (native deadlock/crash). Leak the handle
                # instead — the daemon thread and device die with the process.
                logger.warning(
                    "Capture thread still alive after 2.0s (blocked in read()?) — "
                    "skipping cap.release() to avoid a concurrent read/release race"
                )
                return
        if self._cap is not None:
            self._cap.release()
        logger.info(
            "VideoStream stopped — captured=%d dropped=%d",
            self.frames_captured, self.frames_dropped,
        )

    def get_frame(self, timeout: float = 0.1):
        """Return the next available frame, or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __iter__(self):
        while not self._stop_event.is_set():
            frame = self.get_frame()
            if frame is not None:
                yield frame

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def _capture_loop(self):
        period = self._frame_period()          # 0.0 when pacing is disabled
        next_deadline = time.monotonic()

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                if self.loop and not isinstance(self.source, int):
                    # Rewind the file and keep going. If the very next read also
                    # fails the file is unreadable, not just finished — stop.
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                    if not ret:
                        logger.warning("Loop rewind failed; stopping capture.")
                        break
                    self.loops_completed += 1
                else:
                    logger.warning("Frame read failed; source may be exhausted.")
                    break

            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            self.frames_captured += 1

            # Drop oldest frame rather than blocking the capture thread
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
            self._queue.put_nowait(frame)

            # Real-time pacing: sleep so frames enter at the source frame rate.
            if period > 0:
                next_deadline += period
                sleep_for = next_deadline - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_deadline = time.monotonic()   # fell behind; don't bank debt

    def _frame_period(self) -> float:
        """Seconds per frame for real-time pacing, or 0.0 if pacing is off."""
        if not self.realtime:
            return 0.0
        fps = self._fps_override or (self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0)
        if not fps or fps <= 0 or fps > 240:      # missing/garbage metadata
            fps = 30.0
        return 1.0 / fps
