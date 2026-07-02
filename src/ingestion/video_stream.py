import cv2
import queue
import threading
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
    ):
        self.source = source
        self.width = width
        self.height = height
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self.frames_captured = 0
        self.frames_dropped = 0

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
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
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
