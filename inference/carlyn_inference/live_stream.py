"""Live video of a camera, rendered at its own rate (STREAM_FPS), independent of the analysis rate.

Analysis is expensive (GPU) and runs at the camera's target FPS; the video shown to people only needs the
raw frames with the most recent detections drawn on top, which is cheap. Frames are rendered only while
somebody is watching.
"""
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .config import CameraConfig
from .detection import blur_faces
from .drawing import draw, encode_jpeg
from .geometry import Box
from .rules import PersonObs

# Faces are pixelated where the last analysis saw them. Between analyses people move, so the regions are
# enlarged; and if the analysis stalls, the video freezes instead of showing faces at outdated positions.
BLUR_PADDING = 0.5
MAX_OVERLAY_AGE_SEC = 1.0


@dataclass(frozen=True)
class Overlay:
    """Result of the last analyzed frame, drawn over the following raw frames."""
    camera: CameraConfig
    persons: list[PersonObs]
    pose_keypoints: list[tuple[Box, np.ndarray]]
    counts: dict[str, int]
    fps: float
    at: float = field(default_factory=time.time)


class LiveStream:
    def __init__(self, read_frame: Callable[[], tuple[np.ndarray | None, float]], fps: int, jpeg_quality: int, max_width: int,
                 blur: bool, name: str = ""):
        self._read_frame = read_frame
        self.fps, self._jpeg_quality, self._max_width, self._blur = max(1, fps), jpeg_quality, max_width, blur
        self.overlay: Overlay | None = None
        self.latest_jpeg: bytes | None = None
        self.rendered = 0   # frames rendered so far (to measure the real rate)
        self._viewers = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"stream-{name}")
        self._thread.start()

    @contextmanager
    def watching(self):
        """Marks a viewer for the duration of the block; rendering only happens while there is one."""
        with self._lock:
            self._viewers += 1
        self._wake.set()
        try:
            yield
        finally:
            with self._lock:
                self._viewers -= 1

    def render(self, frame: np.ndarray, now: float | None = None) -> bytes | None:
        """JPEG of a raw frame with the latest detections, or None while there is no recent analysis."""
        overlay = self.overlay
        if overlay is None or (now or time.time()) - overlay.at > MAX_OVERLAY_AGE_SEC:
            return None
        out = draw(frame, overlay.camera, overlay.persons, overlay.counts, overlay.fps)
        if self._blur:
            out = blur_faces(out, overlay.persons, overlay.pose_keypoints, padding=BLUR_PADDING)
        return encode_jpeg(out, self._jpeg_quality, self._max_width)

    def close(self) -> None:
        self._running = False
        self._wake.set()
        self._thread.join(timeout=2)

    def _loop(self) -> None:
        last_ts = 0.0
        while self._running:
            if self._viewers <= 0:
                self._wake.clear()
                self._wake.wait(1.0)
                continue
            started = time.time()
            frame, ts = self._read_frame()
            if frame is not None and ts != last_ts:
                last_ts = ts
                jpeg = self.render(frame)
                if jpeg is not None:
                    self.latest_jpeg = jpeg
                    self.rendered += 1
            time.sleep(max(0.001, 1 / self.fps - (time.time() - started)))
