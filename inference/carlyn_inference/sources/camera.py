"""Camera reader reused from ml/inference/run_camara.py (LectorCamara).

Reads the source in a background thread and keeps only the most recent frame, so slow
inference never accumulates delay. Reconnects with exponential backoff. Adapted for the
service: status for heartbeats, and video files that loop at their real frame rate (tests).
"""
import logging
import os
import threading
import time
from pathlib import Path

# Force RTSP over TCP (more stable than UDP); must be set before importing cv2.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402

log = logging.getLogger(__name__)


class CameraReader:
    """Reads a webcam index, RTSP/HTTP URL or video file, keeping only the latest frame."""

    def __init__(self, source, name: str = ""):
        self.source = source
        self.name = name or str(source)
        self.frame = None
        self.ts = 0.0
        self.connected = False
        self.error: str | None = None
        self.active = True
        self.lock = threading.Lock()
        self._is_file = isinstance(source, str) and Path(source).is_file()
        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"reader-{self.name}")
        self.thread.start()

    def _open(self):
        if isinstance(self.source, str):
            return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        return cv2.VideoCapture(self.source, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)

    def _loop(self):
        wait = 1
        while self.active:
            cap = self._open()
            if not cap.isOpened():
                self.connected, self.error = False, "No se pudo abrir la fuente de video"
                log.warning("[%s] not connected, retrying in %ss", self.name, wait)
                time.sleep(wait)
                wait = min(wait * 2, 30)
                continue
            log.info("[%s] connected", self.name)
            self.connected, self.error, wait = True, None, 1
            # Video files are played at their own frame rate and looped.
            frame_interval = 1 / (cap.get(cv2.CAP_PROP_FPS) or 25) if self._is_file else 0
            while self.active:
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    if self._is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.connected, self.error = False, "Señal perdida"
                    log.warning("[%s] signal lost, reconnecting", self.name)
                    break
                with self.lock:
                    self.frame, self.ts = frame, time.time()
                if frame_interval:
                    time.sleep(max(0.0, frame_interval - (time.monotonic() - started)))
            cap.release()

    def read(self):
        """Latest frame and its capture timestamp (frame is None until the first one arrives)."""
        with self.lock:
            return self.frame, self.ts

    def close(self):
        self.active = False
        self.thread.join(timeout=3)  # lets the thread call cap.release(); otherwise the process may hang
