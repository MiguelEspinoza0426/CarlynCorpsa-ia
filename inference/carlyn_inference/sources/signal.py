"""Frames that carry no video: the DVR keeps the RTSP stream alive but sends a black screen, a uniform color or its
logo placeholder ("annke" on black) when the channel has no camera signal. Such a camera must not look Online.
"""
import cv2
import numpy as np

PROBE_SIZE = (64, 36)         # small grayscale copy: cheap, and compression noise averages out
UNIFORM_TOLERANCE = 12        # gray levels around the median considered "the same color"
UNIFORM_FRACTION = 0.97       # a logo or the date overlay may cover the rest
FROZEN_DIFF = 0.3             # mean absolute difference below which two frames are the same image


def probe(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(gray, PROBE_SIZE, interpolation=cv2.INTER_AREA).astype(np.int16)


def is_placeholder(frame: np.ndarray) -> bool:
    """Black, uniform, or uniform with a small logo/overlay: almost every pixel close to the median gray level."""
    return _uniform(probe(frame))


def _uniform(small: np.ndarray) -> bool:
    return float(np.mean(np.abs(small - np.median(small)) <= UNIFORM_TOLERANCE)) >= UNIFORM_FRACTION


def is_same_image(a: np.ndarray, b: np.ndarray) -> bool:
    """Two probes of an unchanged image: a real camera always has some sensor or compression noise."""
    return float(np.mean(np.abs(a - b))) < FROZEN_DIFF


class SignalMonitor:
    """Tracks for how long a camera has sent placeholder or frozen frames. One per camera."""

    def __init__(self, placeholder_sec: float = 5.0, frozen_sec: float = 30.0):
        self.placeholder_sec, self.frozen_sec = placeholder_sec, frozen_sec
        self._placeholder_since: float | None = None
        self._frozen_since: float | None = None
        self._last_probe: np.ndarray | None = None

    def update(self, frame: np.ndarray, now: float) -> None:
        small = probe(frame)
        placeholder = _uniform(small)
        self._placeholder_since = (now if self._placeholder_since is None else self._placeholder_since) if placeholder else None
        frozen = self._last_probe is not None and is_same_image(small, self._last_probe)
        self._frozen_since = (now if self._frozen_since is None else self._frozen_since) if frozen else None
        self._last_probe = small

    def reason(self, now: float) -> str | None:
        """Why the camera has no usable signal, or None if it has."""
        if self._placeholder_since is not None and now - self._placeholder_since >= self.placeholder_sec:
            return "Sin señal: el DVR envía una imagen negra, uniforme o su logo"
        if self._frozen_since is not None and now - self._frozen_since >= self.frozen_sec:
            return "Sin señal: la imagen está congelada"
        return None
