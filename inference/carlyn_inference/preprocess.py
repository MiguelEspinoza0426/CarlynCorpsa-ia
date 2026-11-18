"""Optional contrast correction before detection, only inside a region of the frame (backlight from the gate, dark corners).

Applied to the frame used for detection; evidence images and the live video keep the original frame.
"""
from dataclasses import dataclass

import cv2
import numpy as np

MODES = ("none", "clahe", "gamma")


@dataclass(frozen=True)
class PreprocessConfig:
    mode: str = "none"
    gamma: float = 1.5            # > 1 brightens dark areas, < 1 darkens washed-out ones
    clip_limit: float = 2.0       # CLAHE contrast limit
    roi: tuple[float, float, float, float] | None = None  # normalized x1, y1, x2, y2; None = the SAHI ROI or the whole frame

    @staticmethod
    def parse(data: dict | None) -> "PreprocessConfig":
        """From the camera's `inference.preprocess` block (docs/contrato-integracion.md)."""
        if not data:
            return PreprocessConfig()
        mode = str(data.get("mode", "none")).lower()
        roi = data.get("roi")
        return PreprocessConfig(mode=mode if mode in MODES else "none", gamma=float(data.get("gamma", 1.5)),
                                clip_limit=float(data.get("clipLimit", 2.0)), roi=tuple(float(v) for v in roi) if roi else None)

    @property
    def enabled(self) -> bool:
        return self.mode != "none"


def gamma_lut(gamma: float) -> np.ndarray:
    return np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)


class Preprocessor:
    """One per camera: the CLAHE object and the gamma table are built once."""

    def __init__(self, config: PreprocessConfig, default_roi: tuple[float, float, float, float] | None = None):
        self.config = config
        self.roi = config.roi or default_roi or (0.0, 0.0, 1.0, 1.0)
        self._clahe = cv2.createCLAHE(clipLimit=config.clip_limit, tileGridSize=(8, 8)) if config.mode == "clahe" else None
        self._lut = gamma_lut(config.gamma) if config.mode == "gamma" else None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enabled:
            return frame
        h, w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self.roi
        x1, y1, x2, y2 = int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h)
        if x2 <= x1 or y2 <= y1:
            return frame
        out = frame.copy()
        region = out[y1:y2, x1:x2]
        if self._clahe is not None:
            # Only the lightness channel: colors (helmets, vests) stay as they are.
            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            out[y1:y2, x1:x2] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            out[y1:y2, x1:x2] = cv2.LUT(region, self._lut)
        return out
