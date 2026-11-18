"""Sliced inference (SAHI): a full-frame pass for nearby persons plus 640 px tiles inside a ROI for distant ones.

Pure numpy: tiles, size filter and the merge of overlapping detections. Adapted from sliced_detector.py.
"""
from dataclasses import dataclass

import numpy as np

PixelRegion = tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels
EDGE_PX = 2                              # a box this close to an inner tile edge was cut by the tile


@dataclass(frozen=True)
class SliceConfig:
    """Per camera (comes with the camera in the node configuration)."""
    enabled: bool = False
    tile_size: int = 640
    overlap: float = 0.2
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # normalized x1, y1, x2, y2
    full_frame: bool = True
    iou: float = 0.5
    min_box_px: int = 8

    @staticmethod
    def parse(data: dict | None) -> "SliceConfig":
        """From the camera's `inference.slice` block (docs/contrato-integracion.md); missing values keep the defaults."""
        if not data:
            return SliceConfig()
        roi = data.get("roi") or (0.0, 0.0, 1.0, 1.0)
        return SliceConfig(enabled=bool(data.get("enabled", False)), tile_size=int(data.get("tileSize", 640)),
                           overlap=float(data.get("overlap", 0.2)), roi=tuple(float(v) for v in roi))


def tile_starts(start: int, end: int, tile: int, step: int) -> list[int]:
    """Fewest tile origins that cover [start, end) with at least the requested overlap, spread evenly."""
    length = end - start
    if length <= tile:
        return [start]
    count = int(np.ceil((length - tile) / step)) + 1
    return [start + round(i * (length - tile) / (count - 1)) for i in range(count)]


def tile_regions(frame_w: int, frame_h: int, config: SliceConfig) -> list[PixelRegion]:
    """Pixel regions to analyze: the whole frame (optional) followed by the tiles of the ROI."""
    regions = [(0, 0, frame_w, frame_h)] if config.full_frame or not config.enabled else []
    if not config.enabled:
        return regions
    rx1, ry1, rx2, ry2 = config.roi
    x1, y1, x2, y2 = int(rx1 * frame_w), int(ry1 * frame_h), int(rx2 * frame_w), int(ry2 * frame_h)
    tile, step = config.tile_size, max(1, int(config.tile_size * (1 - config.overlap)))
    for ty in tile_starts(y1, y2, tile, step):
        for tx in tile_starts(x1, x2, tile, step):
            region = (tx, ty, min(tx + tile, x2), min(ty + tile, y2))
            if region[2] > region[0] and region[3] > region[1]:
                regions.append(region)
    return regions


def cut_by_tile(xyxy: np.ndarray, region: PixelRegion, frame_w: int, frame_h: int) -> np.ndarray:
    """Whether each box (frame pixels) touches an edge of its tile that is not an edge of the frame."""
    x1, y1, x2, y2 = region
    return (((xyxy[:, 0] - x1 <= EDGE_PX) & (x1 > 0)) | ((x2 - xyxy[:, 2] <= EDGE_PX) & (x2 < frame_w))
            | ((xyxy[:, 1] - y1 <= EDGE_PX) & (y1 > 0)) | ((y2 - xyxy[:, 3] <= EDGE_PX) & (y2 < frame_h)))


def merge(xyxy: np.ndarray, conf: np.ndarray, iou: float, min_box_px: int, cut: np.ndarray | None = None,
          ios: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Greedy NMS by confidence over the detections of all regions; returns the kept boxes and confidences.

    Besides IoU, a box cut by a tile edge that lies mostly inside another one (intersection over the smaller box) is the
    same person: both are merged into their union, so the person keeps its full height whichever scored higher.
    Only cut boxes are merged this way; two real persons, one in front of the other, stay apart.
    """
    cut = np.zeros(len(xyxy), dtype=bool) if cut is None else cut
    widths, heights = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
    candidates = np.flatnonzero((widths >= min_box_px) & (heights >= min_box_px))
    order = candidates[np.argsort(-conf[candidates], kind="stable")]
    boxes, scores = [], []
    while len(order):
        best, rest = xyxy[order[0]].copy(), order[1:]
        ix1, iy1 = np.maximum(best[0], xyxy[rest, 0]), np.maximum(best[1], xyxy[rest, 1])
        ix2, iy2 = np.minimum(best[2], xyxy[rest, 2]), np.minimum(best[3], xyxy[rest, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        area_best, area_rest = (best[2] - best[0]) * (best[3] - best[1]), widths[rest] * heights[rest]
        overlapping = inter / (area_best + area_rest - inter) > iou
        contained = (inter / np.minimum(area_best, area_rest) > ios) & (cut[rest] | cut[order[0]])
        for j in rest[contained & ~overlapping]:
            best = np.array([min(best[0], xyxy[j, 0]), min(best[1], xyxy[j, 1]), max(best[2], xyxy[j, 2]), max(best[3], xyxy[j, 3])])
        boxes.append(best)
        scores.append(conf[order[0]])
        order = rest[~(overlapping | contained)]
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(scores, dtype=float)
