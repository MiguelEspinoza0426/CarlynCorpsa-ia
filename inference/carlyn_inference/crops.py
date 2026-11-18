"""Person crops for the second detection stage: EPP is searched inside each person, at the frame's full resolution."""
from .geometry import Box, center, contains

PixelRegion = tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels
MIN_CROP_PX = 16                         # smaller crops carry no usable EPP detail
OWNER_MARGIN = 0.1                       # same tolerance as the single-stage association


def crop_region(box: Box, frame_w: int, frame_h: int, margin: float) -> PixelRegion | None:
    """Pixel region of a person box enlarged by `margin` of its size on each side, clamped to the frame.

    Returns None when the region is too small to analyze.
    """
    x, y, w, h = box
    x1, y1 = (x - w * margin) * frame_w, (y - h * margin) * frame_h
    x2, y2 = (x + w * (1 + margin)) * frame_w, (y + h * (1 + margin)) * frame_h
    region = max(0, int(x1)), max(0, int(y1)), min(frame_w, int(round(x2))), min(frame_h, int(round(y2)))
    if region[2] - region[0] < MIN_CROP_PX or region[3] - region[1] < MIN_CROP_PX:
        return None
    return region


def to_frame_box(xyxy: tuple[float, float, float, float], region: PixelRegion, frame_w: int, frame_h: int) -> Box:
    """Box detected inside a crop (crop pixels, x1 y1 x2 y2) to normalized frame coordinates (x, y, w, h)."""
    ox, oy = region[0], region[1]
    x1, y1, x2, y2 = xyxy
    return (ox + x1) / frame_w, (oy + y1) / frame_h, (x2 - x1) / frame_w, (y2 - y1) / frame_h


def belongs_to(person: Box, epp: Box) -> bool:
    """Crops include a margin and may show neighbours: only EPP centered on the person counts."""
    return contains(person, center(epp), margin=OWNER_MARGIN)
