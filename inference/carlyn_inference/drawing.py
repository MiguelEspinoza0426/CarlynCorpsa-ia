"""Annotations for the MJPEG stream and the event evidence."""
import cv2
import numpy as np

from .config import CameraConfig
from .geometry import Box, foot_point, point_in_polygon
from .rules import EventOut, PersonObs

ZONE_COLORS = {"Work": (0, 200, 0), "Circulation": (255, 170, 0), "Danger": (0, 140, 255), "Restricted": (0, 0, 230)}
OK, ALERT = (60, 200, 60), (40, 40, 230)


def _px(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x, y, bw, bh = box
    return int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h)


def draw(frame: np.ndarray, camera: CameraConfig, persons: list[PersonObs], counts: dict[str, int], fps: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    for zone in camera.zones:
        pts = np.array([[int(x * w), int(y * h)] for x, y in zone.polygon], np.int32)
        color = ZONE_COLORS.get(zone.type, (200, 200, 200))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], True, color, 2)
        label = f"{zone.name}: {counts.get(zone.id, 0)}"
        cv2.putText(out, label, (pts[:, 0].min() + 4, pts[:, 1].min() + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)

    for p in persons:
        missing = sorted(set().union(*[z.required_epp - p.epp for z in camera.zones
                                       if z.required_epp and point_in_polygon(foot_point(p.box), z.polygon)]))
        color = ALERT if missing or p.fallen else OK
        x1, y1, x2, y2 = _px(p.box, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"#{p.track_id}" + (" CAIDA?" if p.fallen else "") + (f" falta: {', '.join(missing)}" if missing else "")
        cv2.putText(out, text, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.putText(out, f"{camera.name}  {fps:.1f} FPS", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return out


def draw_event(frame: np.ndarray, event: EventOut) -> np.ndarray:
    """Highlights the box of the event in the evidence snapshot."""
    out = frame.copy()
    if event.bbox:
        h, w = out.shape[:2]
        x1, y1, x2, y2 = _px(event.bbox, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), ALERT, 3)
    return out


def encode_jpeg(frame: np.ndarray, quality: int, max_width: int | None = None) -> bytes:
    if max_width and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
