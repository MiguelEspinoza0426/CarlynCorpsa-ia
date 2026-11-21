"""Annotations for the MJPEG stream and the event evidence."""
import cv2
import numpy as np

from .config import CameraConfig
from .geometry import Box, foot_point, point_in_polygon
from .rules import EventOut, PersonObs

ZONE_COLORS = {"Work": (0, 200, 0), "Circulation": (255, 170, 0), "Danger": (0, 140, 255), "Restricted": (0, 0, 230)}
OK, ALERT, UNJUDGED = (60, 200, 60), (40, 40, 230), (200, 200, 200)


def missing_epp(camera: CameraConfig, person: PersonObs) -> list[str]:
    """Required EPP the person lacks in the zones where they stand; nothing when their EPP was not judged (too small)."""
    if not person.epp_evaluated:
        return []
    return sorted(set().union(*[z.required_epp - person.epp for z in camera.zones
                                if z.required_epp and point_in_polygon(foot_point(person.box), z.polygon)]))


def _px(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x, y, bw, bh = box
    return int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h)


# A zone covering this share of the image or more is not tinted nor outlined: it would only tint the whole video.
FULL_FRAME_ZONE = 0.9


def zone_area(polygon) -> float:
    """Area of a normalized polygon (shoelace formula), as a share of the image."""
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]))) / 2


def draw(frame: np.ndarray, camera: CameraConfig, persons: list[PersonObs], counts: dict[str, int], fps: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    for zone in camera.zones:
        pts = np.array([[int(x * w), int(y * h)] for x, y in zone.polygon], np.int32)
        color = ZONE_COLORS.get(zone.type, (200, 200, 200))
        if zone_area(list(zone.polygon)) < FULL_FRAME_ZONE:
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(out, [pts], True, color, 2)
        label = f"{zone.name}: {counts.get(zone.id, 0)}"
        cv2.putText(out, label, (pts[:, 0].min() + 4, pts[:, 1].min() + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)

    for p in persons:
        missing = missing_epp(camera, p)
        color = ALERT if missing or p.fallen else OK if p.epp_evaluated else UNJUDGED
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
