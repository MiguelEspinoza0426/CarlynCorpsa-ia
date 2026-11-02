"""Geometry in normalized image coordinates (0–1), independent of the camera resolution."""
import math

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x, y, width, height (top-left corner)


def foot_point(box: Box) -> Point:
    """Bottom-center of a person box: where the person stands, used for zone membership."""
    x, y, w, h = box
    return x + w / 2, y + h


def center(box: Box) -> Point:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray casting test."""
    px, py = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def distance_to_polygon(point: Point, polygon: list[Point]) -> float:
    """0 inside the polygon; otherwise the distance to the closest edge."""
    if point_in_polygon(point, polygon):
        return 0.0
    return min(_distance_to_segment(point, polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon)))


def _distance_to_segment(p: Point, a: Point, b: Point) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def contains(outer: Box, point: Point, margin: float = 0.0) -> bool:
    """Whether the point lies inside the box, optionally enlarged by a fraction of its size."""
    x, y, w, h = outer
    mx, my = w * margin, h * margin
    return x - mx <= point[0] <= x + w + mx and y - my <= point[1] <= y + h + my


def iou(a: Box, b: Box) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    inter = max(0.0, min(ax2, bx2) - max(a[0], b[0])) * max(0.0, min(ay2, by2) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
