"""YOLO detection + tracking, EPP-to-person association, fall detection with pose, and face blurring."""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Box, center, contains, iou
from .rules import ObjectObs, PersonObs

# COCO classes that may obstruct walkways (used only if an object model is configured).
OBSTACLE_CLASSES = {"backpack", "suitcase", "chair", "bench", "handbag", "bottle", "potted plant", "box"}
# COCO pose keypoints: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 11-12 hips.
FACE_KEYPOINTS = [0, 1, 2, 3, 4]
FALL_ASPECT = 1.0           # width / height (in pixels) above which a person may be lying down
FALL_TORSO_DEG = 60         # torso angle from the vertical considered "on the ground"


@dataclass
class Detections:
    persons: list[PersonObs]
    objects: list[ObjectObs]
    epp_boxes: list[tuple[str, Box, float]]
    pose_keypoints: list[tuple[Box, np.ndarray]]   # (person box, keypoints xy normalized, shape (17, 2))


def associate_epp(persons: list[tuple[int, Box, float]], epp_boxes: list[tuple[str, Box, float]]) -> dict[int, set[str]]:
    """Assigns each EPP box to the person whose (slightly enlarged) box contains its center; the smallest one wins."""
    assigned: dict[int, set[str]] = {track_id: set() for track_id, _, _ in persons}
    for label, box, _ in epp_boxes:
        point = center(box)
        owners = [(w * h, track_id) for track_id, (x, y, w, h), _ in persons if contains((x, y, w, h), point, margin=0.1)]
        if owners:
            assigned[min(owners)[1]].add(label)
    return assigned


def torso_angle(keypoints: np.ndarray) -> float | None:
    """Angle between the shoulders→hips line and the vertical, in degrees (0 = standing)."""
    shoulders, hips = keypoints[[5, 6]], keypoints[[11, 12]]
    if (shoulders <= 0).all() or (hips <= 0).all():
        return None
    s, h = shoulders[shoulders.sum(axis=1) > 0].mean(axis=0), hips[hips.sum(axis=1) > 0].mean(axis=0)
    dx, dy = h[0] - s[0], h[1] - s[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))


class Detector:
    """One instance per camera: Ultralytics trackers keep state per model instance."""

    def __init__(self, model_path: str, pose_model_path: str | None, object_model_path: str | None,
                 person_class: str, epp_classes: frozenset[str], device, imgsz: int, conf: float):
        from ultralytics import YOLO  # imported lazily so pure modules can be tested without torch

        self.model = YOLO(model_path)
        self.pose = YOLO(pose_model_path) if pose_model_path else None
        self.objects = YOLO(object_model_path) if object_model_path else None
        self.person_class, self.epp_classes = person_class, epp_classes
        self.device, self.imgsz, self.conf = device, imgsz, conf
        self.model_version = model_path.replace("\\", "/").split("/")[-3] if model_path.count("/") + model_path.count("\\") >= 2 else model_path

    def detect(self, frame: np.ndarray, detect_objects: bool) -> Detections:
        h, w = frame.shape[:2]
        result = self.model.track(frame, persist=True, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                  tracker="bytetrack.yaml", verbose=False)[0]
        names = result.names
        raw_persons, epp_boxes = [], []
        if result.boxes is not None and len(result.boxes):
            ids = result.boxes.id.int().tolist() if result.boxes.id is not None else [None] * len(result.boxes)
            for cls, conf, xyxy, track_id in zip(result.boxes.cls.int().tolist(), result.boxes.conf.tolist(),
                                                 result.boxes.xyxyn.tolist(), ids):
                box = (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
                label = names[cls]
                if label == self.person_class and track_id is not None:
                    raw_persons.append((track_id, box, conf))
                elif label in self.epp_classes:
                    epp_boxes.append((label, box, conf))

        epp = associate_epp(raw_persons, epp_boxes)
        lying = [p for p in raw_persons if p[1][2] * w > FALL_ASPECT * p[1][3] * h]
        keypoints = self._pose(frame) if lying and self.pose else []
        persons = [PersonObs(t, box, round(conf, 3), frozenset(epp[t]), self._is_fallen(box, keypoints, w, h) if (t, box, conf) in lying else False)
                   for t, box, conf in raw_persons]
        objects = self._objects(frame) if detect_objects and self.objects else []
        return Detections(persons, objects, epp_boxes, keypoints)

    def _pose(self, frame: np.ndarray) -> list[tuple[Box, np.ndarray]]:
        result = self.pose.predict(frame, conf=0.3, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if result.keypoints is None or result.boxes is None:
            return []
        out = []
        for xyxy, kp in zip(result.boxes.xyxyn.tolist(), result.keypoints.xyn.cpu().numpy()):
            out.append(((xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]), kp))
        return out

    @staticmethod
    def _is_fallen(box: Box, keypoints: list[tuple[Box, np.ndarray]], w: int, h: int) -> bool:
        match = max(keypoints, key=lambda k: iou(box, k[0]), default=None)
        if match is None or iou(box, match[0]) < 0.3:
            return False  # lying shape without pose confirmation is not enough
        angle = torso_angle(match[1] * np.array([w, h]))
        return angle is not None and angle >= FALL_TORSO_DEG

    def _objects(self, frame: np.ndarray) -> list[ObjectObs]:
        result = self.objects.track(frame, persist=True, conf=0.4, imgsz=self.imgsz, device=self.device,
                                    tracker="bytetrack.yaml", verbose=False)[0]
        if result.boxes is None or result.boxes.id is None:
            return []
        out = []
        for cls, conf, xyxy, track_id in zip(result.boxes.cls.int().tolist(), result.boxes.conf.tolist(),
                                             result.boxes.xyxyn.tolist(), result.boxes.id.int().tolist()):
            label = result.names[cls]
            if label in OBSTACLE_CLASSES:
                out.append(ObjectObs(track_id, label, (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]), round(conf, 3)))
        return out

    def face_keypoints(self, frame: np.ndarray) -> list[tuple[Box, np.ndarray]]:
        return self._pose(frame) if self.pose else []


PIXELATION_BLOCKS = 6       # a face region is reduced to about 6x6 blocks: not recognizable


def blur_faces(frame: np.ndarray, persons: list[PersonObs], keypoints: list[tuple[Box, np.ndarray]]) -> np.ndarray:
    """Pixelates faces: around the face keypoints when available, otherwise the top of each person box."""
    out = frame.copy()
    h, w = out.shape[:2]
    regions = []
    for _, kp in keypoints:
        face = kp[FACE_KEYPOINTS]
        face = face[face.sum(axis=1) > 0]
        if len(face):
            cx, cy = face.mean(axis=0) * [w, h]
            r = max(12.0, float(np.ptp(face[:, 0]) * w) * 1.2, 0.04 * w)
            regions.append((int(cx - r), int(cy - r), int(cx + r), int(cy + r)))
    covered = [r for r in regions]
    for p in persons:
        x, y, bw, bh = p.box
        head = (int(x * w), int(y * h), int((x + bw) * w), int((y + min(bh * 0.22, bw * 1.1)) * h))
        if not any(_overlaps(head, r) for r in covered):
            regions.append(head)
    for x1, y1, x2, y2 in regions:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 > 2 and y2 - y1 > 2:
            roi = out[y1:y2, x1:x2]
            small = cv2.resize(roi, (PIXELATION_BLOCKS, PIXELATION_BLOCKS), interpolation=cv2.INTER_AREA)
            out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
