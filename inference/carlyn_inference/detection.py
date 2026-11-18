"""YOLO detection + tracking (one or two stages), EPP-to-person association, fall detection with pose, and face blurring."""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .crops import belongs_to, crop_region, to_frame_box
from .geometry import Box, center, contains, iou
from .preprocess import Preprocessor
from .slicing import SliceConfig, cut_by_tile, merge, tile_regions
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


def epp_passes(label: str, conf: float, minimums: dict[str, float]) -> bool:
    """Per-class minimum confidence (EppClass.MinConfidence in the platform); the model's own threshold already applied."""
    return conf >= minimums.get(label, 0.0)


def predict_batched(model, images: list[np.ndarray], max_batch: int, **kwargs) -> list:
    """One GPU call per `max_batch` images: TensorRT engines accept up to the batch they were exported with."""
    results = []
    for i in range(0, len(images), max_batch):
        results += model.predict(images[i:i + max_batch], **kwargs)
    return results


RawPerson = tuple[int, Box, float]          # track id, normalized box, confidence
EppBox = tuple[str, Box, float]             # label, normalized box, confidence


def _boxes(result) -> list[tuple[str, Box, float, int | None]]:
    """(label, normalized x-y-w-h box, confidence, track id) of an Ultralytics result."""
    if result.boxes is None or not len(result.boxes):
        return []
    ids = result.boxes.id.int().tolist() if result.boxes.id is not None else [None] * len(result.boxes)
    return [(result.names[cls], (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]), conf, track_id)
            for cls, conf, xyxy, track_id in zip(result.boxes.cls.int().tolist(), result.boxes.conf.tolist(),
                                                 result.boxes.xyxyn.tolist(), ids)]


class SingleStage:
    """One model finds persons and EPP in the whole frame (reduced to imgsz); EPP goes to the person that contains it."""

    def __init__(self, model, person_class: str, epp_classes: frozenset[str], device, imgsz: int, conf: float):
        self.model, self.person_class, self.epp_classes = model, person_class, epp_classes
        self.device, self.imgsz, self.conf = device, imgsz, conf
        self.epp_min_confidence: dict[str, float] = {}

    def run(self, frame: np.ndarray) -> tuple[list[RawPerson], dict[int, set[str]], list[EppBox]]:
        result = self.model.track(frame, persist=True, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                  tracker="bytetrack.yaml", verbose=False)[0]
        persons, epp_boxes = [], []
        for label, box, conf, track_id in _boxes(result):
            if label == self.person_class and track_id is not None:
                persons.append((track_id, box, conf))
            elif label in self.epp_classes and epp_passes(label, conf, self.epp_min_confidence):
                epp_boxes.append((label, box, conf))
        return persons, associate_epp(persons, epp_boxes), epp_boxes


class SlicedPersons:
    """First stage: persons in the whole frame plus, with SAHI, tiles of the camera's ROI, in one batch; then ByteTrack.

    Distant workers (~20x60 px in Camara 1) disappear when the whole frame is reduced to 640 px; inside a tile they
    keep their pixels. Detection and tracking are separate so the tracker's buffer can be set in seconds.
    """

    def __init__(self, model, person_classes: frozenset[str], device, imgsz: int, conf: float, config: SliceConfig, tracker=None,
                 max_batch: int = 8):
        # Every region is inferred at imgsz: tiles smaller than imgsz are enlarged (320 px tiles at 640 = 2x zoom).
        self.model, self.device, self.imgsz, self.conf, self.config = model, device, imgsz, conf, config
        self.max_batch = max_batch
        # Only the person class goes through NMS and the tracker (COCO "person" or our "persona").
        self.classes = [i for i, name in model.names.items() if name in person_classes]
        self.half = device != "cpu"
        self.tracker = tracker or byte_tracker()

    def track(self, frame: np.ndarray) -> list[RawPerson]:
        from ultralytics.engine.results import Boxes

        h, w = frame.shape[:2]
        regions = tile_regions(w, h, self.config)
        results = predict_batched(self.model, [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in regions], self.max_batch,
                                  imgsz=self.imgsz, conf=self.conf, classes=self.classes, device=self.device, half=self.half,
                                  verbose=False)
        found = []
        for r, region in zip(results, regions):
            if r.boxes is not None and len(r.boxes):
                xyxy = np.asarray(r.boxes.xyxy.tolist(), dtype=float).reshape(-1, 4) + [region[0], region[1], region[0], region[1]]
                found.append((xyxy, np.asarray(r.boxes.conf.tolist(), dtype=float), cut_by_tile(xyxy, region, w, h)))
        xyxy, conf, cut = (np.concatenate(parts) for parts in zip(*found)) if found else (np.empty((0, 4)), np.empty(0), None)
        boxes, scores = merge(xyxy, conf, self.config.iou, self.config.min_box_px, cut)
        data = np.column_stack([boxes, scores, np.zeros(len(scores))]).astype(np.float32) if len(scores) else np.empty((0, 6), np.float32)
        tracks = self.tracker.update(Boxes(data, (h, w)), frame)
        # Plain floats: these boxes end up in the JSON of events (numpy float32 is not serializable).
        return [(int(t[4]), (float(t[0] / w), float(t[1] / h), float((t[2] - t[0]) / w), float((t[3] - t[1]) / h)), float(t[5]))
                for t in tracks]


class TrackedPersons(SlicedPersons):
    """First stage without SAHI: only the whole frame, reduced to imgsz."""

    def __init__(self, model, person_classes: frozenset[str], device, imgsz: int, conf: float, tracker=None, max_batch: int = 8):
        super().__init__(model, person_classes, device, imgsz, conf, SliceConfig(), tracker, max_batch)


def byte_tracker(buffer_frames: int | None = None):
    """Ultralytics' ByteTrack with its default settings; `buffer_frames` = frames a lost person is remembered."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml
    args = YAML.load(check_yaml("bytetrack.yaml"))
    if buffer_frames:
        args["track_buffer"] = buffer_frames
    return BYTETracker(IterableSimpleNamespace(**args))


class TwoStage:
    """Persons first (TrackedPersons or SlicedPersons); then EPP in each person's crop, taken from the full-resolution frame.

    Small or distant workers keep their pixels for the EPP model, instead of being reduced with the whole frame.
    """

    def __init__(self, persons: TrackedPersons | SlicedPersons, epp_model, epp_classes: frozenset[str], device,
                 epp_imgsz: int, epp_conf: float, margin: float, min_height_px: int = 0, max_batch: int = 8):
        self.persons, self.epp_model = persons, epp_model
        self.epp_classes, self.device = epp_classes, device
        self.epp_imgsz, self.epp_conf, self.margin = epp_imgsz, epp_conf, margin
        self.min_height_px = min_height_px
        self.max_batch = max_batch
        self.epp_min_confidence: dict[str, float] = {}

    def run(self, frame: np.ndarray) -> tuple[list[RawPerson], dict[int, set[str]], list[EppBox]]:
        persons = self.persons.track(frame)
        epp: dict[int, set[str]] = {track_id: set() for track_id, _, _ in persons}
        h, w = frame.shape[:2]
        # Persons too small to judge their EPP are not cropped (the Detector marks them as not evaluated).
        crops = [(track_id, box, region) for track_id, box, _ in persons
                 if box[3] * h >= self.min_height_px and (region := crop_region(box, w, h, self.margin)) is not None]
        if not crops:
            return persons, epp, []
        # A single batch for all the persons of the frame.
        results = predict_batched(self.epp_model, [frame[y1:y2, x1:x2] for _, _, (x1, y1, x2, y2) in crops], self.max_batch,
                                  conf=self.epp_conf, imgsz=self.epp_imgsz, device=self.device, verbose=False)
        epp_boxes = []
        for (track_id, person_box, region), crop_result in zip(crops, results):
            if crop_result.boxes is None:
                continue
            for cls, conf, xyxy in zip(crop_result.boxes.cls.int().tolist(), crop_result.boxes.conf.tolist(),
                                       crop_result.boxes.xyxy.tolist()):
                label = crop_result.names[cls]
                if label not in self.epp_classes or not epp_passes(label, conf, self.epp_min_confidence):
                    continue
                box = to_frame_box(tuple(xyxy), region, w, h)
                if belongs_to(person_box, box):
                    epp[track_id].add(label)
                    epp_boxes.append((label, box, conf))
        return persons, epp, epp_boxes


class Detector:
    """One instance per camera: Ultralytics trackers keep state per model instance."""

    def __init__(self, stage: SingleStage | TwoStage, pose_model=None, object_model=None, device=None, imgsz: int = 640,
                 min_person_height_px: int = 0, preprocess: Preprocessor | None = None):
        self.stage, self.pose, self.objects = stage, pose_model, object_model
        self.device, self.imgsz = device, imgsz
        self.min_person_height_px = min_person_height_px
        self.preprocess = preprocess

    def set_epp_min_confidence(self, minimums: dict[str, float]) -> None:
        """Applied from the next frame on: changing a class threshold in the platform needs no restart."""
        self.stage.epp_min_confidence = dict(minimums)

    def detect(self, frame: np.ndarray, detect_objects: bool) -> Detections:
        h, w = frame.shape[:2]
        # Persons and EPP are searched in the corrected frame; pose, evidence and live video use the original one.
        raw_persons, epp, epp_boxes = self.stage.run(self.preprocess(frame) if self.preprocess else frame)
        lying = [p for p in raw_persons if p[1][2] * w > FALL_ASPECT * p[1][3] * h]
        keypoints = self._pose(frame) if lying and self.pose else []
        persons = []
        for t, box, conf in raw_persons:
            evaluated = box[3] * h >= self.min_person_height_px
            persons.append(PersonObs(t, box, round(conf, 3), frozenset(epp[t]) if evaluated else frozenset(),
                                     self._is_fallen(box, keypoints, w, h) if (t, box, conf) in lying else False, evaluated))
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


DETECTION_MODES = ("two_stage", "single")
COCO_PERSON = "person"


def create_detector(settings, person_class: str, epp_classes: frozenset[str], with_objects: bool, inference=None,
                    fps: float | None = None) -> Detector:
    """Loads the models of one camera according to the node settings (DETECTION_MODE) and its `inference` block."""
    from ultralytics import YOLO  # imported lazily so pure modules can be tested without torch

    if inference is not None:
        settings = inference.apply_to(settings)   # per-camera person confidence, image size and minimum height

    device = settings.torch_device
    slicing = inference.slicing if inference else None
    epp_model = YOLO(settings.model_path)
    if settings.detection_mode == "single":
        stage = SingleStage(epp_model, person_class, epp_classes, device, settings.imgsz, settings.conf)
    else:
        person_model, classes = YOLO(settings.person_model), frozenset({person_class, COCO_PERSON})
        # A person hidden behind a column for a few seconds keeps the same track id (and the same dwell session).
        tracker = byte_tracker(round(settings.track_buffer_sec * fps) if fps else None)
        if slicing and slicing.enabled:
            persons = SlicedPersons(person_model, classes, device, settings.imgsz, settings.person_conf, slicing, tracker,
                                    settings.max_batch)
        else:
            persons = TrackedPersons(person_model, classes, device, settings.imgsz, settings.person_conf, tracker, settings.max_batch)
        stage = TwoStage(persons, epp_model, epp_classes, device, settings.epp_imgsz, settings.conf, settings.epp_crop_margin,
                         settings.min_person_height_px, settings.max_batch)
    pose = YOLO(settings.pose_model) if settings.pose_model else None
    objects = YOLO(settings.object_model) if with_objects and settings.object_model else None
    preprocess = None
    if inference and inference.preprocess.enabled:
        # Without its own ROI, the correction covers the SAHI region (where distant workers are) or the whole frame.
        preprocess = Preprocessor(inference.preprocess, slicing.roi if slicing and slicing.enabled else None)
    return Detector(stage, pose, objects, device, settings.imgsz, settings.min_person_height_px, preprocess)


PIXELATION_BLOCKS = 6       # a face region is reduced to about 6x6 blocks: not recognizable


def blur_faces(frame: np.ndarray, persons: list[PersonObs], keypoints: list[tuple[Box, np.ndarray]], padding: float = 0.0) -> np.ndarray:
    """Pixelates faces: around the face keypoints when available, otherwise the top of each person box.

    `padding` enlarges every region by that fraction of its size on each side, for frames newer than the detections.
    """
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
        px, py = int((x2 - x1) * padding), int((y2 - y1) * padding)
        x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
        if x2 - x1 > 2 and y2 - y1 > 2:
            roi = out[y1:y2, x1:x2]
            small = cv2.resize(roi, (PIXELATION_BLOCKS, PIXELATION_BLOCKS), interpolation=cv2.INTER_AREA)
            out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
