"""Detection stages with fake models: no torch or weights needed."""
import numpy as np
import pytest

from carlyn_inference.detection import Detector, SingleStage, SlicedPersons, TrackedPersons, TwoStage
from carlyn_inference.slicing import SliceConfig


class _List(list):
    def tolist(self):
        return list(self)

    def int(self):
        return self


class _Boxes:
    def __init__(self, rows, size, ids=None):
        # rows: (class index, confidence, x1, y1, x2, y2 in pixels of an image of `size` = (w, h))
        w, h = size
        self.cls = _List(r[0] for r in rows)
        self.conf = _List(r[1] for r in rows)
        self.xyxy = _List(list(r[2:]) for r in rows)
        self.xyxyn = _List([r[2] / w, r[3] / h, r[4] / w, r[5] / h] for r in rows)
        self.id = _List(ids) if ids is not None else None

    def __len__(self):
        return len(self.cls)


class _Result:
    def __init__(self, names, boxes):
        self.names, self.boxes = names, boxes


class _PersonModel:
    names = {0: "person", 2: "car"}

    def __init__(self, rows, ids):
        self.rows, self.ids, self.calls = rows, ids, []

    def predict(self, crops, **kwargs):
        self.calls.append(kwargs)
        h, w = crops[0].shape[:2]
        return [_Result(self.names, _Boxes(self.rows, (w, h)))]


class _IdTracker:
    """Gives each detection the next of the given ids; None = not confirmed yet (dropped)."""

    def __init__(self, ids):
        self.ids = ids

    def update(self, boxes, img):
        rows = [[*xyxy, track_id, conf, 0, i] for i, (xyxy, conf, track_id) in
                enumerate(zip(boxes.xyxy.tolist(), boxes.conf.tolist(), self.ids)) if track_id is not None]
        return np.array(rows).reshape(-1, 8)


class _EppModel:
    names = {0: "casco", 1: "mandil", 2: "persona"}

    def __init__(self, rows_per_crop):
        self.rows_per_crop, self.crops = rows_per_crop, []

    def predict(self, crops, **kwargs):
        self.crops = crops
        return [_Result(self.names, _Boxes(rows, (c.shape[1], c.shape[0]))) for c, rows in zip(crops, self.rows_per_crop)]


FRAME = np.zeros((1000, 2000, 3), dtype=np.uint8)
EPP = frozenset({"casco", "mandil"})


def _two_stage(person_rows, ids, epp_rows):
    persons, epp = _PersonModel(person_rows, ids), _EppModel(epp_rows)
    tracked = TrackedPersons(persons, frozenset({"persona", "person"}), "cpu", 640, 0.3, _IdTracker(ids))
    return TwoStage(tracked, epp, EPP, "cpu", 320, 0.25, 0.15), persons, epp


def test_two_stage_finds_epp_in_full_resolution_crops():
    # Person of 100x300 px at (1000, 500); its crop with 15 % margin starts at (985, 455).
    stage, persons, epp = _two_stage([(0, 0.8, 1000, 500, 1100, 800)], [7],
                                     [[(1, 0.6, 30, 150, 110, 250), (2, 0.9, 0, 0, 130, 390)]])
    raw, assigned, boxes = stage.run(FRAME)

    assert [t for t, _, _ in raw] == [7]
    assert assigned == {7: {"mandil"}}                 # "persona" inside the crop is not EPP
    assert epp.crops[0].shape[:2] == (390, 130)        # crop taken from the full frame, not from the 640 px input
    label, box, _ = boxes[0]
    assert label == "mandil"
    assert box == pytest.approx(((985 + 30) / 2000, (455 + 150) / 1000, 80 / 2000, 100 / 1000))
    assert persons.calls[0]["classes"] == [0]          # only "person" goes through NMS and the tracker


def test_two_stage_applies_the_minimum_confidence_of_each_class():
    stage, _, _ = _two_stage([(0, 0.8, 1000, 500, 1100, 800)], [7],
                             [[(1, 0.6, 30, 150, 110, 250), (0, 0.4, 40, 20, 90, 60)]])
    detector = Detector(stage)
    detector.set_epp_min_confidence({"casco": 0.5})
    _, assigned, _ = stage.run(FRAME)
    assert assigned == {7: {"mandil"}}                 # casco at 0.4 is below its 0.5 minimum; mandil keeps CONF


def test_two_stage_ignores_epp_of_a_neighbour_seen_in_the_margin():
    # EPP at the left edge of the crop: it belongs to someone standing next to the person.
    stage, _, _ = _two_stage([(0, 0.8, 1000, 500, 1100, 800)], [7], [[(0, 0.7, 0, 100, 4, 120)]])
    _, assigned, boxes = stage.run(FRAME)
    assert assigned == {7: set()}
    assert boxes == []


def test_two_stage_skips_untracked_and_tiny_persons():
    # The 5x10 px box is dropped before tracking (min_box_px); the large one is not confirmed by the tracker yet.
    stage, _, epp = _two_stage([(0, 0.8, 1000, 500, 1100, 800), (0, 0.5, 10, 10, 15, 20)], [None], [[]])
    raw, assigned, _ = stage.run(FRAME)
    assert raw == [] and assigned == {}
    assert epp.crops == []


class _SingleModel:
    names = {0: "persona", 1: "casco"}

    def track(self, frame, **kwargs):
        h, w = frame.shape[:2]
        return [_Result(self.names, _Boxes([(0, 0.9, 1000, 500, 1100, 800), (1, 0.7, 1030, 500, 1070, 540)], (w, h), [4, 5]))]


def test_single_stage_keeps_the_previous_behaviour():
    raw, assigned, boxes = SingleStage(_SingleModel(), "persona", EPP, "cpu", 640, 0.25).run(FRAME)
    assert [t for t, _, _ in raw] == [4]
    assert assigned == {4: {"casco"}}
    assert [b[0] for b in boxes] == ["casco"]


def test_small_persons_are_tracked_but_their_epp_is_not_judged():
    # 100x300 px person and a 150 px minimum: evaluated. A 400 px minimum: tracked, not cropped, EPP not judged.
    stage, _, epp = _two_stage([(0, 0.8, 1000, 500, 1100, 800)], [7], [[(1, 0.6, 30, 150, 110, 250)]])
    stage.min_height_px = 400
    person = Detector(stage, min_person_height_px=400).detect(FRAME, detect_objects=False).persons[0]
    assert epp.crops == []
    assert (person.track_id, person.epp, person.epp_evaluated) == (7, frozenset(), False)


def test_detector_builds_person_observations_from_any_stage():
    stage, _, _ = _two_stage([(0, 0.81234, 1000, 500, 1100, 800)], [7], [[(1, 0.6, 30, 150, 110, 250)]])
    detections = Detector(stage).detect(FRAME, detect_objects=False)
    person = detections.persons[0]
    assert (person.track_id, person.confidence, person.epp, person.fallen) == (7, 0.812, frozenset({"mandil"}), False)


class _TilePersonModel:
    """Finds one distant person (40x120 px) only when it gets a tile, and a nearby one in the whole frame."""
    names = {0: "person"}

    def __init__(self):
        self.calls = []

    def predict(self, crops, **kwargs):
        self.calls.append(([c.shape[:2] for c in crops], kwargs))
        out = []
        for i, crop in enumerate(crops):
            h, w = crop.shape[:2]
            if i == 0:
                rows = [(0, 0.9, 1500, 600, 1700, 1000)]           # nearby person, whole frame
            elif i == 1:
                rows = [(0, 0.6, 100, 50, 140, 170)]               # distant person inside the first tile
            else:
                rows = []
            out.append(_Result(self.names, _Boxes(rows, (w, h))))
        return out


class _FakeTracker:
    def __init__(self):
        self.seen = None

    def update(self, boxes, img):
        self.seen = boxes
        return np.array([[*xyxy, i + 1, conf, 0, i] for i, (xyxy, conf) in enumerate(zip(boxes.xyxy.tolist(), boxes.conf.tolist()))],
                        dtype=np.float32)   # like Ultralytics: float32


def test_sliced_persons_adds_distant_people_from_the_roi_tiles():
    model, tracker = _TilePersonModel(), _FakeTracker()
    config = SliceConfig(enabled=True, roi=(0.0, 0.0, 1.0, 0.3))          # 2000x300 px band: one row of tiles
    persons = SlicedPersons(model, frozenset({"person"}), "cpu", 640, 0.3, config, tracker).track(FRAME)

    shapes, kwargs = model.calls[0]
    assert shapes[0] == (1000, 2000) and all(s[0] == 300 for s in shapes[1:])   # one batch: whole frame + tiles
    assert kwargs["half"] is False and kwargs["classes"] == [0]
    assert len(persons) == 2
    distant = min(persons, key=lambda p: p[1][3])
    assert all(type(v) is float for _, box, conf in persons for v in (*box, conf))   # JSON-serializable
    assert distant[1] == pytest.approx((100 / 2000, 50 / 1000, 40 / 2000, 120 / 1000))


def test_half_precision_can_be_turned_off_on_gpu():
    model = _TilePersonModel()
    SlicedPersons(model, frozenset({"person"}), 0, 640, 0.3, SliceConfig(), _FakeTracker(), half=False).track(FRAME)
    assert model.calls[0][1]["half"] is False


def test_predictions_are_split_in_batches_the_engine_accepts():
    from carlyn_inference.detection import predict_batched

    class Model:
        def __init__(self):
            self.sizes = []

        def predict(self, images, **kwargs):
            self.sizes.append(len(images))
            return list(images)

    model = Model()
    assert predict_batched(model, list(range(19)), 8, conf=0.2) == list(range(19))
    assert model.sizes == [8, 8, 3]


def test_tracker_buffer_is_set_in_frames_from_seconds():
    from carlyn_inference.detection import byte_tracker
    assert byte_tracker(32).max_frames_lost == 32       # 4 s at 8 FPS
    assert byte_tracker().max_frames_lost == 30         # Ultralytics' default
