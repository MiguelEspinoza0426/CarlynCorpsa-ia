import numpy as np

from carlyn_inference.rules import EventOut, PersonObs
from carlyn_inference.worker import evidence_image

PERSON = PersonObs(1, (0.2, 0.1, 0.1, 0.8), 0.9)
EVENT = EventOut("EppMissing", "z1", "r1", 0.0, None, 0.8, 1, 1, PERSON.box, {})


def textured_frame() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)


def test_evidence_pixelates_faces_when_the_camera_asks_for_it():
    frame = textured_frame()
    out = evidence_image(frame, EVENT, [PERSON], [], blur=True)
    assert out[26:40, 46:55].std() < frame[26:40, 46:55].std() * 0.5


def test_evidence_keeps_faces_when_pixelation_is_off():
    frame = textured_frame()
    out = evidence_image(frame, EVENT, [PERSON], [], blur=False)
    assert np.array_equal(out[26:40, 46:55], frame[26:40, 46:55])   # face area untouched
    assert not np.array_equal(out, frame)                            # the event box is still drawn
