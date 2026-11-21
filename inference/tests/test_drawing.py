"""Zones drawn over the live video."""
import numpy as np

from carlyn_inference.drawing import draw, zone_area
from conftest import LEFT, zone

FULL = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_zone_area_is_the_share_of_the_image():
    assert zone_area(FULL) == 1.0
    assert zone_area(LEFT) == 0.5


def test_a_zone_covering_the_whole_image_does_not_tint_the_video(camera_factory):
    frame = np.full((200, 300, 3), 128, np.uint8)
    out = draw(frame, camera_factory(zone("z", "Work", polygon=FULL)), [], {}, 0.0)
    # Away from the zone label and the FPS text, the image keeps its colors (no green tint or border).
    assert (out[60:150, 100:280] == 128).all()


def test_a_partial_zone_is_still_tinted(camera_factory):
    frame = np.full((200, 300, 3), 128, np.uint8)
    out = draw(frame, camera_factory(zone("z", "Work", polygon=LEFT)), [], {}, 0.0)
    assert out[100, 75, 1] > 128 and (out[100, 250] == 128).all()   # left half greener, right half unchanged
