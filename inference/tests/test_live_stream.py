"""The live video is rendered at its own rate from raw frames plus the latest detections."""
import time
from dataclasses import replace

import numpy as np

from carlyn_inference.detection import blur_faces
from carlyn_inference.live_stream import MAX_OVERLAY_AGE_SEC, LiveStream, Overlay
from carlyn_inference.rules import PersonObs
from conftest import zone

PERSON = PersonObs(1, (0.4, 0.2, 0.2, 0.6), 0.9)


def noise(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (240, 320, 3), dtype=np.uint8)


class Source:
    """Camera that delivers a new frame on every read (faster than any stream rate)."""

    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1
        return noise(self.reads), float(self.reads)


def overlay(camera_factory, at: float | None = None) -> Overlay:
    camera = camera_factory(zone("z", "Work"))
    return Overlay(camera, [PERSON], [], {"z": 1}, 5.0) if at is None else Overlay(camera, [PERSON], [], {"z": 1}, 5.0, at)


def test_frames_are_rendered_at_the_stream_rate_only_while_watched(camera_factory):
    source = Source()
    stream = LiveStream(source.read, fps=30, jpeg_quality=70, max_width=320, blur=True)
    try:
        stream.overlay = overlay(camera_factory)
        time.sleep(0.3)
        assert source.reads == 0 and stream.latest_jpeg is None      # nobody watching: nothing is rendered

        with stream.watching():
            deadline = time.time() + 1.0
            while time.time() < deadline:
                stream.overlay = overlay(camera_factory)             # analysis keeps running (5 fps in production)
                time.sleep(0.2)
            assert stream.latest_jpeg[:2] == b"\xff\xd8"
        assert 20 <= stream.rendered <= 34                           # ~30 fps, far above the analysis rate
        assert stream.latest_jpeg is None                            # the next viewer does not start with an old frame
    finally:
        stream.close()


def test_video_freezes_instead_of_showing_faces_at_outdated_positions(camera_factory):
    stream = LiveStream(lambda: (None, 0.0), fps=30, jpeg_quality=70, max_width=320, blur=True)
    try:
        assert stream.render(noise()) is None                                         # no analysis yet
        stream.overlay = overlay(camera_factory, at=time.time() - MAX_OVERLAY_AGE_SEC - 0.5)
        assert stream.render(noise()) is None                                         # analysis stalled
        stream.overlay = overlay(camera_factory)
        assert stream.render(noise()) is not None
    finally:
        stream.close()


def test_a_camera_can_turn_off_the_pixelation_of_its_live_video(camera_factory):
    stream = LiveStream(lambda: (None, 0.0), fps=30, jpeg_quality=95, max_width=320, blur=True)
    try:
        blurred = overlay(camera_factory)
        stream.overlay = blurred
        with_blur = stream.render(noise())
        stream.overlay = replace(blurred, camera=replace(blurred.camera, blur_faces=False), at=time.time())
        assert stream.render(noise()) != with_blur          # same frame and detections, faces left as they are
    finally:
        stream.close()


def test_blur_padding_covers_a_face_that_moved_since_the_last_analysis():
    frame = noise()
    h, w = frame.shape[:2]
    # Just left of the head region of the last detection (person box x from 0.4): only padding reaches it.
    ys, xs = slice(int(0.22 * h), int(0.30 * h)), slice(int(0.36 * w), int(0.39 * w))
    assert np.array_equal(blur_faces(frame, [PERSON], [])[ys, xs], frame[ys, xs])
    assert not np.array_equal(blur_faces(frame, [PERSON], [], padding=0.5)[ys, xs], frame[ys, xs])


def test_overlay_does_not_claim_missing_epp_for_persons_too_small_to_judge(camera_factory):
    from carlyn_inference.drawing import missing_epp
    from conftest import zone
    camera = camera_factory(zone("z", "Work", required=("mandil",), polygon=[(0, 0), (1, 0), (1, 1), (0, 1)]))
    far = PersonObs(1, (0.4, 0.2, 0.05, 0.1), 0.6, frozenset(), epp_evaluated=False)
    near = PersonObs(2, (0.4, 0.2, 0.2, 0.6), 0.9, frozenset())
    assert missing_epp(camera, far) == []
    assert missing_epp(camera, near) == ["mandil"]
