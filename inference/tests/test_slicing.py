import numpy as np

from carlyn_inference.slicing import SliceConfig, cut_by_tile, merge, tile_regions, tile_starts


def test_tile_starts_cover_with_overlap_and_touch_the_end():
    assert tile_starts(0, 500, 640, 512) == [0]
    starts = tile_starts(0, 1920, 640, 512)
    assert starts[0] == 0 and starts[-1] == 1920 - 640
    assert all(b - a <= 512 for a, b in zip(starts, starts[1:]))


def test_disabled_slicing_is_a_single_full_frame_pass():
    assert tile_regions(1920, 1080, SliceConfig()) == [(0, 0, 1920, 1080)]


def test_tiles_stay_inside_the_roi():
    config = SliceConfig(enabled=True, roi=(0.0, 0.0, 1.0, 0.3))
    regions = tile_regions(2560, 1440, config)
    assert regions[0] == (0, 0, 2560, 1440)           # full-frame pass first
    tiles = regions[1:]
    assert len(tiles) == 5                             # 2560 px wide, 432 px tall: one row of 5 tiles
    assert all(y1 == 0 and y2 == 432 for _, y1, _, y2 in tiles)
    assert tiles[-1][2] == 2560


def test_tiles_only_when_the_full_frame_pass_is_off():
    regions = tile_regions(1280, 720, SliceConfig(enabled=True, full_frame=False, roi=(0.0, 0.0, 0.5, 0.5)))
    assert regions == [(0, 0, 640, 360)]


def test_parse_reads_the_camera_block():
    config = SliceConfig.parse({"enabled": True, "tileSize": 512, "overlap": 0.25, "roi": [0, 0, 1, 0.3]})
    assert (config.enabled, config.tile_size, config.overlap, config.roi) == (True, 512, 0.25, (0.0, 0.0, 1.0, 0.3))
    assert SliceConfig.parse(None) == SliceConfig()


def test_merge_keeps_the_best_of_overlapping_boxes():
    xyxy = np.array([[100, 100, 150, 250], [102, 98, 152, 248], [400, 100, 450, 250]], dtype=float)
    boxes, scores = merge(xyxy, np.array([0.6, 0.9, 0.5]), iou=0.5, min_box_px=8)
    assert boxes.tolist() == [[102, 98, 152, 248], [400, 100, 450, 250]]
    assert scores.tolist() == [0.9, 0.5]


def test_merge_joins_a_person_cut_by_a_tile_edge():
    # Upper half of a person from a tile (higher score) and the full box from the whole frame: low IoU, but contained.
    xyxy = np.array([[100, 100, 150, 300], [100, 100, 150, 190]], dtype=float)
    boxes, scores = merge(xyxy, np.array([0.7, 0.8]), iou=0.5, min_box_px=8, cut=np.array([False, True]))
    assert boxes.tolist() == [[100, 100, 150, 300]]
    assert scores.tolist() == [0.8]


def test_merge_keeps_a_person_standing_in_front_of_another():
    # Same boxes, but neither was cut by a tile: they are two people, as Ultralytics' own NMS would keep them.
    xyxy = np.array([[100, 100, 150, 300], [100, 100, 150, 190]], dtype=float)
    boxes, _ = merge(xyxy, np.array([0.7, 0.8]), iou=0.5, min_box_px=8)
    assert len(boxes) == 2


def test_cut_by_tile_ignores_frame_borders():
    # Tile (0, 0)-(640, 432) of a 2560x1440 frame: its left and top edges are the frame's own.
    xyxy = np.array([[0, 0, 40, 100], [600, 300, 639, 431], [200, 100, 260, 300]], dtype=float)
    assert cut_by_tile(xyxy, (0, 0, 640, 432), 2560, 1440).tolist() == [False, True, False]


def test_merge_discards_tiny_boxes():
    xyxy = np.array([[0, 0, 5, 30], [10, 10, 40, 90]], dtype=float)
    boxes, _ = merge(xyxy, np.array([0.9, 0.5]), iou=0.5, min_box_px=8)
    assert boxes.tolist() == [[10, 10, 40, 90]]


def test_merge_of_nothing():
    boxes, scores = merge(np.empty((0, 4)), np.empty(0), iou=0.5, min_box_px=8)
    assert boxes.shape == (0, 4) and scores.shape == (0,)
