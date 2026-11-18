import pytest

from carlyn_inference.crops import belongs_to, crop_region, to_frame_box


def test_crop_region_adds_margin_on_each_side():
    # Person of 100x200 px at (500, 200) in a 1000x1000 frame; 15 % margin = 15 px / 30 px.
    assert crop_region((0.5, 0.2, 0.1, 0.2), 1000, 1000, 0.15) == (485, 170, 615, 430)


def test_crop_region_is_clamped_to_the_frame():
    assert crop_region((0.0, 0.9, 0.2, 0.1), 1000, 1000, 0.15) == (0, 885, 230, 1000)


def test_crop_region_too_small_is_skipped():
    assert crop_region((0.5, 0.5, 0.005, 0.01), 1000, 1000, 0.15) is None


def test_to_frame_box_maps_crop_pixels_back():
    box = to_frame_box((10, 20, 40, 80), (485, 170, 615, 430), 1000, 1000)
    assert box == pytest.approx((0.495, 0.19, 0.03, 0.06))


def test_belongs_to_rejects_neighbour_epp_inside_the_margin():
    person = (0.5, 0.2, 0.1, 0.2)
    assert belongs_to(person, (0.52, 0.25, 0.02, 0.02))
    assert not belongs_to(person, (0.62, 0.25, 0.02, 0.02))  # center at x=0.63, beyond the 10 % tolerance
