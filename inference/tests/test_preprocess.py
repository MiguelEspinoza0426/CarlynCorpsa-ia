import numpy as np

from carlyn_inference.preprocess import PreprocessConfig, Preprocessor


def _frame():
    frame = np.full((100, 200, 3), 40, dtype=np.uint8)
    frame[:, 100:] = 200
    return frame


def test_disabled_returns_the_same_frame():
    frame = _frame()
    assert Preprocessor(PreprocessConfig())(frame) is frame


def test_gamma_brightens_only_inside_the_roi():
    frame = _frame()
    out = Preprocessor(PreprocessConfig(mode="gamma", gamma=2.0, roi=(0.0, 0.0, 0.25, 1.0)))(frame)
    assert out[50, 10, 0] > 40            # inside the ROI: brighter
    assert out[50, 80, 0] == 40           # outside: unchanged
    assert frame[50, 10, 0] == 40         # the original frame (evidence, live video) is not modified


def test_clahe_uses_the_default_roi_when_none_is_given():
    rng = np.random.default_rng(0)
    frame = rng.integers(60, 90, (100, 200, 3), dtype=np.uint8)
    out = Preprocessor(PreprocessConfig(mode="clahe"), default_roi=(0.0, 0.0, 1.0, 0.5))(frame)
    assert out[:50].std() > frame[:50].std()     # more contrast in the top half
    assert (out[50:] == frame[50:]).all()


def test_parse_reads_the_camera_block_and_rejects_unknown_modes():
    config = PreprocessConfig.parse({"mode": "GAMMA", "gamma": 1.8, "roi": [0, 0, 1, 0.5]})
    assert (config.mode, config.gamma, config.roi) == ("gamma", 1.8, (0.0, 0.0, 1.0, 0.5))
    assert PreprocessConfig.parse({"mode": "sharpen"}).mode == "none"
    assert PreprocessConfig.parse(None) == PreprocessConfig()
