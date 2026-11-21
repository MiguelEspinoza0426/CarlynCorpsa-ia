import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import capturar_cuadros as cc  # noqa: E402


def test_hides_rtsp_credentials():
    assert cc.ocultar_credenciales("rtsp://admin:s3cret@10.0.0.5:554/ch1") == "rtsp://***@10.0.0.5:554/ch1"
    assert cc.ocultar_credenciales("videos/a.mp4") == "videos/a.mp4"


def test_frame_name_keeps_camera_day_and_time():
    assert cc.nombre_cuadro("camara1", datetime(2026, 11, 17, 17, 51, 36), 12) == "camara1_20261117-175136_00012"


def test_still_scene_is_not_new_but_a_change_is():
    base = np.full((90, 160), 100, np.uint8)
    assert cc.es_nuevo(None, base)
    noise = (base + np.random.default_rng(0).integers(0, 10, base.shape)).astype(np.uint8)
    assert not cc.es_nuevo(base, noise)             # compression noise everywhere is not a change
    person = base.copy()
    person[40:52, 70:72] = 200                      # a distant person: 24 px of 14 400 (0.17 %)
    assert cc.es_nuevo(base, person)


def test_yolo_line_uses_center_and_clamps_to_frame():
    assert cc.linea_yolo(7, (0.1, 0.2, 0.2, 0.4)) == "7 0.200000 0.400000 0.200000 0.400000"
    assert cc.linea_yolo(0, (-0.1, 0.9, 0.3, 0.2)) == "0 0.100000 0.950000 0.200000 0.100000"


def _video(path: Path, frames: list[np.ndarray], fps: int = 10) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        w.write(f)
    w.release()
    return path


def test_capture_skips_no_signal_and_repeated_frames(tmp_path):
    rng = np.random.default_rng(0)
    black = np.zeros((120, 160, 3), np.uint8)
    scene = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    # 1 s per segment at 10 fps, sampled every second: black, scene, scene (still), other scene
    video = _video(tmp_path / "cam.mp4", [black] * 10 + [scene] * 20 + [other] * 10)

    count = cc.capturar([str(video)], "camara1", tmp_path / "out", cada=1, duracion=None,
                        cambio_minimo=cc.CAMBIO_MINIMO, etiquetar=None)

    assert count == {"leidos": 4, "sin_senal": 1, "repetidos": 1, "guardados": 2}
    names = sorted(p.name for p in (tmp_path / "out/images").iterdir())
    assert len(names) == 2 and all(n.startswith("camara1_") for n in names)


def test_capture_writes_labels_and_continues_numbering(tmp_path):
    rng = np.random.default_rng(1)
    video = _video(tmp_path / "cam.mp4", [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(2)], fps=1)

    def fake_labeler(_img):
        return ["7 0.5 0.5 0.2 0.4"]

    cc.capturar([str(video)], "camara1", tmp_path / "out", 1, None, cc.CAMBIO_MINIMO, fake_labeler)
    cc.capturar([str(video)], "camara1", tmp_path / "out", 1, None, cc.CAMBIO_MINIMO, fake_labeler)

    labels = sorted((tmp_path / "out/labels").iterdir())
    assert len(labels) == 4 and labels[-1].stem.endswith("_00003")
    assert labels[0].read_text(encoding="utf-8") == "7 0.5 0.5 0.2 0.4\n"
    assert "persona" in (tmp_path / "out/data.yaml").read_text(encoding="utf-8")


def test_video_frames_use_the_given_dvr_start_time(tmp_path):
    rng = np.random.default_rng(2)
    video = _video(tmp_path / "cam.mp4", [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(20)])

    cc.capturar([str(video)], "camara1", tmp_path / "out", 1, None, cc.CAMBIO_MINIMO, None,
                datetime(2026, 11, 17, 17, 51, 36))

    assert sorted(p.stem for p in (tmp_path / "out/images").iterdir()) == [
        "camara1_20261117-175136_00000", "camara1_20261117-175137_00001"]


def test_unreadable_source_fails_without_showing_password(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        list(cc.leer_cuadros("rtsp://admin:s3cret@127.0.0.1:1/x", 1, 1))
    assert "s3cret" not in str(e.value)
