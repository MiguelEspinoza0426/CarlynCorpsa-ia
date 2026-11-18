from datetime import datetime

from carlyn_inference.benchmark import markdown, report_name, save, size_bucket, stats


def test_stats_percentiles():
    result = stats([float(v) for v in range(1, 101)])
    assert result["n"] == 100
    assert result["mean"] == 50.5
    assert result["p50"] == 51.0
    assert result["p95"] == 95.0
    assert result["max"] == 100.0


def test_stats_empty():
    assert stats([]) is None


def test_size_bucket_limits():
    assert size_bucket(20) == "<60 px"
    assert size_bucket(60) == "60–150 px"
    assert size_bucket(149.9) == "60–150 px"
    assert size_bucket(150) == "≥150 px"


def test_report_name_is_file_safe():
    assert report_name("línea base", "DESKTOP-01", datetime(2026, 11, 17, 9, 5)) == "2026-11-17_0905_línea-base_desktop-01"


def _report():
    return {
        "label": "linea-base", "mode": "live", "date": "2026-11-17T09:05:00", "notes": "",
        "environment": {"host": "pc", "os": "Windows", "python": "3.12", "torch": "2", "ultralytics": "8", "commit": "abc", "gpu": "GTX 1650"},
        "config": {"model": "epp_v3", "imgsz": 640, "conf": 0.25, "pose_model": "yolo11n-pose.pt"},
        "cameras": {"Camara 1": {"resolution": "1920x1080", "target_fps": 20, "achieved_fps": 8.2,
                                 "detect_ms": {"mean": 95.0, "p95": 120.0}, "pose_ms": None, "persons_per_frame": 1.5,
                                 "person_heights": {"<60 px": 3, "60–150 px": 2, "≥150 px": 1}}},
        "totals": {"Cuadros analizados por segundo (todas las cámaras)": 12.0},
    }


def test_markdown_has_one_row_per_camera():
    text = markdown(_report())
    assert "| Camara 1 | 1920x1080 | 20 | 8.2 | 95.0 / 120.0 | — | 1.5 | <60 px: 3, 60–150 px: 2, ≥150 px: 1 |" in text
    assert "GTX 1650" in text


def test_save_writes_json_and_markdown(tmp_path):
    path = save(_report(), tmp_path)
    assert path.suffix == ".md"
    assert path.with_suffix(".json").exists()
