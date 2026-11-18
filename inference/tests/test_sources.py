"""Camera sources chosen from the stream URL: DVR (RTSP), webcam of the node and desktop window."""
import numpy as np
from fastapi.testclient import TestClient

import carlyn_inference.sources as sources
from carlyn_inference.server import create_app
from carlyn_inference.sources import window
from test_service import make_service


class FakeReader:
    def __init__(self, source, name="", hw_decode="off"):
        self.source, self.name, self.hw_decode = source, name, hw_decode


def opened(monkeypatch, source):
    monkeypatch.setattr(sources, "CameraReader", FakeReader)
    monkeypatch.setattr(window, "WindowReader", lambda spec: ("window", spec))
    return sources.open_source(source, "cam")


def test_webcam_sources_open_the_device_index(monkeypatch):
    assert opened(monkeypatch, "webcam:1").source == 1
    assert opened(monkeypatch, "0").source == 0      # SOURCE_OVERRIDES shortcut


def test_rtsp_and_files_are_opened_as_urls(monkeypatch):
    assert opened(monkeypatch, "rtsp://dvr/Streaming/Channels/102").source == "rtsp://dvr/Streaming/Channels/102"
    assert opened(monkeypatch, "C:/videos/nave.mp4").source == "C:/videos/nave.mp4"


def test_hardware_decoding_parameters():
    import cv2

    from carlyn_inference.sources.camera import capture_params
    assert capture_params("off") == []
    assert capture_params("auto") == [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
    assert capture_params("D3D11") == [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_D3D11]
    assert capture_params("nvdec-typo") == []


def test_window_sources_keep_title_and_region(monkeypatch):
    assert opened(monkeypatch, "window:Hik-Connect:10,20,640,360") == ("window", "Hik-Connect:10,20,640,360")
    assert window.parse("Hik-Connect:10,20,640,360") == ("Hik-Connect", (10, 20, 640, 360))
    assert window.parse("iVMS-4200 - Vista: principal") == ("iVMS-4200 - Vista: principal", None)


def test_credentials_are_only_added_to_network_urls(monkeypatch):
    monkeypatch.setenv("CAM_DVR1_USER", "visor")
    monkeypatch.setenv("CAM_DVR1_PASS", "clave")
    assert sources.with_credentials("rtsp://dvr:554/s", "CAM_DVR1") == "rtsp://visor:clave@dvr:554/s"
    assert sources.with_credentials("webcam:0", "CAM_DVR1") == "webcam:0"
    assert sources.with_credentials("window:iVMS-4200", "CAM_DVR1") == "window:iVMS-4200"


def test_server_lists_windows_and_serves_a_preview(monkeypatch):
    monkeypatch.setattr(window, "list_windows", lambda: [{"title": "Hik-Connect", "width": 1280, "height": 720}])
    monkeypatch.setattr(window, "capture", lambda title: np.zeros((72, 128, 3), np.uint8) if title == "Hik-Connect" else None)
    client = TestClient(create_app(make_service(), internal_key="secret"))
    key = {"X-Internal-Key": "secret"}

    assert client.get("/sources/windows").status_code == 401
    assert client.get("/sources/windows", headers=key).json() == [{"title": "Hik-Connect", "width": 1280, "height": 720}]
    preview = client.get("/sources/window-preview.jpg", params={"title": "Hik-Connect"}, headers=key)
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/jpeg"
    assert client.get("/sources/window-preview.jpg", params={"title": "Otra"}, headers=key).status_code == 404
