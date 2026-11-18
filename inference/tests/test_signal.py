import cv2
import numpy as np

from carlyn_inference.sources.signal import SignalMonitor, is_placeholder


def _scene(seed=0):
    """A textured frame like a real camera, with sensor noise that changes between frames."""
    rng = np.random.default_rng(seed)
    base = np.tile(np.linspace(30, 220, 1280, dtype=np.float32), (720, 1))
    frame = np.clip(base[..., None] + rng.normal(0, 4, (720, 1280, 3)), 0, 255).astype(np.uint8)
    cv2.rectangle(frame, (400, 200), (700, 600), (20, 60, 200), -1)
    return frame


def _logo():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, "annke", (520, 380), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
    cv2.putText(frame, "17-11-2026 17:46:24", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def test_placeholders_are_recognized():
    assert is_placeholder(np.zeros((480, 960, 3), dtype=np.uint8))          # black
    assert is_placeholder(np.full((480, 960, 3), (200, 80, 0), np.uint8))  # blue screen
    assert is_placeholder(_logo())                                           # DVR logo and date on black


def test_a_real_scene_is_not_a_placeholder():
    assert not is_placeholder(_scene())


def test_a_dark_night_scene_with_lights_is_not_a_placeholder():
    night = np.full((720, 1280, 3), 12, np.uint8)
    for x in range(100, 1280, 250):
        cv2.circle(night, (x, 150), 60, (230, 230, 230), -1)                # lamps and their glow
    cv2.rectangle(night, (0, 500), (1280, 720), (40, 45, 50), -1)            # lit floor
    assert not is_placeholder(night)


def test_monitor_needs_the_placeholder_for_some_seconds():
    monitor = SignalMonitor(placeholder_sec=5)
    for t in range(5):
        monitor.update(_logo(), float(t))
    assert monitor.reason(4.0) is None
    monitor.update(_logo(), 5.0)
    assert "logo" in monitor.reason(5.0)
    monitor.update(_scene(1), 6.0)                                          # signal is back
    assert monitor.reason(6.0) is None


def test_monitor_detects_a_frozen_image():
    monitor, frozen = SignalMonitor(frozen_sec=30), _scene()
    for t in range(0, 36, 5):
        monitor.update(frozen, float(t))
    assert monitor.reason(30.0) is None                                     # the same image since t=5: 25 s
    assert "congelada" in monitor.reason(35.0)
    live = SignalMonitor(frozen_sec=30)
    for t in range(0, 31, 5):
        live.update(_scene(t), float(t))                                     # noise changes every frame
    assert live.reason(30.0) is None


def test_worker_reports_no_signal_instead_of_online(camera_factory):
    from types import SimpleNamespace

    from carlyn_inference.config import NodeConfig
    from carlyn_inference.settings import Settings
    from carlyn_inference.worker import CameraWorker

    camera = camera_factory()
    node = NodeConfig("v1", "gpu-01", frozenset(), "persona", [camera])
    worker = CameraWorker(camera, node, Settings(), outbox=None)
    worker.reader, worker.last_frame_at = SimpleNamespace(connected=True, error=None), __import__("time").time()
    assert worker.status == "Online"
    worker.no_signal = "Sin señal: el DVR envía una imagen negra, uniforme o su logo"
    beat = worker.heartbeat()
    assert beat["status"] == "NoSignal" and beat["error"].startswith("Sin señal")
    worker.reader.connected = False
    assert worker.status == "Reconnecting"            # a broken stream is still reported as such
