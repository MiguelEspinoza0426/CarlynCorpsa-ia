"""One thread per camera: capture → detection/tracking → rules → events, metrics and stream frames."""
import logging
import threading
import time

from .api_client import Outbox, event_payload, iso
from .config import CameraConfig, NodeConfig
from .detection import Detector, blur_faces
from .drawing import draw, draw_event, encode_jpeg
from .rules import RuleEngine
from .settings import Settings
from .sources import open_source, with_credentials

log = logging.getLogger(__name__)

# Single GPU: inferences of all cameras are serialized, which is more predictable than running them concurrently.
GPU_LOCK = threading.Lock()


class CameraWorker(threading.Thread):
    def __init__(self, camera: CameraConfig, node: NodeConfig, settings: Settings, outbox: Outbox):
        super().__init__(daemon=True, name=f"camera-{camera.name}")
        self.camera, self.node, self.settings, self.outbox = camera, node, settings, outbox
        self.engine = RuleEngine(camera)
        self._engine_lock = threading.Lock()
        self._running = True
        self.reader = None
        self.detector: Detector | None = None
        self.fps = 0.0
        self.latency_ms: int | None = None
        self.last_frame_at: float | None = None
        self.error: str | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_frame = None
        self.latest_persons = []
        self.frame_ready = threading.Condition()

    @property
    def source(self) -> str:
        overrides = self.settings.source_overrides
        return overrides.get(self.camera.id) or overrides.get(self.camera.name) or with_credentials(self.camera.stream_url, self.camera.credentials_ref)

    @property
    def status(self) -> str:
        if self.reader is None:
            return "Reconnecting"
        if not getattr(self.reader, "connected", True):
            return "Reconnecting"
        if self.last_frame_at is None or time.time() - self.last_frame_at > 10:
            return "Reconnecting"
        return "Online"

    def update_camera(self, camera: CameraConfig) -> None:
        """New zones or rules: the rules engine restarts (temporary track state is discarded)."""
        with self._engine_lock:
            self.camera = camera
            self.engine = RuleEngine(camera)

    def stop(self) -> None:
        self._running = False
        if self.reader is not None:
            self.reader.close()

    def run(self) -> None:
        try:
            self.detector = Detector(self.settings.model_path, self.settings.pose_model or None, self.settings.object_model or None,
                                     self.node.person_class, self.node.epp_classes, self.settings.torch_device,
                                     self.settings.imgsz, self.settings.conf)
            self.reader = open_source(self.source, self.camera.name)
        except Exception as ex:  # noqa: BLE001 - reported through the heartbeat
            self.error = f"No se pudo iniciar: {ex}"
            log.exception("Camera %s could not start", self.camera.name)
            return

        last_ts, last_metrics = 0.0, time.time()
        while self._running:
            started = time.time()
            frame, ts = self.reader.read()
            if frame is None or ts == last_ts:
                time.sleep(0.01)
                continue
            last_ts = ts
            try:
                self._process(frame, ts)
                self.error = None
            except Exception as ex:  # noqa: BLE001 - keep the camera alive
                self.error = str(ex)
                log.exception("Error processing a frame of %s", self.camera.name)
                time.sleep(1)
            if time.time() - last_metrics >= self.settings.metrics_flush_sec:
                with self._engine_lock:
                    self.outbox.add_metrics(self.camera.id, *self.engine.take_metrics())
                last_metrics = time.time()
            time.sleep(max(0.0, 1 / self.camera.target_fps - (time.time() - started)))

    def _process(self, frame, ts: float) -> None:
        now = time.time()
        with self._engine_lock:
            camera, engine = self.camera, self.engine
        detect_objects = any(z.rule("UnsafeCondition") for z in camera.zones)
        with GPU_LOCK:
            detections = self.detector.detect(frame, detect_objects)
        events = engine.update(now, detections.persons, detections.objects)

        if events:
            with GPU_LOCK:
                keypoints = self.detector.face_keypoints(frame)
            for event in events:
                snapshot = blur_faces(draw_event(frame, event), detections.persons, keypoints)
                self.outbox.add_event(event_payload(camera.id, event, self.settings.model_version),
                                      encode_jpeg(snapshot, 85, max_width=1280))
                log.info("[%s] %s in zone %s (track %s)", camera.name, event.type, event.zone_id, event.track_id)

        annotated = draw(frame, camera, detections.persons, engine.zone_counts(), self.fps)
        if self.settings.blur_stream:
            annotated = blur_faces(annotated, detections.persons, detections.pose_keypoints)
        jpeg = encode_jpeg(annotated, self.settings.jpeg_quality, self.settings.stream_max_width)

        elapsed = time.time() - (self.last_frame_at or now)
        self.fps = 0.8 * self.fps + 0.2 * (1 / elapsed) if self.last_frame_at and elapsed > 0 else self.fps
        self.latency_ms = int((time.time() - ts) * 1000)
        self.last_frame_at = now
        with self.frame_ready:
            self.latest_jpeg, self.latest_frame, self.latest_persons = jpeg, frame, detections.persons
            self.frame_ready.notify_all()

    def snapshot_jpeg(self) -> bytes | None:
        """Current frame without annotations (zone editor), with faces blurred."""
        with self.frame_ready:
            frame, persons = self.latest_frame, self.latest_persons
        if frame is None:
            return None
        keypoints = []
        if self.detector is not None:
            with GPU_LOCK:
                keypoints = self.detector.face_keypoints(frame)
        return encode_jpeg(blur_faces(frame, persons, keypoints), 90)

    def heartbeat(self) -> dict:
        return {
            "cameraId": self.camera.id,
            "status": self.status,
            "fps": round(self.fps, 1) if self.fps else None,
            "latencyMs": self.latency_ms,
            "lastFrameAt": None if self.last_frame_at is None else iso(self.last_frame_at),
            "error": self.error or (getattr(self.reader, "error", None) if self.reader else None),
        }
