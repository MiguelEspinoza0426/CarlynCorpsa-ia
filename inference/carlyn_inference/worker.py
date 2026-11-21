"""One thread per camera: capture → detection/tracking → rules → events and metrics. The live video is rendered by LiveStream."""
import logging
import threading
import time

from .api_client import Outbox, event_payload, iso
from .config import CameraConfig, InferenceConfig, NodeConfig
from .detection import Detector, blur_faces, create_detector
from .drawing import draw_event, encode_jpeg
from .gpu import GPU
from .live_stream import LiveStream, Overlay
from .rules import EventOut, PersonObs, RuleEngine
from .settings import Settings
from .sources import open_source, with_credentials
from .sources.signal import SignalMonitor

log = logging.getLogger(__name__)


def evidence_image(frame, event: EventOut, persons: list[PersonObs], keypoints, blur: bool):
    """Evidence of an alert: the event box drawn over the frame, with faces pixelated when the camera asks for it."""
    image = draw_event(frame, event)
    return blur_faces(image, persons, keypoints) if blur else image



class CameraWorker(threading.Thread):
    def __init__(self, camera: CameraConfig, node: NodeConfig, settings: Settings, outbox: Outbox):
        super().__init__(daemon=True, name=f"camera-{camera.name}")
        self.camera, self.node, self.settings, self.outbox = camera, node, settings, outbox
        self.engine = RuleEngine(camera, track_lost_sec=settings.track_buffer_sec)
        self._engine_lock = threading.Lock()
        self._running = True
        self.reader = None
        self.detector: Detector | None = None
        self.fps = 0.0
        self.latency_ms: int | None = None
        self.last_frame_at: float | None = None
        self.error: str | None = None
        self.signal = SignalMonitor(settings.no_signal_sec)
        self.no_signal: str | None = None      # reason while the DVR sends no real image
        self.stream = LiveStream(self._read_frame, settings.stream_fps, settings.jpeg_quality, settings.stream_max_width,
                                 settings.blur_stream, camera.name)
        self.latest_frame = None
        self.latest_persons = []
        self.frame_ready = threading.Condition()

    @property
    def source(self) -> str:
        overrides = self.settings.source_overrides
        return overrides.get(self.camera.id) or overrides.get(self.camera.name) or with_credentials(self.camera.stream_url, self.camera.credentials_ref)

    @property
    def inference(self) -> InferenceConfig:
        overrides = self.settings.inference_overrides
        override = overrides.get(self.camera.id) or overrides.get(self.camera.name)
        return InferenceConfig.parse(override) if override else self.camera.inference

    @property
    def status(self) -> str:
        if self.no_signal and getattr(self.reader, "connected", True):
            # The stream is alive but carries no image: not Online (see settings.no_signal_status).
            return self.settings.no_signal_status
        if self.reader is None:
            return "Reconnecting"
        if not getattr(self.reader, "connected", True):
            return "Reconnecting"
        if self.last_frame_at is None or time.time() - self.last_frame_at > 10:
            return "Reconnecting"
        return "Online"

    def update_camera(self, camera: CameraConfig, node: NodeConfig | None = None) -> None:
        """New zones or rules: the rules engine restarts (temporary track state is discarded)."""
        with self._engine_lock:
            self.camera = camera
            self.engine = RuleEngine(camera, track_lost_sec=self.settings.track_buffer_sec)
        if node is not None:
            self.node = node
            if self.detector is not None:
                self.detector.set_epp_min_confidence(node.epp_min_confidence)

    def stop(self) -> None:
        self._running = False
        self.stream.close()
        if self.reader is not None:
            self.reader.close()

    def run(self) -> None:
        try:
            self.detector = create_detector(self.settings, self.node.person_class, self.node.epp_classes, with_objects=True,
                                            inference=self.inference, fps=self.camera.target_fps)
            self.detector.set_epp_min_confidence(self.node.epp_min_confidence)
            self.reader = open_source(self.source, self.camera.name, self.settings.decode_hw)
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
            self.signal.update(frame, time.time())
            self.no_signal = self.signal.reason(time.time())
            if self.no_signal:
                # Nothing to analyze: the GPU is left to the other cameras.
                time.sleep(max(0.0, 1 / self.camera.target_fps - (time.time() - started)))
                continue
            try:
                self._process(frame, ts)
                self.error = None
            except Exception as ex:  # noqa: BLE001 - keep the camera alive
                self.error = str(ex)
                log.exception("Error processing a frame of %s", self.camera.name)
                time.sleep(1)
            if time.time() - last_metrics >= self.settings.metrics_flush_sec and self._running:
                # Not after stop(): a removed camera must not send metrics of its (deleted) zones.
                with self._engine_lock:
                    self.outbox.add_metrics(self.camera.id, *self.engine.take_metrics())
                last_metrics = time.time()
            time.sleep(max(0.0, 1 / self.camera.target_fps - (time.time() - started)))

    def _process(self, frame, ts: float) -> None:
        now = time.time()
        with self._engine_lock:
            camera, engine = self.camera, self.engine
        detect_objects = any(z.rule("UnsafeCondition") for z in camera.zones)
        with GPU.turn(self.camera.id, self.camera.target_fps):
            detections = self.detector.detect(frame, detect_objects)
        events = engine.update(now, detections.persons, detections.objects)

        if events:
            keypoints = []
            if camera.blur_faces:   # without pixelation the face keypoints are not needed: no GPU spent on them
                with GPU.turn(self.camera.id, self.camera.target_fps):
                    keypoints = self.detector.face_keypoints(frame)
            for event in events:
                snapshot = evidence_image(frame, event, detections.persons, keypoints, camera.blur_faces)
                self.outbox.add_event(event_payload(camera.id, event, self.settings.model_version),
                                      encode_jpeg(snapshot, 85, max_width=1280))
                log.info("[%s] %s in zone %s (track %s)", camera.name, event.type, event.zone_id, event.track_id)

        # The live video draws these detections over the raw frames that follow, at its own rate.
        self.stream.overlay = Overlay(camera, detections.persons, detections.pose_keypoints, engine.zone_counts(), self.fps)

        elapsed = time.time() - (self.last_frame_at or now)
        self.fps = 0.8 * self.fps + 0.2 * (1 / elapsed) if self.last_frame_at and elapsed > 0 else self.fps
        self.latency_ms = int((time.time() - ts) * 1000)
        self.last_frame_at = now
        with self.frame_ready:
            self.latest_frame, self.latest_persons = frame, detections.persons
            self.frame_ready.notify_all()

    def _read_frame(self):
        return self.reader.read() if self.reader is not None else (None, 0.0)

    def snapshot_jpeg(self) -> bytes | None:
        """Current frame without annotations (zone editor), with faces blurred."""
        with self.frame_ready:
            frame, persons = self.latest_frame, self.latest_persons
        if frame is None:
            return None
        keypoints = []
        if self.detector is not None:
            with GPU.turn(self.camera.id, self.camera.target_fps):
                keypoints = self.detector.face_keypoints(frame)
        return encode_jpeg(blur_faces(frame, persons, keypoints), 90)

    def heartbeat(self) -> dict:
        return {
            "cameraId": self.camera.id,
            "status": self.status,
            "fps": round(self.fps, 1) if self.fps and not self.no_signal else None,   # nothing is analyzed without signal
            "latencyMs": self.latency_ms,
            "lastFrameAt": None if self.last_frame_at is None else iso(self.last_frame_at),
            "error": self.no_signal or self.error or (getattr(self.reader, "error", None) if self.reader else None),
        }
