"""Orchestrator: downloads the configuration from the API, starts/stops camera workers and sends heartbeats."""
import logging
import threading

import httpx

from . import config as config_module
from .api_client import ApiClient, Outbox
from .config import NodeConfig
from .settings import Settings
from .worker import CameraWorker

log = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, settings: Settings, api: ApiClient | None = None, worker_factory=CameraWorker):
        self.settings = settings
        self.api = api or ApiClient(settings.api_base_url, settings.ingest_api_key)
        self.outbox = Outbox(self.api)
        self.worker_factory = worker_factory
        self.workers: dict[str, CameraWorker] = {}
        self.config: NodeConfig | None = None
        self._etag: str | None = None
        self._reload = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.last_config_error: str | None = None

    # ---- lifecycle ----

    def start(self) -> None:
        self.outbox.start()
        threading.Thread(target=self._config_loop, daemon=True, name="config").start()
        threading.Thread(target=self._heartbeat_loop, daemon=True, name="heartbeat").start()

    def stop(self) -> None:
        self._stop.set()
        self._reload.set()
        with self._lock:
            for worker in self.workers.values():
                worker.stop()
        self.outbox.stop()

    def request_reload(self) -> None:
        self._reload.set()

    # ---- configuration ----

    def _config_loop(self) -> None:
        while not self._stop.is_set():
            self.refresh_config()
            self._reload.wait(self.settings.config_poll_sec)
            self._reload.clear()

    def refresh_config(self) -> bool:
        """Fetches the configuration (ETag) and applies it. Returns True if it changed."""
        try:
            status, data, etag = self.api.get_config(self._etag)
            self.last_config_error = None
        except httpx.HTTPError as ex:
            self.last_config_error = str(ex)
            log.warning("Could not download the configuration: %s", ex)
            return False
        if status == 304:
            return False
        self._etag = etag
        self.apply(config_module.parse(data))
        return True

    def apply(self, node: NodeConfig) -> None:
        with self._lock:
            self.config = node
            wanted = {c.id: c for c in node.cameras}
            for camera_id in list(self.workers):
                if camera_id not in wanted:
                    log.info("Stopping camera %s (removed or disabled)", self.workers[camera_id].camera.name)
                    self.workers.pop(camera_id).stop()
            for camera in node.cameras:
                worker = self.workers.get(camera.id)
                # The source, the rate and the slicing are fixed when a worker starts: changing them restarts it.
                if worker is not None and (worker.camera.source_key != camera.source_key or worker.camera.target_fps != camera.target_fps
                                           or worker.camera.slicing != camera.slicing):
                    worker.stop()
                    worker = None
                if worker is None:
                    log.info("Starting camera %s", camera.name)
                    worker = self.workers[camera.id] = self.worker_factory(camera, node, self.settings, self.outbox)
                    worker.start()
                else:
                    worker.update_camera(camera, node)

    # ---- heartbeat and queries ----

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.settings.heartbeat_sec):
            try:
                self.api.heartbeat(self.settings.model_version, self.camera_statuses())
            except httpx.HTTPError as ex:
                log.warning("Heartbeat failed: %s", ex)

    def camera_statuses(self) -> list[dict]:
        with self._lock:
            return [w.heartbeat() for w in self.workers.values()]

    def worker(self, camera_id: str) -> CameraWorker | None:
        with self._lock:
            return self.workers.get(camera_id)
