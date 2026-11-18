"""Client of the .NET ingestion API with an outbox: data is retried until the API accepts it.

Event ingestion is idempotent (clientEventId), so retrying after a timeout never duplicates events.
"""
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from .rules import DwellSession, EventOut, InactivityInterval, OccupancySample

log = logging.getLogger(__name__)

MAX_EVENTS_PER_BATCH = 100
MAX_METRICS_PER_BATCH = 1000
MAX_QUEUE = 5000


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


class ApiClient:
    def __init__(self, base_url: str, api_key: str, transport: httpx.BaseTransport | None = None):
        self.http = httpx.Client(base_url=base_url, headers={"X-Api-Key": api_key}, timeout=10, transport=transport)

    def get_config(self, etag: str | None) -> tuple[int, dict | None, str | None]:
        headers = {"If-None-Match": etag} if etag else {}
        response = self.http.get("/api/ingest/config", headers=headers)
        if response.status_code == 304:
            return 304, None, etag
        response.raise_for_status()
        return 200, response.json(), response.headers.get("ETag")

    def post_events(self, events: list[dict]) -> dict:
        response = self.http.post("/api/ingest/events", json={"events": events})
        response.raise_for_status()
        return response.json()

    def put_evidence(self, client_event_id: str, snapshot_jpeg: bytes) -> None:
        response = self.http.put(f"/api/ingest/events/{client_event_id}/evidence",
                                 files={"snapshot": ("snapshot.jpg", snapshot_jpeg, "image/jpeg")})
        if response.status_code != 404:   # the event may have been rejected: nothing to attach to
            response.raise_for_status()

    def post_metrics(self, kind: str, key: str, items: list[dict]) -> dict:
        response = self.http.post(f"/api/ingest/metrics/{kind}", json={key: items})
        response.raise_for_status()
        return response.json()

    def heartbeat(self, model_version: str, cameras: list[dict]) -> None:
        response = self.http.post("/api/ingest/heartbeat", json={"sentAt": iso(time.time()), "modelVersion": model_version, "cameras": cameras})
        response.raise_for_status()

    def post_logs(self, logs: list[dict]) -> None:
        response = self.http.post("/api/ingest/logs", json={"logs": logs})
        response.raise_for_status()

    def close(self) -> None:
        self.http.close()


@dataclass
class _PendingEvent:
    payload: dict
    snapshot: bytes | None
    sent: bool = False


def event_payload(camera_id: str, event: EventOut, model_version: str) -> dict:
    return {
        "clientEventId": str(uuid.uuid4()),
        "type": event.type,
        "cameraId": camera_id,
        "zoneId": event.zone_id,
        "ruleId": event.rule_id,
        "startedAt": iso(event.started_at),
        "endedAt": iso(event.ended_at) if event.ended_at is not None else None,
        "confidence": event.confidence,
        "personCount": event.person_count,
        "trackId": event.track_id,
        "bbox": [round(v, 4) for v in event.bbox] if event.bbox else None,
        "details": event.details,
        "modelVersion": model_version,
    }


class Outbox:
    """Background sender with retries and exponential backoff. Bounded: drops the oldest items if the API is down for long."""

    def __init__(self, api: ApiClient, flush_interval: float = 1.0):
        self.api = api
        self.flush_interval = flush_interval
        self._events: deque[_PendingEvent] = deque()
        self._metrics: dict[str, deque] = {"occupancy": deque(), "dwell": deque(), "inactivity": deque()}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._backoff = 0.0
        # camera id -> zone ids of the current configuration; None until the first one arrives (no filtering).
        self._active: dict[str, frozenset[str]] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="outbox")
        self.sent_events = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self.flush()

    def set_active(self, zones_by_camera: dict[str, frozenset[str]]) -> None:
        """New configuration: queued data of removed cameras or zones is dropped (the API would reject it forever)."""
        with self._lock:
            self._active = dict(zones_by_camera)
            before = self._pending_unlocked()
            self._events = deque(e for e in self._events if self._is_active(e.payload))
            for kind, queue in self._metrics.items():
                self._metrics[kind] = deque(item for item in queue if self._is_active(item))
            dropped = before - self._pending_unlocked()
        if dropped:
            log.info("Dropped %d queued items of removed cameras or zones", dropped)

    def _is_active(self, item: dict) -> bool:
        if self._active is None:
            return True
        zones = self._active.get(item["cameraId"])
        return zones is not None and (item.get("zoneId") is None or item["zoneId"] in zones)

    def add_event(self, payload: dict, snapshot: bytes | None) -> None:
        with self._lock:
            if not self._is_active(payload):
                return
            self._events.append(_PendingEvent(payload, snapshot))
            while len(self._events) > MAX_QUEUE:
                log.warning("Outbox full, dropping event %s", self._events.popleft().payload["clientEventId"])

    def add_metrics(self, camera_id: str, occupancy: list[OccupancySample], dwell: list[DwellSession], inactivity: list[InactivityInterval]) -> None:
        with self._lock:
            self._metrics["occupancy"].extend(m for m in ({
                "cameraId": camera_id, "zoneId": o.zone_id, "windowStart": iso(o.window_start), "windowSec": o.window_sec,
                "avgCount": o.avg_count, "maxCount": o.max_count, "minCount": o.min_count} for o in occupancy) if self._is_active(m))
            self._metrics["dwell"].extend(m for m in ({
                "cameraId": camera_id, "zoneId": d.zone_id, "trackId": d.track_id, "enteredAt": iso(d.entered_at),
                "exitedAt": iso(d.exited_at), "fromZoneId": d.from_zone_id} for d in dwell) if self._is_active(m))
            self._metrics["inactivity"].extend(m for m in ({
                "cameraId": camera_id, "zoneId": i.zone_id, "trackId": i.track_id, "startedAt": iso(i.started_at),
                "endedAt": iso(i.ended_at)} for i in inactivity) if self._is_active(m))
            for queue in self._metrics.values():
                while len(queue) > MAX_QUEUE:
                    queue.popleft()

    def pending(self) -> int:
        with self._lock:
            return self._pending_unlocked()

    def _pending_unlocked(self) -> int:
        return len(self._events) + sum(len(q) for q in self._metrics.values())

    def _run(self) -> None:
        while not self._stop.wait(self.flush_interval + self._backoff):
            try:
                self.flush()
                self._backoff = 0.0
            except httpx.HTTPError as ex:
                self._backoff = min(60.0, max(1.0, self._backoff * 2))
                log.warning("API not reachable (%s); retrying in %.0fs with %d items pending", ex, self._backoff, self.pending())
            except Exception:  # noqa: BLE001 - a bug must not stop the sender thread silently
                self._backoff = min(60.0, max(1.0, self._backoff * 2))
                log.exception("Unexpected error sending data; retrying in %.0fs with %d items pending", self._backoff, self.pending())

    def flush(self) -> None:
        self._flush_events()
        for kind, key in (("occupancy", "samples"), ("dwell", "sessions"), ("inactivity", "intervals")):
            self._flush_metrics(kind, key)

    def _flush_events(self) -> None:
        while True:
            with self._lock:
                batch = [e for e in list(self._events)[:MAX_EVENTS_PER_BATCH]]
            if not batch:
                return
            unsent = [e for e in batch if not e.sent]
            if unsent:
                result = self.api.post_events([e.payload for e in unsent])
                for item in result.get("rejected", []):
                    log.warning("Event %s rejected by the API: %s", item["clientEventId"], item["reason"])
                for e in unsent:
                    e.sent = True
                self.sent_events += len(result.get("accepted", []))
            rejected = {r["clientEventId"] for r in result.get("rejected", [])} if unsent else set()
            for e in batch:
                if e.snapshot is not None and e.payload["clientEventId"] not in rejected:
                    self.api.put_evidence(e.payload["clientEventId"], e.snapshot)
                    e.snapshot = None
            with self._lock:
                for _ in batch:
                    self._events.popleft()

    def _flush_metrics(self, kind: str, key: str) -> None:
        with self._lock:
            batch = list(self._metrics[kind])[:MAX_METRICS_PER_BATCH]
        if not batch:
            return
        result = self.api.post_metrics(kind, key, batch)
        for reason in result.get("rejected", []):
            log.warning("%s item rejected by the API: %s", kind, reason)
        with self._lock:
            for _ in batch:
                self._metrics[kind].popleft()
