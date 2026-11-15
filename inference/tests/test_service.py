import json
import threading

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from carlyn_inference import config as config_module
from carlyn_inference.api_client import ApiClient, Outbox, event_payload
from carlyn_inference.detection import associate_epp, blur_faces, torso_angle
from carlyn_inference.live_stream import LiveStream
from carlyn_inference.rules import EventOut, OccupancySample, PersonObs
from carlyn_inference.server import create_app
from carlyn_inference.service import InferenceService
from carlyn_inference.settings import Settings
from carlyn_inference.sources import with_credentials

CONFIG = {
    "version": "abc123", "nodeId": "gpu-01",
    "model": {"eppClasses": ["careta_soldar", "guantes"], "personClass": "persona"},
    "cameras": [{
        "id": "cam-1", "name": "Nave 1", "streamUrl": "rtsp://192.168.100.14:554/Streaming/Channels/102",
        "credentialsRef": "CAM_DVR1", "targetFps": 5,
        "zones": [{
            "id": "z-1", "name": "Soldadura", "type": "Work", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "requiredEpp": ["guantes"],
            "rules": [{"id": "r-1", "type": "EppMissing", "minDurationSec": 3, "cooldownSec": 60, "minConfidence": 0.45, "severity": "High"}],
            "thresholds": [{"metric": "MaxOccupancy", "warningValue": 3, "criticalValue": 4}],
        }],
    }],
}


class FakeApi:
    """Records the HTTP calls made to the .NET API."""

    def __init__(self, fail_events: int = 0):
        self.calls: list[tuple[str, str, object]] = []
        self.fail_events = fail_events

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.headers.get("content-type") == "application/json" else request.content
        self.calls.append((request.method, request.url.path, body))
        assert request.headers["X-Api-Key"] == "node-key"
        if request.url.path == "/api/ingest/config":
            if request.headers.get("If-None-Match") == '"abc123"':
                return httpx.Response(304)
            return httpx.Response(200, json=CONFIG, headers={"ETag": '"abc123"'})
        if request.url.path == "/api/ingest/events":
            if self.fail_events > 0:
                self.fail_events -= 1
                return httpx.Response(503)
            return httpx.Response(200, json={"accepted": [e["clientEventId"] for e in body["events"]], "duplicates": [], "rejected": []})
        if request.url.path.startswith("/api/ingest/metrics/"):
            return httpx.Response(200, json={"accepted": len(next(iter(body.values()))), "rejected": []})
        return httpx.Response(204)

    def client(self) -> ApiClient:
        return ApiClient("http://api", "node-key", transport=httpx.MockTransport(self.handler))


# ---- configuration ----

def test_config_is_parsed():
    node = config_module.parse(CONFIG)
    zone = node.cameras[0].zones[0]
    assert node.person_class == "persona" and "guantes" in node.epp_classes
    assert zone.required_epp == {"guantes"} and zone.rule("EppMissing").min_duration_sec == 3
    assert zone.threshold("MaxOccupancy").critical == 4


def test_credentials_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("CAM_DVR1_USER", "visor")
    monkeypatch.setenv("CAM_DVR1_PASS", "cl@ve#1")
    url = with_credentials("rtsp://192.168.100.14:554/Streaming/Channels/102", "CAM_DVR1")
    assert url == "rtsp://visor:cl%40ve%231@192.168.100.14:554/Streaming/Channels/102"
    assert with_credentials("rtsp://cam/1", None) == "rtsp://cam/1"


# ---- API client and outbox ----

def test_config_download_uses_etag():
    api = FakeApi().client()
    status, data, etag = api.get_config(None)
    assert status == 200 and data["version"] == "abc123"
    assert api.get_config(etag)[0] == 304


def test_outbox_sends_events_then_evidence_and_metrics():
    fake = FakeApi()
    outbox = Outbox(fake.client())
    event = EventOut("EppMissing", "z-1", "r-1", 1_700_000_000.0, 1_700_000_003.5, 0.82, 1, 17, (0.1, 0.2, 0.1, 0.5),
                     {"missingEpp": ["guantes"], "presentEpp": []})
    payload = event_payload("cam-1", event, "epp_v3_completo")
    outbox.add_event(payload, b"\xff\xd8jpeg")
    outbox.add_metrics("cam-1", [OccupancySample("z-1", 1_700_000_000.0, 10, 1.5, 2, 1)], [], [])

    outbox.flush()

    paths = [c[1] for c in fake.calls]
    assert paths == ["/api/ingest/events", f"/api/ingest/events/{payload['clientEventId']}/evidence", "/api/ingest/metrics/occupancy"]
    sent = fake.calls[0][2]["events"][0]
    assert sent["startedAt"] == "2023-11-14T22:13:20Z" and sent["bbox"] == [0.1, 0.2, 0.1, 0.5] and sent["modelVersion"] == "epp_v3_completo"
    assert outbox.pending() == 0


def test_outbox_keeps_events_when_the_api_is_down():
    fake = FakeApi(fail_events=1)
    outbox = Outbox(fake.client())
    outbox.add_event(event_payload("cam-1", EventOut("Inactivity", "z-1", None, 1.0, 2.0, 0.9, 1, 3, None, {}), "v3"), None)

    with pytest.raises(httpx.HTTPStatusError):
        outbox.flush()
    assert outbox.pending() == 1
    outbox.flush()
    assert outbox.pending() == 0
    first, retry = [c[2]["events"][0]["clientEventId"] for c in fake.calls if c[1] == "/api/ingest/events"]
    assert first == retry   # same id: the API deduplicates retries


# ---- detection helpers (no GPU) ----

def test_epp_is_assigned_to_the_person_that_contains_it():
    persons = [(1, (0.1, 0.1, 0.2, 0.6), 0.9), (2, (0.6, 0.1, 0.2, 0.6), 0.9)]
    epp = [("guantes", (0.12, 0.5, 0.04, 0.04), 0.7), ("careta_soldar", (0.65, 0.1, 0.08, 0.08), 0.8), ("guantes", (0.45, 0.9, 0.02, 0.02), 0.6)]
    assert associate_epp(persons, epp) == {1: {"guantes"}, 2: {"careta_soldar"}}


def test_torso_angle_distinguishes_standing_from_lying():
    standing, lying = np.zeros((17, 2)), np.zeros((17, 2))
    standing[[5, 6]] = [[100, 100], [140, 100]]
    standing[[11, 12]] = [[105, 220], [135, 220]]
    lying[[5, 6]] = [[100, 300], [100, 330]]
    lying[[11, 12]] = [[220, 305], [220, 325]]
    assert torso_angle(standing) < 10
    assert torso_angle(lying) > 80


def test_faces_are_pixelated_in_the_top_of_person_boxes():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)   # textured image (details of a face)
    out = blur_faces(frame, [PersonObs(1, (0.2, 0.1, 0.1, 0.8), 0.9)], [])
    head_before, head_after = frame[20:42, 40:60].astype(float), out[20:42, 40:60].astype(float)
    assert head_after.std() < head_before.std() * 0.5             # detail removed
    assert len(np.unique(head_after.reshape(-1, 3), axis=0)) <= 36  # at most 6x6 blocks
    assert (frame[150:, :] == out[150:, :]).all()                 # the rest is untouched


# ---- service and HTTP server ----

class FakeWorker:
    def __init__(self, camera, node, settings, outbox):
        self.camera, self.started, self.stopped, self.updated = camera, False, False, 0
        self.latest_jpeg = b"\xff\xd8annotated"

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def update_camera(self, camera):
        self.camera, self.updated = camera, self.updated + 1

    def snapshot_jpeg(self):
        return b"\xff\xd8raw"

    def heartbeat(self):
        return {"cameraId": self.camera.id, "status": "Online", "fps": 5.0, "latencyMs": 100, "lastFrameAt": None, "error": None}


def make_service() -> InferenceService:
    return InferenceService(Settings(ingest_api_key="node-key"), api=FakeApi().client(), worker_factory=FakeWorker)


def test_service_starts_restarts_and_stops_workers():
    service = make_service()
    assert service.refresh_config()
    worker = service.workers["cam-1"]
    assert worker.started

    same_source = config_module.parse({**CONFIG, "version": "v2"})
    service.apply(same_source)
    assert service.workers["cam-1"] is worker and worker.updated == 1

    moved = json.loads(json.dumps(CONFIG))
    moved["cameras"][0]["streamUrl"] = "rtsp://192.168.100.14:554/Streaming/Channels/202"
    service.apply(config_module.parse(moved))
    assert worker.stopped and service.workers["cam-1"] is not worker

    service.apply(config_module.parse({**CONFIG, "cameras": []}))
    assert service.workers == {}


def test_server_requires_internal_key_and_serves_snapshot():
    service = make_service()
    service.refresh_config()
    client = TestClient(create_app(service, internal_key="secret"))

    assert client.get("/health").json()["cameras"] == {"cam-1": "Online"}
    assert client.get("/cameras/cam-1/snapshot.jpg").status_code == 401
    response = client.get("/cameras/cam-1/snapshot.jpg", headers={"X-Internal-Key": "secret"})
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert client.get("/cameras/unknown/snapshot.jpg", headers={"X-Internal-Key": "secret"}).status_code == 404


def test_reload_endpoint_triggers_a_configuration_refresh():
    service = make_service()
    client = TestClient(create_app(service, internal_key="secret"))
    assert client.post("/config/reload", headers={"X-Internal-Key": "secret"}).status_code == 202
    assert service._reload.is_set()
