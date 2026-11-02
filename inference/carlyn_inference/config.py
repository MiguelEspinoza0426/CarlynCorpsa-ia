"""Node configuration received from the API (GET /api/ingest/config). See docs/contrato-integracion.md."""
from dataclasses import dataclass, field

from .geometry import Point


@dataclass(frozen=True)
class Rule:
    id: str
    type: str
    min_duration_sec: int
    cooldown_sec: int
    min_confidence: float
    severity: str


@dataclass(frozen=True)
class Threshold:
    warning: float | None
    critical: float


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    type: str  # Work | Circulation | Danger | Restricted
    polygon: list[Point]
    required_epp: frozenset[str]
    rules: dict[str, Rule]
    thresholds: dict[str, Threshold]

    def rule(self, event_type: str) -> Rule | None:
        return self.rules.get(event_type)

    def threshold(self, metric: str) -> Threshold | None:
        return self.thresholds.get(metric)


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    stream_url: str
    credentials_ref: str | None
    target_fps: int
    zones: list[Zone] = field(default_factory=list)

    @property
    def source_key(self) -> tuple:
        """Changes when the video source must be reopened."""
        return self.stream_url, self.credentials_ref


@dataclass(frozen=True)
class NodeConfig:
    version: str
    node_id: str
    epp_classes: frozenset[str]
    person_class: str
    cameras: list[CameraConfig]


def parse(data: dict) -> NodeConfig:
    def zone(z: dict) -> Zone:
        return Zone(
            id=z["id"],
            name=z["name"],
            type=z["type"],
            polygon=[(p[0], p[1]) for p in z["polygon"]],
            required_epp=frozenset(z.get("requiredEpp", [])),
            rules={r["type"]: Rule(r["id"], r["type"], r["minDurationSec"], r["cooldownSec"], r["minConfidence"], r["severity"])
                   for r in z.get("rules", [])},
            thresholds={t["metric"]: Threshold(t.get("warningValue"), t["criticalValue"]) for t in z.get("thresholds", [])},
        )

    return NodeConfig(
        version=data["version"],
        node_id=data["nodeId"],
        epp_classes=frozenset(data["model"]["eppClasses"]),
        person_class=data["model"]["personClass"],
        cameras=[CameraConfig(c["id"], c["name"], c["streamUrl"], c.get("credentialsRef"), c["targetFps"],
                              [zone(z) for z in c.get("zones", [])]) for c in data["cameras"]],
    )
