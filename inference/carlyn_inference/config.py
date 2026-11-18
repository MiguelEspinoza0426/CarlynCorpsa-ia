"""Node configuration received from the API (GET /api/ingest/config). See docs/contrato-integracion.md."""
from dataclasses import dataclass, field

from .geometry import Point
from .preprocess import PreprocessConfig
from .slicing import SliceConfig


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
class InferenceConfig:
    """Per-camera detection settings (`inference` block); everything off when the API does not send it."""
    slicing: SliceConfig = SliceConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    # None: the node default (PERSON_CONF, IMGSZ, MIN_PERSON_HEIGHT_PX).
    person_confidence: float | None = None
    img_size: int | None = None
    min_person_height_px: int | None = None

    @staticmethod
    def parse(data: dict | None) -> "InferenceConfig":
        data = data or {}
        number = lambda key, kind: kind(data[key]) if data.get(key) is not None else None  # noqa: E731
        return InferenceConfig(SliceConfig.parse(data.get("slice")), PreprocessConfig.parse(data.get("preprocess")),
                               number("personConfidence", float), number("imgSize", int), number("minPersonHeightPx", int))

    def apply_to(self, settings):
        """The node settings with this camera's thresholds on top."""
        from dataclasses import replace
        overrides = {k: v for k, v in (("person_conf", self.person_confidence), ("imgsz", self.img_size),
                                       ("min_person_height_px", self.min_person_height_px)) if v is not None}
        return replace(settings, **overrides) if overrides else settings


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    stream_url: str
    credentials_ref: str | None
    target_fps: int
    zones: list[Zone] = field(default_factory=list)
    # Pixelate faces in the live video (evidence images and snapshots are always pixelated).
    blur_faces: bool = True
    # SAHI and contrast correction; changing them restarts the camera's worker (new models).
    inference: InferenceConfig = InferenceConfig()

    @property
    def slicing(self) -> SliceConfig:
        return self.inference.slicing

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
    # Minimum confidence per EPP class; classes not listed use CONF.
    epp_min_confidence: dict[str, float] = field(default_factory=dict)


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
        epp_min_confidence={k: float(v) for k, v in (data["model"].get("eppMinConfidence") or {}).items()},
        person_class=data["model"]["personClass"],
        cameras=[CameraConfig(c["id"], c["name"], c["streamUrl"], c.get("credentialsRef"), c["targetFps"],
                              [zone(z) for z in c.get("zones", [])], c.get("blurFaces", True),
                              InferenceConfig.parse(c.get("inference"))) for c in data["cameras"]],
    )
