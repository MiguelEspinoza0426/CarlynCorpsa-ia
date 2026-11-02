import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carlyn_inference.config import CameraConfig, Rule, Threshold, Zone  # noqa: E402

LEFT = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
RIGHT = [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]


def rule(type_: str, min_duration: int = 0, cooldown: int = 60, min_conf: float = 0.4) -> Rule:
    return Rule(id=f"r-{type_}", type=type_, min_duration_sec=min_duration, cooldown_sec=cooldown, min_confidence=min_conf, severity="High")


def zone(id_: str, type_: str, polygon=LEFT, required=(), rules=(), thresholds=None) -> Zone:
    return Zone(id=id_, name=id_, type=type_, polygon=polygon, required_epp=frozenset(required),
                rules={r.type: r for r in rules}, thresholds={k: Threshold(*v) for k, v in (thresholds or {}).items()})


@pytest.fixture
def camera_factory():
    def make(*zones: Zone) -> CameraConfig:
        return CameraConfig(id="cam-1", name="Nave 1", stream_url="rtsp://cam/1", credentials_ref=None, target_fps=5, zones=list(zones))
    return make
