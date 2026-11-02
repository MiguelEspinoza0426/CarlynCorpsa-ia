"""Engine inputs (per-frame observations) and outputs (events and anonymous metrics).

Track ids are temporary and never identify people.
"""
from dataclasses import dataclass

from ..geometry import Box


@dataclass(frozen=True)
class PersonObs:
    track_id: int
    box: Box
    confidence: float
    epp: frozenset[str] = frozenset()
    fallen: bool = False


@dataclass(frozen=True)
class ObjectObs:
    track_id: int
    label: str
    box: Box
    confidence: float


@dataclass
class EventOut:
    type: str
    zone_id: str | None
    rule_id: str | None
    started_at: float
    ended_at: float | None
    confidence: float
    person_count: int
    track_id: int | None
    bbox: Box | None
    details: dict


@dataclass(frozen=True)
class OccupancySample:
    zone_id: str
    window_start: float
    window_sec: int
    avg_count: float
    max_count: int
    min_count: int


@dataclass(frozen=True)
class DwellSession:
    zone_id: str
    track_id: int
    entered_at: float
    exited_at: float
    from_zone_id: str | None


@dataclass(frozen=True)
class InactivityInterval:
    zone_id: str
    track_id: int
    started_at: float
    ended_at: float
