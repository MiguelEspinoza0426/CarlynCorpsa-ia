"""Mutable state shared by the engine and the rules: tracks, presences, cooldowns and metric buffers."""
from collections import deque
from dataclasses import dataclass, field

from ..config import CameraConfig, Rule, Zone
from ..geometry import Point
from .models import DwellSession, EventOut, EventType, InactivityInterval, OccupancySample, PersonObs

TRACK_LOST_SEC = 2.0          # a track not seen for this long is closed
MOVE_EPS = 0.02               # displacement (fraction of the image) considered movement
DEFAULT_INACTIVITY_SEC = 300


@dataclass
class Presence:
    entered_at: float
    from_zone_id: str | None
    epp: deque = field(default_factory=deque)       # (timestamp, missing classes)
    anchor: Point | None = None
    still_since: float | None = None
    inactivity_reported: bool = False
    entry_reported: bool = False
    dwell_reported: bool = False


@dataclass
class Track:
    last_seen: float
    zones: dict[str, Presence] = field(default_factory=dict)
    last_exited_zone: str | None = None
    proximity_since: dict[str, float] = field(default_factory=dict)
    fallen_since: float | None = None
    fall_reported: bool = False


@dataclass
class ZoneState:
    window_start: float | None = None
    counts: list[int] = field(default_factory=list)
    over_since: float | None = None


@dataclass
class ObjectState:
    first_seen: float
    last_seen: float
    anchor: Point
    reported: bool = False


class EngineState:
    """Everything that survives between frames. Owned by the engine, read and updated by the rules."""

    def __init__(self, camera: CameraConfig):
        self.camera = camera
        self.tracks: dict[int, Track] = {}
        self.zones: dict[str, ZoneState] = {z.id: ZoneState() for z in camera.zones}
        self.objects: dict[int, ObjectState] = {}
        self.last_emitted: dict[tuple, float] = {}
        self.occupancy: list[OccupancySample] = []
        self.dwell: list[DwellSession] = []
        self.inactivity: list[InactivityInterval] = []

    def cooldown_ok(self, rule: Rule, key, now: float) -> bool:
        """True (and the emission is recorded) if the rule may fire again for this key."""
        last = self.last_emitted.get((rule.id, key))
        if last is not None and now - last < rule.cooldown_sec:
            return False
        self.last_emitted[(rule.id, key)] = now
        return True

    def close_inactivity(self, zone_id: str, track_id: int, presence: Presence, limit: float, now: float) -> None:
        if presence.still_since is not None and now - presence.still_since >= limit:
            self.inactivity.append(InactivityInterval(zone_id, track_id, presence.still_since, now))
        presence.still_since = None


def critical(zone: Zone, metric: str) -> float | None:
    threshold = zone.threshold(metric)
    return threshold.critical if threshold else None


def inactivity_limit(zone: Zone) -> float:
    rule = zone.rule(EventType.INACTIVITY)
    return critical(zone, "InactivitySeconds") or (rule.min_duration_sec if rule and rule.min_duration_sec else DEFAULT_INACTIVITY_SEC)


def person_event(rule: Rule, zone: Zone, started: float, ended: float | None, p: PersonObs, count: int, details: dict) -> EventOut:
    return EventOut(rule.type, zone.id, rule.id, started, ended, round(p.confidence, 3), count, p.track_id, p.box, details)
