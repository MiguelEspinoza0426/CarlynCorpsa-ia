"""Zone rules engine. Pure Python (no GPU, no camera) so it can be tested in isolation.

Input per frame: tracked people (temporary track id, box, detected EPP, fall flag) and objects.
Output: events and anonymous metrics per zone. Track ids are temporary and never identify people.
"""
import math
from collections import Counter, deque
from dataclasses import dataclass, field

from ..config import CameraConfig, Rule, Zone
from ..geometry import Point, distance_to_polygon, foot_point, point_in_polygon
from .models import DwellSession, EventOut, InactivityInterval, ObjectObs, OccupancySample, PersonObs

TRACK_LOST_SEC = 2.0          # a track not seen for this long is closed
EPP_MAJORITY = 0.7            # an item is missing if absent in ≥ 70 % of the frames of the window
EPP_MIN_FRAMES = 3
MOVE_EPS = 0.02               # displacement (fraction of the image) considered movement
OCCUPANCY_WINDOW_SEC = 10
MIN_DWELL_SEC = 1.0
MIN_FALL_SEC = 2.0
DEFAULT_INACTIVITY_SEC = 300
DEFAULT_PROXIMITY = 0.05


@dataclass
class _Presence:
    entered_at: float
    from_zone_id: str | None
    epp: deque = field(default_factory=deque)       # (timestamp, missing classes)
    anchor: Point | None = None
    still_since: float | None = None
    inactivity_reported: bool = False
    entry_reported: bool = False
    dwell_reported: bool = False


@dataclass
class _Track:
    last_seen: float
    zones: dict[str, _Presence] = field(default_factory=dict)
    last_exited_zone: str | None = None
    proximity_since: dict[str, float] = field(default_factory=dict)
    fallen_since: float | None = None
    fall_reported: bool = False


@dataclass
class _ZoneState:
    window_start: float | None = None
    counts: list[int] = field(default_factory=list)
    over_since: float | None = None


@dataclass
class _Object:
    first_seen: float
    last_seen: float
    anchor: Point
    reported: bool = False


class RuleEngine:
    def __init__(self, camera: CameraConfig):
        self.camera = camera
        self._tracks: dict[int, _Track] = {}
        self._zones: dict[str, _ZoneState] = {z.id: _ZoneState() for z in camera.zones}
        self._objects: dict[int, _Object] = {}
        self._last_emitted: dict[tuple, float] = {}
        self._occupancy: list[OccupancySample] = []
        self._dwell: list[DwellSession] = []
        self._inactivity: list[InactivityInterval] = []

    # ---- public API ----

    def update(self, now: float, persons: list[PersonObs], objects: list[ObjectObs] = ()) -> list[EventOut]:
        events: list[EventOut] = []
        zones = self.camera.zones

        # 1. Zone membership by foot point; zone exits close dwell sessions.
        for p in persons:
            track = self._tracks.setdefault(p.track_id, _Track(last_seen=now))
            track.last_seen = now
            foot = foot_point(p.box)
            inside = {z.id for z in zones if point_in_polygon(foot, z.polygon)}
            for zone_id in set(track.zones) - inside:
                self._exit(p.track_id, track, zone_id, now)
            for zone_id in inside - set(track.zones):
                track.zones[zone_id] = _Presence(entered_at=now, from_zone_id=track.last_exited_zone)
        counts = {z.id: sum(1 for p in persons if z.id in self._tracks[p.track_id].zones) for z in zones}

        # 2. Per-person rules.
        for p in persons:
            track = self._tracks[p.track_id]
            foot = foot_point(p.box)
            for zone in zones:
                presence = track.zones.get(zone.id)
                if presence is not None:
                    events += self._person_in_zone(zone, p, presence, foot, counts[zone.id], now)
                else:
                    events += self._proximity(zone, p, track, foot, now)
            events += self._fall(p, track, counts, now)

        # 3. Tracks that disappeared.
        seen = {p.track_id for p in persons}
        for track_id, track in list(self._tracks.items()):
            if track_id not in seen and now - track.last_seen > TRACK_LOST_SEC:
                for zone_id in list(track.zones):
                    self._exit(track_id, track, zone_id, track.last_seen)
                del self._tracks[track_id]

        # 4. Zone-level rules and occupancy windows.
        for zone in zones:
            events += self._occupancy_rule(zone, counts[zone.id], persons, now)
            self._sample_occupancy(zone, counts[zone.id], now)
        events += self._unsafe_conditions(objects, now)
        return events

    def take_metrics(self) -> tuple[list[OccupancySample], list[DwellSession], list[InactivityInterval]]:
        """Returns and clears the metrics produced since the last call."""
        out = self._occupancy, self._dwell, self._inactivity
        self._occupancy, self._dwell, self._inactivity = [], [], []
        return out

    def zone_counts(self) -> dict[str, int]:
        return {z.id: sum(1 for t in self._tracks.values() if z.id in t.zones) for z in self.camera.zones}

    # ---- person rules ----

    def _person_in_zone(self, zone: Zone, p: PersonObs, presence: _Presence, foot: Point, count: int, now: float) -> list[EventOut]:
        events = []
        stay = now - presence.entered_at

        rule = zone.rule("EppMissing")
        if rule and zone.required_epp and p.confidence >= rule.min_confidence:
            window = max(rule.min_duration_sec, 1)
            presence.epp.append((now, zone.required_epp - p.epp))
            while presence.epp and presence.epp[0][0] < now - window:
                presence.epp.popleft()
            if stay >= window and len(presence.epp) >= EPP_MIN_FRAMES:
                missing = Counter(item for _, absent in presence.epp for item in absent)
                majority = sorted(item for item, n in missing.items() if n / len(presence.epp) >= EPP_MAJORITY)
                if majority and self._cooldown_ok(rule, p.track_id, now):
                    events.append(self._event(rule, zone, now - window, now, p, count,
                                              {"missingEpp": majority, "presentEpp": sorted(zone.required_epp - set(majority))}))

        rule = zone.rule("RestrictedZoneEntry")
        if rule and not presence.entry_reported and stay >= rule.min_duration_sec and p.confidence >= rule.min_confidence \
                and self._cooldown_ok(rule, p.track_id, now):
            presence.entry_reported = True
            events.append(self._event(rule, zone, presence.entered_at, now, p, count, {"durationSec": round(stay, 1)}))

        rule = zone.rule("DangerZoneDwell")
        if rule and not presence.dwell_reported:
            limit = max(rule.min_duration_sec, _critical(zone, "MaxDwellSeconds") or 0)
            if stay >= limit and p.confidence >= rule.min_confidence and self._cooldown_ok(rule, p.track_id, now):
                presence.dwell_reported = True
                events.append(self._event(rule, zone, presence.entered_at, now, p, count, {"durationSec": round(stay, 1)}))

        events += self._inactivity_rule(zone, p, presence, foot, count, now)
        return events

    def _inactivity_rule(self, zone: Zone, p: PersonObs, presence: _Presence, foot: Point, count: int, now: float) -> list[EventOut]:
        rule = zone.rule("Inactivity")
        if rule is None and zone.threshold("InactivitySeconds") is None:
            return []
        limit = _inactivity_limit(zone)
        if presence.anchor is None or math.dist(foot, presence.anchor) > MOVE_EPS:
            self._close_inactivity(zone.id, p.track_id, presence, limit, now)
            presence.anchor, presence.still_since, presence.inactivity_reported = foot, now, False
            return []
        still = now - presence.still_since
        if rule and not presence.inactivity_reported and still >= limit and self._cooldown_ok(rule, p.track_id, now):
            presence.inactivity_reported = True
            return [self._event(rule, zone, presence.still_since, now, p, count, {"durationSec": round(still, 1)})]
        return []

    def _proximity(self, zone: Zone, p: PersonObs, track: _Track, foot: Point, now: float) -> list[EventOut]:
        rule = zone.rule("Proximity")
        if rule is None:
            return []
        limit = _critical(zone, "ProximityDistance") or DEFAULT_PROXIMITY
        distance = distance_to_polygon(foot, zone.polygon)
        if distance > limit:
            track.proximity_since.pop(zone.id, None)
            return []
        since = track.proximity_since.setdefault(zone.id, now)
        if now - since >= rule.min_duration_sec and p.confidence >= rule.min_confidence and self._cooldown_ok(rule, p.track_id, now):
            return [self._event(rule, zone, since, now, p, 1, {"distance": round(distance, 3)})]
        return []

    def _fall(self, p: PersonObs, track: _Track, counts: dict[str, int], now: float) -> list[EventOut]:
        if not p.fallen:
            track.fallen_since, track.fall_reported = None, False
            return []
        track.fallen_since = track.fallen_since or now
        zone = next((z for z in self.camera.zones if z.id in track.zones and z.rule("PossibleAccident")), None)
        if zone is None or track.fall_reported:
            return []
        rule = zone.rule("PossibleAccident")
        duration = now - track.fallen_since
        if duration >= max(rule.min_duration_sec, MIN_FALL_SEC) and p.confidence >= rule.min_confidence:
            track.fall_reported = True
            return [self._event(rule, zone, track.fallen_since, None, p, counts[zone.id],
                                {"kind": "PersonOnGround", "durationSec": round(duration, 1)})]
        return []

    # ---- zone rules ----

    def _occupancy_rule(self, zone: Zone, count: int, persons: list[PersonObs], now: float) -> list[EventOut]:
        rule, limit = zone.rule("OccupancyExceeded"), _critical(zone, "MaxOccupancy")
        state = self._zones[zone.id]
        if rule is None or limit is None or count < limit:
            state.over_since = None
            return []
        state.over_since = state.over_since or now
        if now - state.over_since >= rule.min_duration_sec and self._cooldown_ok(rule, "zone", now):
            inside = [p for p in persons if zone.id in self._tracks[p.track_id].zones]
            confidence = sum(p.confidence for p in inside) / len(inside)
            return [EventOut("OccupancyExceeded", zone.id, rule.id, state.over_since, now, round(confidence, 3), count, None, None,
                             {"count": count, "threshold": limit})]
        return []

    def _unsafe_conditions(self, objects: list[ObjectObs], now: float) -> list[EventOut]:
        events = []
        for obj in objects:
            foot = foot_point(obj.box)
            zone = next((z for z in self.camera.zones if z.rule("UnsafeCondition") and point_in_polygon(foot, z.polygon)), None)
            if zone is None:
                self._objects.pop(obj.track_id, None)
                continue
            state = self._objects.get(obj.track_id)
            if state is None or math.dist(foot, state.anchor) > MOVE_EPS * 2:
                state = self._objects[obj.track_id] = _Object(first_seen=now, last_seen=now, anchor=foot)
            state.last_seen = now
            rule = zone.rule("UnsafeCondition")
            stay = now - state.first_seen
            if not state.reported and stay >= rule.min_duration_sec and obj.confidence >= rule.min_confidence \
                    and self._cooldown_ok(rule, ("object", obj.track_id), now):
                state.reported = True
                events.append(EventOut("UnsafeCondition", zone.id, rule.id, state.first_seen, now, round(obj.confidence, 3), 0, None,
                                       obj.box, {"objectClass": obj.label, "durationSec": round(stay, 1)}))
        for track_id, state in list(self._objects.items()):
            if now - state.last_seen > TRACK_LOST_SEC * 5:
                del self._objects[track_id]
        return events

    # ---- metrics ----

    def _sample_occupancy(self, zone: Zone, count: int, now: float) -> None:
        state = self._zones[zone.id]
        if state.window_start is None:
            state.window_start = now
        state.counts.append(count)
        if now - state.window_start >= OCCUPANCY_WINDOW_SEC:
            self._occupancy.append(OccupancySample(zone.id, state.window_start, OCCUPANCY_WINDOW_SEC,
                                                   round(sum(state.counts) / len(state.counts), 2), max(state.counts), min(state.counts)))
            state.window_start, state.counts = now, []

    def _exit(self, track_id: int, track: _Track, zone_id: str, now: float) -> None:
        presence = track.zones.pop(zone_id)
        if now - presence.entered_at >= MIN_DWELL_SEC:
            self._dwell.append(DwellSession(zone_id, track_id, presence.entered_at, now, presence.from_zone_id))
        zone = next(z for z in self.camera.zones if z.id == zone_id)
        self._close_inactivity(zone_id, track_id, presence, _inactivity_limit(zone), now)
        track.last_exited_zone = zone_id

    def _close_inactivity(self, zone_id: str, track_id: int, presence: _Presence, limit: float, now: float) -> None:
        if presence.still_since is not None and now - presence.still_since >= limit:
            self._inactivity.append(InactivityInterval(zone_id, track_id, presence.still_since, now))
        presence.still_since = None

    # ---- helpers ----

    def _cooldown_ok(self, rule: Rule, key, now: float) -> bool:
        last = self._last_emitted.get((rule.id, key))
        if last is not None and now - last < rule.cooldown_sec:
            return False
        self._last_emitted[(rule.id, key)] = now
        return True

    @staticmethod
    def _event(rule: Rule, zone: Zone, started: float, ended: float | None, p: PersonObs, count: int, details: dict) -> EventOut:
        return EventOut(rule.type, zone.id, rule.id, started, ended, round(p.confidence, 3), count, p.track_id, p.box, details)


def _critical(zone: Zone, metric: str) -> float | None:
    threshold = zone.threshold(metric)
    return threshold.critical if threshold else None


def _inactivity_limit(zone: Zone) -> float:
    rule = zone.rule("Inactivity")
    return _critical(zone, "InactivitySeconds") or (rule.min_duration_sec if rule and rule.min_duration_sec else DEFAULT_INACTIVITY_SEC)
