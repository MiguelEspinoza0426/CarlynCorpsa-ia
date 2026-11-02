"""Coordinates the rules: updates tracks and zone membership, runs the registered rules and collects the metrics.

The engine knows no concrete rule; see registry.py to add one.
"""
from collections.abc import Mapping

from ..config import CameraConfig, Zone
from ..geometry import foot_point, point_in_polygon
from .base import FrameRule, PersonRule, PersonZoneRule, RuleStrategy, ZoneRule
from .models import DwellSession, EventOut, EventType, InactivityInterval, ObjectObs, OccupancySample, PersonObs
from .registry import default_rules
from .state import TRACK_LOST_SEC, EngineState, Presence, Track, inactivity_limit

OCCUPANCY_WINDOW_SEC = 10
MIN_DWELL_SEC = 1.0


class RuleEngine:
    def __init__(self, camera: CameraConfig, rules: Mapping[EventType, RuleStrategy] | None = None):
        self.camera = camera
        self._state = EngineState(camera)
        table = list((rules if rules is not None else default_rules()).values())
        self._person_zone_rules = [r for r in table if isinstance(r, PersonZoneRule)]
        self._person_rules = [r for r in table if isinstance(r, PersonRule)]
        self._zone_rules = [r for r in table if isinstance(r, ZoneRule)]
        self._frame_rules = [r for r in table if isinstance(r, FrameRule)]

    # ---- public API ----

    def update(self, now: float, persons: list[PersonObs], objects: list[ObjectObs] = ()) -> list[EventOut]:
        state, zones = self._state, self.camera.zones
        events: list[EventOut] = []

        # 1. Zone membership by foot point; zone exits close dwell sessions.
        for p in persons:
            track = state.tracks.setdefault(p.track_id, Track(last_seen=now))
            track.last_seen = now
            foot = foot_point(p.box)
            inside = {z.id for z in zones if point_in_polygon(foot, z.polygon)}
            for zone_id in set(track.zones) - inside:
                self._exit(p.track_id, track, zone_id, now)
            for zone_id in inside - set(track.zones):
                track.zones[zone_id] = Presence(entered_at=now, from_zone_id=track.last_exited_zone)
        counts = {z.id: sum(1 for p in persons if z.id in state.tracks[p.track_id].zones) for z in zones}

        # 2. Per-person rules.
        for p in persons:
            track = state.tracks[p.track_id]
            foot = foot_point(p.box)
            for zone in zones:
                for rule in self._person_zone_rules:
                    events += rule.evaluate(state, zone, p, track, foot, counts[zone.id], now)
            for rule in self._person_rules:
                events += rule.evaluate(state, p, track, counts, now)

        # 3. Tracks that disappeared.
        seen = {p.track_id for p in persons}
        for track_id, track in list(state.tracks.items()):
            if track_id not in seen and now - track.last_seen > TRACK_LOST_SEC:
                for zone_id in list(track.zones):
                    self._exit(track_id, track, zone_id, track.last_seen)
                del state.tracks[track_id]

        # 4. Zone-level rules and occupancy windows, then frame-level rules.
        for zone in zones:
            for rule in self._zone_rules:
                events += rule.evaluate(state, zone, counts[zone.id], persons, now)
            self._sample_occupancy(zone, counts[zone.id], now)
        for rule in self._frame_rules:
            events += rule.evaluate(state, objects, now)
        return events

    def take_metrics(self) -> tuple[list[OccupancySample], list[DwellSession], list[InactivityInterval]]:
        """Returns and clears the metrics produced since the last call."""
        state = self._state
        out = state.occupancy, state.dwell, state.inactivity
        state.occupancy, state.dwell, state.inactivity = [], [], []
        return out

    def zone_counts(self) -> dict[str, int]:
        return {z.id: sum(1 for t in self._state.tracks.values() if z.id in t.zones) for z in self.camera.zones}

    # ---- metrics ----

    def _sample_occupancy(self, zone: Zone, count: int, now: float) -> None:
        zone_state = self._state.zones[zone.id]
        if zone_state.window_start is None:
            zone_state.window_start = now
        zone_state.counts.append(count)
        if now - zone_state.window_start >= OCCUPANCY_WINDOW_SEC:
            counts = zone_state.counts
            self._state.occupancy.append(OccupancySample(zone.id, zone_state.window_start, OCCUPANCY_WINDOW_SEC,
                                                         round(sum(counts) / len(counts), 2), max(counts), min(counts)))
            zone_state.window_start, zone_state.counts = now, []

    def _exit(self, track_id: int, track: Track, zone_id: str, now: float) -> None:
        presence = track.zones.pop(zone_id)
        if now - presence.entered_at >= MIN_DWELL_SEC:
            self._state.dwell.append(DwellSession(zone_id, track_id, presence.entered_at, now, presence.from_zone_id))
        zone = next(z for z in self.camera.zones if z.id == zone_id)
        self._state.close_inactivity(zone_id, track_id, presence, inactivity_limit(zone), now)
        track.last_exited_zone = zone_id
