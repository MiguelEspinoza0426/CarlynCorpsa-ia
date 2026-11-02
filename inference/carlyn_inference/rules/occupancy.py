from ..config import Zone
from .base import ZoneRule
from .models import EventOut, EventType, PersonObs
from .state import EngineState, critical


class OccupancyExceededRule(ZoneRule):
    """People inside the zone ≥ `MaxOccupancy` for `minDurationSec`."""
    event_type = EventType.OCCUPANCY_EXCEEDED

    def evaluate(self, state: EngineState, zone: Zone, count: int, persons: list[PersonObs], now: float) -> list[EventOut]:
        rule, limit = zone.rule(self.event_type), critical(zone, "MaxOccupancy")
        zone_state = state.zones[zone.id]
        if rule is None or limit is None or count < limit:
            zone_state.over_since = None
            return []
        zone_state.over_since = zone_state.over_since or now
        if now - zone_state.over_since >= rule.min_duration_sec and state.cooldown_ok(rule, "zone", now):
            inside = [p for p in persons if zone.id in state.tracks[p.track_id].zones]
            confidence = sum(p.confidence for p in inside) / len(inside)
            return [EventOut(rule.type, zone.id, rule.id, zone_state.over_since, now, round(confidence, 3), count, None, None,
                             {"count": count, "threshold": limit})]
        return []
