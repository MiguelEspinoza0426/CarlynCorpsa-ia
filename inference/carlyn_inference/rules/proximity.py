from ..config import Zone
from ..geometry import Point, distance_to_polygon
from .base import PersonZoneRule
from .models import EventOut, EventType, PersonObs
from .state import EngineState, Track, critical, person_event

DEFAULT_PROXIMITY = 0.05


class ProximityRule(PersonZoneRule):
    """A person outside the zone stays within `ProximityDistance` of its polygon."""
    event_type = EventType.PROXIMITY

    def evaluate(self, state: EngineState, zone: Zone, person: PersonObs, track: Track, foot: Point,
                 count: int, now: float) -> list[EventOut]:
        rule = zone.rule(self.event_type)
        if zone.id in track.zones or rule is None:
            return []
        limit = critical(zone, "ProximityDistance") or DEFAULT_PROXIMITY
        distance = distance_to_polygon(foot, zone.polygon)
        if distance > limit:
            track.proximity_since.pop(zone.id, None)
            return []
        since = track.proximity_since.setdefault(zone.id, now)
        if now - since >= rule.min_duration_sec and person.confidence >= rule.min_confidence \
                and state.cooldown_ok(rule, person.track_id, now):
            return [person_event(rule, zone, since, now, person, 1, {"distance": round(distance, 3)})]
        return []
