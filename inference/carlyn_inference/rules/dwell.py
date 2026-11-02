from ..config import Zone
from ..geometry import Point
from .base import PersonZoneRule
from .models import EventOut, EventType, PersonObs
from .state import EngineState, Track, critical, person_event


class DangerZoneDwellRule(PersonZoneRule):
    """Continuous stay in a danger zone of at least `MaxDwellSeconds`. Reported once per stay."""
    event_type = EventType.DANGER_ZONE_DWELL

    def evaluate(self, state: EngineState, zone: Zone, person: PersonObs, track: Track, foot: Point,
                 count: int, now: float) -> list[EventOut]:
        presence = track.zones.get(zone.id)
        rule = zone.rule(self.event_type)
        if presence is None or rule is None or presence.dwell_reported:
            return []
        stay = now - presence.entered_at
        limit = max(rule.min_duration_sec, critical(zone, "MaxDwellSeconds") or 0)
        if stay >= limit and person.confidence >= rule.min_confidence and state.cooldown_ok(rule, person.track_id, now):
            presence.dwell_reported = True
            return [person_event(rule, zone, presence.entered_at, now, person, count, {"durationSec": round(stay, 1)})]
        return []
