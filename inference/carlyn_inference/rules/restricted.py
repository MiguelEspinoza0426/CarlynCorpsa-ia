from ..config import Zone
from ..geometry import Point
from .base import PersonZoneRule
from .models import EventOut, EventType, PersonObs
from .state import EngineState, Track, person_event


class RestrictedZoneEntryRule(PersonZoneRule):
    """A person stays inside a restricted zone for `minDurationSec`. Reported once per stay."""
    event_type = EventType.RESTRICTED_ZONE_ENTRY

    def evaluate(self, state: EngineState, zone: Zone, person: PersonObs, track: Track, foot: Point,
                 count: int, now: float) -> list[EventOut]:
        presence = track.zones.get(zone.id)
        rule = zone.rule(self.event_type)
        if presence is None or rule is None or presence.entry_reported:
            return []
        stay = now - presence.entered_at
        if stay >= rule.min_duration_sec and person.confidence >= rule.min_confidence \
                and state.cooldown_ok(rule, person.track_id, now):
            presence.entry_reported = True
            return [person_event(rule, zone, presence.entered_at, now, person, count, {"durationSec": round(stay, 1)})]
        return []
