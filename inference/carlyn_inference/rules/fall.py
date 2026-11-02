from .base import PersonRule
from .models import EventOut, EventType, PersonObs
from .state import EngineState, Track, person_event

MIN_FALL_SEC = 2.0


class PossibleAccidentRule(PersonRule):
    """A person flagged as fallen for at least 2 s inside a zone with the rule. Always needs human validation."""
    event_type = EventType.POSSIBLE_ACCIDENT

    def evaluate(self, state: EngineState, person: PersonObs, track: Track, counts: dict[str, int], now: float) -> list[EventOut]:
        if not person.fallen:
            track.fallen_since, track.fall_reported = None, False
            return []
        track.fallen_since = track.fallen_since or now
        zone = next((z for z in state.camera.zones if z.id in track.zones and z.rule(self.event_type)), None)
        if zone is None or track.fall_reported:
            return []
        rule = zone.rule(self.event_type)
        duration = now - track.fallen_since
        if duration >= max(rule.min_duration_sec, MIN_FALL_SEC) and person.confidence >= rule.min_confidence:
            track.fall_reported = True
            return [person_event(rule, zone, track.fallen_since, None, person, counts[zone.id],
                                 {"kind": "PersonOnGround", "durationSec": round(duration, 1)})]
        return []
