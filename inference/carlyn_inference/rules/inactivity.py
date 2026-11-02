import math

from ..config import Zone
from ..geometry import Point
from .base import PersonZoneRule
from .models import EventOut, EventType, PersonObs
from .state import MOVE_EPS, EngineState, Track, inactivity_limit, person_event


class InactivityRule(PersonZoneRule):
    """No displacement for `InactivitySeconds`. It also tracks still periods for the inactivity metric,
    so it runs when the zone only has the threshold (no rule)."""
    event_type = EventType.INACTIVITY

    def evaluate(self, state: EngineState, zone: Zone, person: PersonObs, track: Track, foot: Point,
                 count: int, now: float) -> list[EventOut]:
        presence = track.zones.get(zone.id)
        rule = zone.rule(self.event_type)
        if presence is None or (rule is None and zone.threshold("InactivitySeconds") is None):
            return []
        limit = inactivity_limit(zone)
        if presence.anchor is None or math.dist(foot, presence.anchor) > MOVE_EPS:
            state.close_inactivity(zone.id, person.track_id, presence, limit, now)
            presence.anchor, presence.still_since, presence.inactivity_reported = foot, now, False
            return []
        still = now - presence.still_since
        if rule and not presence.inactivity_reported and still >= limit and state.cooldown_ok(rule, person.track_id, now):
            presence.inactivity_reported = True
            return [person_event(rule, zone, presence.still_since, now, person, count, {"durationSec": round(still, 1)})]
        return []
