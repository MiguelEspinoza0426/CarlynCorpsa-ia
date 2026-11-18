from collections import Counter

from ..config import Zone
from ..geometry import Point
from .base import PersonZoneRule
from .models import EventOut, EventType, PersonObs
from .state import EngineState, Track, person_event

EPP_MAJORITY = 0.7            # an item is missing if absent in ≥ 70 % of the frames of the window
EPP_MIN_FRAMES = 3


class EppMissingRule(PersonZoneRule):
    """Required EPP absent in most frames of a sliding window (tolerates brief occlusions)."""
    event_type = EventType.EPP_MISSING

    def evaluate(self, state: EngineState, zone: Zone, person: PersonObs, track: Track, foot: Point,
                 count: int, now: float) -> list[EventOut]:
        presence = track.zones.get(zone.id)
        rule = zone.rule(self.event_type)
        if presence is None or rule is None or not zone.required_epp or person.confidence < rule.min_confidence \
                or not person.epp_evaluated:
            return []
        window = max(rule.min_duration_sec, 1)
        presence.epp.append((now, zone.required_epp - person.epp))
        while presence.epp and presence.epp[0][0] < now - window:
            presence.epp.popleft()
        if now - presence.entered_at < window or len(presence.epp) < EPP_MIN_FRAMES:
            return []
        missing = Counter(item for _, absent in presence.epp for item in absent)
        majority = sorted(item for item, n in missing.items() if n / len(presence.epp) >= EPP_MAJORITY)
        if majority and state.cooldown_ok(rule, person.track_id, now):
            return [person_event(rule, zone, now - window, now, person, count,
                                 {"missingEpp": majority, "presentEpp": sorted(zone.required_epp - set(majority))})]
        return []
