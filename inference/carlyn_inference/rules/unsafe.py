import math

from ..geometry import foot_point, point_in_polygon
from .base import FrameRule
from .models import EventOut, EventType, ObjectObs
from .state import MOVE_EPS, TRACK_LOST_SEC, EngineState, ObjectState


class UnsafeConditionRule(FrameRule):
    """A static object inside a zone with the rule for `minDurationSec` (e.g. something blocking a walkway)."""
    event_type = EventType.UNSAFE_CONDITION

    def evaluate(self, state: EngineState, objects: list[ObjectObs], now: float) -> list[EventOut]:
        events = []
        for obj in objects:
            foot = foot_point(obj.box)
            zone = next((z for z in state.camera.zones if z.rule(self.event_type) and point_in_polygon(foot, z.polygon)), None)
            if zone is None:
                state.objects.pop(obj.track_id, None)
                continue
            tracked = state.objects.get(obj.track_id)
            if tracked is None or math.dist(foot, tracked.anchor) > MOVE_EPS * 2:
                tracked = state.objects[obj.track_id] = ObjectState(first_seen=now, last_seen=now, anchor=foot)
            tracked.last_seen = now
            rule = zone.rule(self.event_type)
            stay = now - tracked.first_seen
            if not tracked.reported and stay >= rule.min_duration_sec and obj.confidence >= rule.min_confidence \
                    and state.cooldown_ok(rule, ("object", obj.track_id), now):
                tracked.reported = True
                events.append(EventOut(rule.type, zone.id, rule.id, tracked.first_seen, now, round(obj.confidence, 3), 0, None,
                                       obj.box, {"objectClass": obj.label, "durationSec": round(stay, 1)}))
        for track_id, tracked in list(state.objects.items()):
            if now - tracked.last_seen > TRACK_LOST_SEC * 5:
                del state.objects[track_id]
        return events
