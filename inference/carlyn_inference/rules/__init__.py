"""Zone rules engine. Pure Python (no GPU, no camera) so it can be tested in isolation.

Input per frame: tracked people (temporary track id, box, detected EPP, fall flag) and objects.
Output: events and anonymous metrics per zone. Track ids are temporary and never identify people.
"""
from .base import FrameRule, PersonRule, PersonZoneRule, RuleStrategy, ZoneRule
from .engine import RuleEngine
from .models import DwellSession, EventOut, EventType, InactivityInterval, ObjectObs, OccupancySample, PersonObs
from .registry import default_rules

__all__ = ["DwellSession", "EventOut", "EventType", "FrameRule", "InactivityInterval", "ObjectObs", "OccupancySample", "PersonObs",
           "PersonRule", "PersonZoneRule", "RuleEngine", "RuleStrategy", "ZoneRule", "default_rules"]
