"""Zone rules engine. Pure Python (no GPU, no camera) so it can be tested in isolation.

Input per frame: tracked people (temporary track id, box, detected EPP, fall flag) and objects.
Output: events and anonymous metrics per zone. Track ids are temporary and never identify people.
"""
from .engine import RuleEngine
from .models import DwellSession, EventOut, InactivityInterval, ObjectObs, OccupancySample, PersonObs

__all__ = ["DwellSession", "EventOut", "InactivityInterval", "ObjectObs", "OccupancySample", "PersonObs", "RuleEngine"]
