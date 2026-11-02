"""Rules table, keyed by event type. Adding a rule = one new class + one entry here; the engine does not change.

Order only decides the order of the events within a frame.
"""
from .base import RuleStrategy
from .dwell import DangerZoneDwellRule
from .epp import EppMissingRule
from .fall import PossibleAccidentRule
from .inactivity import InactivityRule
from .models import EventType
from .occupancy import OccupancyExceededRule
from .proximity import ProximityRule
from .restricted import RestrictedZoneEntryRule
from .unsafe import UnsafeConditionRule


def default_rules() -> dict[EventType, RuleStrategy]:
    rules = (EppMissingRule(), RestrictedZoneEntryRule(), DangerZoneDwellRule(), InactivityRule(), ProximityRule(),
             PossibleAccidentRule(), OccupancyExceededRule(), UnsafeConditionRule())
    return {r.event_type: r for r in rules}
