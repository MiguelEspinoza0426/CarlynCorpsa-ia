"""Rule interfaces. Each rule type is one class; the engine only knows these four scopes.

- PersonZoneRule: once per tracked person and zone (the person may be inside or outside the zone).
- PersonRule: once per tracked person.
- ZoneRule: once per zone, with the number of people inside.
- FrameRule: once per frame, with the detected objects.
"""
from abc import ABC, abstractmethod
from typing import ClassVar

from ..config import Zone
from ..geometry import Point
from .models import EventOut, EventType, ObjectObs, PersonObs
from .state import EngineState, Track


class PersonZoneRule(ABC):
    event_type: ClassVar[EventType]

    @abstractmethod
    def evaluate(self, state: EngineState, zone: Zone, person: PersonObs, track: Track, foot: Point,
                 count: int, now: float) -> list[EventOut]:
        """`count` is the number of people inside the zone; `track.zones` tells whether this person is inside."""


class PersonRule(ABC):
    event_type: ClassVar[EventType]

    @abstractmethod
    def evaluate(self, state: EngineState, person: PersonObs, track: Track, counts: dict[str, int], now: float) -> list[EventOut]:
        ...


class ZoneRule(ABC):
    event_type: ClassVar[EventType]

    @abstractmethod
    def evaluate(self, state: EngineState, zone: Zone, count: int, persons: list[PersonObs], now: float) -> list[EventOut]:
        ...


class FrameRule(ABC):
    event_type: ClassVar[EventType]

    @abstractmethod
    def evaluate(self, state: EngineState, objects: list[ObjectObs], now: float) -> list[EventOut]:
        ...


RuleStrategy = PersonZoneRule | PersonRule | ZoneRule | FrameRule
