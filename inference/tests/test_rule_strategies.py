"""Each rule class in isolation (without the engine) and the rules table."""
from carlyn_inference.rules import EventType, PersonObs, RuleEngine, ZoneRule, default_rules
from carlyn_inference.rules.dwell import DangerZoneDwellRule
from carlyn_inference.rules.epp import EppMissingRule
from carlyn_inference.rules.fall import PossibleAccidentRule
from carlyn_inference.rules.inactivity import InactivityRule
from carlyn_inference.rules.models import EventOut, ObjectObs
from carlyn_inference.rules.occupancy import OccupancyExceededRule
from carlyn_inference.rules.proximity import ProximityRule
from carlyn_inference.rules.restricted import RestrictedZoneEntryRule
from carlyn_inference.rules.state import EngineState, Presence, Track
from carlyn_inference.rules.unsafe import UnsafeConditionRule
from conftest import rule, zone

FOOT = (0.15, 0.7)
BOX = (0.1, 0.2, 0.1, 0.5)


def inside(zone_id: str, entered_at: float = 0.0) -> Track:
    return Track(last_seen=0.0, zones={zone_id: Presence(entered_at=entered_at, from_zone_id=None)})


def person(epp=(), fallen=False, conf=0.8):
    return PersonObs(1, BOX, conf, frozenset(epp), fallen)


def run(rule_obj, state, z, track, seconds, step=0.5, **kw):
    events, t = [], 0.0
    while t <= seconds:
        events += rule_obj.evaluate(state, z, kw.get("p", person()), track, kw.get("foot", FOOT), 1, t)
        t += step
    return events


def test_epp_rule_ignores_persons_whose_epp_was_not_judged(camera_factory):
    z = zone("z", "Work", required=("guantes",), rules=(rule("EppMissing", min_duration=1, min_conf=0.1),))
    small = PersonObs(1, BOX, 0.9, frozenset(), epp_evaluated=False)
    assert run(EppMissingRule(), EngineState(camera_factory(z)), z, inside("z"), 5, p=small) == []
    assert run(EppMissingRule(), EngineState(camera_factory(z)), z, inside("z"), 5, p=person()) != []   # same person, judged: fires


def test_rules_table_is_keyed_by_event_type():
    table = default_rules()
    assert set(table) == set(EventType)
    assert all(key == r.event_type for key, r in table.items())


def test_epp_rule(camera_factory):
    z = zone("weld", "Work", required=["guantes"], rules=[rule("EppMissing", min_duration=2)])
    events = run(EppMissingRule(), EngineState(camera_factory(z)), z, inside("weld"), seconds=3)
    assert len(events) == 1 and events[0].details["missingEpp"] == ["guantes"]


def test_restricted_entry_rule_only_applies_inside(camera_factory):
    z = zone("panel", "Restricted", rules=[rule("RestrictedZoneEntry", min_duration=1)])
    state = EngineState(camera_factory(z))
    assert run(RestrictedZoneEntryRule(), state, z, Track(last_seen=0.0), seconds=3) == []
    assert len(run(RestrictedZoneEntryRule(), state, z, inside("panel"), seconds=3)) == 1


def test_danger_dwell_rule(camera_factory):
    z = zone("grinder", "Danger", rules=[rule("DangerZoneDwell")], thresholds={"MaxDwellSeconds": (None, 2)})
    events = run(DangerZoneDwellRule(), EngineState(camera_factory(z)), z, inside("grinder"), seconds=4)
    assert len(events) == 1 and events[0].started_at == 0.0


def test_inactivity_rule_records_interval_when_moving(camera_factory):
    z = zone("weld", "Work", rules=[rule("Inactivity", cooldown=0)], thresholds={"InactivitySeconds": (None, 2)})
    state, track, r = EngineState(camera_factory(z)), inside("weld"), InactivityRule()
    assert len(run(r, state, z, track, seconds=3)) == 1
    r.evaluate(state, z, person(), track, (0.4, 0.7), 1, 4.0)
    assert len(state.inactivity) == 1


def test_proximity_rule_only_applies_outside(camera_factory):
    z = zone("grinder", "Danger", rules=[rule("Proximity", min_duration=1)])
    state = EngineState(camera_factory(z))
    near = (0.53, 0.7)
    assert run(ProximityRule(), state, z, inside("grinder"), seconds=2, foot=near) == []
    assert len(run(ProximityRule(), state, z, Track(last_seen=0.0), seconds=2, foot=near)) == 1


def test_possible_accident_rule(camera_factory):
    z = zone("weld", "Work", rules=[rule("PossibleAccident")])
    state, track, r = EngineState(camera_factory(z)), inside("weld"), PossibleAccidentRule()
    assert r.evaluate(state, person(fallen=True), track, {"weld": 1}, 1.0) == []
    assert len(r.evaluate(state, person(fallen=True), track, {"weld": 1}, 3.0)) == 1


def test_occupancy_rule(camera_factory):
    z = zone("weld", "Work", rules=[rule("OccupancyExceeded", min_duration=1)], thresholds={"MaxOccupancy": (None, 2)})
    state = EngineState(camera_factory(z))
    crowd = [PersonObs(i, BOX, 0.6) for i in (1, 2)]
    for p in crowd:
        state.tracks[p.track_id] = inside("weld")
    r = OccupancyExceededRule()
    assert r.evaluate(state, z, 2, crowd, 1.0) == []
    events = r.evaluate(state, z, 2, crowd, 2.0)
    assert len(events) == 1 and events[0].confidence == 0.6


def test_unsafe_condition_rule(camera_factory):
    z = zone("hall", "Circulation", rules=[rule("UnsafeCondition", min_duration=2)])
    state, r = EngineState(camera_factory(z)), UnsafeConditionRule()
    obj = ObjectObs(7, "box", (0.1, 0.5, 0.1, 0.1), 0.7)
    assert r.evaluate(state, [obj], 0.0) == []
    assert len(r.evaluate(state, [obj], 2.0)) == 1


def test_a_new_rule_is_added_without_changing_the_engine(camera_factory):
    class CrowdedZone(ZoneRule):
        event_type = "CrowdedZone"

        def evaluate(self, state, z, count, persons, now):
            return [EventOut(self.event_type, z.id, None, now, now, 1.0, count, None, None, {})] if count >= 2 else []

    engine = RuleEngine(camera_factory(zone("weld", "Work")), {**default_rules(), "CrowdedZone": CrowdedZone()})
    events = engine.update(0.0, [PersonObs(1, BOX, 0.8), PersonObs(2, BOX, 0.8)])
    assert [e.type for e in events] == ["CrowdedZone"]
