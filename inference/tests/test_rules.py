from carlyn_inference.geometry import distance_to_polygon, point_in_polygon
from carlyn_inference.rules import ObjectObs, PersonObs, RuleEngine
from conftest import LEFT, RIGHT, rule, zone

# Boxes (x, y, w, h); the foot point is the bottom center.
IN_LEFT = (0.1, 0.2, 0.1, 0.5)        # foot (0.15, 0.7)
IN_RIGHT = (0.7, 0.2, 0.1, 0.5)       # foot (0.75, 0.7)
NEAR_LEFT = (0.5, 0.2, 0.06, 0.5)     # foot (0.53, 0.7): 0.03 from the left zone


def person(box=IN_LEFT, track=1, epp=(), conf=0.8, fallen=False):
    return PersonObs(track, box, conf, frozenset(epp), fallen)


def feed(engine, observations, seconds, start=0.0, step=0.2):
    events, t = [], start
    while t <= start + seconds + 1e-9:
        events += engine.update(t, observations(t) if callable(observations) else observations)
        t = round(t + step, 6)
    return events


# ---- geometry ----

def test_a_person_hidden_briefly_keeps_one_dwell_session(camera_factory):
    def seen(lost_sec):
        engine = RuleEngine(camera_factory(zone("weld", "Work")), track_lost_sec=lost_sec)
        # Visible 0-4 s, hidden behind a column 4-7.4 s, visible again until 9 s, then gone for good.
        feed(engine, lambda t: [person()] if t < 4 or 7.4 <= t < 9 else [], seconds=15)
        return len(engine.take_metrics()[1])
    assert seen(4.0) == 1                     # one dwell session
    assert seen(2.0) == 2                     # the old 2 s patience split it in two


def test_point_in_polygon_and_distance():
    assert point_in_polygon((0.25, 0.5), LEFT)
    assert not point_in_polygon((0.75, 0.5), LEFT)
    assert distance_to_polygon((0.25, 0.5), LEFT) == 0
    assert abs(distance_to_polygon((0.6, 0.5), LEFT) - 0.1) < 1e-9


# ---- EPP ----

def test_missing_epp_is_reported_after_the_window(camera_factory):
    welding = zone("weld", "Work", required=["careta_soldar", "guantes"], rules=[rule("EppMissing", min_duration=3)])
    engine = RuleEngine(camera_factory(welding))

    events = feed(engine, [person(epp=["careta_soldar"])], seconds=4)

    assert len(events) == 1
    event = events[0]
    assert event.type == "EppMissing" and event.zone_id == "weld" and event.track_id == 1
    assert event.details == {"missingEpp": ["guantes"], "presentEpp": ["careta_soldar"]}
    assert event.ended_at - event.started_at == 3


def test_brief_occlusion_does_not_trigger_missing_epp(camera_factory):
    welding = zone("weld", "Work", required=["guantes"], rules=[rule("EppMissing", min_duration=3)])
    engine = RuleEngine(camera_factory(welding))
    # Gloves are hidden in 1 of every 4 frames (25 % < 70 %).
    frames = iter(range(1000))
    events = feed(engine, lambda t: [person(epp=[] if next(frames) % 4 == 0 else ["guantes"])], seconds=6)
    assert events == []


def test_cooldown_limits_repeated_alerts(camera_factory):
    welding = zone("weld", "Work", required=["guantes"], rules=[rule("EppMissing", min_duration=1, cooldown=60)])
    engine = RuleEngine(camera_factory(welding))
    assert len(feed(engine, [person()], seconds=30)) == 1
    assert len(feed(engine, [person()], seconds=40, start=30.2)) == 1  # after 60 s, a new alert


def test_low_confidence_people_are_ignored(camera_factory):
    welding = zone("weld", "Work", required=["guantes"], rules=[rule("EppMissing", min_duration=1, min_conf=0.6)])
    assert feed(RuleEngine(camera_factory(welding)), [person(conf=0.5)], seconds=5) == []


# ---- restricted, danger, proximity ----

def test_restricted_entry_is_reported_once_per_stay(camera_factory):
    panel = zone("panel", "Restricted", rules=[rule("RestrictedZoneEntry", min_duration=1, cooldown=0)])
    engine = RuleEngine(camera_factory(panel))
    events = feed(engine, [person()], seconds=10)
    assert [e.type for e in events] == ["RestrictedZoneEntry"]

    feed(engine, [person(box=IN_RIGHT)], seconds=1, start=10.2)   # leaves
    again = feed(engine, [person()], seconds=2, start=11.4)        # comes back
    assert len(again) == 1


def test_danger_zone_dwell_uses_the_max_dwell_threshold(camera_factory):
    grinder = zone("grinder", "Danger", rules=[rule("DangerZoneDwell", min_duration=5)], thresholds={"MaxDwellSeconds": (None, 20)})
    engine = RuleEngine(camera_factory(grinder))
    assert feed(engine, [person()], seconds=19) == []
    events = feed(engine, [person()], seconds=2, start=19.2)
    assert len(events) == 1 and events[0].details["durationSec"] >= 20


def test_proximity_to_a_danger_zone(camera_factory):
    grinder = zone("grinder", "Danger", rules=[rule("Proximity", min_duration=1)], thresholds={"ProximityDistance": (None, 0.05)})
    engine = RuleEngine(camera_factory(grinder))
    events = feed(engine, [person(box=NEAR_LEFT)], seconds=2)
    assert len(events) == 1 and events[0].type == "Proximity" and events[0].details["distance"] == 0.03
    assert feed(RuleEngine(camera_factory(grinder)), [person(box=IN_RIGHT)], seconds=3) == []


# ---- occupancy ----

def test_occupancy_exceeded_and_samples(camera_factory):
    welding = zone("weld", "Work", rules=[rule("OccupancyExceeded", min_duration=2)], thresholds={"MaxOccupancy": (2, 3)})
    engine = RuleEngine(camera_factory(welding))
    crowd = [person(track=i) for i in range(1, 4)]

    events = feed(engine, crowd, seconds=11)

    assert len(events) == 1
    assert events[0].person_count == 3 and events[0].details == {"count": 3, "threshold": 3}
    occupancy, _, _ = engine.take_metrics()
    assert occupancy and occupancy[0].max_count == 3 and occupancy[0].window_sec == 10


# ---- dwell, transitions and inactivity ----

def test_dwell_session_and_transition_between_zones(camera_factory):
    left, right = zone("a", "Work", LEFT), zone("b", "Work", RIGHT)
    engine = RuleEngine(camera_factory(left, right))
    feed(engine, [person(box=IN_LEFT)], seconds=5)
    feed(engine, [person(box=IN_RIGHT)], seconds=5, start=5.2)
    feed(engine, [], seconds=3, start=10.4)   # track lost

    _, dwell, _ = engine.take_metrics()
    assert [(d.zone_id, d.from_zone_id) for d in dwell] == [("a", None), ("b", "a")]
    assert all(d.track_id == 1 for d in dwell)


def test_inactivity_event_and_interval(camera_factory):
    welding = zone("weld", "Work", rules=[rule("Inactivity", min_duration=300, cooldown=0)], thresholds={"InactivitySeconds": (None, 60)})
    engine = RuleEngine(camera_factory(welding))

    events = feed(engine, [person()], seconds=70, step=1)
    assert [e.type for e in events] == ["Inactivity"]

    feed(engine, [person(box=(0.2, 0.2, 0.1, 0.5))], seconds=1, start=71, step=1)   # moves
    _, _, inactivity = engine.take_metrics()
    assert len(inactivity) == 1 and inactivity[0].ended_at - inactivity[0].started_at >= 60


def test_small_jitter_is_not_movement(camera_factory):
    welding = zone("weld", "Work", rules=[rule("Inactivity", cooldown=0)], thresholds={"InactivitySeconds": (None, 30)})
    engine = RuleEngine(camera_factory(welding))
    jitter = lambda t: [person(box=(0.1 + (0.005 if int(t) % 2 else 0), 0.2, 0.1, 0.5))]
    assert len(feed(engine, jitter, seconds=35, step=1)) == 1


# ---- possible accident ----

def test_fall_is_reported_after_minimum_time_and_needs_validation_rule(camera_factory):
    welding = zone("weld", "Work", rules=[rule("PossibleAccident", min_duration=2)])
    engine = RuleEngine(camera_factory(welding))
    assert feed(engine, [person(fallen=True)], seconds=1.6) == []
    events = feed(engine, [person(fallen=True)], seconds=1, start=1.8)
    assert len(events) == 1
    assert events[0].type == "PossibleAccident" and events[0].ended_at is None and events[0].details["kind"] == "PersonOnGround"


def test_fall_outside_zones_with_the_rule_is_ignored(camera_factory):
    hallway = zone("hall", "Circulation", RIGHT, rules=[rule("PossibleAccident")])
    assert feed(RuleEngine(camera_factory(hallway)), [person(fallen=True)], seconds=5) == []


# ---- unsafe conditions ----

def test_static_object_in_circulation_zone(camera_factory):
    hallway = zone("hall", "Circulation", rules=[rule("UnsafeCondition", min_duration=30)])
    engine = RuleEngine(camera_factory(hallway))
    box = ObjectObs(7, "suitcase", (0.1, 0.5, 0.1, 0.1), 0.7)
    assert feed(engine, [], seconds=1) == []
    events = []
    t = 0.0
    while t <= 31:
        events += engine.update(t, [], [box])
        t += 1
    assert len(events) == 1 and events[0].details["objectClass"] == "suitcase"
