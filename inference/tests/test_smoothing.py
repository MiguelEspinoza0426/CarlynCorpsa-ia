from carlyn_inference.rules import PersonObs, RuleEngine
from carlyn_inference.rules.smoothing import CountSmoother, k_of_n
from conftest import rule, zone


def test_k_of_n_needs_the_count_in_three_of_five_frames():
    assert k_of_n([2, 2, 3, 2, 2], 3) == 2          # one frame with 3 people: noise
    assert k_of_n([2, 3, 3, 2, 3], 3) == 3          # three frames with 3: real
    assert k_of_n([1, 0, 1, 1, 0], 3) == 1          # a person missed in two frames is still counted


def test_k_of_n_with_few_frames_requires_all_of_them():
    assert k_of_n([], 3) == 0
    assert k_of_n([2], 3) == 2
    assert k_of_n([2, 1], 3) == 1


def test_smoother_keeps_one_history_per_zone():
    smoother = CountSmoother()
    for count in (1, 1, 1, 4, 1):
        a = smoother.update("a", count)
        b = smoother.update("b", 0)
    assert (a, b) == (1, 0)                          # the single frame with 4 does not reach the zone count


def test_occupancy_rule_fires_on_a_frame_where_nobody_was_detected(camera_factory):
    crowd = zone("z", "Work", rules=(rule("OccupancyExceeded", min_duration=0, cooldown=0),), thresholds={"MaxOccupancy": (None, 2)})
    engine = RuleEngine(camera_factory(crowd))
    people = [PersonObs(i, (0.1, 0.2, 0.1, 0.5), 0.8) for i in (1, 2)]
    for t in range(4):
        engine.update(t * 0.2, people)
    events = engine.update(0.8, [])                  # detector missed everyone once: the smoothed count is still 2
    assert [e.type for e in events] == ["OccupancyExceeded"]
    assert engine.zone_counts() == {"z": 2}
