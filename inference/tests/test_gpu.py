import threading
import time
from collections import Counter

from carlyn_inference.gpu import GpuScheduler


def _saturate(rates: dict[str, float], seconds: float = 1.0, work: float = 0.004) -> Counter:
    """Each camera asks for the GPU in a loop (as fast as it can): the GPU is the bottleneck."""
    scheduler, turns, stop = GpuScheduler(), Counter(), threading.Event()

    def camera(name, fps):
        while not stop.is_set():
            with scheduler.turn(name, fps):
                turns[name] += 1
                time.sleep(work)

    threads = [threading.Thread(target=camera, args=item, daemon=True) for item in rates.items()]
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join()
    return turns


def test_a_saturated_gpu_is_shared_in_proportion_to_target_fps():
    turns = _saturate({"Camara 1": 20, "c": 2, "CAM 3": 2})
    total = sum(turns.values())
    assert 0.75 <= turns["Camara 1"] / total <= 0.92          # 20 / 24 = 0.83
    assert turns["c"] >= 0.05 * total and turns["CAM 3"] >= 0.05 * total   # 2 / 24 = 0.08 each: not starved


def test_equal_rates_get_equal_turns():
    turns = _saturate({"a": 5, "b": 5})
    assert abs(turns["a"] - turns["b"]) <= max(3, 0.1 * sum(turns.values()))


def test_turns_are_exclusive():
    scheduler, inside, overlaps = GpuScheduler(), [0], [0]

    def run(name):
        for _ in range(50):
            with scheduler.turn(name, 5):
                inside[0] += 1
                overlaps[0] += inside[0] > 1
                time.sleep(0.0005)
                inside[0] -= 1

    threads = [threading.Thread(target=run, args=(n,)) for n in "abc"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlaps[0] == 0
