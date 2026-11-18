"""Smoothed people count per zone: a count holds only if it was reached in at least K of the last N frames.

A person that flickers in and out of detection (backlight, partial occlusion) neither inflates nor drops the count for
a single frame, so occupancy metrics and the occupancy rule do not react to one-frame noise.
"""
from collections import deque

SMOOTH_K = 3
SMOOTH_N = 5


def k_of_n(history: list[int], k: int) -> int:
    """Largest count reached in at least k of the frames in history (all of them while there are fewer than k)."""
    if not history:
        return 0
    ordered = sorted(history, reverse=True)
    return ordered[min(k, len(ordered)) - 1]


class CountSmoother:
    def __init__(self, k: int = SMOOTH_K, n: int = SMOOTH_N):
        self.k, self.n = k, n
        self._history: dict[str, deque[int]] = {}

    def update(self, zone_id: str, count: int) -> int:
        history = self._history.setdefault(zone_id, deque(maxlen=self.n))
        history.append(count)
        return k_of_n(list(history), self.k)
