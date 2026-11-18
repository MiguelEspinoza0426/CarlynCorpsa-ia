"""Single-GPU arbiter: inferences of all cameras run one at a time, and the GPU is shared in proportion to targetFps.

With a plain lock, whoever asks most often wins: a camera at 20 FPS could leave cameras at 2 FPS waiting. Here each
camera has a virtual clock that advances 1/targetFps per turn, and the free GPU goes to the waiting request whose clock
is furthest behind (weighted fair queuing). While the GPU is not saturated every camera simply gets its rate.
"""
import itertools
import threading
from contextlib import contextmanager


class GpuScheduler:
    def __init__(self):
        self._cond = threading.Condition()
        self._busy = False
        self._clock: dict[str, float] = {}                  # virtual finish time per camera
        self._waiting: dict[int, tuple[float, int]] = {}    # ticket -> (virtual start, ticket): order of service
        self._tickets = itertools.count()
        self._now = 0.0                                      # virtual time of the last turn granted

    @contextmanager
    def turn(self, key: str, fps: float):
        """Holds the GPU for one inference of camera `key`, whose target rate is `fps`."""
        with self._cond:
            ticket = next(self._tickets)
            # A camera that was idle does not accumulate credit: it starts from the current virtual time.
            start = max(self._clock.get(key, 0.0), self._now)
            self._clock[key] = start + 1.0 / max(fps, 0.1)
            self._waiting[ticket] = (start, ticket)
            while self._busy or min(self._waiting.values())[1] != ticket:
                self._cond.wait()
            del self._waiting[ticket]
            self._busy, self._now = True, start
        try:
            yield
        finally:
            with self._cond:
                self._busy = False
                self._cond.notify_all()

    def forget(self, key: str) -> None:
        """A removed camera."""
        with self._cond:
            self._clock.pop(key, None)


GPU = GpuScheduler()
