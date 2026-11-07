"""Sends this node's log records to the API (POST /api/ingest/logs), where administrators read them in the portal."""
import logging
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone

MAX_BATCH = 500     # API limit per request
MAX_QUEUE = 5000    # older records are dropped while the API is unreachable

LEVELS = {logging.DEBUG: "Debug", logging.INFO: "Information", logging.WARNING: "Warning", logging.ERROR: "Error", logging.CRITICAL: "Critical"}
# HTTP client libraries and this module log about the sending itself: forwarding them would loop.
SKIPPED_LOGGERS = ("httpx", "httpcore", "urllib3", __name__)
_FORMATTER = logging.Formatter()


def level_name(levelno: int) -> str:
    return next((name for value, name in sorted(LEVELS.items(), reverse=True) if levelno >= value), "Debug")


class ApiLogHandler(logging.Handler):
    """Queues records without blocking the caller; a background thread sends them in batches every few seconds."""

    def __init__(self, send: Callable[[list[dict]], None], level: int = logging.INFO, interval: float = 5.0):
        super().__init__(level)
        self._send = send
        self._interval = interval
        self._queue: deque[dict] = deque(maxlen=MAX_QUEUE)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="remote-logs")

    def start(self) -> "ApiLogHandler":
        self._thread.start()
        return self

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(SKIPPED_LOGGERS):
            return
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat().replace("+00:00", "Z"),
                "level": level_name(record.levelno),
                "logger": record.name,
                "message": record.getMessage(),
                "exception": _FORMATTER.formatException(record.exc_info) if record.exc_info else None,
            }
        except Exception:  # noqa: BLE001 - a bad log call must never break the caller
            self.handleError(record)
            return
        with self._lock:
            self._queue.append(entry)

    def flush(self) -> bool:
        """Sends what is queued. Returns False (and keeps the records) if the API did not accept them."""
        while True:
            with self._lock:
                batch = [self._queue.popleft() for _ in range(min(MAX_BATCH, len(self._queue)))]
            if not batch:
                return True
            try:
                self._send(batch)
            except Exception:  # noqa: BLE001 - retried on the next round
                with self._lock:
                    self._queue.extendleft(reversed(batch))
                return False

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3)
        self.flush()
        super().close()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.flush()
