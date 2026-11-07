"""The node's logs are sent to the API in batches, without loops and without losing them while the API is down."""
import logging

from carlyn_inference.remote_logs import MAX_BATCH, ApiLogHandler, level_name


def make_logger(handler: ApiLogHandler, name: str = "carlyn_inference.worker") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers, logger.propagate, logger.level = [handler], False, logging.DEBUG
    return logger


def test_records_are_sent_in_batches_with_the_api_format():
    sent: list[list[dict]] = []
    handler = ApiLogHandler(sent.append)
    log = make_logger(handler)

    log.debug("ignored: below the handler level")
    log.warning("Camera %s reconnecting", "Nave 1")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("Camera failed")
    assert handler.flush()

    (batch,) = sent
    assert [(e["level"], e["logger"], e["message"]) for e in batch] == [
        ("Warning", "carlyn_inference.worker", "Camera Nave 1 reconnecting"),
        ("Error", "carlyn_inference.worker", "Camera failed"),
    ]
    assert batch[0]["exception"] is None and "RuntimeError: boom" in batch[1]["exception"]
    assert batch[0]["timestamp"].endswith("Z")


def test_http_client_logs_are_not_forwarded():
    sent: list[list[dict]] = []
    handler = ApiLogHandler(sent.append)
    make_logger(handler, "httpx").warning("POST /api/ingest/logs failed")
    assert handler.flush() and sent == []


def test_records_are_kept_while_the_api_is_down_and_sent_later():
    attempts: list[int] = []

    def send(batch: list[dict]) -> None:
        attempts.append(len(batch))
        if len(attempts) == 1:
            raise ConnectionError("API down")

    handler = ApiLogHandler(send)
    log = make_logger(handler)
    for i in range(MAX_BATCH + 10):
        log.info("record %d", i)

    assert not handler.flush()          # first batch failed: everything stays queued
    assert handler.flush()              # API back: two batches (500 + 10)
    assert attempts == [MAX_BATCH, MAX_BATCH, 10]


def test_python_levels_map_to_the_api_levels():
    assert [level_name(n) for n in (5, 10, 20, 25, 30, 40, 50)] == ["Debug", "Debug", "Information", "Information", "Warning", "Error", "Critical"]
