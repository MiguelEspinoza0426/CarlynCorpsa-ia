"""Entry point: python -m carlyn_inference (run from the inference/ folder)."""
import logging

import uvicorn

from .server import create_app
from .service import InferenceService
from .settings import Settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    if not settings.ingest_api_key:
        raise SystemExit("Configure INGEST_API_KEY (clave del nodo registrada en la API, Ingest:Nodes).")
    service = InferenceService(settings)
    service.start()
    try:
        uvicorn.run(create_app(service, settings.internal_key), host=settings.host, port=settings.port, log_level="warning")
    finally:
        service.stop()


if __name__ == "__main__":
    main()
