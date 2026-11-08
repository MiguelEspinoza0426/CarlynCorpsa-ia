"""Entry point: python -m carlyn_inference (run from the inference/ folder)."""
import logging

import uvicorn

from .remote_logs import ApiLogHandler
from .server import create_app
from .service import InferenceService
from .settings import Settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    if not settings.ingest_api_key:
        raise SystemExit("Configure INGEST_API_KEY (clave del nodo registrada en la API, Ingest:Nodes).")
    service = InferenceService(settings)
    remote_logs = None
    if settings.remote_log_level not in ("", "OFF", "NONE"):
        # The node's logs also go to the API, where administrators read them (Seguridad > Logs del sistema).
        remote_logs = ApiLogHandler(service.api.post_logs, logging.getLevelName(settings.remote_log_level)).start()
        logging.getLogger().addHandler(remote_logs)
    service.start()
    try:
        # log_config=None: uvicorn's own logging setup (dictConfig) would close the handlers above, remote_logs included.
        uvicorn.run(create_app(service, settings.internal_key), host=settings.host, port=settings.port, log_level="warning",
                    log_config=None)
    finally:
        service.stop()
        if remote_logs is not None:
            logging.getLogger().removeHandler(remote_logs)
            remote_logs.close()


if __name__ == "__main__":
    main()
