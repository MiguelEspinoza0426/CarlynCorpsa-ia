"""HTTP endpoints used by the .NET API: health, snapshot, MJPEG stream and configuration reload."""
import asyncio
import hmac

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from .service import InferenceService

BOUNDARY = "frame"


def create_app(service: InferenceService, internal_key: str) -> FastAPI:
    app = FastAPI(title="CARLYNCORPSA - servicio de inferencia", version="1.0")

    def require_key(x_internal_key: str = Header(default="")) -> None:
        if internal_key and not hmac.compare_digest(x_internal_key, internal_key):
            raise HTTPException(status_code=401, detail="X-Internal-Key inválida")

    def worker_or_404(camera_id: str):
        worker = service.worker(camera_id)
        if worker is None:
            raise HTTPException(status_code=404, detail="La cámara no está activa en este nodo")
        return worker

    @app.get("/health")
    def health():
        statuses = service.camera_statuses()
        return {
            "status": "ok" if service.config is not None else "starting",
            "nodeId": service.config.node_id if service.config else None,
            "modelVersion": service.settings.model_version,
            "configError": service.last_config_error,
            "pendingUploads": service.outbox.pending(),
            "cameras": {s["cameraId"]: s["status"] for s in statuses},
        }

    @app.get("/cameras/{camera_id}/snapshot.jpg", dependencies=[Depends(require_key)])
    def snapshot(camera_id: str):
        jpeg = worker_or_404(camera_id).snapshot_jpeg()
        if jpeg is None:
            raise HTTPException(status_code=404, detail="Aún no hay imagen de la cámara")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/cameras/{camera_id}/stream.mjpeg", dependencies=[Depends(require_key)])
    async def stream(camera_id: str):
        worker = worker_or_404(camera_id)

        async def frames():
            last = None
            while True:
                jpeg = worker.latest_jpeg
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + \
                        str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(1 / max(1, worker.camera.target_fps))

        return StreamingResponse(frames(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                                 headers={"Cache-Control": "no-store"})

    @app.post("/config/reload", status_code=202, dependencies=[Depends(require_key)])
    def reload():
        service.request_reload()
        return {"reloading": True}

    return app
