"""Service settings from environment variables (see inference/README.md)."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "si", "sí")


@dataclass(frozen=True)
class Settings:
    api_base_url: str = "http://localhost:5000"
    ingest_api_key: str = ""
    internal_key: str = ""
    model_path: str = str(REPO / "ml/runs/epp_v3_completo/weights/best.pt")
    pose_model: str = "yolo11n-pose.pt"
    object_model: str = ""
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.25
    host: str = "0.0.0.0"
    port: int = 8100
    config_poll_sec: int = 30
    heartbeat_sec: int = 15
    metrics_flush_sec: int = 10
    stream_max_width: int = 960
    jpeg_quality: int = 75
    blur_stream: bool = True
    source_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def model_version(self) -> str:
        """'epp_v3_completo' for ml/runs/epp_v3_completo/weights/best.pt, else the file name."""
        path = Path(self.model_path)
        return path.parent.parent.name if path.parent.name == "weights" else path.stem

    @property
    def torch_device(self):
        if self.device != "auto":
            return int(self.device) if self.device.isdigit() else self.device
        import torch
        return 0 if torch.cuda.is_available() else "cpu"

    @staticmethod
    def from_env() -> "Settings":
        env = os.environ.get
        defaults = Settings()
        return Settings(
            api_base_url=env("API_BASE_URL", defaults.api_base_url).rstrip("/"),
            ingest_api_key=env("INGEST_API_KEY", ""),
            internal_key=env("INTERNAL_KEY", ""),
            model_path=env("MODEL_PATH", defaults.model_path),
            pose_model=env("POSE_MODEL", defaults.pose_model),
            object_model=env("OBJECT_MODEL", defaults.object_model),
            device=env("DEVICE", defaults.device),
            imgsz=int(env("IMGSZ", defaults.imgsz)),
            conf=float(env("CONF", defaults.conf)),
            host=env("HOST", defaults.host),
            port=int(env("PORT", defaults.port)),
            config_poll_sec=int(env("CONFIG_POLL_SEC", defaults.config_poll_sec)),
            heartbeat_sec=int(env("HEARTBEAT_SEC", defaults.heartbeat_sec)),
            metrics_flush_sec=int(env("METRICS_FLUSH_SEC", defaults.metrics_flush_sec)),
            stream_max_width=int(env("STREAM_MAX_WIDTH", defaults.stream_max_width)),
            jpeg_quality=int(env("JPEG_QUALITY", defaults.jpeg_quality)),
            blur_stream=_bool("BLUR_STREAM", defaults.blur_stream),
            # JSON map camera id or name -> source ("0", video file, rtsp://...), for tests without the DVR.
            source_overrides=json.loads(env("SOURCE_OVERRIDES", "{}")),
        )
