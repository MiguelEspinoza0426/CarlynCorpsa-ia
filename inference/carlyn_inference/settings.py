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
    # two_stage: persons with PERSON_MODEL, then EPP on each person's crop at full resolution; single: MODEL_PATH alone.
    detection_mode: str = "two_stage"
    person_model: str = "yolo11s.pt"
    # Low on purpose: ByteTrack only starts tracks at >= 0.25; weaker detections just keep existing tracks alive.
    person_conf: float = 0.2
    epp_imgsz: int = 640
    epp_crop_margin: float = 0.15
    # Persons shorter than this (source pixels) are counted and tracked, but their EPP is not judged.
    min_person_height_px: int = 150
    # Seconds a person who stops being detected (column, truck, backlight) keeps its track id.
    track_buffer_sec: float = 4.0
    # Seconds of black/uniform/logo frames before a camera is reported without signal (frozen image: 30 s).
    no_signal_sec: float = 5.0
    # Video decoding of RTSP/file sources: auto (GPU decoder if available, else CPU), off, d3d11, vaapi, mfx.
    decode_hw: str = "auto"
    # Heartbeat status for a camera without signal. "Offline" until the API knows "NoSignal" (step 6).
    no_signal_status: str = "Offline"
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
    stream_fps: int = 30
    jpeg_quality: int = 75
    blur_stream: bool = True
    source_overrides: dict[str, str] = field(default_factory=dict)
    # Camera id or name -> `inference` block (slice, preprocess), to try them before the API sends them.
    inference_overrides: dict[str, dict] = field(default_factory=dict)
    # Lowest level sent to the API for the administrators' log page; OFF disables it.
    remote_log_level: str = "INFO"

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
            detection_mode=env("DETECTION_MODE", defaults.detection_mode).strip().lower(),
            person_model=env("PERSON_MODEL", defaults.person_model),
            person_conf=float(env("PERSON_CONF", defaults.person_conf)),
            epp_imgsz=int(env("EPP_IMGSZ", defaults.epp_imgsz)),
            epp_crop_margin=float(env("EPP_CROP_MARGIN", defaults.epp_crop_margin)),
            min_person_height_px=int(env("MIN_PERSON_HEIGHT_PX", defaults.min_person_height_px)),
            track_buffer_sec=float(env("TRACK_BUFFER_SEC", defaults.track_buffer_sec)),
            no_signal_sec=float(env("NO_SIGNAL_SEC", defaults.no_signal_sec)),
            decode_hw=env("DECODE_HW", defaults.decode_hw).strip().lower(),
            no_signal_status=env("NO_SIGNAL_STATUS", defaults.no_signal_status),
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
            stream_fps=int(env("STREAM_FPS", defaults.stream_fps)),
            jpeg_quality=int(env("JPEG_QUALITY", defaults.jpeg_quality)),
            blur_stream=_bool("BLUR_STREAM", defaults.blur_stream),
            # JSON map camera id or name -> source ("0", video file, rtsp://...), for tests without the DVR.
            source_overrides=json.loads(env("SOURCE_OVERRIDES", "{}")),
            inference_overrides=json.loads(env("INFERENCE_OVERRIDES", "{}")),
            remote_log_level=env("REMOTE_LOG_LEVEL", defaults.remote_log_level).strip().upper(),
        )
