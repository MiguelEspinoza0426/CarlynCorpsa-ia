"""Pipeline benchmark (HU-35): inference time per stage and FPS per camera, saved to docs/benchmarks for comparison.

Two modes, both using the same Detector and CameraWorker as the service:

    # Offline: a video or a folder of images, frame by frame (inference time per stage, persons by size)
    python -m carlyn_inference.benchmark offline --source C:/ruta/video.mp4 --label linea-base

    # Live: the node's real cameras (configuration from the API, credentials from the environment) for N seconds.
    # Stop the inference service first: both would compete for the GPU and the DVR. Nothing is sent to the API.
    python -m carlyn_inference.benchmark live --seconds 120 --label linea-base

Each run writes <fecha>_<label>_<equipo>.json (raw data) and .md (summary) into docs/benchmarks/.
"""
import argparse
import json
import platform
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .settings import REPO, Settings
from .config import InferenceConfig

OUT_DIR = REPO / "docs" / "benchmarks"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
# Person heights in source pixels: below ~60 px the model barely sees EPP; above 150 px EPP is reliable.
SIZE_BUCKETS = ((0, 60, "<60 px"), (60, 150, "60–150 px"), (150, 10**6, "≥150 px"))


# ---------- pure helpers (tested) ----------

def stats(values: list[float]) -> dict | None:
    """Mean, p50, p95 and max of a list of durations in ms."""
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]  # noqa: E731
    return {"n": len(values), "mean": round(statistics.mean(values), 1), "p50": round(pick(0.5), 1),
            "p95": round(pick(0.95), 1), "max": round(ordered[-1], 1)}


def size_bucket(height_px: float) -> str:
    return next(label for low, high, label in SIZE_BUCKETS if low <= height_px < high)


def report_name(label: str, host: str, when: datetime) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in f"{label}_{host}").strip("-").lower()
    return f"{when:%Y-%m-%d_%H%M}_{safe}"


def markdown(report: dict) -> str:
    env, cfg = report["environment"], report["config"]
    lines = [
        f"# Benchmark «{report['label']}» ({report['mode']})",
        "",
        f"- Fecha: {report['date']}",
        f"- Equipo: {env['host']} · GPU: {env['gpu']} · {env['os']}",
        f"- Versiones: Python {env['python']} · torch {env['torch']} · ultralytics {env['ultralytics']} · commit {env['commit']}",
        f"- Modelo: `{cfg['model']}` · imgsz {cfg['imgsz']} · conf {cfg['conf']} · pose `{cfg['pose_model'] or '—'}`",
    ]
    if cfg.get("detection_mode") == "two_stage":
        lines.append(f"- Dos etapas: personas con `{cfg['person_model']}` (conf {cfg['person_conf']}) · EPP en recortes a "
                     f"imgsz {cfg['epp_imgsz']} con margen {cfg['epp_crop_margin']:.0%}")
    if report.get("source"):
        lines.append(f"- Fuente: {report['source']}")
    if report.get("notes"):
        lines.append(f"- Notas: {report['notes']}")
    lines += ["", "## Por cámara / fuente", "",
              "| Fuente | Resolución | FPS objetivo | FPS logrados | Detección + tracking (media / p95 ms) | Pose (media ms) | Personas por cuadro | Personas por altura | Personas con algún EPP | EPP por clase |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, cam in report["cameras"].items():
        det, pose = cam.get("detect_ms") or {}, cam.get("pose_ms") or {}
        sizes = ", ".join(f"{k}: {v}" for k, v in cam.get("person_heights", {}).items()) or "—"
        epp = ", ".join(f"{k}: {v}" for k, v in cam.get("epp_by_class", {}).items()) or "—"
        lines.append(f"| {name} | {cam.get('resolution') or '—'} | {cam.get('target_fps') or '—'} | {cam.get('achieved_fps') or '—'} | "
                     f"{det.get('mean', '—')} / {det.get('p95', '—')} | {pose.get('mean', '—')} | {cam.get('persons_per_frame', '—')} | {sizes} | "
                     f"{cam.get('persons_with_epp', '—')} | {epp} |")
    sliced = {name: cam["slicing"] for name, cam in report["cameras"].items() if cam.get("slicing")}
    if sliced:
        lines += ["", "SAHI: " + "; ".join(f"{name}: mosaicos de {s['tile_size']} px, solape {s['overlap']:.0%}, ROI {s['roi']}"
                                            for name, s in sliced.items())]
    corrected = {name: cam["preprocess"] for name, cam in report["cameras"].items() if cam.get("preprocess")}
    if corrected:
        lines += ["", "Corrección de contraste: " + "; ".join(f"{name}: {mode}" for name, mode in corrected.items())]
    total = report.get("totals", {})
    if total:
        lines += ["", "## Totales", ""]
        lines += [f"- {key}: {value}" for key, value in total.items()]
    return "\n".join(lines) + "\n"


def save(report: dict, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / report_name(report["label"], report["environment"]["host"], datetime.fromisoformat(report["date"]))
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    base.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    return base.with_suffix(".md")


# ---------- environment ----------

def environment() -> dict:
    import torch
    import ultralytics
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except OSError:
        commit = "?"
    return {"host": platform.node(), "os": platform.platform(terse=True), "python": platform.python_version(),
            "torch": torch.__version__, "ultralytics": ultralytics.__version__, "commit": commit or "?",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sin GPU"}


def run_config(settings: Settings) -> dict:
    return {"model": settings.model_version, "model_path": settings.model_path, "imgsz": settings.imgsz,
            "conf": settings.conf, "pose_model": settings.pose_model, "device": str(settings.torch_device),
            "detection_mode": settings.detection_mode, "person_model": settings.person_model, "person_conf": settings.person_conf,
            "epp_imgsz": settings.epp_imgsz, "epp_crop_margin": settings.epp_crop_margin}


class _Timings:
    """Wraps Detector methods to record their duration per camera thread, without changing the pipeline."""

    def __init__(self):
        self.detect, self.pose = defaultdict(list), defaultdict(list)
        self.persons, self.heights, self.resolution = defaultdict(list), defaultdict(lambda: defaultdict(int)), {}
        self.with_epp, self.epp = defaultdict(int), defaultdict(lambda: defaultdict(int))

    def install(self) -> None:
        from .detection import Detector
        original_detect, original_pose = Detector.detect, Detector._pose
        timings = self

        def detect(self_, frame, detect_objects):
            key = threading.current_thread().name
            start = time.perf_counter()
            result = original_detect(self_, frame, detect_objects)
            timings.detect[key].append((time.perf_counter() - start) * 1000)
            h, w = frame.shape[:2]
            timings.resolution[key] = f"{w}x{h}"
            timings.persons[key].append(len(result.persons))
            for p in result.persons:
                timings.heights[key][size_bucket(p.box[3] * h)] += 1
                timings.with_epp[key] += bool(p.epp)
                for label in p.epp:
                    timings.epp[key][label] += 1
            return result

        def pose(self_, frame):
            start = time.perf_counter()
            result = original_pose(self_, frame)
            timings.pose[threading.current_thread().name].append((time.perf_counter() - start) * 1000)
            return result

        Detector.detect, Detector._pose = detect, pose

    def camera(self, key: str) -> dict:
        persons = self.persons.get(key, [])
        return {"resolution": self.resolution.get(key), "detect_ms": stats(self.detect.get(key, [])),
                "pose_ms": stats(self.pose.get(key, [])), "frames": len(persons),
                "persons_per_frame": round(statistics.mean(persons), 2) if persons else 0,
                "person_heights": {label: self.heights[key].get(label, 0) for _, _, label in SIZE_BUCKETS},
                "persons_with_epp": f"{self.with_epp[key] / max(1, sum(persons)):.0%}",
                "epp_by_class": dict(sorted(self.epp[key].items()))}


# ---------- offline mode ----------

def read_frames(source: str, limit: int, resize: tuple[int, int] | None, stride: int = 1) -> list[np.ndarray]:
    path = Path(source)
    if path.is_dir():
        files = sorted(f for f in path.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)[:limit]
        frames = [cv2.imread(str(f)) for f in files]
    else:
        cap, frames, index = cv2.VideoCapture(source), [], 0
        while len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:   # every stride-th frame: covers a long recording with few frames
                frames.append(frame)
            index += 1
        cap.release()
    frames = [f for f in frames if f is not None]
    if not frames:
        raise SystemExit(f"No se pudieron leer imágenes de {source}")
    return [cv2.resize(f, resize) for f in frames] if resize else frames


def offline(args, settings: Settings) -> dict:
    from .detection import blur_faces, create_detector
    from .drawing import encode_jpeg

    timings = _Timings()
    timings.install()
    frames = read_frames(args.source, args.frames, args.resize, args.stride)
    inference = InferenceConfig.parse({"slice": json.loads(args.slice) if args.slice else None,
                                       "preprocess": json.loads(args.preprocess) if args.preprocess else None})
    slicing = inference.slicing
    detector = create_detector(settings, "persona", frozenset(args.epp_classes), with_objects=False, inference=inference)
    for frame in frames[:5]:            # warm-up: CUDA kernels and cuDNN autotuning
        detector.detect(frame, False)
    timings.__init__()
    key, encode_ms = threading.current_thread().name, []
    started = time.perf_counter()
    for frame in frames:
        detections = detector.detect(frame, False)
        detector.face_keypoints(frame)   # same cost as an event snapshot or a blurred live frame
        t = time.perf_counter()
        encode_jpeg(blur_faces(frame, detections.persons, detections.pose_keypoints), settings.jpeg_quality, settings.stream_max_width)
        encode_ms.append((time.perf_counter() - t) * 1000)
    elapsed = time.perf_counter() - started

    camera = timings.camera(key)
    camera["encode_ms"] = stats(encode_ms)
    camera["achieved_fps"] = round(len(frames) / elapsed, 1)
    detect_mean = camera["detect_ms"]["mean"]
    if slicing and slicing.enabled:
        camera["slicing"] = {"tile_size": slicing.tile_size, "overlap": slicing.overlap, "roi": list(slicing.roi)}
    if inference.preprocess.enabled:
        camera["preprocess"] = inference.preprocess.mode
    sampled = f", 1 de cada {args.stride}" if args.stride > 1 else ""
    return {"source": f"{args.source} ({len(frames)} cuadros{sampled})", "cameras": {Path(args.source).name: camera},
            "totals": {"Capacidad de detección (sin pose ni video)": f"{1000 / detect_mean:.1f} cuadros/s en total",
                       "Cuadros/s con detección + pose + codificación": camera["achieved_fps"]}}


# ---------- live mode ----------

class _NullOutbox:
    """Discards events and metrics: a benchmark must not create data in production."""

    def add_event(self, *_):
        pass

    def add_metrics(self, *_):
        pass


def live(args, settings: Settings) -> dict:
    from . import config as config_module
    from .api_client import ApiClient
    from .worker import CameraWorker

    if not settings.ingest_api_key:
        raise SystemExit("Configure INGEST_API_KEY y API_BASE_URL como en el servicio (solo se lee la configuración).")
    api = ApiClient(settings.api_base_url, settings.ingest_api_key)
    _, data, _ = api.get_config(None)
    api.close()
    node = config_module.parse(data)
    cameras = [c for c in node.cameras if not args.camera or c.name in args.camera]
    if not cameras:
        raise SystemExit("No hay cámaras activas en la configuración del nodo.")

    timings = _Timings()
    timings.install()
    workers = [CameraWorker(c, node, settings, _NullOutbox()) for c in cameras]
    for w in workers:
        w.start()
    print(f"Calentando {args.warmup} s con {len(workers)} cámara(s)…")
    time.sleep(args.warmup)
    timings.__init__()
    started = time.perf_counter()
    time.sleep(args.seconds)
    elapsed = time.perf_counter() - started
    statuses = {w.name: (w.status, w.latency_ms, w.error) for w in workers}
    for w in workers:
        w.stop()

    report, busy_ms = {}, 0.0
    for w in workers:
        camera = timings.camera(w.name)
        camera["target_fps"] = w.camera.target_fps
        slicing, preprocess = w.inference.slicing, w.inference.preprocess
        if slicing.enabled:
            camera["slicing"] = {"tile_size": slicing.tile_size, "overlap": slicing.overlap, "roi": list(slicing.roi)}
        if preprocess.enabled:
            camera["preprocess"] = preprocess.mode
        camera["achieved_fps"] = round(camera["frames"] / elapsed, 1)
        camera["status"], camera["latency_ms"], camera["error"] = statuses[w.name]
        busy_ms += sum(timings.detect.get(w.name, [])) + sum(timings.pose.get(w.name, []))
        report[w.camera.name] = camera
    return {"source": f"cámaras del nodo {node.node_id} durante {args.seconds} s", "cameras": report,
            "totals": {"Cuadros analizados por segundo (todas las cámaras)": round(sum(c["frames"] for c in report.values()) / elapsed, 1),
                       "Ocupación de la GPU por inferencia": f"{busy_ms / (elapsed * 1000):.0%} del tiempo"}}


# ---------- CLI ----------

def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark del pipeline de inferencia (resultado en docs/benchmarks)")
    p.add_argument("--label", default="linea-base", help="nombre corto de la medición, p. ej. linea-base, dos-etapas, sahi")
    p.add_argument("--notes", default="", help="contexto libre que se guarda con el resultado")
    p.add_argument("--model", help="pesos a medir (por defecto MODEL_PATH, como el servicio)")
    p.add_argument("--imgsz", type=int, help="tamaño de inferencia (por defecto IMGSZ)")
    p.add_argument("--mode-detection", choices=("two_stage", "single"), help="por defecto DETECTION_MODE")
    p.add_argument("--epp-imgsz", type=int, help="tamaño de inferencia de los recortes de EPP (por defecto EPP_IMGSZ)")
    sub = p.add_subparsers(dest="mode", required=True)
    off = sub.add_parser("offline", help="video o carpeta de imágenes")
    off.add_argument("--source", required=True)
    off.add_argument("--frames", type=int, default=300)
    off.add_argument("--stride", type=int, default=1, help="tomar 1 de cada N cuadros del video")
    off.add_argument("--resize", type=lambda s: tuple(int(v) for v in s.lower().split("x")), help="p. ej. 1280x720")
    off.add_argument("--preprocess", help='corrección de contraste, p. ej. {"mode": "clahe"} o {"mode": "gamma", "gamma": 1.8}')
    off.add_argument("--slice", help='SAHI como en la cámara, p. ej. {"enabled": true, "roi": [0, 0, 1, 0.3]}')
    off.add_argument("--epp-classes", nargs="*", default=["careta_soldar", "guantes", "calzado", "mandil", "traje", "protector_facial", "lentes"])
    on = sub.add_parser("live", help="cámaras reales del nodo (detenga antes el servicio)")
    on.add_argument("--seconds", type=int, default=120)
    on.add_argument("--warmup", type=int, default=20)
    on.add_argument("--camera", nargs="*", help="nombres de cámaras a medir (por defecto, todas)")
    args = p.parse_args()

    settings = Settings.from_env()
    overrides = {k: v for k, v in (("model_path", args.model), ("imgsz", args.imgsz), ("detection_mode", args.mode_detection),
                                   ("epp_imgsz", args.epp_imgsz)) if v}
    if overrides:
        settings = Settings(**{**settings.__dict__, **overrides})

    result = (offline if args.mode == "offline" else live)(args, settings)
    report = {"label": args.label, "mode": args.mode, "date": datetime.now().isoformat(timespec="seconds"),
              "notes": args.notes, "environment": environment(), "config": run_config(settings), **result}
    path = save(report)
    print(markdown(report))
    print(f"Guardado en {path} (y .json)")


if __name__ == "__main__":
    main()
