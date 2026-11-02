"""Hardware benchmark (HU-35): how many cameras can this machine analyze at a given FPS?

Usage (from inference/):
    python -m carlyn_inference.benchmark --model ../ml/runs/epp_v3_completo/weights/best.pt --imgsz 640
    python -m carlyn_inference.benchmark --model ... --export-engine    # TensorRT FP16 (NVIDIA)
"""
import argparse
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from .settings import REPO


def frames_from(source: str | None, count: int, size=(1280, 720)) -> list[np.ndarray]:
    if source:
        cap = cv2.VideoCapture(source)
        frames = []
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if not frames:
                    raise SystemExit(f"No se pudo leer {source}")
                continue
            frames.append(frame)
        return frames
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8) for _ in range(count)]


def measure(model_path: str, frames: list[np.ndarray], imgsz: int, device, track: bool) -> list[float]:
    from ultralytics import YOLO
    model = YOLO(model_path)
    run = (lambda f: model.track(f, persist=True, imgsz=imgsz, device=device, verbose=False, tracker="bytetrack.yaml")) if track \
        else (lambda f: model.predict(f, imgsz=imgsz, device=device, verbose=False))
    for f in frames[:10]:      # warm-up (CUDA kernels, TensorRT engine)
        run(f)
    times = []
    for f in frames:
        start = time.perf_counter()
        run(f)
        times.append((time.perf_counter() - start) * 1000)
    return times


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(REPO / "ml/runs/epp_v3_completo/weights/best.pt"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--source", help="video para medir con imágenes reales (por defecto, imágenes sintéticas 1280x720)")
    p.add_argument("--target-fps", type=float, default=5, help="FPS por cámara que se quiere sostener")
    p.add_argument("--export-engine", action="store_true", help="exportar a TensorRT FP16 y medir el motor exportado")
    args = p.parse_args()

    import torch
    device = (0 if torch.cuda.is_available() else "cpu") if args.device == "auto" else (int(args.device) if args.device.isdigit() else args.device)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sin GPU"
    model_path = args.model
    if args.export_engine:
        from ultralytics import YOLO
        model_path = YOLO(args.model).export(format="engine", half=True, imgsz=args.imgsz, device=device)

    frames = frames_from(args.source, args.frames)
    times = measure(model_path, frames, args.imgsz, device, track=True)
    mean, p95 = statistics.mean(times), sorted(times)[int(len(times) * 0.95) - 1]
    # 25 % headroom for pose (on demand), drawing, JPEG encoding and the API.
    budget_fps = 1000 / mean * 0.75
    cameras = int(budget_fps // args.target_fps)

    print(f"\nEquipo: {platform.node()} | GPU: {gpu} | dispositivo: {device}")
    print(f"Modelo: {Path(model_path).name} | imgsz {args.imgsz} | {len(times)} frames")
    print(f"Inferencia + tracking: media {mean:.1f} ms | p95 {p95:.1f} ms | máx. {1000 / mean:.1f} FPS totales")
    print(f"Con {args.target_fps:g} FPS por cámara y 25 % de margen: hasta {cameras} cámara(s) simultáneas")


if __name__ == "__main__":
    main()
