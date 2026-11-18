"""Exports the node's models to TensorRT FP16 engines (HU-35). Run it ON THE INFERENCE PC: an engine only works on the
GPU model and TensorRT version that built it (an engine built on the RTX 5070 does not run on the GTX 1650).

Usage (from inference/, with the same environment variables as the service):
    python -m carlyn_inference.export                 # persons, EPP and pose with the current settings
    python -m carlyn_inference.export --max-batch 12  # more SAHI tiles or persons per frame

Then point the service to the printed .engine files (PERSON_MODEL, MODEL_PATH, POSE_MODEL). The engines are built with
a dynamic batch: the SAHI tiles of a frame and the persons' crops still go to the GPU in a single call.
"""
import argparse
import shutil
import time
from pathlib import Path

from .settings import Settings


def engine_name(model_path: str, imgsz: int, max_batch: int) -> Path:
    """yolo11s.pt -> yolo11s_fp16_640_b8.engine, next to the weights (keeps engines of several sizes apart)."""
    path = Path(model_path)
    return path.with_name(f"{path.stem}_fp16_{imgsz}_b{max_batch}.engine")


def export(model_path: str, imgsz: int, max_batch: int, device) -> Path:
    from ultralytics import YOLO

    target = engine_name(model_path, imgsz, max_batch)
    started = time.perf_counter()
    built = Path(YOLO(model_path).export(format="engine", half=True, dynamic=True, batch=max_batch, imgsz=imgsz,
                                         device=device, simplify=True, verbose=False))
    shutil.move(built, target)
    print(f"  {Path(model_path).name} -> {target.name} ({time.perf_counter() - started:.0f} s)")
    return target


def main() -> None:
    p = argparse.ArgumentParser(description="Exportar los modelos del nodo a TensorRT FP16")
    p.add_argument("--max-batch", type=int, help="imágenes por llamada a la GPU (por defecto MAX_BATCH)")
    p.add_argument("--only", choices=("persons", "epp", "pose"), nargs="*", help="exportar solo algunos modelos")
    args = p.parse_args()

    settings = Settings.from_env()
    args.max_batch = args.max_batch or settings.max_batch
    if str(settings.torch_device) == "cpu":
        raise SystemExit("TensorRT necesita una GPU NVIDIA.")
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)} · los motores solo sirven en esta GPU")
    jobs = {"persons": ("PERSON_MODEL", settings.person_model, settings.imgsz),
            "epp": ("MODEL_PATH", settings.model_path, settings.epp_imgsz),
            "pose": ("POSE_MODEL", settings.pose_model, settings.imgsz)}
    lines = []
    for key, (variable, path, imgsz) in jobs.items():
        if args.only and key not in args.only or not path or path.endswith(".engine"):
            continue
        lines.append(f"{variable}={export(path, imgsz, args.max_batch, settings.torch_device)}")
    print("\nConfigure el servicio con:\n" + "\n".join(lines + [f"MAX_BATCH={args.max_batch}"]))


if __name__ == "__main__":
    main()
