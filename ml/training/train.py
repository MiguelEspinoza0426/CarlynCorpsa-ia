import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    # Ej.: python ml/training/train.py --modelo yolo11m.pt --imgsz 800 --nombre epp_unificado_v3
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="yolo11s.pt", help="pesos base (yolo11n/s/m/l.pt)")
    p.add_argument("--imgsz", type=int, default=640, help="resolución; más alta ayuda con objetos pequeños")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--nombre", default="epp_unificado_v2", help="carpeta del run en ml/runs")
    p.add_argument("--data", default="epp_unificado", help="dataset en ml/datasets (p. ej. epp_unificado_persona)")
    args = p.parse_args()

    model = YOLO(args.modelo)
    model.train(
        data=str(ROOT / "ml/datasets" / args.data / "data.yaml"),   # generado por merge_datasets.py
        epochs=args.epochs,
        patience=30,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=4,
        project=str(ROOT / "ml/runs"),  # ruta absoluta: evita que Ultralytics la anide en runs/detect/
        name=args.nombre,
    )
