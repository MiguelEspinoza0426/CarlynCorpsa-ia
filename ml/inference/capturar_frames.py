# ml/inference/capturar_frames.py
"""Guarda frames de las cámaras de la empresa para etiquetarlos y reentrenar.

Toma un frame cada --intervalo segundos por cámara y descarta los casi idénticos
al anterior (escena quieta). Con --prelabel, el modelo actual pre-etiqueta cada
imagen en formato YOLO, para solo corregir en Roboflow/CVAT en vez de dibujar todo.

Uso:
  python ml/inference/capturar_frames.py                     # todas las de camaras.yaml
  python ml/inference/capturar_frames.py soldadura1 --prelabel
  python ml/inference/capturar_frames.py 0 --intervalo 2     # prueba con la webcam
  python ml/inference/capturar_frames.py "ventana:iVMS-4200" # vista en vivo de iVMS-4200
Ctrl+C para terminar.
"""
import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from camaras import cargar
from run_camara import MODELO, ROOT, crear_lector

SALIDA = ROOT / "ml/datasets/empresa_raw"


def huella(frame):
    """Miniatura en gris para comparar frames rápido."""
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("camaras", nargs="*", help="nombres de camaras.yaml, índice de webcam o URL rtsp")
    p.add_argument("--intervalo", type=float, default=5.0, help="segundos entre capturas por cámara")
    p.add_argument("--umbral", type=float, default=6.0, help="diferencia mínima (0-255) para guardar")
    p.add_argument("--prelabel", action="store_true", help="pre-etiquetar con el modelo actual")
    p.add_argument("--conf", type=float, default=0.35)
    args = p.parse_args()

    nombres = args.camaras or cargar()["camaras"]
    sesion = SALIDA / time.strftime("%Y%m%d_%H%M%S")
    (sesion / "images").mkdir(parents=True)

    model = None
    if args.prelabel:
        from ultralytics import YOLO
        model = YOLO(str(MODELO))
        (sesion / "labels").mkdir()
        nombres_clases = [model.names[i] for i in sorted(model.names)]
        (sesion / "data.yaml").write_text(yaml.safe_dump(
            {"train": "images", "val": "images", "nc": len(nombres_clases), "names": nombres_clases},
            sort_keys=False, allow_unicode=True), encoding="utf-8")

    camaras = {}
    for n in map(str, nombres):
        if n.isdigit():
            etiqueta = f"webcam{n}"
        elif n.startswith("ventana:"):
            etiqueta = "ventana_" + n.split(":")[1]
        else:
            etiqueta = n.rsplit("/", 1)[-1]
        etiqueta = re.sub(r"[^\w-]+", "_", etiqueta)   # válido como nombre de archivo
        camaras[etiqueta] = {"lector": crear_lector(n), "proxima": 0.0, "huella": None}
    print(f"Guardando en {sesion}  (Ctrl+C para terminar)")

    total = 0
    try:
        while True:
            ahora = time.time()
            for etiqueta, c in camaras.items():
                if ahora < c["proxima"]:
                    continue
                frame, ts = c["lector"].leer()
                if frame is None or ahora - ts > 2:   # sin señal o frame viejo
                    continue
                c["proxima"] = ahora + args.intervalo
                h = huella(frame)
                if c["huella"] is not None and np.abs(h - c["huella"]).mean() < args.umbral:
                    continue
                c["huella"] = h

                nombre = f"{etiqueta}_{time.strftime('%Y%m%d_%H%M%S')}"
                cv2.imwrite(str(sesion / "images" / f"{nombre}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if model is not None:
                    r = model.predict(frame, conf=args.conf, verbose=False)[0]
                    lineas = [f"{int(k)} " + " ".join(f"{v:.6f}" for v in caja)
                              for k, caja in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
                    (sesion / "labels" / f"{nombre}.txt").write_text("\n".join(lineas))
                total += 1
                print(f"[{etiqueta}] {nombre}.jpg  (total {total})")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        for c in camaras.values():
            c["lector"].cerrar()
        print(f"\n{total} imágenes guardadas en {sesion}")


if __name__ == "__main__":
    main()
