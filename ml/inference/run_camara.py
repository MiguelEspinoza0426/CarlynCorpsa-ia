import os
import time
from pathlib import Path

import sys

import cv2
from ultralytics import YOLO

from camaras import fuente as resolver_fuente

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
from carlyn_inference.sources.camera import CameraReader  # noqa: E402
MODELO = Path(os.environ.get("MODELO", ROOT / "ml/runs/epp_v3_completo/weights/best.pt"))
# Umbral de confianza: más bajo = detecta más (y se equivoca más). El modelo es más preciso
# que exhaustivo (falla por omisión), así que conviene un umbral bajo.
CONF = float(os.environ.get("CONF", 0.25))


class LectorCamara(CameraReader):
    """Lector reutilizado por el servicio de inferencia (inference/carlyn_inference/sources/camera.py)."""

    def leer(self):
        return self.read()

    def cerrar(self):
        self.close()


def crear_lector(valor):
    """'ventana:iVMS-4200[:x,y,w,h]' -> captura de esa ventana; lo demás -> cámara (webcam, nombre o rtsp)."""
    if valor.startswith("ventana:"):
        from carlyn_inference.sources.window import LectorVentana
        return LectorVentana(valor[len("ventana:"):])
    return LectorCamara(resolver_fuente(valor))


if __name__ == "__main__":
    # Uso: python run_camara.py [0 | soldadura1 | rtsp://... | ventana:iVMS-4200]   (por defecto: webcam 0)
    fuente = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CAMARA_URL", "0")

    if not MODELO.is_file():
        raise SystemExit(f"No se encontró el modelo: {MODELO}\n"
                         "Entrena primero (ml/training/train.py) o indica otro con la variable MODELO.")
    model = YOLO(str(MODELO))
    lector = crear_lector(fuente)
    ultimo_ts, fps, t0 = 0.0, 0.0, time.time()

    try:
        while True:
            frame, ts = lector.leer()
            if frame is None or ts == ultimo_ts:
                time.sleep(0.005)
                continue
            ultimo_ts = ts

            r = model.predict(frame, conf=CONF, imgsz=640, device=0, verbose=False)[0]
            salida = r.plot()

            ahora = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(ahora - t0, 1e-6))
            t0 = ahora
            retraso = ahora - ts   # latencia captura -> procesado
            cv2.putText(salida, f"{fps:.1f} FPS | retraso {retraso*1000:.0f} ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("CarlynCorpsa - EPP", salida)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        lector.cerrar()
        cv2.destroyAllWindows()