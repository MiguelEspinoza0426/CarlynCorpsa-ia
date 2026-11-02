import os
import threading
import time
from pathlib import Path

# Forzar RTSP sobre TCP (más estable que UDP); debe ir antes de importar cv2
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
from ultralytics import YOLO

from camaras import fuente as resolver_fuente

ROOT = Path(__file__).resolve().parents[2]
MODELO = Path(os.environ.get("MODELO", ROOT / "ml/runs/epp_welding_v1/weights/best.pt"))
# Umbral de confianza: más bajo = detecta más (y se equivoca más). El modelo es más preciso
# que exhaustivo (falla por omisión), así que conviene un umbral bajo.
CONF = float(os.environ.get("CONF", 0.25))


class LectorCamara:
    """Lee la cámara en un hilo y conserva solo el frame más reciente."""

    def __init__(self, fuente):
        self.fuente = fuente
        self.frame = None
        self.ts = 0.0
        self.activo = True
        self.lock = threading.Lock()
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _abrir(self):
        if isinstance(self.fuente, str):
            return cv2.VideoCapture(self.fuente, cv2.CAP_FFMPEG)
        return cv2.VideoCapture(self.fuente, cv2.CAP_DSHOW)   # en Windows abre la webcam más rápido que MSMF

    def _bucle(self):
        espera = 1
        while self.activo:
            cap = self._abrir()
            if not cap.isOpened():
                print(f"[cámara] sin conexión, reintento en {espera}s")
                time.sleep(espera)
                espera = min(espera * 2, 30)
                continue
            print("[cámara] conectada")
            espera = 1
            while self.activo:
                ok, frame = cap.read()
                if not ok:
                    print("[cámara] señal perdida, reconectando...")
                    break
                with self.lock:
                    self.frame, self.ts = frame, time.time()
            cap.release()

    def leer(self):
        with self.lock:
            return self.frame, self.ts

    def cerrar(self):
        self.activo = False
        self.hilo.join(timeout=3)   # deja que el hilo haga cap.release(); si no, el proceso queda colgado


def crear_lector(valor):
    """'ventana:iVMS-4200[:x,y,w,h]' -> captura de esa ventana; lo demás -> cámara (webcam, nombre o rtsp)."""
    if valor.startswith("ventana:"):
        from captura_ventana import LectorVentana
        return LectorVentana(valor[len("ventana:"):])
    return LectorCamara(resolver_fuente(valor))


if __name__ == "__main__":
    import sys

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