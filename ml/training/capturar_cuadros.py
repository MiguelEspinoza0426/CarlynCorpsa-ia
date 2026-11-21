# ml/training/capturar_cuadros.py
"""Captura cuadros de una cámara (videos grabados o RTSP del DVR) para etiquetarlos en Roboflow.

Guarda un cuadro cada --cada segundos, salvo los que no aportan nada nuevo:
- sin señal (negro, color uniforme o logo del DVR), con el mismo criterio del nodo de IA;
- casi iguales al último guardado (escena quieta): menos del --cambio-minimo % de píxeles cambió.
  En una cámara fija una persona lejana ocupa pocos píxeles, así que el umbral es bajo; el ruido
  de compresión no cuenta porque solo se suman píxeles que cambiaron más de CAMBIO_PIXEL niveles.

Los nombres llevan cámara, día y hora (camara1_20261117-175136_00012.jpg) para poder separar
train/valid/test por día después y no evaluar con cuadros vecinos de los de entrenamiento.

Con --pre-etiquetar se escriben etiquetas YOLO con las 8 clases del dataset unificado:
personas con el detector de personas y EPP en el recorte de cada persona (igual que el nodo).
Roboflow las importa como sugerencias: solo hay que corregirlas, no dibujar desde cero.

Ejemplos:
  python ml/training/capturar_cuadros.py --camara camara1 --fuente "videos/*.mp4" --cada 2
  python ml/training/capturar_cuadros.py --camara camara1 --fuente rtsp://... --duracion 600 --pre-etiquetar
"""
import argparse
import glob
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
from carlyn_inference.crops import crop_region, to_frame_box  # noqa: E402
from carlyn_inference.sources.signal import is_placeholder  # noqa: E402

CLASES = ["careta_soldar", "guantes", "calzado", "mandil", "traje", "protector_facial", "lentes", "persona"]
PERSONA = CLASES.index("persona")
MINIATURA = (160, 90)   # tamaño de comparación: suficiente para ver moverse a una persona lejana
CAMBIO_PIXEL = 25       # niveles de gris para que un píxel cuente como cambiado (no ruido de compresión)
CAMBIO_MINIMO = 0.1     # % de píxeles cambiados para considerar el cuadro nuevo


@dataclass
class Cuadro:
    imagen: np.ndarray
    momento: datetime   # hora real de la escena (stream) o inicio del archivo + posición (video)


def ocultar_credenciales(fuente: str) -> str:
    """rtsp://usuario:clave@host/... -> rtsp://***@host/... (nunca se imprime la clave)."""
    return re.sub(r"//[^/@]+@", "//***@", fuente)


def nombre_cuadro(camara: str, momento: datetime, indice: int) -> str:
    return f"{camara}_{momento:%Y%m%d-%H%M%S}_{indice:05d}"


def miniatura(img: np.ndarray) -> np.ndarray:
    pequena = cv2.resize(img, MINIATURA, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(pequena, cv2.COLOR_BGR2GRAY), (3, 3), 0)


def es_nuevo(anterior: np.ndarray | None, actual: np.ndarray, cambio_minimo: float = CAMBIO_MINIMO) -> bool:
    """Una escena quieta no aporta ejemplos nuevos: exige que cambie al menos `cambio_minimo` % de píxeles."""
    if anterior is None:
        return True
    return float(np.mean(cv2.absdiff(anterior, actual) > CAMBIO_PIXEL)) * 100 >= cambio_minimo


def linea_yolo(clase: int, caja: tuple[float, float, float, float]) -> str:
    """Caja normalizada (x, y, w, h desde la esquina) -> línea YOLO (centro), recortada al cuadro."""
    x, y, w, h = caja
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(1.0, x + w), min(1.0, y + h)
    return f"{clase} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def leer_cuadros(fuente: str, cada: float, duracion: float | None, inicio_video: datetime | None = None):
    """Genera un cuadro cada `cada` segundos de la fuente (archivo o stream).

    En un video, la hora de cada cuadro es `inicio_video` (la hora del DVR al empezar la grabación)
    más su posición; sin ella se usa la fecha del archivo, que puede ser la de la descarga.
    """
    es_archivo = Path(fuente).is_file()
    cap = cv2.VideoCapture(fuente, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {ocultar_credenciales(fuente)}")
    try:
        if es_archivo:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            inicio = inicio_video or datetime.fromtimestamp(Path(fuente).stat().st_mtime)
            paso = max(1, round(fps * cada))
            n = 0
            while (duracion is None or n / fps < duracion) and cap.grab():
                if n % paso == 0:
                    ok, img = cap.retrieve()
                    if ok:
                        yield Cuadro(img, datetime.fromtimestamp(inicio.timestamp() + n / fps))
                n += 1
        else:
            inicio = time.monotonic()
            siguiente = 0.0
            while duracion is None or time.monotonic() - inicio < duracion:
                ok, img = cap.read()   # leer siempre: si no, el buffer del stream se atrasa
                if not ok:
                    raise SystemExit("Se cortó el stream")
                if time.monotonic() - inicio >= siguiente:
                    siguiente += cada
                    yield Cuadro(img, datetime.now())
    finally:
        cap.release()


class PreEtiquetador:
    """Personas con el detector de personas y EPP dentro del recorte de cada una."""

    def __init__(self, modelo_personas: str, modelo_epp: str, conf: float, imgsz: int, margen: float = 0.15):
        from ultralytics import YOLO
        self.personas, self.epp = YOLO(modelo_personas), YOLO(modelo_epp)
        self.conf, self.imgsz, self.margen = conf, imgsz, margen
        self.epp_clases = {i: n for i, n in self.epp.names.items() if n in CLASES and n != "persona"}

    def __call__(self, img: np.ndarray) -> list[str]:
        alto, ancho = img.shape[:2]
        r = self.personas.predict(img, conf=self.conf, imgsz=self.imgsz, classes=[0], verbose=False)[0]
        lineas = []
        for x1, y1, x2, y2 in r.boxes.xyxy.tolist():
            caja = (x1 / ancho, y1 / alto, (x2 - x1) / ancho, (y2 - y1) / alto)
            lineas.append(linea_yolo(PERSONA, caja))
            region = crop_region(caja, ancho, alto, self.margen)
            if region is None:
                continue
            recorte = img[region[1]:region[3], region[0]:region[2]]
            e = self.epp.predict(recorte, conf=self.conf, imgsz=640, verbose=False)[0]
            for xyxy, c in zip(e.boxes.xyxy.tolist(), e.boxes.cls.tolist()):
                if int(c) in self.epp_clases:
                    lineas.append(linea_yolo(CLASES.index(self.epp_clases[int(c)]), to_frame_box(xyxy, region, ancho, alto)))
        return lineas


def capturar(fuentes: list[str], camara: str, salida: Path, cada: float, duracion: float | None,
             cambio_minimo: float, etiquetar: Callable[[np.ndarray], list[str]] | None,
             inicio_video: datetime | None = None) -> dict[str, int]:
    (salida / "images").mkdir(parents=True, exist_ok=True)
    if etiquetar:
        (salida / "labels").mkdir(exist_ok=True)
        (salida / "data.yaml").write_text(yaml.safe_dump(
            {"path": ".", "train": "images", "val": "images", "nc": len(CLASES), "names": CLASES},
            allow_unicode=True), encoding="utf-8")
    cuenta = {"leidos": 0, "sin_senal": 0, "repetidos": 0, "guardados": 0}
    indice = len(list((salida / "images").glob("*.jpg")))   # continuar sin pisar capturas previas
    for fuente in fuentes:
        print(f"Fuente: {ocultar_credenciales(fuente)}")
        anterior = None
        for cuadro in leer_cuadros(fuente, cada, duracion, inicio_video):
            cuenta["leidos"] += 1
            if is_placeholder(cuadro.imagen):
                cuenta["sin_senal"] += 1
                continue
            pequeno = miniatura(cuadro.imagen)
            if not es_nuevo(anterior, pequeno, cambio_minimo):
                cuenta["repetidos"] += 1
                continue
            anterior = pequeno
            nombre = nombre_cuadro(camara, cuadro.momento, indice)
            cv2.imwrite(str(salida / "images" / f"{nombre}.jpg"), cuadro.imagen, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if etiquetar:
                (salida / "labels" / f"{nombre}.txt").write_text("".join(f"{linea}\n" for linea in etiquetar(cuadro.imagen)), encoding="utf-8")
            indice += 1
            cuenta["guardados"] += 1
    return cuenta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Captura cuadros de una cámara para etiquetar.")
    p.add_argument("--camara", required=True, help="prefijo de los archivos, p. ej. camara1")
    p.add_argument("--fuente", required=True, nargs="+",
                   help="videos (acepta comodines) o URL RTSP del stream principal del DVR")
    p.add_argument("--salida", type=Path, help="carpeta destino (por defecto ml/datasets/captura_<camara>)")
    p.add_argument("--cada", type=float, default=2.0, help="segundos entre cuadros")
    p.add_argument("--duracion", type=float, help="segundos a capturar por fuente (obligatorio en streams largos)")
    p.add_argument("--inicio", type=datetime.fromisoformat,
                   help='hora del DVR al empezar el video, p. ej. "2026-11-17 17:51:36" (un video por llamada)')
    p.add_argument("--cambio-minimo", type=float, default=CAMBIO_MINIMO,
                   help="%% de píxeles que deben cambiar; más alto: descarta más cuadros parecidos")
    p.add_argument("--pre-etiquetar", action="store_true", help="escribe etiquetas YOLO sugeridas")
    p.add_argument("--modelo-personas", default=str(ROOT / "yolo11s.pt"))
    p.add_argument("--modelo-epp", default=str(ROOT / "ml/runs/epp_v5/weights/best.pt"))
    p.add_argument("--conf", type=float, default=0.15, help="baja a propósito: es más fácil borrar que dibujar")
    p.add_argument("--imgsz", type=int, default=1280, help="tamaño para detectar personas (las lejanas necesitan más)")
    a = p.parse_args()

    fuentes = [f for patron in a.fuente for f in (sorted(glob.glob(patron)) if "://" not in patron else [patron]) or [patron]]
    salida = a.salida or ROOT / "ml/datasets" / f"captura_{a.camara}"
    etiquetador = PreEtiquetador(a.modelo_personas, a.modelo_epp, a.conf, a.imgsz) if a.pre_etiquetar else None
    c = capturar(fuentes, a.camara, salida, a.cada, a.duracion, a.cambio_minimo, etiquetador, a.inicio)
    print(f"Leídos {c['leidos']} · sin señal {c['sin_senal']} · repetidos {c['repetidos']} · guardados {c['guardados']}")
    print(f"Carpeta: {salida}")
