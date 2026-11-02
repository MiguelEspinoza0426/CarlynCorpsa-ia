# ml/inference/captura_ventana.py
"""Usa una ventana de Windows (p. ej. la vista en vivo de iVMS-4200) como si fuera una cámara.

Fuente: "ventana:<parte del título>" o "ventana:<título>:x,y,w,h"
  - Sin región: al iniciar se abre una ventana para marcar con el ratón el recuadro
    de la cámara (Enter = confirmar, sin marcar nada = ventana completa). Se imprime
    la fuente con la región para reutilizarla sin volver a marcar.
  - La región es relativa al área interna de la ventana, así que sirve aunque la muevas.
La ventana debe estar visible (no minimizada ni tapada): se captura lo que hay en pantalla.
"""
import ctypes
import re
import threading
import time
from ctypes import wintypes

import cv2
import mss
import numpy as np

user32 = ctypes.windll.user32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # coordenadas reales con escalado de Windows
except (AttributeError, OSError):
    user32.SetProcessDPIAware()

FPS_CAPTURA = 15


def buscar_ventana(titulo):
    """Primer ventana visible cuyo título contiene `titulo` (sin distinguir mayúsculas)."""
    encontradas = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            largo = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(largo + 1)
            user32.GetWindowTextW(hwnd, buf, largo + 1)
            if titulo.lower() in buf.value.lower():
                encontradas.append((hwnd, buf.value))
        return True

    user32.EnumWindows(callback, 0)
    return encontradas[0] if encontradas else (None, None)


def area_cliente(hwnd):
    """Rectángulo del contenido de la ventana (sin bordes ni barra de título) en pantalla."""
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    origen = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origen))
    return origen.x, origen.y, rect.right, rect.bottom


def parsear(fuente):
    """'iVMS-4200:10,20,640,360' -> ('iVMS-4200', (10, 20, 640, 360)); sin región -> (título, None)."""
    m = re.fullmatch(r"(.+):(\d+),(\d+),(\d+),(\d+)", fuente)
    if m:
        return m.group(1), tuple(int(v) for v in m.groups()[1:])
    return fuente, None


class LectorVentana:
    """Misma interfaz que LectorCamara: leer() -> (frame, ts), cerrar()."""

    def __init__(self, fuente):
        self.titulo, self.region = parsear(fuente)
        self.frame = None
        self.ts = 0.0
        self.activo = True
        self.lock = threading.Lock()
        if self.region is None:
            self.region = self._elegir_region()
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _capturar(self, sct):
        hwnd, _ = buscar_ventana(self.titulo)
        if hwnd is None or user32.IsIconic(hwnd):
            return None
        x, y, w, h = area_cliente(hwnd)
        if self.region:
            rx, ry, rw, rh = self.region
            x, y, w, h = x + rx, y + ry, min(rw, w - rx), min(rh, h - ry)
        if w <= 0 or h <= 0:
            return None
        img = np.asarray(sct.grab({"left": x, "top": y, "width": w, "height": h}))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _elegir_region(self):
        hwnd, nombre = buscar_ventana(self.titulo)
        if hwnd is None:
            raise SystemExit(f"No hay ninguna ventana visible con '{self.titulo}' en el título.")
        with mss.mss() as sct:
            completa = self._capturar(sct)
        if completa is None:
            raise SystemExit(f"La ventana '{nombre}' está minimizada; restáurala y vuelve a intentar.")
        x, y, w, h = cv2.selectROI("Marca la camara y presiona Enter", completa, showCrosshair=False)
        cv2.destroyWindow("Marca la camara y presiona Enter")
        if w == 0 or h == 0:
            return None   # ventana completa
        print(f'[ventana] Para reutilizar esta región: "ventana:{self.titulo}:{x},{y},{w},{h}"')
        return (x, y, w, h)

    def _bucle(self):
        avisado = False
        with mss.mss() as sct:
            while self.activo:
                inicio = time.time()
                frame = self._capturar(sct)
                if frame is None:
                    if not avisado:
                        print(f"[ventana] '{self.titulo}' no está visible, esperando...")
                        avisado = True
                else:
                    if avisado:
                        print("[ventana] capturando")
                        avisado = False
                    with self.lock:
                        self.frame, self.ts = frame, time.time()
                time.sleep(max(0.0, 1 / FPS_CAPTURA - (time.time() - inicio)))

    def leer(self):
        with self.lock:
            return self.frame, self.ts

    def cerrar(self):
        self.activo = False
        self.hilo.join(timeout=3)
