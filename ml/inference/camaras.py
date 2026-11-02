# ml/inference/camaras.py
"""Resuelve el nombre de una cámara de la empresa a su URL RTSP (vía MediaMTX + Tailscale)."""
import os
from pathlib import Path
from urllib.parse import quote

import yaml

CONFIG = Path(__file__).with_name("camaras.yaml")


def cargar():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def url(nombre):
    cfg = cargar()
    if nombre not in cfg["camaras"]:
        raise SystemExit(f"Cámara '{nombre}' no está en {CONFIG.name}: {cfg['camaras']}")
    usuario, clave = os.environ.get("CAMARAS_USER", ""), os.environ.get("CAMARAS_PASS", "")
    auth = f"{quote(usuario, safe='')}:{quote(clave, safe='')}@" if usuario else ""
    return f"rtsp://{auth}{cfg['servidor']}:{cfg['puerto']}/{nombre}"


def fuente(valor):
    """'0' -> webcam 0, 'rtsp://...' -> tal cual, 'soldadura1' -> URL de camaras.yaml."""
    if valor.isdigit():
        return int(valor)
    if "://" in valor:
        return valor
    return url(valor)
