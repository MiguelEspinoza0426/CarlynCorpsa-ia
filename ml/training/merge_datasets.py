# ml/training/merge_datasets.py
"""Fusiona todos los datasets YOLO de ml/datasets en uno solo con clases unificadas.

Cada subcarpeta con data.yaml se incluye automáticamente (salvo EXCLUIDOS).
Las clases se renombran con MAPEO; las que no aparecen ahí se descartan.
Las imágenes repetidas entre datasets se incluyen una sola vez; si una copia está en
valid/test, se descartan las de train para no evaluar con fotos que el modelo ya vio.
Uso: python ml/training/merge_datasets.py [--persona]

--persona agrega la clase "persona" y genera ml/datasets/epp_unificado_persona. Como varios
datasets no etiquetan personas, un modelo COCO (yolo11m) las pre-etiqueta en todas las
imágenes y se suman a las que ya venían etiquetadas (sin duplicar cajas).
"""
import argparse
import hashlib
import os
import re
import shutil
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "ml/datasets"
SALIDA = DATASETS / "epp_unificado"
EXCLUIDOS = {"d.v1i.yolov8", "epp_unificado", "epp_unificado_persona", "_rar"}

# Datasets enormes de EPP de construcción: se toma una muestra fija para que no dominen
# el entrenamiento (y no multipliquen su duración) frente a los de soldadura.
LIMITE_IMAGENES = {"dataset_ppe_final": {"train": 6000, "valid": 400, "test": 400}}

CLASES = ["careta_soldar", "guantes", "calzado", "mandil", "traje",
          "protector_facial", "lentes"]

# nombre original (minúsculas, espacios -> "_") -> clase unificada
MAPEO = {
    "welding_mask": "careta_soldar", "helm-las": "careta_soldar", "welding_helmet": "careta_soldar",
    "welding_sheild": "careta_soldar",   # así, mal escrito, viene en "Welding and Cutting Safety"
    "máscara_de_soldar": "careta_soldar", "mascara_de_soldar": "careta_soldar",   # proyecto Yolo_entrenamiento
    "welding_gloves": "guantes", "leather_gloves": "guantes", "gloves": "guantes",
    "safety_shoes": "calzado", "shoes": "calzado", "welding_boots": "calzado", "boots": "calzado",
    "welding_apron": "mandil",
    "welding_coverall": "traje", "protective_suit": "traje", "welding_jacket": "traje",
    "face_shield": "protector_facial",
    "goggles": "lentes", "safety_glasses": "lentes", "glasses": "lentes",
    # "respirator" se descarta: solo hay 1 ejemplo en todos los datasets
     # Dataset propio de la empresa (Roboflow: CarlynCorpsa-EPP-Detection)
    "calzado_de_seguridad": "calzado", "careta_de_soldar": "careta_soldar", "chaqueta_de_soldar": "traje",
    # Capturas de las cámaras etiquetadas con los nombres unificados (capturar_cuadros.py)
    **{clase: clase for clase in CLASES},
}

# Solo con --persona
MAPEO_PERSONA = {"person": "persona", "persona": "persona", "welder": "persona", "non_welder": "persona"}
CONF_PERSONA = 0.5   # confianza mínima del modelo COCO para aceptar una persona
detector_personas = None

# valid/test primero: si una imagen está repetida, se conserva la copia de evaluación
SPLITS = [("valid", "valid"), ("val", "valid"), ("test", "test"), ("train", "train")]
EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIF_MAX_DUPLICADO = 10   # bits distintos (de 64) para considerar dos imágenes iguales


def normalizar(nombre):
    return nombre.strip().lower().replace(" ", "_")


def a_caja(valores):
    """Convierte una línea YOLO (caja o polígono) a caja: cx cy w h."""
    if len(valores) == 4:
        return valores
    xs, ys = valores[0::2], valores[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def larga(ruta):
    r"""Ruta con prefijo \\?\ para que Windows acepte archivos de más de 260 caracteres."""
    return "\\\\?\\" + str(Path(ruta).resolve()) if os.name == "nt" else str(ruta)


def leer_imagen(img, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(larga(img), np.uint8), flags)


def enlazar(origen, destino):
    try:
        os.link(larga(origen), destino)   # hardlink: no duplica espacio en disco
    except OSError:
        shutil.copy2(larga(origen), destino)


def nombre_origen(img):
    """Roboflow renombra 'foto.jpg' a 'foto_jpg.rf.<hash>.jpg'; devuelve 'foto'."""
    return re.split(r"_(?:jpe?g|png|bmp|webp)\.rf\.", img.name, flags=re.I)[0].lower()


def dhash(img):
    """Huella perceptual de 64 bits: igual para la misma foto aunque se haya recomprimido."""
    gris = leer_imagen(img, cv2.IMREAD_GRAYSCALE)
    if gris is None:
        return None
    peq = cv2.resize(gris, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (peq[:, 1:] > peq[:, :-1]).flatten()
    return sum(1 << i for i, b in enumerate(bits) if b)


def iou(a, b):
    """IoU de dos cajas YOLO normalizadas (cx, cy, w, h)."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0


def agregar_personas(img, lineas):
    """Suma las personas que detecta el modelo COCO y que no estaban ya etiquetadas."""
    cls_persona = CLASES.index("persona")
    existentes = [caja for cls, caja in lineas if cls == cls_persona]
    r = detector_personas.predict(leer_imagen(img), conf=CONF_PERSONA, classes=[0], verbose=False)[0]
    nuevas = 0
    for caja in r.boxes.xywhn.tolist():
        if all(iou(caja, e) < 0.5 for e in existentes):
            lineas.append((cls_persona, caja))
            existentes.append(caja)
            nuevas += 1
    return nuevas


def es_duplicada(img, vistas):
    """Mismo nombre de origen y casi la misma imagen que una ya incluida."""
    clave, h = nombre_origen(img), dhash(img)
    if h is None:
        return False
    if any(bin(h ^ otra).count("1") <= DIF_MAX_DUPLICADO for otra in vistas[clave]):
        return True
    vistas[clave].append(h)
    return False


def leer_dataset(carpeta):
    nombres = yaml.safe_load((carpeta / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(nombres, dict):
        nombres = [nombres[k] for k in sorted(nombres)]
    remap = {i: CLASES.index(MAPEO[normalizar(n)])
             for i, n in enumerate(nombres) if normalizar(n) in MAPEO}
    descartadas = [n for n in nombres if normalizar(n) not in MAPEO]
    print(f"{carpeta.name}: {len(remap)}/{len(nombres)} clases usadas"
          + (f" (descartadas: {', '.join(descartadas)})" if descartadas else ""))
    return remap


def fusionar_split(carpeta, remap, split_orig, split, conteo, vistas):
    # Estructura Roboflow (train/images) o Ultralytics (images/train)
    dir_img, dir_lab = carpeta / split_orig / "images", carpeta / split_orig / "labels"
    if not dir_img.is_dir():
        dir_img, dir_lab = carpeta / "images" / split_orig, carpeta / "labels" / split_orig
    if not dir_img.is_dir():
        return
    prefijo = re.sub(r"[^\w-]+", "_", carpeta.name.split(".v")[0])
    usadas = omitidas = duplicadas = personas_auto = 0
    imagenes = sorted(i for i in dir_img.iterdir() if i.suffix.lower() in EXT_IMG)
    limite = LIMITE_IMAGENES.get(carpeta.name, {}).get(split)
    if limite and len(imagenes) > limite:
        imagenes = sorted(random.Random(0).sample(imagenes, limite))   # muestra reproducible
    for img in imagenes:
        if img.suffix.lower() not in EXT_IMG:
            continue
        etiqueta = dir_lab / f"{img.stem}.txt"
        etiqueta = Path(larga(etiqueta))
        lineas_orig = etiqueta.read_text().split("\n") if etiqueta.is_file() else []
        lineas_orig = [l.split() for l in lineas_orig if l.strip()]
        lineas = []
        for partes in lineas_orig:
            cls = int(partes[0])
            if cls in remap:
                caja = a_caja([float(v) for v in partes[1:]])
                lineas.append((remap[cls], caja))
        # Si todas sus etiquetas eran de clases descartadas, la imagen puede tener EPP
        # sin etiquetar; se omite para no enseñarle al modelo que eso es fondo.
        if lineas_orig and not lineas:
            omitidas += 1
            continue
        if es_duplicada(img, vistas):
            duplicadas += 1
            continue
        if detector_personas is not None:
            personas_auto += agregar_personas(img, lineas)
        for cls, _ in lineas:
            conteo[split][CLASES[cls]] += 1
        nombre = f"{prefijo}__{img.stem}"
        if len(nombre) > 100:   # Windows no admite rutas de más de 260 caracteres
            nombre = f"{nombre[:80]}_{hashlib.md5(nombre.encode()).hexdigest()[:8]}"
        enlazar(img, SALIDA / split / "images" / f"{nombre}{img.suffix}")
        (SALIDA / split / "labels" / f"{nombre}.txt").write_text(
            "\n".join(f"{cls} " + " ".join(f"{v:.6f}" for v in caja) for cls, caja in lineas))
        usadas += 1
    print(f"  {carpeta.name} / {split_orig}: {usadas} imágenes"
          + (f", {omitidas} sin EPP" if omitidas else "")
          + (f", {duplicadas} duplicadas" if duplicadas else "")
          + (f", {personas_auto} personas pre-etiquetadas" if personas_auto else ""))


def main():
    global SALIDA, CLASES, detector_personas
    p = argparse.ArgumentParser()
    p.add_argument("--persona", action="store_true", help="agregar la clase persona (pre-etiquetada con COCO)")
    args = p.parse_args()
    if args.persona:
        from ultralytics import YOLO
        SALIDA = DATASETS / "epp_unificado_persona"
        CLASES = CLASES + ["persona"]
        MAPEO.update(MAPEO_PERSONA)
        detector_personas = YOLO(str(ROOT / "yolo11m.pt"))   # COCO: la clase 0 es persona

    if SALIDA.exists():
        shutil.rmtree(SALIDA)
    for split in ("train", "valid", "test"):
        (SALIDA / split / "images").mkdir(parents=True)
        (SALIDA / split / "labels").mkdir(parents=True)

    carpetas = [c for c in sorted(DATASETS.iterdir())
                if c.is_dir() and c.name not in EXCLUIDOS and not c.name.startswith("captura_")   # sin revisar
                and (c / "data.yaml").is_file()]
    remaps = {c: leer_dataset(c) for c in carpetas}

    conteo = {s: Counter() for s in ("train", "valid", "test")}
    vistas = defaultdict(list)   # nombre de origen -> huellas de las imágenes ya incluidas
    for split_orig, split in SPLITS:
        print(f"\n[{split_orig}]")
        for carpeta in carpetas:
            fusionar_split(carpeta, remaps[carpeta], split_orig, split, conteo, vistas)

    (SALIDA / "data.yaml").write_text(yaml.safe_dump({
        "path": SALIDA.as_posix(),
        "train": "train/images", "val": "valid/images", "test": "test/images",
        "nc": len(CLASES), "names": CLASES,
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print("\nObjetos por clase:")
    print(f"  {'clase':18}" + "".join(f"{s:>8}" for s in conteo))
    for c in CLASES:
        print(f"  {c:18}" + "".join(f"{conteo[s][c]:>8}" for s in conteo))
    imagenes = {s: len(list((SALIDA / s / "images").iterdir())) for s in conteo}
    print(f"  {'IMÁGENES':18}" + "".join(f"{imagenes[s]:>8}" for s in conteo))
    print(f"\nDataset listo: {SALIDA / 'data.yaml'}")


if __name__ == "__main__":
    main()
