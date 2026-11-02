# ml/training/comparar.py
"""Compara varios modelos entrenados sobre el mismo split, clase por clase.

Cada modelo se evalúa con la resolución con la que fue entrenado. Sirve aunque los modelos
tengan distinto número de clases (p. ej. 7 de EPP vs 7 + persona), siempre que las clases
comunes tengan el mismo índice. Con --empresa también mide la clase persona en las fotos
reales de la empresa (ml/datasets/empresa_raw/dataset_carlyn_empresa).
Uso: python ml/training/comparar.py epp_unificado_v1 epp_unificado_v2 epp_v3_completo [--split test] [--empresa]
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "ml/datasets/epp_unificado"
EMPRESA = ROOT / "ml/datasets/empresa_raw/dataset_carlyn_empresa"


def evaluar(model, carpeta, split, tmp):
    """Valida `model` sobre `carpeta` usando los nombres de clase del propio modelo."""
    names = [model.names[i] for i in sorted(model.names)]
    data = Path(tmp) / "data.yaml"
    data.write_text(yaml.safe_dump({"path": Path(carpeta).as_posix(), "train": "valid/images",
                                    "val": "valid/images", "test": "test/images",
                                    "nc": len(names), "names": names}, allow_unicode=True), encoding="utf-8")
    r = model.val(data=str(data), split=split, imgsz=model.overrides.get("imgsz", 640), batch=8,
                  workers=0, plots=False, verbose=False, project=str(tmp), name="val")
    return r, names


def resumen(run, split, tmp):
    model = YOLO(str(ROOT / "ml/runs" / run / "weights/best.pt"))
    r, names = evaluar(model, DATASET, split, tmp)
    b = r.box
    return {"imgsz": model.overrides.get("imgsz", 640), "P": b.mp, "R": b.mr, "mAP50": b.map50,
            "mAP50-95": b.map, "ms": r.speed["inference"],
            "clases": {names[int(c)]: b.ap50[i] for i, c in enumerate(b.ap_class_index)}}


def persona_empresa(run, tmp):
    """mAP50 de persona en las fotos de la empresa (etiquetas originales: 0 = Person)."""
    model = YOLO(str(ROOT / "ml/runs" / run / "weights/best.pt"))
    names = [model.names[i] for i in sorted(model.names)]
    if "persona" not in names:
        return None
    idx = names.index("persona")
    carpeta = Path(tmp) / "empresa"
    (carpeta / "valid/images").mkdir(parents=True)
    (carpeta / "valid/labels").mkdir(parents=True)
    for n, lab in enumerate(sorted((EMPRESA / "labels").glob("*.txt"))):
        img = next((p for p in (EMPRESA / "images").glob(lab.stem + ".*") if p.suffix.lower() != ".json"), None)
        if img is None:
            continue
        shutil.copy2(img, carpeta / "valid/images" / f"e{n}{img.suffix}")
        lineas = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        (carpeta / "valid/labels" / f"e{n}.txt").write_text(
            "\n".join(f"{idx} " + " ".join(p[1:]) for p in lineas if p[0] == "0"))
    r, _ = evaluar(model, carpeta, "val", tmp)
    i = list(r.box.ap_class_index).index(idx)
    return {"P": r.box.p[i], "R": r.box.r[i], "mAP50": r.box.ap50[i], "imagenes": n + 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--empresa", action="store_true", help="medir persona en las fotos de la empresa")
    args = p.parse_args()

    res, emp = {}, {}
    for run in args.runs:
        with tempfile.TemporaryDirectory() as tmp:
            res[run] = resumen(run, args.split, tmp)
        if args.empresa:
            with tempfile.TemporaryDirectory() as tmp:
                emp[run] = persona_empresa(run, tmp)

    ancho = max(len(r) for r in args.runs) + 2
    fila = lambda nombre, vals: print(f"{nombre:20}" + "".join(f"{v:>{ancho}}" for v in vals))

    print(f"\n=== Split: {args.split} de {DATASET.name} (mismas imágenes para todos) ===")
    fila("", args.runs)
    fila("resolución", [str(res[r]["imgsz"]) for r in args.runs])
    for k in ("P", "R", "mAP50", "mAP50-95"):
        fila(k, [f"{res[r][k]:.3f}" for r in args.runs])
    fila("ms/imagen (GPU)", [f"{res[r]['ms']:.1f}" for r in args.runs])
    print("\nmAP50 por clase:")
    for c in res[args.runs[0]]["clases"]:
        fila("  " + c, [f"{res[r]['clases'].get(c, 0):.3f}" for r in args.runs])

    if args.empresa:
        print("\nPersona en fotos de la empresa (mAP50 / recall):")
        fila("  persona", [f"{emp[r]['mAP50']:.2f}/{emp[r]['R']:.2f}" if emp[r] else "-" for r in args.runs])


if __name__ == "__main__":
    main()
