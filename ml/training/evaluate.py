from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    model = YOLO(str(ROOT / "ml/runs/epp_welding_v1/weights/best.pt"))
    m = model.val(
        data=str(ROOT / "ml/datasets/weldingpe_v1/data.yaml"),
        split="test",
        imgsz=640,
        device=0,
        project=str(ROOT / "ml/runs"),
        name="epp_welding_v1_test",
    )
    print(f"\nPrecisión: {m.box.mp:.3f} | Recall: {m.box.mr:.3f} | mAP50: {m.box.map50:.3f} | mAP50-95: {m.box.map:.3f}")
    for i, c in enumerate(m.box.ap_class_index):
        print(f"  {model.names[int(c)]}: AP50 = {m.box.ap50[i]:.3f}")