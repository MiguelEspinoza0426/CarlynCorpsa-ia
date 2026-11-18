from carlyn_inference.export import engine_name


def test_engine_name_keeps_size_and_batch_apart():
    assert engine_name("../yolo11s.pt", 640, 8).name == "yolo11s_fp16_640_b8.engine"
    assert engine_name("ml/runs/epp_v3_completo/weights/best.pt", 640, 12).as_posix().endswith("weights/best_fp16_640_b12.engine")
