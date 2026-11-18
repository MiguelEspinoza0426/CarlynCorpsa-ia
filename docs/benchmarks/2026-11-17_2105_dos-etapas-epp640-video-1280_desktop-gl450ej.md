# Benchmark «dos-etapas-epp640-video-1280» (offline)

- Fecha: 2026-11-17T21:05:52
- Equipo: DESKTOP-GL450EJ · GPU: NVIDIA GeForce RTX 5070 · Windows-11
- Versiones: Python 3.12.10 · torch 2.11.0+cu128 · ultralytics 8.4.184 · commit 5326126
- Modelo: `epp_v3_completo` · imgsz 640 · conf 0.25 · pose `yolo11n-pose.pt`
- Dos etapas: personas con `../yolo11s.pt` (conf 0.3) · EPP en recortes a imgsz 640 con margen 15%
- Fuente: ../ml/datasets/empresa_raw/dataset_carlyn_empresa/images_empresa/WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 (109 cuadros)
- Notas: RTX 5070; video escalado a 1280x720; dos etapas, EPP a imgsz 640

## Por cámara / fuente

| Fuente | Resolución | FPS objetivo | FPS logrados | Detección + tracking (media / p95 ms) | Pose (media ms) | Personas por cuadro | Personas por altura | Personas con algún EPP | EPP por clase |
|---|---|---|---|---|---|---|---|---|---|
| WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 | 1280x720 | — | 13.5 | 47.2 / 70.8 | 21.3 | 2.5 | <60 px: 8, 60–150 px: 36, ≥150 px: 229 | 22% | calzado: 21, traje: 47 |

## Totales

- Capacidad de detección (sin pose ni video): 21.2 cuadros/s en total
- Cuadros/s con detección + pose + codificación: 13.5
