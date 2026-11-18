# Benchmark «linea-base-video-848» (offline)

- Fecha: 2026-11-17T21:00:07
- Equipo: DESKTOP-GL450EJ · GPU: NVIDIA GeForce RTX 5070 · Windows-11
- Versiones: Python 3.12.10 · torch 2.11.0+cu128 · ultralytics 8.4.184 · commit 319aff8
- Modelo: `epp_v3_completo` · imgsz 640 · conf 0.25 · pose `yolo11n-pose.pt`
- Fuente: ../ml/datasets/empresa_raw/dataset_carlyn_empresa/images_empresa/WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 (109 cuadros)
- Notas: RTX 5070 de desarrollo; video WhatsApp de la empresa 848x480 (109 cuadros), no es Camara 1 a resolución real

## Por cámara / fuente

| Fuente | Resolución | FPS objetivo | FPS logrados | Detección + tracking (media / p95 ms) | Pose (media ms) | Personas por cuadro | Personas por altura |
|---|---|---|---|---|---|---|---|
| WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 | 848x480 | — | 20.0 | 25.2 / 38.2 | 23.7 | 1.71 | <60 px: 2, 60–150 px: 120, ≥150 px: 64 |

## Totales

- Capacidad de detección (sin pose ni video): 39.7 cuadros/s en total
- Cuadros/s con detección + pose + codificación: 20.0
