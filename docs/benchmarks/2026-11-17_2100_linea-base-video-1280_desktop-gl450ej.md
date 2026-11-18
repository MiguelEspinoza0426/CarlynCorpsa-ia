# Benchmark «linea-base-video-1280» (offline)

- Fecha: 2026-11-17T21:00:19
- Equipo: DESKTOP-GL450EJ · GPU: NVIDIA GeForce RTX 5070 · Windows-11
- Versiones: Python 3.12.10 · torch 2.11.0+cu128 · ultralytics 8.4.184 · commit 319aff8
- Modelo: `epp_v3_completo` · imgsz 640 · conf 0.25 · pose `yolo11n-pose.pt`
- Fuente: ../ml/datasets/empresa_raw/dataset_carlyn_empresa/images_empresa/WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 (109 cuadros)
- Notas: RTX 5070 de desarrollo; mismo video escalado a 1280x720 (resolución del substream)

## Por cámara / fuente

| Fuente | Resolución | FPS objetivo | FPS logrados | Detección + tracking (media / p95 ms) | Pose (media ms) | Personas por cuadro | Personas por altura |
|---|---|---|---|---|---|---|---|
| WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 | 1280x720 | — | 18.8 | 24.1 / 37.9 | 23.7 | 1.7 | <60 px: 0, 60–150 px: 1, ≥150 px: 184 |

## Totales

- Capacidad de detección (sin pose ni video): 41.5 cuadros/s en total
- Cuadros/s con detección + pose + codificación: 18.8
