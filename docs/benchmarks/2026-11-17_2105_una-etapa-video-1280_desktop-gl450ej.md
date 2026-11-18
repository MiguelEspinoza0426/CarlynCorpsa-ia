# Benchmark «una-etapa-video-1280» (offline)

- Fecha: 2026-11-17T21:05:24
- Equipo: DESKTOP-GL450EJ · GPU: NVIDIA GeForce RTX 5070 · Windows-11
- Versiones: Python 3.12.10 · torch 2.11.0+cu128 · ultralytics 8.4.184 · commit 5326126
- Modelo: `epp_v3_completo` · imgsz 640 · conf 0.25 · pose `yolo11n-pose.pt`
- Fuente: ../ml/datasets/empresa_raw/dataset_carlyn_empresa/images_empresa/WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 (109 cuadros)
- Notas: RTX 5070; video escalado a 1280x720; modo anterior (una etapa) con conteo de EPP

## Por cámara / fuente

| Fuente | Resolución | FPS objetivo | FPS logrados | Detección + tracking (media / p95 ms) | Pose (media ms) | Personas por cuadro | Personas por altura | Personas con algún EPP | EPP por clase |
|---|---|---|---|---|---|---|---|---|---|
| WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 | 1280x720 | — | 19.8 | 23.1 / 37.1 | 21.8 | 1.7 | <60 px: 0, 60–150 px: 1, ≥150 px: 184 | 3% | calzado: 5 |

## Totales

- Capacidad de detección (sin pose ni video): 43.3 cuadros/s en total
- Cuadros/s con detección + pose + codificación: 19.8
