# Benchmark «dos-etapas-video-1920» (offline)

- Fecha: 2026-11-17T21:11:47
- Equipo: DESKTOP-GL450EJ · GPU: NVIDIA GeForce RTX 5070 · Windows-11
- Versiones: Python 3.12.10 · torch 2.11.0+cu128 · ultralytics 8.4.184 · commit a42be13
- Modelo: `epp_v3_completo` · imgsz 640 · conf 0.25 · pose `yolo11n-pose.pt`
- Dos etapas: personas con `../yolo11s.pt` (conf 0.3) · EPP en recortes a imgsz 640 con margen 15%
- Fuente: ../ml/datasets/empresa_raw/dataset_carlyn_empresa/images_empresa/WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 (109 cuadros)
- Notas: RTX 5070; video escalado a 1920x1080; dos etapas sin SAHI

## Por cámara / fuente

| Fuente | Resolución | FPS objetivo | FPS logrados | Detección + tracking (media / p95 ms) | Pose (media ms) | Personas por cuadro | Personas por altura | Personas con algún EPP | EPP por clase |
|---|---|---|---|---|---|---|---|---|---|
| WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 | 1920x1080 | — | 13.4 | 48.9 / 81.4 | 22.5 | 2.5 | <60 px: 0, 60–150 px: 26, ≥150 px: 246 | 21% | calzado: 23, guantes: 1, traje: 45 |

## Totales

- Capacidad de detección (sin pose ni video): 20.4 cuadros/s en total
- Cuadros/s con detección + pose + codificación: 13.4
