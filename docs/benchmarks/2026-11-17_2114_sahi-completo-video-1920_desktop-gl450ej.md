# Benchmark «sahi-completo-video-1920» (offline)

- Fecha: 2026-11-17T21:14:38
- Equipo: DESKTOP-GL450EJ · GPU: NVIDIA GeForce RTX 5070 · Windows-11
- Versiones: Python 3.12.10 · torch 2.11.0+cu128 · ultralytics 8.4.184 · commit a42be13
- Modelo: `epp_v3_completo` · imgsz 640 · conf 0.25 · pose `yolo11n-pose.pt`
- Dos etapas: personas con `../yolo11s.pt` (conf 0.3) · EPP en recortes a imgsz 640 con margen 15%
- Fuente: ../ml/datasets/empresa_raw/dataset_carlyn_empresa/images_empresa/WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 (109 cuadros)
- Notas: RTX 5070; video escalado a 1920x1080; SAHI en todo el cuadro (peor caso)

## Por cámara / fuente

| Fuente | Resolución | FPS objetivo | FPS logrados | Detección + tracking (media / p95 ms) | Pose (media ms) | Personas por cuadro | Personas por altura | Personas con algún EPP | EPP por clase |
|---|---|---|---|---|---|---|---|---|---|
| WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4 | 1920x1080 | — | 10.9 | 67.9 / 94.4 | 20.0 | 3.09 | <60 px: 5, 60–150 px: 46, ≥150 px: 286 | 19% | calzado: 24, guantes: 4, traje: 46 |

SAHI: WhatsApp Video 2026-10-21 at 11.19.34 AM (1).mp4: mosaicos de 640 px, solape 20%, ROI [0.0, 0.0, 1.0, 1.0]

## Totales

- Capacidad de detección (sin pose ni video): 14.7 cuadros/s en total
- Cuadros/s con detección + pose + codificación: 10.9
