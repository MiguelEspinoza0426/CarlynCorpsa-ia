# Benchmark «cam1-a-8-paso3-epp90» (offline)

- Fecha: 2026-11-17T21:53:22
- Equipo: DESKTOP-GL450EJ · GPU: NVIDIA GeForce RTX 5070 · Windows-11
- Versiones: Python 3.12.10 · torch 2.11.0+cu128 · ultralytics 8.4.184 · commit 9bb763c
- Modelo: `epp_v3_completo` · imgsz 640 · conf 0.25 · pose `yolo11n-pose.pt`
- Dos etapas: personas con `../yolo11s.pt` (conf 0.2) · EPP en recortes a imgsz 640 con margen 15%
- Fuente: ../videos/WhatsApp Video 2026-11-17 at 5.51.36 PM.mp4 (1200 cuadros)
- Notas: RTX 5070; grabación de Camara 1, 1200 cuadros seguidos; paso 3 completo con MIN_PERSON_HEIGHT_PX=90 para ver el EPP con esta resolución

## Por cámara / fuente

| Fuente | Resolución | FPS objetivo | FPS logrados | Detección + tracking (media / p95 ms) | Pose (media ms) | Personas por cuadro | Personas por altura | Personas con algún EPP | EPP por clase |
|---|---|---|---|---|---|---|---|---|---|
| WhatsApp Video 2026-11-17 at 5.51.36 PM.mp4 | 960x480 | — | 22.6 | 27.3 / 40.8 | 15.7 | 0.93 | <60 px: 436, 60–150 px: 680, ≥150 px: 0 | 0% | — |

SAHI: WhatsApp Video 2026-11-17 at 5.51.36 PM.mp4: mosaicos de 320 px, solape 20%, ROI [0.0, 0.06, 1.0, 0.5]

Corrección de contraste: WhatsApp Video 2026-11-17 at 5.51.36 PM.mp4: clahe

## Totales

- Capacidad de detección (sin pose ni video): 36.6 cuadros/s en total
- Cuadros/s con detección + pose + codificación: 22.6
