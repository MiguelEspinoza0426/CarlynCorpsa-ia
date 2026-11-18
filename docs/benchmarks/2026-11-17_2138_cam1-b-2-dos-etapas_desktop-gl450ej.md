# Benchmark «cam1-B-2-dos-etapas» (offline)

- Fecha: 2026-11-17T21:38:35
- Equipo: DESKTOP-GL450EJ · GPU: NVIDIA GeForce RTX 5070 · Windows-11
- Versiones: Python 3.12.10 · torch 2.11.0+cu128 · ultralytics 8.4.184 · commit 76b7898
- Modelo: `epp_v3_completo` · imgsz 640 · conf 0.25 · pose `yolo11n-pose.pt`
- Dos etapas: personas con `../yolo11s.pt` (conf 0.2) · EPP en recortes a imgsz 640 con margen 15%
- Fuente: ../videos/WhatsApp Video 2026-11-17 at 5.57.29 PM.mp4 (1200 cuadros)
- Notas: RTX 5070; grabación de Camara 1 (960x480, 12 fps, WhatsApp), 1200 cuadros seguidos; dos etapas

## Por cámara / fuente

| Fuente | Resolución | FPS objetivo | FPS logrados | Detección + tracking (media / p95 ms) | Pose (media ms) | Personas por cuadro | Personas por altura | Personas con algún EPP | EPP por clase |
|---|---|---|---|---|---|---|---|---|---|
| WhatsApp Video 2026-11-17 at 5.57.29 PM.mp4 | 960x480 | — | 28.4 | 18.5 / 36.8 | 15.5 | 0.21 | <60 px: 180, 60–150 px: 68, ≥150 px: 0 | 0% | — |

## Totales

- Capacidad de detección (sin pose ni video): 54.1 cuadros/s en total
- Cuadros/s con detección + pose + codificación: 28.4
