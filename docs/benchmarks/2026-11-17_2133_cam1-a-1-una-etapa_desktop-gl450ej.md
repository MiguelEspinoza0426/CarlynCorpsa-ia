# Benchmark «cam1-A-1-una-etapa» (offline)

- Fecha: 2026-11-17T21:33:52
- Equipo: DESKTOP-GL450EJ · GPU: NVIDIA GeForce RTX 5070 · Windows-11
- Versiones: Python 3.12.10 · torch 2.11.0+cu128 · ultralytics 8.4.184 · commit 76b7898
- Modelo: `epp_v3_completo` · imgsz 640 · conf 0.25 · pose `yolo11n-pose.pt`
- Fuente: ../videos/WhatsApp Video 2026-11-17 at 5.51.36 PM.mp4 (1200 cuadros)
- Notas: RTX 5070; grabación de Camara 1 (960x480, 12 fps, WhatsApp), 1200 cuadros seguidos; modo anterior

## Por cámara / fuente

| Fuente | Resolución | FPS objetivo | FPS logrados | Detección + tracking (media / p95 ms) | Pose (media ms) | Personas por cuadro | Personas por altura | Personas con algún EPP | EPP por clase |
|---|---|---|---|---|---|---|---|---|---|
| WhatsApp Video 2026-11-17 at 5.51.36 PM.mp4 | 960x480 | — | 29.8 | 16.9 / 26.3 | 15.5 | 0.05 | <60 px: 0, 60–150 px: 61, ≥150 px: 0 | 0% | — |

## Totales

- Capacidad de detección (sin pose ni video): 59.2 cuadros/s en total
- Cuadros/s con detección + pose + codificación: 29.8
