# Benchmark «cam1-A-una-etapa-epp_v4» (offline)

- Fecha: 2026-11-20T17:36:14
- Equipo: DESKTOP-GL450EJ · GPU: NVIDIA GeForce RTX 5070 · Windows-11
- Versiones: Python 3.12.10 · torch 2.11.0+cu128 · ultralytics 8.4.184 · commit caf297e
- Modelo: `epp_v4` · imgsz 640 · conf 0.25 · pose `yolo11n-pose.pt`
- Fuente: ../videos/WhatsApp Video 2026-11-17 at 5.51.36 PM.mp4 (1200 cuadros)
- Notas: RTX 5070; grabación de Camara 1, 1200 cuadros seguidos; una etapa con epp_v4 (personas del propio modelo)

## Por cámara / fuente

| Fuente | Resolución | FPS objetivo | FPS logrados | Detección + tracking (media / p95 ms) | Pose (media ms) | Personas por cuadro | Personas por altura | Personas con algún EPP | EPP por clase |
|---|---|---|---|---|---|---|---|---|---|
| WhatsApp Video 2026-11-17 at 5.51.36 PM.mp4 | 960x480 | — | 30.3 | 16.9 / 32.7 | 14.9 | 0.29 | <60 px: 0, 60–150 px: 345, ≥150 px: 0 | 0% | — |

## Totales

- Capacidad de detección (sin pose ni video): 59.2 cuadros/s en total
- Cuadros/s con detección + pose + codificación: 30.3
