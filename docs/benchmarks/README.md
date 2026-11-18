# Benchmarks del pipeline de inferencia

Mediciones para comparar el rendimiento antes y después de cada mejora del servicio de IA: detección en dos etapas, SAHI, TensorRT, etc. Cada ejecución de `python -m carlyn_inference.benchmark` deja dos archivos en esta carpeta:

- `<fecha>_<etiqueta>_<equipo>.json`: datos completos (entorno, configuración, tiempos por etapa, personas por tamaño);
- `<fecha>_<etiqueta>_<equipo>.md`: resumen en tabla.

## Qué se mide

| Columna | Significado |
|---|---|
| FPS objetivo | `targetFps` de la cámara en la plataforma |
| FPS logrados | Cuadros analizados por segundo en la medición |
| Detección + tracking | Tiempo de `Detector.detect`: YOLO + ByteTrack + asociación de EPP; incluye la pose cuando hay alguien acostado |
| Pose | Tiempo de yolo11n-pose: confirma caídas y ubica los rostros a pixelar |
| Personas por cuadro | Promedio de personas con ID de seguimiento |
| Personas por altura | Detecciones según la altura de la persona en píxeles de la imagen original: `<60 px` (fondo de la nave), `60–150 px` y `≥150 px` (EPP evaluable) |

## Línea base (antes de los cambios del 17/11/2026)

### Medida en la PC de desarrollo (RTX 5070)

Con el único video de la empresa disponible en el repositorio, que es 848×480 y 109 cuadros. No es Camara 1 a resolución real; sirve para medir el costo de cada etapa.

| Medición | Resolución | Detección + tracking (media / p95) | Pose | Cuadros/s del pipeline completo |
|---|---|---|---|---|
| [linea-base-video-848](2026-11-17_2100_linea-base-video-848_desktop-gl450ej.md) | 848×480 | 25,2 / 38,2 ms | 23,7 ms | 20,0 |
| [linea-base-video-1280](2026-11-17_2100_linea-base-video-1280_desktop-gl450ej.md) | 1280×720 | 24,1 / 37,9 ms | 23,7 ms | 18,8 |

La resolución de entrada casi no cambia el tiempo, porque YOLO siempre reduce la imagen a 640 px. Esa misma reducción explica por qué no se detectan las personas pequeñas del fondo (hallazgo 1).

### Medida en producción (PC de la empresa, GTX 1650)

Datos reportados desde la plataforma (heartbeat) el 16–17/11/2026:

| Cámara | targetFps | FPS logrados | Observación |
|---|---|---|---|
| Camara 1 | 20 (main stream /Channels/101) | 5,3 antes del cambio → 8,2 | Limitada por la GPU |
| c | 2 | — | — |
| CAM 3 | 2 | — | — |
| Video en vivo (MJPEG anotado) | — | 2,5–8 | — |

**Pendiente:** ejecutar en la PC de la empresa, con el servicio detenido unos 3 minutos, para tener tiempos por etapa y FPS por cámara en la GTX 1650:

```powershell
cd C:\...\CarlynCorpsa-ia\inference
# mismas variables de entorno que el servicio: API_BASE_URL, INGEST_API_KEY, MODEL_PATH, credenciales de las cámaras
C:\venvs\carlyn\Scripts\python -m carlyn_inference.benchmark --label linea-base-produccion live --seconds 120
```

El modo `live` solo lee la configuración de la API. Los eventos y métricas que generan las reglas se descartan, así que no crea datos en producción.
