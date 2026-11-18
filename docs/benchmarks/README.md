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

### Con conteo de EPP (mismo video a 1280×720, RTX 5070)

| Medición | Detección + tracking (media / p95) | Cuadros/s del pipeline | Personas por cuadro | Personas por altura (<60 / 60–150 / ≥150 px) | Personas con algún EPP | EPP por clase |
|---|---|---|---|---|---|---|
| [una-etapa](2026-11-17_2105_una-etapa-video-1280_desktop-gl450ej.md) (línea base) | 23,1 / 37,1 ms | 19,8 | 1,7 | 0 / 1 / 184 | 3 % | calzado 5 |

## Paso 1: detección en dos etapas

La primera etapa detecta personas con YOLO11s (COCO) y ByteTrack. La segunda corre `epp_v3_completo` en un solo batch sobre el recorte de cada persona, tomado del cuadro a resolución completa con un margen del 15 %. Solo cuenta el EPP cuyo centro cae dentro de la persona.

| Medición | Detección + tracking (media / p95) | Cuadros/s del pipeline | Personas por cuadro | Personas por altura (<60 / 60–150 / ≥150 px) | Personas con algún EPP | EPP por clase |
|---|---|---|---|---|---|---|
| [dos-etapas, EPP a 320](2026-11-17_2105_dos-etapas-epp320-video-1280_desktop-gl450ej.md) | 46,8 / 72,8 ms | 13,6 | 2,5 | 8 / 36 / 229 | 8 % | calzado 8, guantes 6, traje 9 |
| [dos-etapas, EPP a 640](2026-11-17_2105_dos-etapas-epp640-video-1280_desktop-gl450ej.md) (por defecto) | 47,2 / 70,8 ms | 13,5 | 2,5 | 8 / 36 / 229 | 22 % | calzado 21, traje 47 |

- **Personas:** +47 % por cuadro, incluidas 44 detecciones de personas pequeñas (<150 px) que antes no aparecían. Sin etiquetas no se puede separar cuántas son falsos positivos; eso se medirá con el dataset del paso 8.
- **EPP:** el porcentaje de personas con algún EPP sube de 3 % a 22 %, porque el modelo ve a cada persona con más píxeles. A 320 px rinde menos: el modelo se entrenó a 640.
- **Costo:** la detección pasa de ~23 a ~47 ms por cuadro (dos modelos). En la GTX 1650 esto pesa más; lo compensan los pasos 3 (no evaluar EPP en personas pequeñas) y 5 (TensorRT FP16). `DETECTION_MODE=single` vuelve al modo anterior.

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
