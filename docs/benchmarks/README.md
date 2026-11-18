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

## Paso 2: SAHI por cámara

La primera etapa analiza en un solo batch el cuadro completo y mosaicos de 640 px con 20 % de solape, solo dentro de la ROI de la cámara. Las detecciones se fusionan con NMS; una persona cortada por el borde de un mosaico se une con su caja completa. Después pasan por ByteTrack (el de Ultralytics, con los mismos parámetros).

Mismo video escalado a 1920×1080 en la RTX 5070:

| Medición | Detección + tracking (media / p95) | Cuadros/s del pipeline | Personas por cuadro | Personas por altura (<60 / 60–150 / ≥150 px) | Personas con algún EPP |
|---|---|---|---|---|---|
| [dos etapas, sin SAHI](2026-11-17_2111_dos-etapas-video-1920_desktop-gl450ej.md) | 48,9 / 81,4 ms | 13,4 | 2,5 | 0 / 26 / 246 | 21 % |
| [SAHI en la franja superior (30 %)](2026-11-17_2114_sahi-roi30-video-1920_desktop-gl450ej.md) | 53,1 / 79,2 ms | 13,5 | 2,41 | 1 / 23 / 239 | 22 % |
| [SAHI en todo el cuadro](2026-11-17_2114_sahi-completo-video-1920_desktop-gl450ej.md) | 67,9 / 94,4 ms | 10,9 | 3,09 | 5 / 46 / 286 | 19 % |

- **Costo:** SAHI limitado a una franja como «Fondo nave» suma ~4 ms, porque los mosaicos van en el mismo batch. En todo el cuadro suma ~19 ms.
- **Límite de esta medición:** el video tiene 848×480 de origen. Al escalarlo no aparecen personas lejanas nuevas, así que no mide la ganancia real en el fondo de Camara 1. Eso queda para el modo `live` en la PC de la empresa con `INFERENCE_OVERRIDES`, y para el recall por distancia del paso 8.
- **Fusión:** una primera versión unía también cajas contenidas una en otra aunque no las hubiera cortado un mosaico. Eso juntaba a dos personas, una delante de la otra (2,28 personas por cuadro contra 2,48 de Ultralytics). Ahora solo se unen las cajas cortadas por el borde de un mosaico.

## Grabaciones reales de Camara 1 (17/11/2026)

Hay dos grabaciones de la cámara en `videos/` (no versionadas): 960×480, 12 fps, comprimidas por WhatsApp. Se midieron 1200 cuadros seguidos de cada una (100 s), en la RTX 5070.

- **A (17:45–17:54):** dos o tres personas trabajan al fondo de la nave, junto a los cilindros rojos. Miden 30–60 px y las tapan el banco y la columna.
- **B (17:53–17:57):** una persona trabaja junto a la columna, casi oculta.

| Configuración | A: personas por cuadro | A: menores de 60 px | B: personas por cuadro | Detección + tracking A (media / p95) | Cuadros/s A |
|---|---|---|---|---|---|
| [1. Una etapa (anterior)](2026-11-17_2133_cam1-a-1-una-etapa_desktop-gl450ej.md) | 0,05 | 0 | 0,00 | 16,9 / 26,3 ms | 29,8 |
| [2. Dos etapas](2026-11-17_2134_cam1-a-2-dos-etapas_desktop-gl450ej.md) | 0,62 | 56 | 0,21 | 24,1 / 46,3 ms | 24,3 |
| [3. + SAHI, mosaicos de 640](2026-11-17_2135_cam1-a-3-sahi-640_desktop-gl450ej.md) | 0,72 | 128 | 0,16 | 28,4 / 50,4 ms | 22,3 |
| [4. + SAHI, mosaicos de 320 ampliados a 640](2026-11-17_2137_cam1-a-4-sahi-320_desktop-gl450ej.md) | **0,79** | **199** | 0,15 | 33,6 / 54,6 ms | 19,9 |

SAHI con ROI `[0, 0.06, 1, 0.5]`: el fondo de la nave, sin la franja de la fecha que imprime el DVR.

- **Hallazgo 1 confirmado:** con el modelo anterior, la cámara prácticamente no ve personas (0,05 por cuadro en A, 0 en B). `epp_v3_completo` no generaliza a personas lejanas y parcialmente ocultas.
- **Dos etapas:** YOLO11s (COCO) multiplica por 12 las personas detectadas en A.
- **SAHI:** suma un 16–27 % en A, sobre todo personas de menos de 60 px (56 → 199). Los mosaicos de 320 ampliados a 640 equivalen a mosaicos de 640 sobre el main stream de doble resolución. En B no suma: la persona no está lejos sino oculta tras la columna.
- **Falsos positivos:** con confianza baja aparecían cajas diminutas y detecciones sobre la fecha del DVR. Por eso:
  - SAHI descarta cajas de menos de 16 px;
  - la ROI excluye la franja superior;
  - `PERSON_CONF` es 0,2: ByteTrack solo inicia un track desde 0,25, y las detecciones más débiles solo mantienen tracks existentes.
- **EPP:** ninguna persona de estas grabaciones llega a 150 px de alto, así que el EPP casi no se reconoce (≤1 %). Con 960×480 esto es esperable; el paso 3 debe fijar `minPersonHeightPx` según la resolución real del main stream.

## Paso 3: filtros y robustez (grabaciones de Camara 1)

| Configuración (video A) | Personas por cuadro | Menores de 60 px | Detección + tracking (media / p95) | Cuadros/s |
|---|---|---|---|---|
| [5. SAHI 320 + CLAHE en la ROI](2026-11-17_2145_cam1-a-5-sahi-320-clahe_desktop-gl450ej.md) | **0,92** | 434 | 26,3 / 38,9 ms | 23,0 |
| [6. SAHI 320 + gamma 0,7](2026-11-17_2146_cam1-a-6-sahi-320-gamma_desktop-gl450ej.md) | 0,78 | 298 | 23,9 / 34,9 ms | 24,5 |
| [7. Paso 3 completo](2026-11-17_2150_cam1-a-7-paso3_desktop-gl450ej.md) | 0,93 | 436 | 28,4 / 39,7 ms | 21,6 |
| [7. Paso 3 completo, video B](2026-11-17_2151_cam1-b-7-paso3_desktop-gl450ej.md) | 0,17 | 133 | 26,2 / 37,7 ms | 23,1 |

El paso 3 completo combina: dos etapas, SAHI 320→640 en la ROI `[0, 0.06, 1, 0.5]`, CLAHE, EPP solo desde 150 px, buffer de 4 s y conteo 3 de 5.

- **CLAHE** sube las personas detectadas en el fondo de A de 0,79 a 0,92 por cuadro, con menos de 60 px: 199 → 434. Gamma no aporta en esta escena. Falta revisar a ojo si CLAHE agrega falsos positivos; el conteo suavizado y el umbral de 0,25 de ByteTrack los limitan.
- **La detección baja de ~34 a ~27 ms** respecto de la configuración 4: ya no se buscan EPP en personas de menos de 150 px, que en estas grabaciones son todas.
- **EPP:** con `MIN_PERSON_HEIGHT_PX=90` ([8](2026-11-17_2153_cam1-a-8-paso3-epp90_desktop-gl450ej.md)) tampoco aparece EPP: el modelo `epp_v3_completo` no reconoce el EPP de estos trabajadores ni sobre recortes a resolución completa. Confirma el hallazgo 2; lo resuelve el reentrenamiento del paso 8.
- **Resumen Camara 1, video A:** de 0,05 personas por cuadro (modelo anterior) a 0,93 (paso 3), con un costo de detección de 17 → 28 ms en la RTX 5070.

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
