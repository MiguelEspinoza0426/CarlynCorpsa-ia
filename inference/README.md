# Servicio de inferencia – CARLYNCORPSA

Servicio Python que analiza las cámaras con YOLO y envía eventos y métricas a la API .NET. Puede ejecutarse en otra PC con GPU (por ejemplo, la PC de la empresa junto al DVR).

```
Cámaras / DVR ──RTSP──► CameraReader ──► Detector (YOLO + ByteTrack + postura) ──► RuleEngine (zonas y reglas)
                                                                                      │
API .NET ◄── eventos + evidencias (rostros pixelados), métricas, heartbeat ◄── Outbox (reintentos)
API .NET ──► /cameras/{id}/snapshot.jpg, /cameras/{id}/stream.mjpeg, /config/reload
```

## Qué detecta

| Regla | Cómo |
|---|---|
| Incumplimiento de EPP | El EPP se asigna a la persona que lo contiene. Un elemento falta si no aparece en ≥ 70 % de los frames de la ventana (`minDurationSec`), para tolerar oclusiones. |
| Ingreso a zona restringida | Punto de apoyo (centro inferior de la caja) dentro del polígono durante `minDurationSec`. |
| Permanencia en zona de peligro | Estancia continua ≥ `MaxDwellSeconds`. |
| Proximidad | Distancia al polígono ≤ `ProximityDistance` (fracción del ancho de la imagen). |
| Aforo excedido | Personas en la zona ≥ `MaxOccupancy` durante `minDurationSec`. |
| Inactividad | Sin desplazamiento (> 2 % de la imagen) durante `InactivitySeconds`. |
| Posible accidente | Persona más ancha que alta **y** torso a ≥ 60° de la vertical (modelo de postura) durante ≥ 2 s. Siempre requiere validación humana en la plataforma. |
| Condición insegura | Objeto estático en zona de circulación (solo si se configura `OBJECT_MODEL`, p. ej. `yolo11n.pt`; clases COCO). |

**Métricas anónimas:**
- ocupación por zona, en ventanas de 10 s;
- sesiones de permanencia y desplazamientos entre zonas;
- intervalos de inactividad.

Los IDs de track son temporales: **no hay reconocimiento facial ni identidades**.

**Privacidad:** los rostros se **pixelan** en evidencias, snapshots y el video en vivo. Se usan los puntos de la cara del modelo de postura y, si no hay, la parte superior de la caja de la persona.

## Instalación

```bash
python -m venv C:\venvs\carlyn
C:\venvs\carlyn\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
C:\venvs\carlyn\Scripts\pip install -r requirements-dev.txt
```

## Configuración (variables de entorno)

| Variable | Descripción | Por defecto |
|---|---|---|
| `API_BASE_URL` | URL de la API .NET | `http://localhost:5000` |
| `INGEST_API_KEY` | Clave del nodo (debe coincidir con `Ingest:Nodes` de la API) | **obligatoria** |
| `INTERNAL_KEY` | Clave que la API envía en `X-Internal-Key` (`Inference:InternalKey`) | vacío (sin control) |
| `MODEL_PATH` | Pesos YOLO de EPP + persona | `../ml/runs/epp_v3_completo/weights/best.pt` |
| `POSE_MODEL` | Modelo de postura (caídas y pixelado de rostros); vacío lo desactiva | `yolo11n-pose.pt` |
| `OBJECT_MODEL` | Modelo COCO para condiciones inseguras; vacío lo desactiva | vacío |
| `DEVICE` | `auto`, `0` (GPU) o `cpu` | `auto` |
| `IMGSZ`, `CONF` | Resolución y confianza mínima de detección | `640`, `0.25` |
| `PORT` | Puerto HTTP del servicio | `8100` |
| `<REF>_USER`, `<REF>_PASS` | Credenciales de las cámaras con `credentialsRef = REF` (p. ej. `CAM_DVR1_USER`) | — |
| `SOURCE_OVERRIDES` | JSON `{ "nombre o id de cámara": "fuente" }` para pruebas sin el DVR (video, `0` = webcam) | `{}` |

## Ejecución

```bash
cd inference
set INGEST_API_KEY=dev-ingest-key-change-me-0123456789
set INTERNAL_KEY=dev-internal-key
C:\venvs\carlyn\Scripts\python -m carlyn_inference
```

El servicio descarga su configuración de la API (cámaras activas, zonas y reglas) y la vuelve a consultar cada 30 s. También la recarga al instante cuando la API llama a `/config/reload`.

**Probar sin el DVR** usando un video de la empresa:

```bash
set SOURCE_OVERRIDES={"Soldadura - Nave 1": "C:/ruta/video.mp4"}
```

## Benchmark del equipo (HU-35)

```bash
C:\venvs\carlyn\Scripts\python -m carlyn_inference.benchmark --source C:/ruta/video.mp4 --imgsz 640 --target-fps 5
```

Indica el tiempo por frame y cuántas cámaras soporta el equipo a esos FPS. Medido en la RTX 5070 de desarrollo con YOLO11m a 640 px:
- **~33 ms por frame** con tracking (22 ms de GPU + tracking en CPU);
- unas **4 cámaras a 5 FPS**.

En la GTX 1650 de la empresa conviene un modelo YOLO11s (`ml/training/train.py --modelo yolo11s.pt`) y 3–5 FPS por cámara.

## Tests

```bash
C:\venvs\carlyn\Scripts\python -m pytest -q
```

Cubren el motor de reglas (sin GPU), la asociación EPP–persona, el ángulo del torso, el pixelado, el cliente de la API con reintentos, el orquestador y el servidor HTTP.
