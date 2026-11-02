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

## Motor de reglas (`carlyn_inference/rules/`)

Cada regla es una clase. `RuleEngine` no conoce ninguna regla concreta: solo coordina.
- Actualiza los tracks y la pertenencia a zonas.
- Ejecuta las reglas registradas en una tabla indexada por `EventType`.
- Acumula las métricas (ocupación, permanencia e inactividad).

| Archivo | Contenido |
|---|---|
| `models.py` | `EventType`, observaciones (`PersonObs`, `ObjectObs`), `EventOut` y métricas |
| `state.py` | `EngineState`: tracks, presencias por zona, objetos, *cooldowns* y búferes de métricas, compartido por todas las reglas |
| `base.py` | Las cuatro interfaces de regla, según su alcance |
| `epp.py`, `restricted.py`, `dwell.py`, `inactivity.py`, `proximity.py`, `fall.py`, `occupancy.py`, `unsafe.py` | Una regla por archivo |
| `registry.py` | `default_rules()`: la tabla `EventType → regla` |
| `engine.py` | `RuleEngine` |

| Interfaz | Se evalúa | Ejemplos |
|---|---|---|
| `PersonZoneRule` | por persona y zona (dentro o fuera; `track.zones` lo indica) | EPP, ingreso restringido, permanencia, inactividad, proximidad |
| `PersonRule` | por persona | posible accidente |
| `ZoneRule` | por zona, con el número de personas dentro | aforo |
| `FrameRule` | una vez por frame, con los objetos | condición insegura |

### Cómo agregar una regla

1. Agrega el tipo a `EventType` en `models.py`. Debe coincidir con el `type` de la regla configurada en la API.
2. Crea `rules/<nombre>.py` con una clase que herede de la interfaz adecuada:
   - define `event_type`;
   - implementa `evaluate(...)`;
   - lee la configuración con `zone.rule(self.event_type)` y `zone.threshold(...)`;
   - usa `state.cooldown_ok(rule, clave, now)` antes de emitir;
   - guarda el estado propio en `EngineState` (o en `Presence`/`Track` si es por persona).
3. Regístrala en `default_rules()` de `registry.py`. El orden de la tabla solo define el orden de los eventos dentro de un frame.
4. Agrega un test en `tests/test_rule_strategies.py` que evalúe la clase sin el motor.

`RuleEngine`, `worker.py` y las demás reglas no cambian. Para pruebas también se puede pasar una tabla propia: `RuleEngine(camera, {**default_rules(), "MiRegla": MiRegla()})`.

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

En PowerShell, desde la carpeta `inference`:

```powershell
$env:INGEST_API_KEY = "dev-ingest-key-change-me-0123456789"
$env:INTERNAL_KEY = "dev-internal-key"
C:\venvs\carlyn\Scripts\python -m carlyn_inference
```

En PowerShell `set` no define variables de entorno: hay que usar `$env:NOMBRE = "valor"`. Las variables solo valen para esa ventana.

El servicio descarga su configuración de la API (cámaras activas, zonas y reglas) y la vuelve a consultar cada 30 s. También la recarga al instante cuando la API llama a `/config/reload`.

## Fuentes de video

La fuente de cada cámara se elige en la plataforma (**Parámetros → Cámaras → Nueva cámara → Fuente de video**), sin tocar este servicio:

| Opción del formulario | `streamUrl` | Qué captura |
|---|---|---|
| Cámara IP / DVR | `rtsp://192.168.100.14:554/Streaming/Channels/102` | El canal del DVR. Usuario y clave se leen de `<REF>_USER` / `<REF>_PASS` en este equipo |
| Ventana (Hik-Connect / iVMS) | `window:Hik-Connect:0,0,960,540` | Una ventana del escritorio de este equipo. El formulario lista las ventanas abiertas y muestra una vista previa para arrastrar la región de una cámara de la cuadrícula |
| Cámara USB | `webcam:0` | Una cámara conectada a este equipo (0 = la primera) |

Al guardar, el servicio abre la nueva fuente en segundos. Si se cambia la fuente de una cámara, la reabre sin reiniciarse.

**Captura de ventanas en el PC de la empresa.** Se captura lo que se ve en pantalla, así que:
- Hik-Connect o iVMS-4200 debe estar abierto con la vista en vivo, **sin minimizar ni tapar**. Conviene un monitor (o escritorio) dedicado.
- La sesión de Windows debe estar **iniciada y desbloqueada**. Desactiva el bloqueo de pantalla y la suspensión, y no cierres la sesión de Escritorio remoto: con la pantalla bloqueada no hay imagen.
- Este servicio debe ejecutarse en esa misma sesión de usuario, no como servicio de Windows, que no ve el escritorio.
- No cambies el diseño de la cuadrícula ni el tamaño de la ventana después de marcar las regiones; si cambian, vuelve a marcarlas.
- La región está en píxeles del área interna de la ventana, así que funciona aunque la ventana se mueva de sitio.

La «Cámara USB» y la «Ventana» siempre son del **equipo donde corre este servicio**, no del navegador del usuario. Así funciona aunque la plataforma se abra desde un dominio en otro equipo.

**Pruebas locales.** `SOURCE_OVERRIDES` reemplaza la fuente sin tocar la plataforma, por ejemplo con un video de la empresa:

```powershell
$env:SOURCE_OVERRIDES = '{"Soldadura - Nave 1": "C:/ruta/video.mp4"}'
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
