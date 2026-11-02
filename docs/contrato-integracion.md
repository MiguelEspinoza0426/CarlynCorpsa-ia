# Contrato de integración Python ↔ API .NET (definitivo)

> Este documento reemplaza la sección 4 de `fase-0-exploracion-y-plan.md`. La fuente de verdad es el documento OpenAPI (`/openapi/v1.json`, etiqueta **Ingest**).

## 1. Autenticación

| Dirección | Mecanismo | Configuración |
|---|---|---|
| Python → API (`/api/ingest/*`) | Header `X-Api-Key` | API: `Ingest:Nodes` = lista de `{ NodeId, ApiKey }` (clave de 16+ caracteres). Variables: `Ingest__Nodes__0__NodeId`, `Ingest__Nodes__0__ApiKey`. |
| API → Python (snapshot, recarga) | Header `X-Internal-Key` | API: `Inference:BaseUrl`, `Inference:InternalKey`. |

- Un token JWT de usuario **no** sirve en `/api/ingest`, y la API key **no** sirve en los demás endpoints.
- El `nodeId` se toma de la API key; no se envía en los cuerpos.
- Fechas en UTC (ISO-8601). Coordenadas normalizadas 0–1. Enums como texto (`"EppMissing"`).

## 2. Endpoints de ingesta

| Método | Ruta | Respuesta |
|---|---|---|
| GET | `/api/ingest/config` | `200` + header `ETag`; `304` si `If-None-Match` coincide |
| POST | `/api/ingest/events` | `200` `{ accepted, duplicates, rejected[{clientEventId, reason}] }` |
| PUT | `/api/ingest/events/{clientEventId}/evidence` | `204`; multipart con `snapshot` (JPEG/PNG ≤ 2 MB) y/o `clip` (MP4 ≤ 20 MB) |
| POST | `/api/ingest/metrics/occupancy` | `200` `{ accepted, rejected[] }` |
| POST | `/api/ingest/metrics/dwell` | `200` `{ accepted, rejected[] }` |
| POST | `/api/ingest/metrics/inactivity` | `200` `{ accepted, rejected[] }` |
| POST | `/api/ingest/heartbeat` | `204` |

### 2.1 `GET /api/ingest/config`
```json
{
  "version": "3f9a1c0d2b7e4a51",
  "nodeId": "gpu-01",
  "model": { "eppClasses": ["calzado","careta_soldar","guantes","lentes","mandil","protector_facial","traje"], "personClass": "persona" },
  "cameras": [
    {
      "id": "01a1...", "name": "Soldadura - Nave 1",
      "streamUrl": "rtsp://192.168.100.14:554/Streaming/Channels/102",
      "credentialsRef": "CAM_DVR1", "targetFps": 5,
      "zones": [
        {
          "id": "01a2...", "name": "Área de soldadura", "type": "Work",
          "polygon": [[0.1,0.25],[0.6,0.25],[0.6,0.95],[0.1,0.95]],
          "requiredEpp": ["calzado","careta_soldar","guantes","mandil"],
          "rules": [
            { "id": "01a3...", "type": "EppMissing", "minDurationSec": 3, "cooldownSec": 60, "minConfidence": 0.45, "severity": "High" },
            { "id": "01a4...", "type": "OccupancyExceeded", "minDurationSec": 10, "cooldownSec": 120, "minConfidence": 0.4, "severity": "Medium" }
          ],
          "thresholds": [
            { "metric": "MaxOccupancy", "warningValue": 3, "criticalValue": 4 },
            { "metric": "InactivitySeconds", "warningValue": null, "criticalValue": 300 }
          ]
        }
      ]
    }
  ]
}
```
- Solo incluye cámaras, zonas y reglas **activas**.
- `version` es un hash del contenido: cambia con cualquier cambio de configuración, incluidas las eliminaciones.
- Los parámetros numéricos de las reglas vienen en `thresholds`: `MaxOccupancy`, `MaxDwellSeconds`, `InactivitySeconds` y `ProximityDistance` (fracción del ancho de la imagen).
- Las credenciales de las cámaras **no** viajan: el nodo las lee de `CAM_DVR1_USER` y `CAM_DVR1_PASS`.
- `streamUrl` indica la fuente de video. El administrador la elige en el formulario de la cámara:

  | Formato | Fuente | `credentialsRef` |
  |---|---|---|
  | `rtsp://…`, `rtsps://…`, `http://…`, `https://…` | Cámara IP o canal del DVR, sin usuario ni clave en la URL | Sí |
  | `webcam:<n>` | Cámara conectada al equipo de inferencia (`0` = la primera) | No |
  | `window:<título>[:x,y,ancho,alto]` | Ventana del escritorio del equipo de inferencia, p. ej. la vista en vivo de Hik-Connect o iVMS-4200. Basta con parte del título. La región, en píxeles del área interna de la ventana, recorta una cámara de la cuadrícula; sin región se usa la ventana completa | No |

  Si cambia `streamUrl` o `targetFps`, el nodo reabre la fuente sin reiniciarse.
- La API llama a `POST {Inference:BaseUrl}/config/reload` al guardar cambios. Aun así, el nodo debe consultar `config` periódicamente (cada ~30 s, con `If-None-Match`).

### 2.2 `POST /api/ingest/events`
```json
{
  "events": [
    {
      "clientEventId": "6c1e8f9a-2d4b-4b1e-9b8e-1f2a3c4d5e6f",
      "type": "EppMissing",
      "cameraId": "01a1...", "zoneId": "01a2...", "ruleId": "01a3...",
      "startedAt": "2026-11-01T15:04:05.120Z", "endedAt": "2026-11-01T15:04:09.480Z",
      "confidence": 0.82, "personCount": 2, "trackId": 17,
      "bbox": [0.41, 0.33, 0.12, 0.40],
      "details": { "missingEpp": ["guantes"], "presentEpp": ["careta_soldar","mandil","calzado"] },
      "modelVersion": "epp_v3_completo"
    }
  ]
}
```
- **Máximo 100 eventos por lote.**
- **Idempotente:** un `clientEventId` repetido se devuelve en `duplicates`, así que el nodo puede reintentar con seguridad.
- **Rechazos por evento** (no del lote completo), con motivo:
  - cámara desconocida;
  - zona que no pertenece a la cámara;
  - regla que no pertenece a la zona o cuyo tipo no coincide;
  - `startedAt` más de 5 minutos en el futuro;
  - `bbox` que no tiene 4 valores;
  - `details` que no es un objeto.
- **Severidad:** la de la regla, o la predeterminada del tipo si no hay regla.
- **Validación:** los `PossibleAccident` quedan `PendingValidation`.
- Cada evento aceptado dispara `alertRaised` por SignalR.

**`details` sugerido por tipo:**

| Tipo | `details` |
|---|---|
| `EppMissing` | `{ "missingEpp": [..], "presentEpp": [..] }` |
| `PossibleAccident` | `{ "kind": "PersonOnGround", "durationSec": 4.2, "torsoAngleDeg": 78 }` |
| `OccupancyExceeded` | `{ "count": 5, "threshold": 4 }` |
| `UnsafeCondition` | `{ "objectClass": "box", "durationSec": 45 }` |
| `Proximity` | `{ "distance": 0.03 }` |
| `RestrictedZoneEntry`, `DangerZoneDwell`, `Inactivity` | `{ "durationSec": 25 }` |

### 2.3 Métricas
```json
{ "samples":   [ { "cameraId": "..", "zoneId": "..", "windowStart": "2026-11-01T15:04:00Z", "windowSec": 10, "avgCount": 2.4, "maxCount": 3, "minCount": 2 } ] }
{ "sessions":  [ { "cameraId": "..", "zoneId": "..", "trackId": 17, "enteredAt": "..", "exitedAt": "..", "fromZoneId": null } ] }
{ "intervals": [ { "cameraId": "..", "zoneId": "..", "trackId": 17, "startedAt": "..", "endedAt": ".." } ] }
```
- **Máximo 1000 elementos por lote.**
- Los elementos cuya zona no pertenece a la cámara se rechazan uno por uno.
- Cada lote de ocupación emite `occupancyUpdated` (la muestra más reciente por zona).

### 2.4 `POST /api/ingest/heartbeat` (cada ~15 s)
```json
{
  "sentAt": "2026-11-01T15:04:15Z",
  "modelVersion": "epp_v3_completo",
  "cameras": [ { "cameraId": "..", "status": "Online", "fps": 9.8, "latencyMs": 140, "lastFrameAt": "..", "error": null } ]
}
```
- `status` ∈ `Online`, `Reconnecting`, `Offline`.
- La API usa **su propia hora** como hora del último heartbeat.
- Sin heartbeat en `Ingest:HeartbeatTimeoutSeconds` (60 s), la cámara pasa a `Offline` y se emite `cameraStatusChanged`.

## 3. Endpoints del servicio Python (consumidos por la API)

| Método | Ruta | Uso |
|---|---|---|
| GET | `/health` | Estado del servicio, GPU y versión del modelo |
| GET | `/cameras/{cameraId}/snapshot.jpg` | Frame actual sin anotaciones (editor de zonas). `404` si la cámara no está activa |
| GET | `/cameras/{cameraId}/stream.mjpeg` | Stream con detecciones dibujadas (Fase 5/6, vía proxy de la API) |
| GET | `/sources/windows` | Ventanas visibles del equipo de inferencia que pueden usarse como fuente: `[{ "title", "width", "height" }]` |
| GET | `/sources/window-preview.jpg?title=…` | Imagen actual de una ventana a tamaño real, para elegir la región. `404` si no está visible. No se almacena |
| POST | `/config/reload` | Recarga inmediata de la configuración |

Todos exigen `X-Internal-Key`.

## 4. Tiempo real – SignalR `/hubs/alerts`

- **Autenticación:** JWT de usuario, en el header o en la query string `access_token`.
- **Métodos del cliente:** `JoinCamera(cameraId)` y `LeaveCamera(cameraId)`.

| Mensaje | Contenido |
|---|---|
| `alertRaised` | `{ eventId, type, severity, cameraId, cameraName, zoneId, zoneName, occurredAt, confidence, personCount, validationStatus }` |
| `occupancyUpdated` | `{ cameraId, zoneId, count, warningThreshold, criticalThreshold, at }` |
| `cameraStatusChanged` | `{ cameraId, status, at }` |
| `eventValidated` | `{ eventId, result, validatedBy, at }` |
