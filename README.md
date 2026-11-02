# CarlynCorpsa-ia · Visión artificial

Proyecto de inteligencia artificial de la plataforma de CARLYNCORPSA S.A. Es independiente del back end y del front end:
se comunica con la API (**CarlynCorpsa-be**) solo por HTTP, según el [contrato de integración](docs/contrato-integracion.md).

| Carpeta | Contenido |
|---|---|
| [`inference/`](inference/README.md) | Servicio de inferencia: YOLO + ByteTrack + postura, motor de reglas por zona, pixelado de rostros, video MJPEG y envío de eventos/métricas a la API. |
| `ml/training/` | Pipeline del modelo de EPP: fusión de datasets, entrenamiento, evaluación y comparación de versiones. |
| `ml/inference/` | Scripts de prueba de cámaras y captura de frames usados durante el entrenamiento. |
| `infra/empresa/` | Configuración de MediaMTX para reemitir las cámaras del DVR de la empresa por RTSP. |

Los datasets (`ml/datasets`), las corridas de entrenamiento (`ml/runs`) y los pesos (`*.pt`) no se versionan.
El modelo vigente es `ml/runs/epp_v3_completo/weights/best.pt` (YOLO11m, 8 clases); se distribuye aparte.

## Inicio rápido

```bash
python -m venv C:\venvs\carlyn
C:\venvs\carlyn\Scripts\pip install -r inference/requirements.txt
```

```bash
cd inference
set INGEST_API_KEY=dev-ingest-key-change-me-0123456789
set INTERNAL_KEY=dev-internal-key
C:\venvs\carlyn\Scripts\python -m carlyn_inference
```

La API debe estar en `http://localhost:5000` (o en `API_BASE_URL`). Todas las variables están en [inference/README.md](inference/README.md).

## Pruebas

```bash
cd inference
C:\venvs\carlyn\Scripts\python -m pytest -q
```
