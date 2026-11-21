# Entrenamiento del modelo EPP

Scripts para pasar de grabaciones de las cámaras a un modelo nuevo. Se ejecutan desde la raíz del repositorio con el entorno `C:\venvs\carlyn`.

| Script | Para qué |
|---|---|
| `capturar_cuadros.py` | Saca cuadros de videos o del RTSP del DVR y, opcionalmente, los pre-etiqueta |
| `merge_datasets.py` | Fusiona todos los datasets de `ml/datasets` con las 8 clases unificadas |
| `train.py` | Entrena (fine-tuning) sobre el dataset fusionado |
| `comparar.py` / `evaluate.py` | Comparan modelos sobre el split de test |

Clases unificadas: `careta_soldar`, `guantes`, `calzado`, `mandil`, `traje`, `protector_facial`, `lentes`, `persona`.

## 1. Capturar cuadros

Use el **stream principal** del DVR (resolución completa), no videos reenviados por WhatsApp. A 960×480 las personas del fondo miden menos de 150 px y su EPP no se distingue ni etiquetando a mano.

```bash
# Del DVR en vivo: 10 minutos, un cuadro cada 2 s, con etiquetas sugeridas
python ml/training/capturar_cuadros.py --camara camara1 --fuente "rtsp://USUARIO:CLAVE@IP:554/..." --duracion 600 --pre-etiquetar

# De un video grabado: indique la hora del DVR al inicio para que el nombre tenga la fecha real
python ml/training/capturar_cuadros.py --camara camara1 --fuente "videos/grabacion.mp4" --inicio "2026-11-17 17:51:36" --pre-etiquetar
```

El script:
- **Descarta los cuadros sin señal** (negro, color uniforme o logo del DVR).
- **Descarta las escenas quietas**: guarda un cuadro solo si cambió al menos el `--cambio-minimo` % de los píxeles (0,1 % por defecto) respecto del último guardado.
- **Nombra cada cuadro** con cámara, día y hora: `camara1_20261117-175136_00012.jpg`.
- **No pisa capturas previas**: si la carpeta ya tiene cuadros, sigue la numeración.
- **No imprime la clave** de la URL RTSP.

La salida va a `ml/datasets/captura_<camara>/`, con las carpetas `images/` y `labels/` y el archivo `data.yaml`. La fusión ignora las carpetas `captura_*`, porque contienen sugerencias sin revisar.

Las **sugerencias** (`--pre-etiquetar`) salen de dos modelos:
- las personas, de `yolo11s.pt` a 1280 px;
- el EPP, de `epp_v5`, buscado en el recorte de cada persona.

La confianza es baja a propósito (0,15) porque es más rápido borrar una caja que dibujarla, así que **hay que revisarlas todas**. Con las cámaras actuales casi todas las sugerencias de EPP son falsas.

## 2. Etiquetar en Roboflow

1. **Crear o abrir el proyecto** con exactamente las 8 clases unificadas, escritas igual que arriba.
2. **Subir la carpeta de captura completa** (con `images/`, `labels/` y `data.yaml`). Roboflow toma las etiquetas como anotaciones.
3. **Revisar cada imagen**:
   - Corregir y borrar las cajas sugeridas.
   - Etiquetar también las personas del fondo y las parcialmente tapadas.
   - Dibujar el EPP solo si se distingue. Una caja dudosa enseña mal al modelo.
4. **Separar por día al generar la versión**: valid y test con días o grabaciones distintos de los de train (el día está en el nombre). Si se reparten al azar, los cuadros vecinos de una misma escena terminan en train y en test, y la evaluación sale inflada.
5. **Exportar en formato YOLOv11** y descomprimir el zip en `ml/datasets/`.

Para que el modelo mejore en la cámara hacen falta varios cientos de cuadros revisados, con todas las clases que aparezcan.

## 3. Fusionar y entrenar

Si el export usa otros nombres de clase, agréguelos a `MAPEO` en `merge_datasets.py`. La fusión imprime cuántas clases usó de cada dataset: compruebe que no descarte ninguna.

```bash
python ml/training/merge_datasets.py --persona
python ml/training/train.py --modelo ml/runs/epp_v5/weights/best.pt --data epp_unificado_persona --imgsz 640 --batch 16 --epochs 60 --nombre epp_v6
```

Después compare contra el modelo anterior:
- en el test general;
- en las imágenes de la cámara;
- con `python -m carlyn_inference.benchmark` (desde `inference/`) sobre una grabación que no se haya usado para entrenar.
