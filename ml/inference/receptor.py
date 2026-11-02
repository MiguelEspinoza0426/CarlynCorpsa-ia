import os, time
from pathlib import Path
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
import uvicorn

TOKEN = os.environ["TOKEN"]
SALIDA = Path("ml/datasets/empresa_raw")
app = FastAPI()

@app.post("/frame/{camara}")
async def frame(camara: str, file: UploadFile = File(...), x_token: str = Header(default="")):
    if x_token != TOKEN:
        raise HTTPException(401)
    if not camara.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400)
    carpeta = SALIDA / camara
    carpeta.mkdir(parents=True, exist_ok=True)
    t = time.time()
    nombre = time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"_{int(t * 1000) % 1000:03d}.jpg"
    (carpeta / nombre).write_bytes(await file.read())
    return {"ok": True}

if __name__ == "__main__":
    # BIND = tu IP de Tailscale, así solo se alcanza por la VPN
    uvicorn.run(app, host=os.environ["BIND"], port=8000)