from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
import numpy as np

from .audio import load_audio_mono
from .preprocess import preprocess_audio, PreprocessConfig
from .embeddings import EmbeddingExtractor, EmbeddingConfig
from .model import SimpleSpeakerId
from .storage import Storage

app = FastAPI(title="Speaker ID API")


def build_components(db: str, sample_rate: int, backend: str, no_preprocess: bool):
    storage = Storage(Path(db))
    storage.ensure()
    extractor = EmbeddingExtractor(EmbeddingConfig(sample_rate=sample_rate, backend=backend))
    preproc_cfg = PreprocessConfig(sample_rate=sample_rate)
    return storage, extractor, preproc_cfg, no_preprocess


@app.post("/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...), db: str = Form(str(Path.home() / ".speaker_id_db")), sample_rate: int = Form(16000), backend: str = Form("mfcc"), no_preprocess: bool = Form(False)):
    storage, extractor, preproc_cfg, no_preprocess_flag = build_components(db, sample_rate, backend, no_preprocess)
    data = await file.read()
    tmp = Path("/tmp/uploaded_audio")
    tmp.write_bytes(data)
    audio, _ = load_audio_mono(tmp, target_sr=sample_rate)
    if not no_preprocess_flag:
        audio = preprocess_audio(audio, preproc_cfg)
    emb = extractor.extract(audio)
    storage.add(name, emb)
    return {"status": "ok", "name": name}


@app.post("/identify")
async def identify(file: UploadFile = File(...), db: str = Form(str(Path.home() / ".speaker_id_db")), sample_rate: int = Form(16000), backend: str = Form("mfcc"), threshold: float = Form(0.55), no_preprocess: bool = Form(False)):
    storage, extractor, preproc_cfg, no_preprocess_flag = build_components(db, sample_rate, backend, no_preprocess)
    enrollments = storage.all_enrollments()
    if not enrollments:
        return JSONResponse(status_code=400, content={"error": "no_enrollments"})
    data = await file.read()
    tmp = Path("/tmp/uploaded_audio")
    tmp.write_bytes(data)
    audio, _ = load_audio_mono(tmp, target_sr=sample_rate)
    if not no_preprocess_flag:
        audio = preprocess_audio(audio, preproc_cfg)
    emb = extractor.extract(audio)
    model = SimpleSpeakerId(threshold=threshold)
    name, score, scores = model.identify(emb, enrollments)
    return {"name": name, "score": score, "scores": scores}


@app.get("/")
async def index() -> HTMLResponse:
    html = """
    <html>
    <head><title>Speaker ID</title></head>
    <body>
      <h1>Speaker ID</h1>
      <h2>Enroll</h2>
      <form id="enrollForm" enctype="multipart/form-data">
        Name: <input name="name" />
        File: <input type="file" name="file" />
        <button type="submit">Enroll</button>
      </form>
      <pre id="enrollOut"></pre>
      <h2>Identify</h2>
      <form id="identifyForm" enctype="multipart/form-data">
        File: <input type="file" name="file" />
        <button type="submit">Identify</button>
      </form>
      <pre id="identifyOut"></pre>
      <script>
      document.getElementById('enrollForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const fd = new FormData(e.target);
        const res = await fetch('/enroll', { method: 'POST', body: fd });
        document.getElementById('enrollOut').textContent = await res.text();
      });
      document.getElementById('identifyForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const fd = new FormData(e.target);
        const res = await fetch('/identify', { method: 'POST', body: fd });
        document.getElementById('identifyOut').textContent = await res.text();
      });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
