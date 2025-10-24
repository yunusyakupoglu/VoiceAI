from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pathlib import Path
import numpy as np
import json
from typing import AsyncIterator

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


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.post("/monitor/enroll")
async def monitor_enroll(name: str = Form(...), file: UploadFile = File(...), db: str = Form(str(Path.home() / ".speaker_id_db")), sample_rate: int = Form(16000), backend: str = Form("mfcc"), no_preprocess: bool = Form(False)):
    async def gen() -> AsyncIterator[str]:
        yield _sse({"step": "received", "filename": file.filename})
        data = await file.read()
        yield _sse({"step": "bytes", "n": len(data)})
        tmp = Path("/tmp/uploaded_audio")
        tmp.write_bytes(data)
        yield _sse({"step": "decode_start"})
        audio, _ = load_audio_mono(tmp, target_sr=sample_rate)
        yield _sse({"step": "decode_done", "samples": int(audio.size)})
        if not no_preprocess:
            yield _sse({"step": "preprocess_start"})
            audio_p = preprocess_audio(audio, PreprocessConfig(sample_rate=sample_rate))
            yield _sse({"step": "preprocess_done", "samples": int(audio_p.size)})
        else:
            audio_p = audio
        yield _sse({"step": "embed_start", "backend": backend})
        extractor = EmbeddingExtractor(EmbeddingConfig(sample_rate=sample_rate, backend=backend))
        emb = extractor.extract(audio_p)
        yield _sse({"step": "embed_done", "dim": int(emb.size)})
        storage = Storage(Path(db))
        storage.ensure()
        yield _sse({"step": "store_start", "name": name})
        storage.add(name, emb)
        yield _sse({"step": "store_done", "name": name})
        yield _sse({"step": "complete", "status": "ok", "name": name})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/monitor/identify")
async def monitor_identify(file: UploadFile = File(...), db: str = Form(str(Path.home() / ".speaker_id_db")), sample_rate: int = Form(16000), backend: str = Form("mfcc"), threshold: float = Form(0.55), top_k: int = Form(3), no_preprocess: bool = Form(False)):
    async def gen() -> AsyncIterator[str]:
        yield _sse({"step": "received", "filename": file.filename})
        data = await file.read()
        yield _sse({"step": "bytes", "n": len(data)})
        tmp = Path("/tmp/uploaded_audio")
        tmp.write_bytes(data)
        storage = Storage(Path(db))
        enrollments = storage.all_enrollments()
        if not enrollments:
            yield _sse({"step": "error", "error": "no_enrollments"})
            return
        yield _sse({"step": "decode_start"})
        audio, _ = load_audio_mono(tmp, target_sr=sample_rate)
        yield _sse({"step": "decode_done", "samples": int(audio.size)})
        if not no_preprocess:
            yield _sse({"step": "preprocess_start"})
            audio_p = preprocess_audio(audio, PreprocessConfig(sample_rate=sample_rate))
            yield _sse({"step": "preprocess_done", "samples": int(audio_p.size)})
        else:
            audio_p = audio
        yield _sse({"step": "embed_start", "backend": backend})
        extractor = EmbeddingExtractor(EmbeddingConfig(sample_rate=sample_rate, backend=backend))
        emb = extractor.extract(audio_p)
        yield _sse({"step": "embed_done", "dim": int(emb.size)})
        yield _sse({"step": "match_start", "threshold": threshold})
        model = SimpleSpeakerId(threshold=threshold)
        name, score, scores = model.identify(emb, enrollments)
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: top_k or None]
        yield _sse({"step": "match_done", "name": name, "score": score, "top": sorted_scores})
        yield _sse({"step": "complete", "status": "ok", "name": name, "score": score})

    return StreamingResponse(gen(), media_type="text/event-stream")


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
      <h2>Monitoring (SSE)</h2>
      <h3>Enroll (stream)</h3>
      <form id="enrollMonForm" enctype="multipart/form-data">
        Name: <input name="name" />
        File: <input type="file" name="file" />
        <button type="submit">Enroll with monitoring</button>
      </form>
      <pre id="enrollMonOut"></pre>
      <h3>Identify (stream)</h3>
      <form id="identifyMonForm" enctype="multipart/form-data">
        File: <input type="file" name="file" />
        <button type="submit">Identify with monitoring</button>
      </form>
      <pre id="identifyMonOut"></pre>
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
      async function streamToPre(url, form, preId) {
        const pre = document.getElementById(preId);
        pre.textContent = '';
        const res = await fetch(url, { method: 'POST', body: new FormData(form) });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          pre.textContent += decoder.decode(value, { stream: true });
          pre.scrollTop = pre.scrollHeight;
        }
      }
      document.getElementById('enrollMonForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await streamToPre('/monitor/enroll', e.target, 'enrollMonOut');
      });
      document.getElementById('identifyMonForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await streamToPre('/monitor/identify', e.target, 'identifyMonOut');
      });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
