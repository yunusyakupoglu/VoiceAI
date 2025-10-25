from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pathlib import Path
import numpy as np
import json
from typing import AsyncIterator
import collections

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


@app.websocket("/ws/identify")
async def ws_identify(websocket: WebSocket):
    """Live identification from microphone via WebSocket.

    Client sends binary frames of PCM int16 mono audio.
    Query params:
      - db: storage directory
      - backend: mfcc|ecapa
      - threshold: float
      - sr: client sample rate (default 48000)
      - window: seconds (default 1.5)
      - hop: seconds (default 0.5)
      - no_preprocess: '1' to disable preprocessing
      - top_k: int (default 3)
    """
    await websocket.accept()

    qp = websocket.query_params
    db = qp.get("db") or str(Path.home() / ".speaker_id_db")
    backend = qp.get("backend") or "mfcc"
    threshold = float(qp.get("threshold") or 0.55)
    client_sr = int(qp.get("sr") or 48000)
    window_s = float(qp.get("window") or 1.5)
    hop_s = float(qp.get("hop") or 0.5)
    no_pre = (qp.get("no_preprocess") == "1")
    top_k = int(qp.get("top_k") or 3)

    storage = Storage(Path(db))
    enrollments = storage.all_enrollments()
    if not enrollments:
        await websocket.send_json({"type": "error", "error": "no_enrollments"})
        await websocket.close()
        return

    extractor = EmbeddingExtractor(EmbeddingConfig(sample_rate=16000, backend=backend))
    model = SimpleSpeakerId(threshold=threshold)
    preproc_cfg = PreprocessConfig(sample_rate=16000)

    window_samples_client = int(window_s * client_sr)
    hop_samples_client = int(hop_s * client_sr)
    buf = np.zeros(0, dtype=np.float32)
    samples_since_infer = 0

    async def infer_if_ready() -> None:
        nonlocal buf, samples_since_infer
        if buf.size < window_samples_client or samples_since_infer < hop_samples_client:
            return
        segment = buf[-window_samples_client:]
        # Resample to 16k for feature pipeline
        import librosa
        seg16 = librosa.resample(segment, orig_sr=client_sr, target_sr=16000, res_type="kaiser_fast")
        if not no_pre:
            seg16 = preprocess_audio(seg16, preproc_cfg)
        emb = extractor.extract(seg16)
        name, score, scores = model.identify(emb, enrollments)
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: top_k or None]
        await websocket.send_json({"type": "result", "name": name, "score": score, "top": top})
        samples_since_infer = 0

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data_b = msg.get("bytes")
            data_t = msg.get("text")
            if data_t:
                # optional pings/commands
                if data_t == "ping":
                    await websocket.send_text("pong")
                continue
            if not data_b:
                continue
            # decode little-endian int16 -> float32 [-1,1]
            chunk = np.frombuffer(data_b, dtype=np.int16).astype(np.float32) / 32768.0
            if buf.size == 0:
                buf = chunk
            else:
                buf = np.concatenate([buf, chunk])
            # cap buffer to 5x window to avoid unbounded growth
            max_keep = max(window_samples_client * 5, window_samples_client)
            if buf.size > max_keep:
                buf = buf[-max_keep:]
            samples_since_infer += chunk.size
            await infer_if_ready()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


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
      <h2>Live (WebSocket)</h2>
      <div>
        DB: <input id="liveDb" value="/tmp/.sid_api_db" style="width:320px"/>
        Backend:
        <select id="liveBackend">
          <option value="mfcc">mfcc</option>
          <option value="ecapa">ecapa</option>
        </select>
        Threshold: <input id="liveTh" value="0.55" size="4"/>
        Window(s): <input id="liveWin" value="1.5" size="3"/>
        Hop(s): <input id="liveHop" value="0.5" size="3"/>
        <button id="liveStart">Start</button>
        <button id="liveStop">Stop</button>
      </div>
      <pre id="liveOut"></pre>
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

      // Live WebSocket mic streaming
      let ws = null;
      let audioCtx = null;
      let proc = null;
      let src = null;
      let stream = null;
      function floatTo16BitPCM(input) {
        const out = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
          let s = Math.max(-1, Math.min(1, input[i]));
          out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return out;
      }
      async function startLive() {
        if (ws) return;
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        src = audioCtx.createMediaStreamSource(stream);
        proc = audioCtx.createScriptProcessor(4096, 1, 1);
        src.connect(proc);
        proc.connect(audioCtx.destination);

        const db = encodeURIComponent(document.getElementById('liveDb').value);
        const backend = document.getElementById('liveBackend').value;
        const th = document.getElementById('liveTh').value;
        const win = document.getElementById('liveWin').value;
        const hop = document.getElementById('liveHop').value;
        const sr = audioCtx.sampleRate;
        const url = `ws://${location.host}/ws/identify?db=${db}&backend=${backend}&threshold=${th}&window=${win}&hop=${hop}&sr=${sr}`;
        ws = new WebSocket(url);
        const out = document.getElementById('liveOut');
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === 'result') {
              out.textContent = `name=${msg.name} score=${msg.score.toFixed(3)}\n` + JSON.stringify(msg.top);
            } else if (msg.type === 'error') {
              out.textContent += `\nERROR: ${msg.error}`;
            }
          } catch {}
        };
        ws.onopen = () => {
          out.textContent = `Live started (sr=${sr})...`;
        };
        ws.onclose = () => {
          out.textContent += `\nClosed.`;
        };
        proc.onaudioprocess = (e) => {
          if (!ws || ws.readyState !== WebSocket.OPEN) return;
          const buf = e.inputBuffer.getChannelData(0);
          const pcm16 = floatTo16BitPCM(buf);
          ws.send(pcm16.buffer);
        };
      }
      function stopLive() {
        if (proc) { proc.disconnect(); proc = null; }
        if (src) { src.disconnect(); src = null; }
        if (audioCtx) { audioCtx.close(); audioCtx = null; }
        if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
        if (ws) { ws.close(); ws = null; }
      }
      document.getElementById('liveStart').addEventListener('click', startLive);
      document.getElementById('liveStop').addEventListener('click', stopLive);
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
