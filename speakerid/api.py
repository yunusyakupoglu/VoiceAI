from __future__ import annotations

import json
import tempfile
from pathlib import Path
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from .embed import SpeakerEmbedder
from .store import SpeakerDatabase
from .audio import resample_waveform


APP_TITLE = "SpeakerID Web"
DEFAULT_DB = str(Path.cwd() / "data" / "speakers.json")


app = FastAPI(title=APP_TITLE)


_EMBEDDER: Optional[SpeakerEmbedder] = None


def get_embedder() -> SpeakerEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SpeakerEmbedder()
    return _EMBEDDER


def get_db(db_path: Optional[str] = None) -> SpeakerDatabase:
    # Allow overriding DB path via environment variable SPEAKERID_DB
    env_path = os.getenv("SPEAKERID_DB")
    path = db_path or env_path or DEFAULT_DB
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SpeakerDatabase(path=path)


INDEX_HTML = """
<!DOCTYPE html>
<html lang=\"tr\">
  <head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>SpeakerID Web</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #222; }
      h1 { margin-bottom: 4px; }
      h2 { margin-top: 28px; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-top: 12px; }
      .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
      label { display: inline-block; min-width: 120px; }
      button { background: #0d6efd; color: white; border: 0; padding: 8px 14px; border-radius: 6px; cursor: pointer; }
      button.secondary { background: #6c757d; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
      table { border-collapse: collapse; margin-top: 8px; }
      td, th { border: 1px solid #ddd; padding: 6px 10px; }
      .status { margin-top: 8px; }
      .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; background:#eef; }
      input, select { padding: 6px 8px; border: 1px solid #ccc; border-radius: 6px; }
    </style>
  </head>
  <body>
    <h1>SpeakerID Web</h1>
    <p>Dosyadan veya mikrofondan konuşmacı tanıma. Veri deposu: <code>data/speakers.json</code></p>

    <div class=\"card\">
      <h2>Kayıt (Dosyadan)</h2>
      <div class=\"row\">
        <label for=\"enroll-name\">İsim:</label>
        <input id=\"enroll-name\" type=\"text\" placeholder=\"Ali\" />
        <input id=\"enroll-file\" type=\"file\" accept=\"audio/*\" />
        <button id=\"enroll-btn\">Kaydet</button>
      </div>
      <div id=\"enroll-status\" class=\"status\"></div>
    </div>

    <div class=\"card\">
      <h2>Tanıma (Dosyadan)</h2>
      <div class=\"row\">
        <input id=\"identify-file\" type=\"file\" accept=\"audio/*\" />
        <button id=\"identify-btn\">Tanı</button>
      </div>
      <div id=\"identify-result\" class=\"status\"></div>
    </div>

    <div class=\"card\">
      <h2>Canlı Tanıma (Mikrofon)</h2>
      <div class=\"row\">
        <button id=\"live-start\">Başlat</button>
        <button id=\"live-stop\" class=\"secondary\" disabled>Durdur</button>
        <span id=\"live-indicator\" class=\"pill\">Hazır</span>
      </div>
      <div id=\"live-result\" class=\"status\"></div>
      <table>
        <thead><tr><th>Kişi</th><th>Skor</th></tr></thead>
        <tbody id=\"live-table\"></tbody>
      </table>
    </div>

    <script>
      // --- Enroll (file) ---
      const enrollBtn = document.getElementById('enroll-btn');
      enrollBtn.onclick = async () => {
        const name = document.getElementById('enroll-name').value.trim();
        const fileInput = document.getElementById('enroll-file');
        const statusEl = document.getElementById('enroll-status');
        if (!name || fileInput.files.length === 0) {
          statusEl.textContent = 'İsim ve dosya gerekli.';
          return;
        }
        const form = new FormData();
        form.append('name', name);
        form.append('file', fileInput.files[0]);
        statusEl.textContent = 'Yükleniyor...';
        try {
          const res = await fetch('/enroll', { method: 'POST', body: form });
          const data = await res.json();
          statusEl.textContent = data.message || JSON.stringify(data);
        } catch (e) {
          statusEl.textContent = 'Hata: ' + e;
        }
      };

      // --- Identify (file) ---
      const identifyBtn = document.getElementById('identify-btn');
      identifyBtn.onclick = async () => {
        const fileInput = document.getElementById('identify-file');
        const resultEl = document.getElementById('identify-result');
        if (fileInput.files.length === 0) {
          resultEl.textContent = 'Dosya seçiniz.';
          return;
        }
        const form = new FormData();
        form.append('file', fileInput.files[0]);
        resultEl.textContent = 'Yükleniyor...';
        try {
          const res = await fetch('/identify', { method: 'POST', body: form });
          const data = await res.json();
          resultEl.textContent = data.best ? `Tahmin: ${data.best} (skor=${data.score.toFixed(3)})` : 'Bilinmiyor';
        } catch (e) {
          resultEl.textContent = 'Hata: ' + e;
        }
      };

      // --- Live identify (mic) ---
      let audioCtx;
      let ws;
      let processor;
      let mediaStream;
      const startBtn = document.getElementById('live-start');
      const stopBtn = document.getElementById('live-stop');
      const indicator = document.getElementById('live-indicator');
      const liveTable = document.getElementById('live-table');
      const liveResult = document.getElementById('live-result');

      function updateTable(top) {
        liveTable.innerHTML = '';
        (top || []).forEach(([name, score]) => {
          const tr = document.createElement('tr');
          const td1 = document.createElement('td');
          const td2 = document.createElement('td');
          td1.textContent = name;
          td2.textContent = score.toFixed(3);
          tr.appendChild(td1); tr.appendChild(td2);
          liveTable.appendChild(tr);
        });
      }

      startBtn.onclick = async () => {
        if (ws) return;
        startBtn.disabled = true;
        indicator.textContent = 'Başlatılıyor...';
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          const source = audioCtx.createMediaStreamSource(mediaStream);
          const bufferSize = 2048; // small chunks
          processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
          const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
          ws = new WebSocket(`${wsProto}://${location.host}/ws/identify`);

          ws.onopen = () => {
            indicator.textContent = 'Dinleniyor';
            stopBtn.disabled = false;
            ws.send(JSON.stringify({ type: 'config', sr: audioCtx.sampleRate }));
            processor.onaudioprocess = (e) => {
              const input = e.inputBuffer.getChannelData(0);
              const buf = new Float32Array(input.length);
              buf.set(input);
              ws.send(buf.buffer);
            };
            source.connect(processor);
            processor.connect(audioCtx.destination);
          };

          ws.onmessage = (evt) => {
            try {
              const msg = JSON.parse(evt.data);
              if (msg.type === 'result') {
                liveResult.textContent = msg.best ? `Tahmin: ${msg.best} (skor=${msg.score.toFixed(3)})` : 'Bilinmiyor';
                updateTable(msg.top);
              } else if (msg.type === 'error') {
                liveResult.textContent = 'Hata: ' + msg.message;
              }
            } catch {}
          };

          ws.onclose = () => {
            indicator.textContent = 'Hazır';
          };
        } catch (e) {
          indicator.textContent = 'Mikrofon izni reddedildi veya hata.';
          startBtn.disabled = false;
        }
      };

      stopBtn.onclick = () => {
        if (!ws) return;
        ws.close(); ws = null;
        if (processor) { processor.disconnect(); processor.onaudioprocess = null; processor = null; }
        if (audioCtx) { audioCtx.close(); audioCtx = null; }
        if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
        stopBtn.disabled = true; startBtn.disabled = false; indicator.textContent = 'Hazır';
      };
    </script>
  </body>
  </html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


@app.post("/enroll")
async def http_enroll(name: str = Form(...), file: UploadFile = File(...)):
    embedder = get_embedder()
    db = get_db()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "audio").suffix, delete=True) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        result = embedder.compute_embedding_from_file(tmp.name)
    db.enroll(name, result.embedding)
    return JSONResponse({"ok": True, "message": f"Kayıt başarılı: {name}"})


@app.post("/identify")
async def http_identify(file: UploadFile = File(...), threshold: float = 0.6, top_k: int = 3):
    embedder = get_embedder()
    db = get_db()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "audio").suffix, delete=True) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        result = embedder.compute_embedding_from_file(tmp.name)
    name, score, ranked = db.identify(result.embedding, threshold=threshold, top_k=top_k)
    return JSONResponse({
        "ok": True,
        "best": name,
        "score": score,
        "top": ranked,
    })


@app.websocket("/ws/identify")
async def ws_identify(ws: WebSocket):
    await ws.accept()
    try:
        embedder = get_embedder()
        db = get_db()
        # Proactively inform client if there are no enrollments yet
        if not db.list_speakers():
            await ws.send_text(json.dumps({
                "type": "error",
                "message": "no_enrollments",
            }))
        sr_client: Optional[int] = None
        buffer = np.zeros(0, dtype=np.float32)
        pending = 0  # samples since last emit
        window_s = 1.5
        hop_s = 0.5
        max_buffer_s = 5.0

        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message:
                try:
                    payload = json.loads(message["text"])  # type: ignore
                except Exception:
                    continue
                if payload.get("type") == "config":
                    sr_client = int(payload.get("sr", 48000))
                    await ws.send_text(json.dumps({"type": "config", "ok": True, "sr": sr_client}))
                continue

            if "bytes" in message:
                if sr_client is None:
                    continue
                chunk = message["bytes"]  # type: ignore
                arr = np.frombuffer(chunk, dtype=np.float32).astype(np.float32)
                if arr.size == 0:
                    continue
                buffer = np.concatenate([buffer, arr])
                max_len = int(sr_client * max_buffer_s)
                if buffer.size > max_len:
                    buffer = buffer[-max_len:]

                pending += arr.size
                samples_per_hop = int(sr_client * hop_s)
                samples_window = int(sr_client * window_s)

                if pending >= samples_per_hop and buffer.size >= samples_window:
                    pending = 0
                    window = buffer[-samples_window:]
                    try:
                        if not db.list_speakers():
                            await ws.send_text(json.dumps({
                                "type": "error",
                                "message": "no_enrollments",
                            }))
                        else:
                            window_16k = resample_waveform(window, sr_client, 16000)
                            result = embedder.compute_embedding_from_waveform(window_16k)
                            name, score, ranked = db.identify(result.embedding, threshold=0.6, top_k=3)
                            await ws.send_text(json.dumps({
                                "type": "result",
                                "best": name,
                                "score": float(score),
                                "top": [(n, float(s)) for (n, s) in ranked],
                            }))
                    except Exception as e:
                        await ws.send_text(json.dumps({"type": "error", "message": str(e)}))

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
