# Speaker Identification (Simple)

Basit bir Python konuşmacı tanıma aracı. MFCC + kosinüs benzerliği ile çalışır; istenirse ECAPA (SpeechBrain) ile gelişmiş gömlemeler desteklenir.

## Kurulum

```bash
python3 -m pip install -r requirements.txt
# ECAPA için (opsiyonel)
# python3 -m pip install -r requirements-ecapa.txt
# Mikrofon + VAD + Form Upload (opsiyonel)
# python3 -m pip install -r requirements-extra.txt
```

## CLI Kullanımı

```bash
# Yardım
python3 -m speaker_id.cli -h

# Kayıt
python3 -m speaker_id.cli --db /workspace/.speaker_db enroll ALICE /path/alice.wav

# Tanıma
python3 -m speaker_id.cli --db /workspace/.speaker_db identify /path/query.wav --top-k 3 --znorm

# VAD & Gürültü azaltma
python3 -m speaker_id.cli --vad --noise-reduction identify /path/query.wav

# List/Sil
python3 -m speaker_id.cli --db /workspace/.speaker_db list
python3 -m speaker_id.cli --db /workspace/.speaker_db remove ALICE

# Mikrofondan
python3 -m speaker_id.cli --db /workspace/.speaker_db mic --seconds 3
python3 -m speaker_id.cli --db /workspace/.speaker_db mic-live --vad --window 1.5 --hop 0.5

# Diyarizasyon (naif)
python3 -m speaker_id.cli diarize /path/audio.wav --assign
```

Genel seçenekler:
- `--backend {auto,mfcc,ecapa}`: ECAPA varsa otomatik seçmek için `auto` kullanın.
- `--no-preprocess`: Ön işlemleri kapatma.
- `--trim-top-db`, `--pre-emphasis`, `--target-rms`: Ön işleme ayarları.
- `--noise-reduction`: Spektral gürültü azaltma.
- `--vad`: WebRTC VAD (sadece sesli kısımları kullanır).
- `--znorm`: Basit skor normalizasyonu.

## API

```bash
uvicorn speaker_id.api:app --host 0.0.0.0 --port 8000
```

- `GET /` Basit web arayüzü
- `POST /enroll` (form-data: name, file)
- `POST /identify` (form-data: file)
- `POST /monitor/enroll` ve `POST /monitor/identify` (SSE monitoring)
- `WS /ws/identify` Canlı mikrofon (tarayıcıda Live bölümü)

## Notlar
- Bu proje eğitim/POC amaçlıdır; gelişmiş üretim için ECAPA, güçlü VAD, skor norm/kalibrasyon ve gelişmiş diyarizasyon önerilir. Bu repo MFCC/ECAPA, VAD, basit z-norm, SSE/WS ve mikrofondan canlı tanımayı örnekler.
