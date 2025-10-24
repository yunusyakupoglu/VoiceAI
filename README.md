# Speaker Identification (Simple)

Basit bir Python konuşmacı tanıma aracı. MFCC + kosinüs benzerliği ile çalışır; istenirse ECAPA (SpeechBrain) ile gelişmiş gömlemeler desteklenir.

## Kurulum

```bash
python3 -m pip install -r requirements.txt
# ECAPA için (opsiyonel)
# python3 -m pip install -r requirements-ecapa.txt
# Mikrofon için (opsiyonel)
# python3 -m pip install -r requirements-extra.txt
```

## CLI Kullanımı

```bash
# Yardım
python3 -m speaker_id.cli -h

# Kayıt
python3 -m speaker_id.cli --db /workspace/.speaker_db enroll ALICE /path/alice.wav

# Tanıma
python3 -m speaker_id.cli --db /workspace/.speaker_db identify /path/query.wav --top-k 3 -v

# List/Sil
python3 -m speaker_id.cli --db /workspace/.speaker_db list
python3 -m speaker_id.cli --db /workspace/.speaker_db remove ALICE

# Mikrofondan
python3 -m speaker_id.cli --db /workspace/.speaker_db mic --seconds 3

# Diyarizasyon (naif)
python3 -m speaker_id.cli diarize /path/audio.wav
```

Genel seçenekler:
- `--backend {auto,mfcc,ecapa}`: ECAPA varsa otomatik seçmek için `auto` kullanın.
- `--no-preprocess`: Ön işlemleri kapatma.
- `--trim-top-db`, `--pre-emphasis`, `--target-rms`: Ön işleme ayarları.

## API

Docker ile hızlı başlatma:

```bash
docker build -t speaker-id .
docker run -p 8000:8000 speaker-id
```

FastAPI uç noktaları:
- `POST /enroll` (form-data: name, file)
- `POST /identify` (form-data: file)

## Notlar
- Bu proje eğitim/POC amaçlıdır; gerçek dünyada daha güçlü modeller (ECAPA) ve veri artırma, VAD, daha gelişmiş skorlama, diyarizasyon vb. önerilir.
