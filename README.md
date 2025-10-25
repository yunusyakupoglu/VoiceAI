# SpeakerID (Türkçe)

Python ile konuşmacı tanıma (kimin konuştuğunu sesten bulma) aracı. Kayıtlı kişilerin referans seslerinden ses gömme vektörleri (embedding) çıkarır ve yeni bir sesin bu kişilerden hangisine en çok benzediğini kozayn benzerlik ile tahmin eder.

- Model: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)
- Örnekleme: 16 kHz, mono
- Veri deposu: Basit JSON (varsayılan: `data/speakers.json`)
- CLI: `python -m speakerid` ile komut satırı
 - CLI: `python -m speakerid` ile komut satırı (mikrofon desteği dahil)

## Özellikler
- Kişi kaydetme (enroll) — referans ses(ler) ile kişiyi ekleyin
- Tanıma (identify) — bilinmeyen bir sesi en yakın kişiye eşleyin
- Doğrulama (verify) — iki sesin aynı kişiye ait olma benzerliğini ölçün
- Listeleme/silme/sıfırlama — kayıtlı kişileri yönetin

## Kurulum

1) Python 3.9+ önerilir. Sanal ortam açın:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
# CPU için örnek PyTorch kurulumu (Linux):
# pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

Notlar:
- MP3/OGG desteği için sisteminizde `ffmpeg` bulunması faydalıdır.
- Linux'ta `libsndfile` ve mikrofon için PortAudio gerekebilir:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1 libportaudio2
  ```

## Hızlı Başlangıç

Yardım ekranı:
```bash
python -m speakerid --help
```

### 1) Kişi kaydetme (enroll)
Referans ses dosyasıyla (örn. WAV/MP3) bir kişiyi ekleyin. Bir kişiye birden fazla örnek ekleyebilirsiniz; bu, doğruluğu artırır.
```bash
python -m speakerid enroll "Ali" /path/to/ali_1.wav
python -m speakerid enroll "Ali" /path/to/ali_2.wav
python -m speakerid enroll "Ayşe" /path/to/ayse_1.wav
```

### 2) Tanıma (identify)
Bilinmeyen bir sesin kim olduğunu tahmin edin. `--threshold` eşiğinin altı bilinmiyor kabul edilir.
```bash
python -m speakerid identify /path/to/unknown.wav --threshold 0.60 --top-k 3
```
Örnek çıktı:
```
Tahmin: Ali (skor=0.812)
En yakın eşleşmeler
Kişi    Benzerlik
Ali     0.812
Ayşe    0.431
```

### 3) Doğrulama (verify)
İki sesin aynı kişiye ait olma benzerliğini 0-1 arası döndürür.
```bash
python -m speakerid verify /path/a.wav /path/b.wav
# çıktı: Benzerlik (0-1): 0.734
```

### 4) Listele / Sil / Sıfırla
```bash
python -m speakerid list
python -m speakerid remove "Ali"
python -m speakerid reset  # tüm kayıtları siler
```

### 5) Mikrofon ile Kayıt ve Tanıma
Mikrofon komutları kısa bir kayıt alır (varsayılan 5 sn) ve işlemi yapar.

- Kişi kaydetme (mikrofon):
```bash
python -m speakerid enroll-mic "Ali" --seconds 6
```

- Tanıma (mikrofon):
```bash
python -m speakerid identify-mic --seconds 4 --threshold 0.6 --top-k 3
```

İpuçları:
- Mikrofon sürücüsü/izin sorunları için `arecord -l` (Linux) ya da sistem ayarlarını kontrol edin.
- Çoklu cihaz varsa `sounddevice` içinde varsayılan cihazı sistemden ayarlayın; gerekirse Python tarafında cihaz indeksi eklenebilir.

## İpuçları
- Kalite: Sessiz ortam, 3–10 sn konuşma, farklı örnekler doğruluğu artırır.
- Eşik (`--threshold`): 0.55–0.70 aralığı pratikte işe yarar; veri kalitesine göre ayarlayın.
- Cihaz: CUDA varsa `--device cuda` ile hızlanır; aksi halde otomatik `cpu` kullanılır.
- Format: WAV önerilir. MP3 için `ffmpeg` gereklidir.

## Mimari
- `speakerid/audio.py`: Ses yükleme ve normalizasyon
- `speakerid/embed.py`: SpeechBrain ECAPA ile embedding çıkarma
- `speakerid/store.py`: JSON veritabanı, kişi merkezleri (centroid) ve kimliklendirme
- `speakerid/cli.py`: Typer tabanlı CLI komutları

## Sorumluluk Reddi ve Gizlilik
Bu proje eğitim ve demo amaçlıdır. Gerçek dünyada ses biyometrisi kullanımında yasal düzenlemelere ve gizlilik ilkelerine uyunuz. Model her zaman hatasız çalışmayabilir; sonuçları ek sinyallerle doğrulayın.

## Lisans
MIT (bkz. `LICENSE`)
