from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa


def load_audio_mono(path: Path | str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load an audio file as mono float32 with a target sample rate.

    Returns (audio, sr). Raises ValueError on empty/invalid audio.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Use soundfile for robust decoding; fallback to librosa if needed
    try:
        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32, copy=False)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")
            sr = target_sr
    except Exception:
        audio, sr = librosa.load(str(path), sr=target_sr, mono=True)

    if audio is None or audio.size == 0:
        raise ValueError(f"Loaded empty audio from: {path}")

    # Simple amplitude normalization to [-1, 1] if needed
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 0:
        audio = audio / max_abs

    return audio, sr
