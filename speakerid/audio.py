from __future__ import annotations

import os
from typing import Tuple

import numpy as np

# Prefer librosa + soundfile for broad codec support
try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover - optional dependency handling
    librosa = None  # type: ignore


def load_audio_mono(file_path: str, target_sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as mono float32 waveform resampled to target_sample_rate.

    Returns (waveform, sample_rate) where waveform has shape (num_samples,).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if librosa is None:
        raise RuntimeError(
            "librosa is required to load audio. Please install dependencies via requirements.txt"
        )

    # librosa.load returns float32 in range [-1, 1], mono by default when mono=True
    waveform, sample_rate = librosa.load(file_path, sr=target_sample_rate, mono=True)

    if waveform.ndim != 1:
        # Ensure mono
        waveform = np.mean(waveform, axis=0)

    # Ensure type is float32 and contiguous
    waveform = np.ascontiguousarray(waveform.astype(np.float32))
    return waveform, target_sample_rate


def normalize_waveform(waveform: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """L2-normalize the waveform to unit energy to reduce loudness variance."""
    norm = float(np.linalg.norm(waveform))
    if norm < eps:
        return waveform
    return waveform / norm
