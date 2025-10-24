from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import librosa


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_mfcc: int = 13
    n_fft: int = 400  # 25ms at 16k
    hop_length: int = 160  # 10ms at 16k
    fmin: int = 20
    fmax: int | None = 7600


def compute_mfcc(audio: np.ndarray, config: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Compute MFCCs and return mean+std pooled vector (2*n_mfcc).
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    # Remove the 0th coefficient energy bias if desired? Keep it for simplicity.
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1) + 1e-8
    feat = np.concatenate([mean, std], axis=0).astype(np.float32)
    # L2 normalize for cosine similarity
    norm = np.linalg.norm(feat) + 1e-8
    return (feat / norm).astype(np.float32)
