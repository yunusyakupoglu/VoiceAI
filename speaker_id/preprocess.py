from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import librosa


@dataclass
class PreprocessConfig:
    sample_rate: int = 16000
    trim_top_db: float = 30.0
    pre_emphasis: float = 0.0  # 0 disables, typical 0.97
    target_rms: float = 0.1  # simple loudness normalization target
    noise_reduction: bool = False
    nr_n_fft: int = 1024
    nr_hop_length: int = 256
    nr_percentile: float = 20.0  # noise floor percentile across time


def trim_silence(audio: np.ndarray, sample_rate: int, top_db: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=512)
    if intervals.size == 0:
        return audio
    pieces = [audio[start:end] for start, end in intervals]
    return np.concatenate(pieces, axis=0) if pieces else audio


def apply_pre_emphasis(audio: np.ndarray, coeff: float) -> np.ndarray:
    if coeff <= 0.0:
        return audio
    if audio.size < 2:
        return audio
    out = np.empty_like(audio)
    out[0] = audio[0]
    out[1:] = audio[1:] - coeff * audio[:-1]
    return out


def rms_normalize(audio: np.ndarray, target_rms: float) -> np.ndarray:
    if target_rms <= 0:
        return audio
    rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
    if rms == 0:
        return audio
    gain = target_rms / rms
    return (audio * gain).astype(np.float32)


def spectral_gate_denoise(
    audio: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
    noise_percentile: float = 20.0,
) -> np.ndarray:
    if audio.size == 0:
        return audio
    # STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    mag = np.abs(stft)
    phase = np.exp(1j * np.angle(stft))
    # Estimate noise floor per frequency bin as a low percentile across time
    noise = np.percentile(mag, noise_percentile, axis=1, keepdims=True)
    # Subtract noise floor and clamp to >= 0
    mag_d = np.maximum(mag - noise, 0.0)
    # Reconstruct
    stft_d = mag_d * phase
    y = librosa.istft(stft_d, hop_length=hop_length, win_length=n_fft, length=audio.size)
    return y.astype(np.float32)


def preprocess_audio(audio: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    y = audio.astype(np.float32, copy=False)
    y = trim_silence(y, sample_rate=config.sample_rate, top_db=config.trim_top_db)
    if config.noise_reduction:
        y = spectral_gate_denoise(
            y,
            n_fft=config.nr_n_fft,
            hop_length=config.nr_hop_length,
            noise_percentile=config.nr_percentile,
        )
    y = apply_pre_emphasis(y, coeff=config.pre_emphasis)
    y = rms_normalize(y, target_rms=config.target_rms)
    return y
