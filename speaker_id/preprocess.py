from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import librosa
from typing import Optional


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
    # VAD (WebRTC)
    use_webrtc_vad: bool = False
    vad_aggressiveness: int = 2  # 0-3
    vad_frame_ms: int = 20  # 10, 20, or 30


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


def apply_webrtc_vad(audio: np.ndarray, sample_rate: int, aggressiveness: int = 2, frame_ms: int = 20) -> np.ndarray:
    """Keep only voiced frames using WebRTC VAD. Requires 16k mono float32 audio.
    Returns possibly shorter audio after concatenating voiced frames.
    """
    try:
        import webrtcvad  # type: ignore
    except Exception:
        return audio
    if sample_rate != 16000:
        # resample for VAD requirements
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000, res_type="kaiser_fast")
        sample_rate = 16000
    vad = webrtcvad.Vad(int(aggressiveness))
    frame_len = int(frame_ms * sample_rate / 1000)
    if frame_len % 2 == 1:
        frame_len += 1
    if frame_len <= 0:
        return audio
    # Convert to int16 little-endian as required by webrtcvad
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16, copy=False)
    voiced = []
    for start in range(0, len(pcm16) - frame_len + 1, frame_len):
        frame = pcm16[start:start + frame_len]
        is_voiced = False
        try:
            is_voiced = vad.is_speech(frame.tobytes(), sample_rate)
        except Exception:
            is_voiced = False
        if is_voiced:
            voiced.append(frame)
    if not voiced:
        return audio
    out = np.concatenate(voiced, axis=0).astype(np.float32) / 32768.0
    return out


def preprocess_audio(audio: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    y = audio.astype(np.float32, copy=False)
    y = trim_silence(y, sample_rate=config.sample_rate, top_db=config.trim_top_db)
    if config.use_webrtc_vad:
        y = apply_webrtc_vad(y, sample_rate=config.sample_rate, aggressiveness=config.vad_aggressiveness, frame_ms=config.vad_frame_ms)
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
