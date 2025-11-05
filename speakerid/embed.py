from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# Suppress excessive warnings from torch/speechbrain during import
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from speechbrain.pretrained import EncoderClassifier  # type: ignore
except Exception as exc:  # pragma: no cover
    EncoderClassifier = None  # type: ignore

from .audio import load_audio_mono, normalize_waveform


@dataclass
class EmbeddingResult:
    embedding: np.ndarray  # shape (D,)
    model_name: str


class SpeakerEmbedder:
    """
    Wrapper around SpeechBrain ECAPA-TDNN speaker embedding model.
    """

    def __init__(self, model_source: str = "speechbrain/spkrec-ecapa-voxceleb", device: Optional[str] = None):
        if EncoderClassifier is None:
            raise RuntimeError(
                "speechbrain is required. Install dependencies via requirements.txt"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_source = model_source
        self._classifier = EncoderClassifier.from_hparams(
            source=model_source, run_opts={"device": device}
        )

    @torch.inference_mode()
    def compute_embedding_from_file(self, file_path: str, sample_rate: int = 16000) -> EmbeddingResult:
        waveform, sr = load_audio_mono(file_path, target_sample_rate=sample_rate)
        waveform = normalize_waveform(waveform)
        tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        # encoder expects [batch, time]
        emb = self._classifier.encode_batch(tensor).squeeze(0).squeeze(0).detach().cpu().numpy()
        emb = self._l2_normalize(emb)
        return EmbeddingResult(embedding=emb, model_name=self.model_source)

    @torch.inference_mode()
    def compute_embedding_from_waveform(self, waveform: np.ndarray) -> EmbeddingResult:
        """Compute embedding from an in-memory mono waveform in [-1, 1]."""
        if waveform.ndim != 1:
            waveform = np.mean(waveform, axis=-1)
        waveform = normalize_waveform(waveform.astype(np.float32))
        tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        emb = self._classifier.encode_batch(tensor).squeeze(0).squeeze(0).detach().cpu().numpy()
        emb = self._l2_normalize(emb)
        return EmbeddingResult(embedding=emb, model_name=self.model_source)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-9) -> float:
        a = vec_a.astype(np.float32)
        b = vec_b.astype(np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < eps:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _l2_normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < eps:
            return vec
        return (vec / norm).astype(np.float32)
