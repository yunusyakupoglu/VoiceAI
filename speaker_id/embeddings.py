from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

from .features import compute_mfcc, FeatureConfig

try:
    from speechbrain.pretrained import EncoderClassifier  # type: ignore
    _HAS_SPEECHBRAIN = True
except Exception:
    _HAS_SPEECHBRAIN = False


Backend = Literal["mfcc", "ecapa"]


@dataclass
class EmbeddingConfig:
    sample_rate: int = 16000
    backend: Backend = "mfcc"
    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"


class EmbeddingExtractor:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._ecapa: Optional[object] = None

    def _ensure_ecapa(self) -> None:
        if self._ecapa is not None:
            return
        if not _HAS_SPEECHBRAIN:
            raise RuntimeError("SpeechBrain not installed. Install extras: pip install -r requirements-ecapa.txt")
        self._ecapa = EncoderClassifier.from_hparams(source=self.config.ecapa_source)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        if self.config.backend == "mfcc":
            feat = compute_mfcc(audio, FeatureConfig(sample_rate=self.config.sample_rate))
            return feat
        elif self.config.backend == "ecapa":
            self._ensure_ecapa()
            assert self._ecapa is not None
            ecapa = self._ecapa
            # SpeechBrain expects [batch, time] tensor; we'll import torch lazily
            import torch  # type: ignore
            with torch.no_grad():
                wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                emb = ecapa.encode_batch(wav)
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                out = emb.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                return out
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
