from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class SimpleSpeakerId:
    threshold: float = 0.55  # tune as needed

    def enroll_vector(self, vectors: np.ndarray) -> np.ndarray:
        """Aggregate multiple enrollment vectors into a single template.

        Currently uses simple mean + L2 normalize.
        """
        if vectors.ndim == 1:
            agg = vectors
        else:
            agg = np.mean(vectors, axis=0)
        norm = np.linalg.norm(agg) + 1e-8
        return (agg / norm).astype(np.float32)

    def identify(self, query: np.ndarray, speakers: Dict[str, np.ndarray]) -> Tuple[str | None, float, Dict[str, float]]:
        """Return (best_speaker_or_None, best_score, all_scores).

        - speakers: mapping name -> [N, D] array of enrollment vectors
        """
        if not speakers:
            return None, 0.0, {}
        # Compute a template per speaker
        templates: Dict[str, np.ndarray] = {}
        for name, vectors in speakers.items():
            templates[name] = self.enroll_vector(vectors)

        best_name = None
        best_score = -1.0
        scores: Dict[str, float] = {}
        for name, templ in templates.items():
            s = cosine_similarity(query, templ)
            scores[name] = s
            if s > best_score:
                best_score = s
                best_name = name

        if best_score < self.threshold:
            return None, best_score, scores
        return best_name, best_score, scores
