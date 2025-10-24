from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

from .embeddings import EmbeddingExtractor, EmbeddingConfig


@dataclass
class DiarizeConfig:
    sample_rate: int = 16000
    window_seconds: float = 1.5
    hop_seconds: float = 0.5
    backend: str = "mfcc"


def sliding_windows(num_samples: int, sr: int, window_s: float, hop_s: float) -> List[Tuple[int, int]]:
    w = int(window_s * sr)
    h = int(hop_s * sr)
    if w <= 0 or h <= 0:
        return []
    starts = list(range(0, max(1, num_samples - w + 1), h))
    if not starts:
        starts = [0]
    spans: List[Tuple[int, int]] = []
    for s in starts:
        e = min(num_samples, s + w)
        spans.append((s, e))
    return spans


def kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if k >= n:
        # each point its own cluster
        return np.arange(n), X.copy()
    centroids = X[rng.choice(n, size=k, replace=False)]
    for _ in range(iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = np.stack([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids


def diarize_embeddings(embs: np.ndarray, num_speakers: int | None = None) -> np.ndarray:
    n = embs.shape[0]
    if num_speakers is None:
        # naive estimate using PCA energy or silhouette; here simple 2 speakers fallback
        k = 2 if n >= 2 else 1
    else:
        k = max(1, int(num_speakers))
    labels, _ = kmeans(embs, k=k)
    return labels
