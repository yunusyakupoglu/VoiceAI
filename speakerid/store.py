from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


@dataclass
class Speaker:
    name: str
    embeddings: List[List[float]] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    @property
    def embedding_matrix(self) -> np.ndarray:
        if not self.embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(self.embeddings, dtype=np.float32)

    def add_embedding(self, embedding: np.ndarray) -> None:
        self.embeddings.append(embedding.astype(np.float32).tolist())
        self.updated_at = _now_iso()

    def centroid(self) -> np.ndarray:
        mat = self.embedding_matrix
        if mat.size == 0:
            return np.zeros((0,), dtype=np.float32)
        centroid_vec = np.mean(mat, axis=0)
        # L2-normalize centroid for cosine comparisons
        norm = np.linalg.norm(centroid_vec)
        if norm > 0:
            centroid_vec = centroid_vec / norm
        return centroid_vec.astype(np.float32)


@dataclass
class SpeakerDatabase:
    path: str
    model_name: str = "speechbrain/spkrec-ecapa-voxceleb"

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._initialize()

    def _initialize(self) -> None:
        data = {
            "speakers": {},
            "metadata": {
                "model": self.model_name,
                "version": 1,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            },
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        data["metadata"]["updated_at"] = _now_iso()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def list_speakers(self) -> List[str]:
        data = self._load()
        return sorted(list(data.get("speakers", {}).keys()))

    def get_speaker(self, name: str) -> Optional[Speaker]:
        data = self._load()
        entry = data.get("speakers", {}).get(name)
        if entry is None:
            return None
        return Speaker(name=name, embeddings=entry.get("embeddings", []), created_at=entry.get("created_at", _now_iso()), updated_at=entry.get("updated_at", _now_iso()))

    def upsert_speaker(self, speaker: Speaker) -> None:
        data = self._load()
        data.setdefault("speakers", {})[speaker.name] = {
            "embeddings": speaker.embeddings,
            "created_at": speaker.created_at,
            "updated_at": speaker.updated_at,
        }
        self._save(data)

    def remove_speaker(self, name: str) -> bool:
        data = self._load()
        speakers = data.get("speakers", {})
        if name in speakers:
            del speakers[name]
            self._save(data)
            return True
        return False

    def reset(self) -> None:
        self._initialize()

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        existing = self.get_speaker(name)
        if existing is None:
            existing = Speaker(name=name)
        existing.add_embedding(embedding)
        self.upsert_speaker(existing)

    def centroids(self) -> Dict[str, np.ndarray]:
        data = self._load()
        result: Dict[str, np.ndarray] = {}
        for name, entry in data.get("speakers", {}).items():
            embeddings = np.asarray(entry.get("embeddings", []), dtype=np.float32)
            if embeddings.size == 0:
                continue
            centroid_vec = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid_vec)
            if norm > 0:
                centroid_vec = centroid_vec / norm
            result[name] = centroid_vec.astype(np.float32)
        return result

    def identify(self, query_embedding: np.ndarray, threshold: float = 0.6, top_k: int = 3) -> Tuple[Optional[str], float, List[Tuple[str, float]]]:
        """
        Identify speaker by comparing cosine similarity to centroids.

        Returns (best_name, best_score, ranked_list) where ranked_list is sorted desc.
        If best_score < threshold, best_name is None (unknown).
        """
        centroids = self.centroids()
        if not centroids:
            return None, 0.0, []

        # Cosine similarities
        sims: List[Tuple[str, float]] = []
        q = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        for name, c in centroids.items():
            denom = float(np.linalg.norm(c))
            if denom == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q, c) / (np.linalg.norm(q) * denom + 1e-9))
            sims.append((name, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = sims[0]
        if best_score < threshold:
            return None, best_score, sims[:top_k]
        return best_name, best_score, sims[:top_k]
