from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import numpy as np


@dataclass
class Storage:
    root: Path

    @property
    def db_path(self) -> Path:
        return self.root / "embeddings.json"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self.db_path.write_text(json.dumps({}), encoding="utf-8")

    def load(self) -> Dict[str, List[List[float]]]:
        """Load embeddings as a mapping: name -> list of embedding vectors.

        The on-disk format is JSON with values either:
        - list[list[float]]: multiple enrollment vectors
        - list[float]: legacy single vector (wrapped to list-of-list)
        """
        self.ensure()
        try:
            raw = json.loads(self.db_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        cleaned: Dict[str, List[List[float]]] = {}
        if not isinstance(raw, dict):
            return cleaned

        for name, samples in raw.items():
            if not isinstance(samples, list):
                continue
            # Empty list -> no samples yet
            if len(samples) == 0:
                cleaned[name] = []
                continue
            # Legacy: flat list of numbers
            if all(isinstance(x, (int, float)) for x in samples):
                cleaned[name] = [[float(x) for x in samples]]
                continue
            # Expected: list of lists of numbers
            vectors: List[List[float]] = []
            for vec in samples:
                if isinstance(vec, list):
                    try:
                        vectors.append([float(x) for x in vec])
                    except Exception:
                        continue
            if vectors:
                cleaned[name] = vectors
        return cleaned

    def save(self, data: Dict[str, List[List[float]]]) -> None:
        self.ensure()
        self.db_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, name: str, embedding: np.ndarray) -> None:
        data = self.load()
        data.setdefault(name, [])
        data[name].append(embedding.astype(float).tolist())
        self.save(data)

    def remove(self, name: str) -> bool:
        data = self.load()
        if name in data:
            del data[name]
            self.save(data)
            return True
        return False

    def list_speakers(self) -> list[str]:
        data = self.load()
        return sorted(list(data.keys()))

    def get_speaker_embeddings(self, name: str) -> np.ndarray | None:
        data = self.load()
        samples = data.get(name)
        if not samples:
            return None
        arr = np.array(samples, dtype=np.float32)
        return arr

    def all_enrollments(self) -> Dict[str, np.ndarray]:
        data = self.load()
        out: Dict[str, np.ndarray] = {}
        for name, samples in data.items():
            out[name] = np.array(samples, dtype=np.float32)
        return out
