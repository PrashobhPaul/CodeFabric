"""Minimal cosine-similarity vector store (pure stdlib).

Vectors are already l2-normalized by the embedders, so cosine reduces
to a dot product. Uses numpy transparently when installed (``ml``
extra) for ~50x faster scoring on large indexes; falls back to plain
Python otherwise. Swappable for LanceDB/Qdrant/Chroma behind the same
three methods if a deployment outgrows in-process search.
"""
from __future__ import annotations

import json
import os

try:  # optional acceleration
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


class VectorStore:
    def __init__(self, dim: int, embedder_name: str = ""):
        self.dim = dim
        self.embedder_name = embedder_name
        self.ids: list[str] = []
        self.vectors: list[list[float]] = []
        self._matrix = None  # numpy cache

    def add(self, doc_id: str, vector: list[float]) -> None:
        self.ids.append(doc_id)
        self.vectors.append(vector)
        self._matrix = None

    def search(self, query: list[float], k: int = 20) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        if _np is not None:
            if self._matrix is None:
                self._matrix = _np.asarray(self.vectors, dtype=_np.float32)
            scores = self._matrix @ _np.asarray(query, dtype=_np.float32)
            top = _np.argsort(-scores)[:k]
            return [(self.ids[i], float(scores[i])) for i in top]
        scored = []
        for i, vec in enumerate(self.vectors):
            s = sum(a * b for a, b in zip(vec, query))
            scored.append((self.ids[i], s))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:k]

    # -- persistence ---------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "embedder": self.embedder_name,
                    "ids": self.ids,
                    "vectors": [[round(x, 6) for x in v] for v in self.vectors],
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "VectorStore | None":
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        store = cls(dim=d["dim"], embedder_name=d.get("embedder", ""))
        store.ids = d["ids"]
        store.vectors = d["vectors"]
        return store
