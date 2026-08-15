"""Pluggable embedding backends.

Default is a deterministic feature-hashing embedder (pure stdlib, works
air-gapped, zero model downloads). Installing the ``ml`` extra unlocks
sentence-transformers models — any Apache/MIT-licensed code embedding
model (e.g. BAAI/bge-m3, jinaai/jina-embeddings-v2-base-code,
nomic-ai/nomic-embed-text) can be dropped in by name.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

from ..tokenizer import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Feature-hashing bag-of-subwords embedding.

    Not neural, but deterministic, dependency-free, and surprisingly
    effective for code because identifiers dominate the signal. Serves
    as the offline fallback so hybrid search always has a dense leg.
    """

    def __init__(self, dim: int = 512):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.md5(token.encode("utf-8", "replace")).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        return idx, sign

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = tokenize(text)
            for tok in tokens:
                idx, sign = self._bucket(tok)
                # Sub-linear tf weighting.
                vec[idx] += sign
            # log-scale then l2 normalize
            vec = [math.copysign(math.log1p(abs(v)), v) for v in vec]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class SentenceTransformerEmbedder:  # pragma: no cover - optional extra
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]


def make_embedder(spec: str | None) -> Embedder:
    """spec: None/'hash' -> HashingEmbedder; 'st:<model>' -> neural."""
    if not spec or spec == "hash" or spec.startswith("hashing"):
        return HashingEmbedder()
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    return SentenceTransformerEmbedder(spec)
