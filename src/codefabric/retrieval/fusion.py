"""Result fusion: Reciprocal Rank Fusion (Cormack et al., 2009).

RRF is rank-based, so BM25 scores and cosine similarities fuse without
any score normalization. A small structural bonus favors definition
chunks (function/class/method) over incidental blocks when ranks tie.
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[str, float]]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float, list[str]]]:
    """Fuse named rankings into (doc_id, fused_score, contributing_sources).

    ``rankings`` maps source name ("bm25", "dense", ...) to an ordered
    [(doc_id, raw_score)] list. ``weights`` optionally scales a source's
    contribution (default 1.0).
    """
    weights = weights or {}
    fused: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for source, ranking in rankings.items():
        w = weights.get(source, 1.0)
        for rank, (doc_id, _score) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank + 1)
            sources.setdefault(doc_id, []).append(source)
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(doc_id, score, sources[doc_id]) for doc_id, score in ranked]
