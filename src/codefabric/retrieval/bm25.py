"""Okapi BM25 with field boosting, implemented from the published
formula (Robertson & Zaragoza) — no third-party code.

Symbol names and docstrings are injected into the token stream with a
repeat factor so definition chunks outrank incidental mentions.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.term_freqs: list[Counter[str]] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.avgdl: float = 0.0

    def add(self, doc_id: str, tokens: list[str]) -> None:
        idx = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        tf = Counter(tokens)
        self.term_freqs.append(tf)
        self.doc_len.append(len(tokens))
        for term in tf:
            self.postings[term].append(idx)

    def finalize(self) -> None:
        n = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0

    def search(self, query_tokens: list[str], k: int = 20) -> list[tuple[str, float]]:
        n = len(self.doc_ids)
        if n == 0 or not query_tokens:
            return []
        scores: defaultdict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            qtf = query_tokens.count(term)
            for idx in posting:
                tf = self.term_freqs[idx][term]
                dl = self.doc_len[idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[idx] += idf * (tf * (self.k1 + 1) / denom) * min(qtf, 2)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.doc_ids[i], s) for i, s in ranked]

    # -- persistence ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "term_freqs": [dict(tf) for tf in self.term_freqs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BM25Index":
        idx = cls(k1=d.get("k1", 1.5), b=d.get("b", 0.75))
        idx.doc_ids = list(d["doc_ids"])
        idx.doc_len = list(d["doc_len"])
        idx.term_freqs = [Counter(tf) for tf in d["term_freqs"]]
        for i, tf in enumerate(idx.term_freqs):
            for term in tf:
                idx.postings[term].append(i)
        idx.finalize()
        return idx
