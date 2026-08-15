"""SearchEngine: the query-side facade over a built index.

Pipeline per query:
    1. code-aware tokenization (camelCase/snake_case subword expansion)
    2. sparse leg  — BM25 over boosted fields
    3. dense leg   — embedding cosine similarity
    4. fusion      — Reciprocal Rank Fusion (rank-based, no calibration)
    5. graph boost — hits whose symbols are structurally connected to
                     other hits get promoted (fabric coherence)
    6. context     — each hit is decorated with graph neighbors
                     (callers/callees/subclasses) for the LLM prompt

Also exposes relational queries: symbol lookup, callers/callees,
impact analysis, and file outlines.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from .graph import KnowledgeGraph
from .indexer import INDEX_DIR, Indexer, IndexConfig
from .models import Chunk, SearchResult, Symbol
from .retrieval.bm25 import BM25Index
from .retrieval.embedders import make_embedder
from .retrieval.fusion import reciprocal_rank_fusion
from .retrieval.vector_store import VectorStore
from .tokenizer import tokenize


class SearchEngine:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.index_dir = os.path.join(self.root, INDEX_DIR)
        if not os.path.isdir(self.index_dir):
            raise FileNotFoundError(
                f"No index at {self.index_dir}. Run `codefabric index` first."
            )
        idx = Indexer(self.root, IndexConfig())
        self.chunks: dict[str, Chunk] = {c.chunk_id: c for c in idx._load_chunks()}
        self.symbols: list[Symbol] = idx._load_symbols()
        self.bm25 = BM25Index.from_dict(idx._load_json("bm25.json") or
                                        {"doc_ids": [], "doc_len": [], "term_freqs": []})
        with open(os.path.join(self.index_dir, "graph.json"), encoding="utf-8") as f:
            self.graph = KnowledgeGraph.from_dict(json.load(f))
        self.vectors = VectorStore.load(os.path.join(self.index_dir, "vectors.json"))
        self._embedder = None
        self._symbol_chunks: dict[str, list[str]] = defaultdict(list)
        for cid, c in self.chunks.items():
            if c.symbol:
                self._symbol_chunks[f"sym:{c.file_path}::{c.symbol}"].append(cid)

    # -- hybrid search -------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        language: str | None = None,
        kind: str | None = None,
        use_dense: bool = True,
        graph_context: bool = True,
        candidates: int = 40,
    ) -> list[SearchResult]:
        qtokens = tokenize(query)
        rankings: dict[str, list[tuple[str, float]]] = {}
        rankings["bm25"] = self.bm25.search(qtokens, k=candidates)

        if use_dense and self.vectors is not None and self.vectors.ids:
            if self._embedder is None:
                self._embedder = make_embedder(
                    "hash" if self.vectors.embedder_name.startswith("hashing")
                    else self.vectors.embedder_name
                )
            qvec = self._embedder.embed([query])[0]
            rankings["dense"] = self.vectors.search(qvec, k=candidates)

        fused = reciprocal_rank_fusion(rankings, weights={"bm25": 1.0, "dense": 0.9})

        # Filter + structural bonus.
        hits: list[tuple[str, float, list[str]]] = []
        fused_ids = {doc_id for doc_id, _, _ in fused}
        for doc_id, score, sources in fused:
            c = self.chunks.get(doc_id)
            if c is None:
                continue
            if language and c.language != language:
                continue
            if kind and c.kind != kind:
                continue
            if c.kind in ("function", "method", "class"):
                score *= 1.15  # prefer definitions over incidental blocks
            score *= self._coherence_bonus(c, fused_ids)
            hits.append((doc_id, score, sources))

        hits.sort(key=lambda h: -h[1])
        results = []
        for doc_id, score, sources in hits[:k]:
            c = self.chunks[doc_id]
            related = self._graph_context(c) if graph_context else []
            results.append(SearchResult(chunk=c, score=score,
                                        sources=sources, related=related))
        return results

    def _coherence_bonus(self, chunk: Chunk, fused_ids: set[str]) -> float:
        """Boost chunks whose graph neighbors also matched the query —
        a cluster of related hits usually marks the right subsystem."""
        if not chunk.symbol:
            return 1.0
        sid = f"sym:{chunk.file_path}::{chunk.symbol}"
        neighbor_syms = (
            self.graph.callers_of(sid)
            | self.graph.callees_of(sid)
            | self.graph.subclasses_of(sid)
        )
        for n in neighbor_syms:
            for cid in self._symbol_chunks.get(n, []):
                if cid in fused_ids:
                    return 1.1
        return 1.0

    def _graph_context(self, chunk: Chunk, limit: int = 5) -> list[dict]:
        if not chunk.symbol:
            return []
        sid = f"sym:{chunk.file_path}::{chunk.symbol}"
        out: list[dict] = []
        for rel, nodes in (
            ("caller", self.graph.callers_of(sid)),
            ("callee", self.graph.callees_of(sid)),
            ("subclass", self.graph.subclasses_of(sid)),
        ):
            for n in sorted(nodes):
                attrs = self.graph.nodes.get(n, {})
                if attrs.get("type") != "symbol":
                    continue
                out.append(
                    {
                        "relation": rel,
                        "symbol": attrs.get("qualified", ""),
                        "file": attrs.get("path", ""),
                        "line": attrs.get("line", 0),
                    }
                )
                if len(out) >= limit:
                    return out
        return out

    # -- relational queries --------------------------------------------

    def find_symbol(self, name: str) -> list[Symbol]:
        exact = [s for s in self.symbols
                 if s.name == name or s.qualified_name == name]
        if exact:
            return exact
        lowered = name.lower()
        return [s for s in self.symbols if lowered in s.qualified_name.lower()][:20]

    def callers(self, name: str) -> list[dict]:
        out = []
        for sym in self.find_symbol(name):
            sid = f"sym:{sym.file_path}::{sym.qualified_name}"
            for n in sorted(self.graph.callers_of(sid)):
                attrs = self.graph.nodes.get(n, {})
                out.append({"symbol": attrs.get("qualified"),
                            "file": attrs.get("path"),
                            "line": attrs.get("line")})
        return out

    def impact(self, name: str, max_depth: int = 4) -> list[dict]:
        out = []
        for sym in self.find_symbol(name):
            sid = f"sym:{sym.file_path}::{sym.qualified_name}"
            for node_id, depth, via in self.graph.impact(sid, max_depth=max_depth):
                attrs = self.graph.nodes.get(node_id, {})
                out.append(
                    {
                        "depth": depth,
                        "via": via,
                        "type": attrs.get("type"),
                        "name": attrs.get("qualified") or attrs.get("path"),
                        "file": attrs.get("path"),
                        "line": attrs.get("line"),
                    }
                )
        return out

    def outline(self, file_path: str) -> list[dict]:
        rel = file_path.replace(os.sep, "/")
        syms = [s for s in self.symbols if s.file_path == rel]
        syms.sort(key=lambda s: s.start_line)
        return [
            {"kind": s.kind, "name": s.qualified_name, "line": s.start_line,
             "signature": s.signature, "doc": (s.docstring or "")[:120]}
            for s in syms
        ]

    def stats(self) -> dict:
        langs: dict[str, int] = defaultdict(int)
        for c in self.chunks.values():
            langs[c.language] += 1
        return {
            "chunks": len(self.chunks),
            "symbols": len(self.symbols),
            "graph_nodes": len(self.graph.nodes),
            "graph_edges": sum(
                len(d) for t in self.graph.out.values() for d in t.values()
            ),
            "chunks_by_language": dict(sorted(langs.items(), key=lambda kv: -kv[1])),
            "embedder": self.vectors.embedder_name if self.vectors else None,
        }
