"""Core data model for the CodeFabric index.

Everything is a plain dataclass serializable to/from JSON so the index
directory stays transparent, diffable, and tool-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Chunk:
    """A retrieval unit produced by structure-aware chunking.

    Chunks follow the cAST principle: boundaries respect syntactic
    structure (function/class/module blocks), small siblings are merged,
    oversized nodes are split recursively.
    """

    chunk_id: str
    file_path: str
    language: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str
    kind: str = "block"  # module | class | function | method | block
    symbol: Optional[str] = None  # qualified name, e.g. "pkg.mod.Class.method"
    docstring: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)


@dataclass
class Symbol:
    """A named definition extracted from source code."""

    name: str
    qualified_name: str
    kind: str  # class | function | method
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    bases: list[str] = field(default_factory=list)  # for classes
    calls: list[str] = field(default_factory=list)  # callee names (best effort)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Symbol":
        return cls(**d)


@dataclass
class SearchResult:
    """A fused search hit with provenance and graph context."""

    chunk: Chunk
    score: float
    sources: list[str] = field(default_factory=list)  # e.g. ["bm25", "dense"]
    related: list[dict[str, Any]] = field(default_factory=list)  # graph neighbors

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "sources": self.sources,
            "chunk": self.chunk.to_dict(),
            "related": self.related,
        }
