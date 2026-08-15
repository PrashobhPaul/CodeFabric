"""Repository indexer: walk -> parse -> chunk -> embed -> graph -> persist.

The index lives in ``<repo>/.codefabric/`` as transparent JSON so it
can be inspected, diffed, and consumed by other tools. Re-indexing is
incremental: only files whose content hash changed are re-parsed and
re-embedded.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import time
from dataclasses import dataclass, field

from .chunking.base import ChunkerDispatch
from .graph import GraphBuilder, KnowledgeGraph
from .languages import CODE_LANGUAGES, DEFAULT_EXCLUDE_DIRS, detect_language
from .models import Chunk, Symbol
from .retrieval.bm25 import BM25Index
from .retrieval.embedders import make_embedder
from .retrieval.vector_store import VectorStore
from .tokenizer import tokenize

INDEX_DIR = ".codefabric"
MAX_FILE_BYTES = 1_500_000


@dataclass
class IndexConfig:
    embedder: str = "hash"  # "hash" | "st:<model-name>"
    max_chunk_chars: int = 2400
    exclude_globs: list[str] = field(default_factory=list)
    include_globs: list[str] = field(default_factory=list)  # empty = all


@dataclass
class IndexStats:
    files_indexed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    chunks: int = 0
    symbols: int = 0
    seconds: float = 0.0


def _file_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _load_gitignore(root: str) -> list[str]:
    patterns: list[str] = []
    path = os.path.join(root, ".gitignore")
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line.rstrip("/"))
    return patterns


def _matches_any(rel_path: str, patterns: list[str]) -> bool:
    base = os.path.basename(rel_path)
    for pat in patterns:
        p = pat.lstrip("/")
        if fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(base, p):
            return True
        if fnmatch.fnmatch(rel_path, p + "/*") or rel_path.startswith(p + "/"):
            return True
    return False


class Indexer:
    def __init__(self, root: str, config: IndexConfig | None = None):
        self.root = os.path.abspath(root)
        self.index_dir = os.path.join(self.root, INDEX_DIR)
        self.config = config or IndexConfig()
        self.chunker = ChunkerDispatch(max_chunk_chars=self.config.max_chunk_chars)

    # -- file discovery -----------------------------------------------

    def discover_files(self) -> list[str]:
        ignore = _load_gitignore(self.root) + self.config.exclude_globs
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [
                d for d in dirnames
                if d not in DEFAULT_EXCLUDE_DIRS and not d.startswith(".")
            ]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                if _matches_any(rel, ignore):
                    continue
                if self.config.include_globs and not _matches_any(
                    rel, self.config.include_globs
                ):
                    continue
                if detect_language(rel) is None:
                    continue
                try:
                    if os.path.getsize(full) > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                found.append(rel)
        found.sort()
        return found

    # -- indexing ------------------------------------------------------

    def build(self, full: bool = False) -> IndexStats:
        t0 = time.time()
        os.makedirs(self.index_dir, exist_ok=True)
        stats = IndexStats()

        manifest = {} if full else self._load_json("manifest.json") or {}
        old_hashes: dict[str, str] = manifest.get("files", {})
        old_chunks = [] if full else self._load_chunks()
        old_symbols = [] if full else self._load_symbols()

        files = self.discover_files()
        current_set = set(files)

        # Partition: unchanged / changed / removed.
        keep_chunks: dict[str, list[Chunk]] = {}
        keep_symbols: dict[str, list[Symbol]] = {}
        for c in old_chunks:
            keep_chunks.setdefault(c.file_path, []).append(c)
        for s in old_symbols:
            keep_symbols.setdefault(s.file_path, []).append(s)

        new_hashes: dict[str, str] = {}
        parsed: dict[str, tuple[list[Chunk], list[Symbol], list[str], str]] = {}
        unchanged: set[str] = set()

        for rel in files:
            full_path = os.path.join(self.root, rel)
            try:
                with open(full_path, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            h = _file_hash(data)
            new_hashes[rel] = h
            if not full and old_hashes.get(rel) == h:
                unchanged.add(rel)
                stats.files_skipped += 1
                continue
            language = detect_language(rel) or "text"
            try:
                source = data.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    source = data.decode("latin-1")
                except Exception:
                    continue
            chunks, symbols, imports = self.chunker.parse(rel, source, language)
            parsed[rel] = (chunks, symbols, imports, language)
            stats.files_indexed += 1

        removed = set(old_hashes) - current_set
        stats.files_removed = len(removed)

        # Assemble final chunk/symbol lists.
        all_chunks: list[Chunk] = []
        all_symbols: list[Symbol] = []
        # imports per unchanged file must be re-derived for graph: cheap
        # to re-parse imports only from stored symbols? Simpler: rebuild
        # graph from stored per-file import manifest.
        old_imports: dict[str, list[str]] = manifest.get("imports", {})
        new_imports: dict[str, list[str]] = {}

        for rel in files:
            if rel in unchanged:
                all_chunks.extend(keep_chunks.get(rel, []))
                all_symbols.extend(keep_symbols.get(rel, []))
                new_imports[rel] = old_imports.get(rel, [])
            elif rel in parsed:
                chunks, symbols, imports, _lang = parsed[rel]
                all_chunks.extend(chunks)
                all_symbols.extend(symbols)
                new_imports[rel] = imports

        stats.chunks = len(all_chunks)
        stats.symbols = len(all_symbols)

        # Knowledge graph (always rebuilt — it is cheap relative to
        # embedding and resolution is global by nature).
        builder = GraphBuilder()
        symbols_by_file: dict[str, list[Symbol]] = {}
        for s in all_symbols:
            symbols_by_file.setdefault(s.file_path, []).append(s)
        for rel in files:
            language = detect_language(rel) or "text"
            builder.add_file(rel, language,
                             symbols_by_file.get(rel, []),
                             new_imports.get(rel, []))
        graph = builder.finalize()

        # BM25 (rebuilt in memory — fast) with field boosting.
        bm25 = BM25Index()
        for c in all_chunks:
            tokens = tokenize(c.text)
            if c.symbol:
                tokens += tokenize(c.symbol) * 3
            if c.docstring:
                tokens += tokenize(c.docstring) * 2
            tokens += tokenize(c.file_path)
            bm25.add(c.chunk_id, tokens)
        bm25.finalize()

        # Vectors: re-embed only changed files, keep the rest.
        embedder = make_embedder(self.config.embedder)
        old_store = VectorStore.load(os.path.join(self.index_dir, "vectors.json"))
        reusable: dict[str, list[float]] = {}
        if (
            not full
            and old_store is not None
            and old_store.embedder_name == embedder.name
        ):
            reusable = dict(zip(old_store.ids, old_store.vectors))

        store = VectorStore(dim=embedder.dim, embedder_name=embedder.name)
        to_embed: list[Chunk] = []
        for c in all_chunks:
            if c.chunk_id in reusable and c.file_path in unchanged:
                store.add(c.chunk_id, reusable[c.chunk_id])
            else:
                to_embed.append(c)
        if to_embed:
            texts = [self._embed_text(c) for c in to_embed]
            for c, vec in zip(to_embed, embedder.embed(texts)):
                store.add(c.chunk_id, vec)

        # Persist.
        self._save_chunks(all_chunks)
        self._save_symbols(all_symbols)
        self._save_json("graph.json", graph.to_dict())
        self._save_json("bm25.json", bm25.to_dict())
        store.save(os.path.join(self.index_dir, "vectors.json"))
        self._save_json(
            "manifest.json",
            {
                "version": 1,
                "root": self.root,
                "embedder": embedder.name,
                "files": new_hashes,
                "imports": new_imports,
                "updated_at": int(t0),
            },
        )
        stats.seconds = round(time.time() - t0, 3)
        return stats

    @staticmethod
    def _embed_text(c: Chunk) -> str:
        parts = [c.file_path]
        if c.symbol:
            parts.append(c.symbol)
        if c.docstring:
            parts.append(c.docstring)
        parts.append(c.text[:4000])
        return "\n".join(parts)

    # -- persistence helpers -------------------------------------------

    def _save_json(self, name: str, obj) -> None:
        with open(os.path.join(self.index_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def _load_json(self, name: str):
        path = os.path.join(self.index_dir, name)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _save_chunks(self, chunks: list[Chunk]) -> None:
        with open(os.path.join(self.index_dir, "chunks.jsonl"), "w",
                  encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict()) + "\n")

    def _load_chunks(self) -> list[Chunk]:
        path = os.path.join(self.index_dir, "chunks.jsonl")
        if not os.path.exists(path):
            return []
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(Chunk.from_dict(json.loads(line)))
        return out

    def _save_symbols(self, symbols: list[Symbol]) -> None:
        with open(os.path.join(self.index_dir, "symbols.jsonl"), "w",
                  encoding="utf-8") as f:
            for s in symbols:
                f.write(json.dumps(s.to_dict()) + "\n")

    def _load_symbols(self) -> list[Symbol]:
        path = os.path.join(self.index_dir, "symbols.jsonl")
        if not os.path.exists(path):
            return []
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(Symbol.from_dict(json.loads(line)))
        return out


def load_graph(root: str) -> KnowledgeGraph:
    path = os.path.join(os.path.abspath(root), INDEX_DIR, "graph.json")
    with open(path, encoding="utf-8") as f:
        return KnowledgeGraph.from_dict(json.load(f))
