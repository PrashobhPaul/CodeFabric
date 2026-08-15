"""Chunker dispatch: pick the best available parser per language.

Order of preference:
  1. Python  -> stdlib ``ast`` chunker (always available, full fidelity)
  2. Others  -> tree-sitter chunker when the ``ast`` extra is installed
  3. Fallback -> heuristic declaration/brace chunker (pure stdlib)
"""
from __future__ import annotations

from ..models import Chunk, Symbol
from .python_chunker import PythonChunker
from .heuristic_chunker import HeuristicChunker

ParseResult = tuple[list[Chunk], list[Symbol], list[str]]


class ChunkerDispatch:
    def __init__(self, max_chunk_chars: int = 2400):
        self.python = PythonChunker(max_chunk_chars=max_chunk_chars)
        self.heuristic = HeuristicChunker(max_chunk_chars=max_chunk_chars)
        self._treesitter = None
        self._treesitter_checked = False

    def _get_treesitter(self):
        if not self._treesitter_checked:
            self._treesitter_checked = True
            try:
                from .treesitter_chunker import TreeSitterChunker  # noqa: WPS433

                self._treesitter = TreeSitterChunker(
                    max_chunk_chars=self.heuristic.max_chunk_chars
                )
            except Exception:
                self._treesitter = None
        return self._treesitter

    def parse(self, file_path: str, source: str, language: str) -> ParseResult:
        if language == "python":
            chunks, symbols, imports = self.python.parse(file_path, source)
            if chunks or not source.strip():
                return chunks, symbols, imports
            # SyntaxError fallback: still index the text.
            return self.heuristic.parse(file_path, source, "text")

        ts = self._get_treesitter()
        if ts is not None and ts.supports(language):
            try:
                return ts.parse(file_path, source, language)
            except Exception:
                pass  # fall through to heuristics
        return self.heuristic.parse(file_path, source, language)
