"""Heuristic structural chunker for brace-style and generic languages.

When tree-sitter is not installed we still want chunk boundaries that
respect declarations rather than blind character windows. This chunker
finds declaration lines with per-language regexes, then walks balanced
braces to locate the end of each block. Anything between declarations
becomes merged block chunks. Files in non-code languages fall back to
blank-line-aware sliding windows.

If the optional ``ast`` extra (tree-sitter + grammars) is installed,
``codefabric.chunking.base`` prefers the tree-sitter chunker and this
module is only a fallback.
"""
from __future__ import annotations

import hashlib
import re

from ..models import Chunk, Symbol

# language -> list of (kind, regex) for declaration starts.
_DECL_PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "javascript": [
        ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>")),
    ],
    "typescript": [
        ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)")),
        ("interface", re.compile(r"^\s*(?:export\s+)?interface\s+(\w+)")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>")),
    ],
    "java": [
        ("class", re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(?:class|interface|enum|record)\s+(\w+)")),
        ("method", re.compile(r"^\s*(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?[\w<>\[\],\s]+\s+(\w+)\s*\([^;]*$")),
    ],
    "go": [
        ("function", re.compile(r"^func\s+(?:\([^)]+\)\s+)?(\w+)")),
        ("class", re.compile(r"^type\s+(\w+)\s+(?:struct|interface)\b")),
    ],
    "rust": [
        ("function", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)")),
        ("class", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+(\w+)")),
        ("class", re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:\w+\s+for\s+)?(\w+)")),
    ],
    "csharp": [
        ("class", re.compile(r"^\s*(?:public|private|protected|internal)?\s*(?:static\s+|sealed\s+|abstract\s+|partial\s+)*(?:class|interface|struct|record|enum)\s+(\w+)")),
        ("method", re.compile(r"^\s*(?:public|private|protected|internal)\s+(?:static\s+|virtual\s+|override\s+|async\s+)*[\w<>\[\],\s]+\s+(\w+)\s*\(")),
    ],
    "c": [
        ("function", re.compile(r"^[\w\*\s]+\b(\w+)\s*\([^;]*\)\s*\{?\s*$")),
    ],
    "cpp": [
        ("class", re.compile(r"^\s*(?:class|struct)\s+(\w+)")),
        ("function", re.compile(r"^[\w:\*&<>,\s]+\b([\w:~]+)\s*\([^;]*\)\s*(?:const)?\s*\{?\s*$")),
    ],
    "kotlin": [
        ("class", re.compile(r"^\s*(?:data\s+|sealed\s+|abstract\s+|open\s+)*(?:class|interface|object)\s+(\w+)")),
        ("function", re.compile(r"^\s*(?:override\s+|suspend\s+|private\s+|public\s+|internal\s+)*fun\s+(?:<[^>]*>\s*)?(\w+)")),
    ],
    "ruby": [
        ("class", re.compile(r"^\s*(?:class|module)\s+(\w+)")),
        ("function", re.compile(r"^\s*def\s+(?:self\.)?([\w?!=\[\]]+)")),
    ],
    "php": [
        ("class", re.compile(r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait)\s+(\w+)")),
        ("function", re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?function\s+(\w+)")),
    ],
    "scala": [
        ("class", re.compile(r"^\s*(?:case\s+)?(?:class|object|trait)\s+(\w+)")),
        ("function", re.compile(r"^\s*(?:override\s+|private\s+|protected\s+)*def\s+(\w+)")),
    ],
    "swift": [
        ("class", re.compile(r"^\s*(?:public\s+|open\s+|final\s+)*(?:class|struct|enum|protocol|extension)\s+(\w+)")),
        ("function", re.compile(r"^\s*(?:public\s+|private\s+|internal\s+|static\s+|override\s+)*func\s+(\w+)")),
    ],
}

_IMPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    "javascript": re.compile(r"""(?:import\s+.*?from\s+|require\s*\(\s*)['"]([^'"]+)['"]"""),
    "typescript": re.compile(r"""(?:import\s+.*?from\s+|require\s*\(\s*)['"]([^'"]+)['"]"""),
    "java": re.compile(r"^import\s+(?:static\s+)?([\w.]+)", re.M),
    "go": re.compile(r'"([\w./-]+)"'),
    "rust": re.compile(r"^\s*use\s+([\w:]+)", re.M),
    "csharp": re.compile(r"^using\s+([\w.]+)", re.M),
    "kotlin": re.compile(r"^import\s+([\w.]+)", re.M),
    "ruby": re.compile(r"""require(?:_relative)?\s+['"]([^'"]+)['"]"""),
    "php": re.compile(r"^use\s+([\w\\]+)", re.M),
    "scala": re.compile(r"^import\s+([\w.]+)", re.M),
}

# Ruby/others use `end` keywords rather than braces.
_END_BLOCK_LANGS = {"ruby"}


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8", "replace")).hexdigest()[:16]


def _block_end_by_braces(lines: list[str], start_idx: int, max_lines: int = 800) -> int:
    """Find the line index (0-based) where the brace block opened at/after
    start_idx closes. Returns start_idx if no block is found."""
    depth = 0
    opened = False
    limit = min(len(lines), start_idx + max_lines)
    for i in range(start_idx, limit):
        line = lines[i]
        # Cheap string/comment stripping — good enough for boundaries.
        line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
        line = re.sub(r"'(?:\\.|[^'\\])*'", "''", line)
        line = re.sub(r"//.*$|#.*$", "", line)
        depth += line.count("{") - line.count("}")
        if line.count("{"):
            opened = True
        if opened and depth <= 0:
            return i
        # Declaration without a body (interface method, prototype).
        if not opened and line.rstrip().endswith(";"):
            return i
    return start_idx if not opened else limit - 1


def _block_end_by_indent(lines: list[str], start_idx: int) -> int:
    """Ruby-style: find matching `end` by def/do/class/if nesting."""
    opener = re.compile(r"^\s*(?:def|class|module|if|unless|case|while|until|begin|do)\b|\bdo\s*(?:\|[^|]*\|)?\s*$")
    depth = 0
    for i in range(start_idx, min(len(lines), start_idx + 800)):
        stripped = lines[i].strip()
        if opener.search(lines[i]) and not stripped.endswith("end"):
            depth += 1
        if re.match(r"^\s*end\b", lines[i]):
            depth -= 1
            if depth <= 0:
                return i
    return start_idx


class HeuristicChunker:
    def __init__(self, max_chunk_chars: int = 2400, window_lines: int = 60,
                 overlap_lines: int = 10):
        self.max_chunk_chars = max_chunk_chars
        self.window_lines = window_lines
        self.overlap_lines = overlap_lines

    def parse(
        self, file_path: str, source: str, language: str
    ) -> tuple[list[Chunk], list[Symbol], list[str]]:
        lines = source.splitlines()
        if not lines:
            return [], [], []

        imports: list[str] = []
        imp_re = _IMPORT_PATTERNS.get(language)
        if imp_re:
            imports = list(dict.fromkeys(imp_re.findall(source)))

        patterns = _DECL_PATTERNS.get(language)
        if not patterns:
            return self._window_chunks(file_path, lines, language), [], imports

        chunks: list[Chunk] = []
        symbols: list[Symbol] = []
        covered_until = -1  # last 0-based line already chunked
        pending_start: int | None = None

        def flush_gap(upto: int) -> None:
            nonlocal pending_start
            if pending_start is None:
                return
            gap = "\n".join(lines[pending_start : upto + 1])
            if gap.strip():
                for c in self._window_chunks(
                    file_path, lines[pending_start : upto + 1], language,
                    line_offset=pending_start,
                ):
                    chunks.append(c)
            pending_start = None

        i = 0
        while i < len(lines):
            if i <= covered_until:
                i += 1
                continue
            matched = None
            for kind, pat in patterns:
                m = pat.match(lines[i])
                if m:
                    matched = (kind, m.group(1))
                    break
            if not matched:
                if pending_start is None:
                    pending_start = i
                i += 1
                continue

            flush_gap(i - 1)
            kind, name = matched
            if language in _END_BLOCK_LANGS:
                end_idx = _block_end_by_indent(lines, i)
            else:
                end_idx = _block_end_by_braces(lines, i)
            end_idx = max(end_idx, i)
            text = "\n".join(lines[i : end_idx + 1])
            sym_kind = "class" if kind in ("class", "interface") else "function"
            symbols.append(
                Symbol(
                    name=name,
                    qualified_name=name,
                    kind=sym_kind,
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=end_idx + 1,
                    signature=lines[i].strip()[:200],
                )
            )
            if len(text) <= self.max_chunk_chars * 2:
                chunks.append(
                    Chunk(
                        chunk_id=_hash_id(file_path, name, str(i + 1)),
                        file_path=file_path,
                        language=language,
                        start_line=i + 1,
                        end_line=end_idx + 1,
                        text=text,
                        kind=sym_kind,
                        symbol=name,
                    )
                )
            else:
                # Oversized block: window it, but keep the declaration
                # line as context on each piece.
                header = lines[i].strip()
                for c in self._window_chunks(
                    file_path, lines[i : end_idx + 1], language, line_offset=i,
                    header=f"// context: {header}",
                ):
                    c.symbol = name
                    chunks.append(c)
            covered_until = end_idx
            i = end_idx + 1

        flush_gap(len(lines) - 1)
        return chunks, symbols, imports

    def _window_chunks(
        self,
        file_path: str,
        lines: list[str],
        language: str,
        line_offset: int = 0,
        header: str | None = None,
    ) -> list[Chunk]:
        """Blank-line-aware sliding windows for non-structural text."""
        chunks: list[Chunk] = []
        n = len(lines)
        start = 0
        while start < n:
            end = min(start + self.window_lines, n)
            # Prefer to break on a blank line near the window end.
            if end < n:
                for j in range(end, max(start + self.window_lines // 2, start + 1), -1):
                    if not lines[j - 1].strip():
                        end = j
                        break
            text = "\n".join(lines[start:end])
            if text.strip():
                body = f"{header}\n{text}" if header else text
                chunks.append(
                    Chunk(
                        chunk_id=_hash_id(file_path, str(line_offset + start), text[:64]),
                        file_path=file_path,
                        language=language,
                        start_line=line_offset + start + 1,
                        end_line=line_offset + end,
                        text=body,
                        kind="block",
                    )
                )
            if end >= n:
                break
            start = max(end - self.overlap_lines, start + 1)
        return chunks
