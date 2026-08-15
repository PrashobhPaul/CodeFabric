"""Optional tree-sitter chunker (requires the ``ast`` extra).

Provides precise AST boundaries for non-Python languages using the
MIT-licensed tree-sitter runtime and grammar pack. Loaded lazily by
``ChunkerDispatch``; the system works without it.
"""
from __future__ import annotations

import hashlib

from ..models import Chunk, Symbol

try:  # pragma: no cover - exercised only when extra is installed
    from tree_sitter_language_pack import get_parser
except ImportError as _e:  # pragma: no cover
    raise ImportError("install codefabric[ast] for tree-sitter support") from _e

# Node types that constitute definition boundaries, per language.
_DEF_NODES: dict[str, dict[str, str]] = {
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "interface_declaration": "class",
        "method_definition": "method",
    },
    "java": {
        "class_declaration": "class",
        "interface_declaration": "class",
        "enum_declaration": "class",
        "record_declaration": "class",
        "method_declaration": "method",
        "constructor_declaration": "method",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "class",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "class",
        "enum_item": "class",
        "trait_item": "class",
        "impl_item": "class",
    },
    "c": {"function_definition": "function", "struct_specifier": "class"},
    "cpp": {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "class",
    },
    "csharp": {
        "class_declaration": "class",
        "interface_declaration": "class",
        "struct_declaration": "class",
        "method_declaration": "method",
    },
    "ruby": {"method": "function", "class": "class", "module": "class"},
    "kotlin": {"function_declaration": "function", "class_declaration": "class"},
    "php": {"function_definition": "function", "class_declaration": "class"},
    "scala": {"function_definition": "function", "class_definition": "class"},
    "swift": {"function_declaration": "function", "class_declaration": "class"},
}

_TS_LANG_IDS = {
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "go": "go",
    "rust": "rust",
    "c": "c",
    "cpp": "cpp",
    "csharp": "csharp",
    "ruby": "ruby",
    "kotlin": "kotlin",
    "php": "php",
    "scala": "scala",
    "swift": "swift",
}


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8", "replace")).hexdigest()[:16]


def _node_name(node) -> str:
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "field_identifier",
                          "name", "constant"):
            return child.text.decode("utf-8", "replace")
    # e.g. `name:` field
    name_field = node.child_by_field_name("name")
    if name_field is not None:
        return name_field.text.decode("utf-8", "replace")
    return "<anonymous>"


class TreeSitterChunker:
    def __init__(self, max_chunk_chars: int = 2400):
        self.max_chunk_chars = max_chunk_chars
        self._parsers: dict[str, object] = {}

    def supports(self, language: str) -> bool:
        return language in _TS_LANG_IDS

    def _parser(self, language: str):
        if language not in self._parsers:
            self._parsers[language] = get_parser(_TS_LANG_IDS[language])
        return self._parsers[language]

    def parse(
        self, file_path: str, source: str, language: str
    ) -> tuple[list[Chunk], list[Symbol], list[str]]:
        parser = self._parser(language)
        tree = parser.parse(source.encode("utf-8"))
        def_map = _DEF_NODES.get(language, {})
        lines = source.splitlines()

        chunks: list[Chunk] = []
        symbols: list[Symbol] = []

        def emit(node, kind: str, prefix: str) -> None:
            name = _node_name(node)
            qualified = f"{prefix}.{name}" if prefix else name
            s, e = node.start_point[0] + 1, node.end_point[0] + 1
            text = "\n".join(lines[s - 1 : e])
            symbols.append(
                Symbol(
                    name=name,
                    qualified_name=qualified,
                    kind="class" if kind == "class" else
                    ("method" if kind == "method" or prefix else "function"),
                    file_path=file_path,
                    start_line=s,
                    end_line=e,
                    signature=lines[s - 1].strip()[:200] if s <= len(lines) else "",
                )
            )
            if len(text) <= self.max_chunk_chars or kind != "class":
                chunks.append(
                    Chunk(
                        chunk_id=_hash_id(file_path, qualified, str(s)),
                        file_path=file_path,
                        language=language,
                        start_line=s,
                        end_line=e,
                        text=text,
                        kind=kind,
                        symbol=qualified,
                    )
                )
            else:
                walk(node, qualified)  # split large classes into members

        def walk(node, prefix: str) -> None:
            for child in node.children:
                kind = def_map.get(child.type)
                if kind:
                    emit(child, kind, prefix)
                else:
                    walk(child, prefix)

        walk(tree.root_node, "")

        # Anything not covered by a definition becomes block chunks via
        # the heuristic windower (import blocks, top-level statements).
        from .heuristic_chunker import HeuristicChunker, _IMPORT_PATTERNS

        covered = [False] * (len(lines) + 1)
        for c in chunks:
            for i in range(c.start_line, min(c.end_line, len(lines)) + 1):
                covered[i] = True
        h = HeuristicChunker(max_chunk_chars=self.max_chunk_chars)
        run_start = None
        for i in range(1, len(lines) + 2):
            in_gap = i <= len(lines) and not covered[i] and lines[i - 1].strip()
            if in_gap and run_start is None:
                run_start = i
            elif not in_gap and run_start is not None:
                gap_lines = lines[run_start - 1 : i - 1]
                chunks.extend(
                    h._window_chunks(file_path, gap_lines, language,
                                     line_offset=run_start - 1)
                )
                run_start = None

        imp_re = _IMPORT_PATTERNS.get(language)
        imports = list(dict.fromkeys(imp_re.findall(source))) if imp_re else []
        return chunks, symbols, imports
