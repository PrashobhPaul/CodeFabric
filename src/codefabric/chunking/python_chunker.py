"""Structure-aware chunking + symbol extraction for Python.

Uses the stdlib ``ast`` module (no external parser needed) and follows
the cAST recipe: recurse into nodes that exceed the size budget, merge
small adjacent siblings, and never cut through a syntactic unit unless
that unit alone exceeds the budget.

Also extracts a symbol table (classes, functions, methods) with
signatures, docstrings, base classes, and best-effort callee names for
the knowledge graph.
"""
from __future__ import annotations

import ast
import hashlib

from ..models import Chunk, Symbol


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8", "replace")).hexdigest()[:16]


def _segment(lines: list[str], start: int, end: int) -> str:
    """Return source lines start..end (1-based inclusive)."""
    return "\n".join(lines[start - 1 : end])


def _signature(node: ast.AST, lines: list[str]) -> str:
    """First line of a def/class statement, trimmed."""
    line = lines[node.lineno - 1].strip()
    return line[:200]


def _called_names(node: ast.AST) -> list[str]:
    """Best-effort list of function/method names called inside a node."""
    names: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Name):
                names.append(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.append(fn.attr)
    # Dedupe preserving order.
    seen: set[str] = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _base_names(node: ast.ClassDef) -> list[str]:
    out = []
    for b in node.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
    return out


class PythonChunker:
    def __init__(self, max_chunk_chars: int = 2400, min_merge_chars: int = 300):
        self.max_chunk_chars = max_chunk_chars
        self.min_merge_chars = min_merge_chars

    def parse(
        self, file_path: str, source: str
    ) -> tuple[list[Chunk], list[Symbol], list[str]]:
        """Return (chunks, symbols, imported_modules)."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return [], [], []

        lines = source.splitlines()
        chunks: list[Chunk] = []
        symbols: list[Symbol] = []
        imports: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append("." * node.level + node.module)

        self._walk_body(
            tree.body, file_path, lines, prefix="", chunks=chunks, symbols=symbols
        )
        return chunks, symbols, imports

    # ------------------------------------------------------------------

    def _walk_body(
        self,
        body: list[ast.stmt],
        file_path: str,
        lines: list[str],
        prefix: str,
        chunks: list[Chunk],
        symbols: list[Symbol],
        class_header: str = "",
    ) -> None:
        """cAST-style traversal of a statement list.

        Definition nodes become their own chunks (recursing when too
        large); runs of small non-definition statements are merged into
        block chunks.
        """
        pending: list[ast.stmt] = []  # small non-def statements to merge

        def flush_pending() -> None:
            if not pending:
                return
            start = pending[0].lineno
            end = max(getattr(s, "end_lineno", s.lineno) for s in pending)
            text = _segment(lines, start, end)
            if text.strip():
                self._emit_block(file_path, lines, start, end, chunks, class_header)
            pending.clear()

        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                flush_pending()
                self._handle_definition(
                    stmt, file_path, lines, prefix, chunks, symbols, class_header
                )
            else:
                pending.append(stmt)
                start = pending[0].lineno
                end = max(getattr(s, "end_lineno", s.lineno) for s in pending)
                if len(_segment(lines, start, end)) >= self.max_chunk_chars:
                    flush_pending()
        flush_pending()

    def _emit_block(
        self,
        file_path: str,
        lines: list[str],
        start: int,
        end: int,
        chunks: list[Chunk],
        class_header: str,
    ) -> None:
        text = _segment(lines, start, end)
        # Split blocks that still exceed the budget on blank lines.
        if len(text) > self.max_chunk_chars * 1.5:
            mid = start + (end - start) // 2
            self._emit_block(file_path, lines, start, mid, chunks, class_header)
            self._emit_block(file_path, lines, mid + 1, end, chunks, class_header)
            return
        body = f"{class_header}{text}" if class_header else text
        chunks.append(
            Chunk(
                chunk_id=_hash_id(file_path, str(start), str(end), text[:64]),
                file_path=file_path,
                language="python",
                start_line=start,
                end_line=end,
                text=body,
                kind="block",
            )
        )

    def _handle_definition(
        self,
        node: ast.stmt,
        file_path: str,
        lines: list[str],
        prefix: str,
        chunks: list[Chunk],
        symbols: list[Symbol],
        class_header: str = "",
    ) -> None:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        name = node.name
        qualified = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        if not prefix:
            qualified = name
        start, end = node.lineno, node.end_lineno or node.lineno
        # Include decorators in the chunk.
        if node.decorator_list:
            start = min(d.lineno for d in node.decorator_list)
        text = _segment(lines, start, end)
        doc = ast.get_docstring(node) or ""
        sig = _signature(node, lines)

        if isinstance(node, ast.ClassDef):
            kind = "class"
        elif class_header or "." in qualified:
            kind = "method"
        else:
            kind = "function"

        symbols.append(
            Symbol(
                name=name,
                qualified_name=qualified,
                kind=kind,
                file_path=file_path,
                start_line=start,
                end_line=end,
                signature=sig,
                docstring=doc[:500],
                bases=_base_names(node) if isinstance(node, ast.ClassDef) else [],
                calls=_called_names(node),
            )
        )

        if len(text) <= self.max_chunk_chars or not isinstance(node, ast.ClassDef):
            # Whole definition fits (functions are kept whole even when
            # large — splitting a function body hurts retrieval quality
            # more than a long chunk does).
            body = f"{class_header}{text}" if class_header else text
            chunks.append(
                Chunk(
                    chunk_id=_hash_id(file_path, qualified, str(start)),
                    file_path=file_path,
                    language="python",
                    start_line=start,
                    end_line=end,
                    text=body,
                    kind=kind,
                    symbol=qualified,
                    docstring=doc[:500] or None,
                )
            )
            # Even when the class is emitted as a single chunk, its
            # methods still belong in the symbol table for the graph.
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                self._collect_nested_symbols(node, file_path, lines, qualified, symbols)
            return

        # Oversized class: emit a header chunk (signature + docstring +
        # class-level assignments) and recurse into methods with the
        # class signature prepended for context.
        header_end = node.body[0].lineno - 1 if node.body else start
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            header_end = first.end_lineno or header_end
        header_text = _segment(lines, start, max(header_end, start))
        chunks.append(
            Chunk(
                chunk_id=_hash_id(file_path, qualified, "header"),
                file_path=file_path,
                language="python",
                start_line=start,
                end_line=max(header_end, start),
                text=header_text,
                kind="class",
                symbol=qualified,
                docstring=doc[:500] or None,
            )
        )
        ctx = f"# context: {sig}\n"
        self._walk_body(  # also collects nested symbols

            node.body,
            file_path,
            lines,
            prefix=qualified,
            chunks=chunks,
            symbols=symbols,
            class_header=ctx,
        )

    def _collect_nested_symbols(
        self,
        node: ast.stmt,
        file_path: str,
        lines: list[str],
        prefix: str,
        symbols: list[Symbol],
    ) -> None:
        """Record symbols for definitions nested inside ``node`` without
        emitting chunks for them (the parent chunk already covers the
        text)."""
        for stmt in ast.iter_child_nodes(node):
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            qualified = f"{prefix}.{stmt.name}"
            start = stmt.lineno
            if stmt.decorator_list:
                start = min(d.lineno for d in stmt.decorator_list)
            if isinstance(stmt, ast.ClassDef):
                kind = "class"
            elif isinstance(node, ast.ClassDef):
                kind = "method"
            else:
                kind = "function"
            symbols.append(
                Symbol(
                    name=stmt.name,
                    qualified_name=qualified,
                    kind=kind,
                    file_path=file_path,
                    start_line=start,
                    end_line=stmt.end_lineno or start,
                    signature=_signature(stmt, lines),
                    docstring=(ast.get_docstring(stmt) or "")[:500],
                    bases=_base_names(stmt) if isinstance(stmt, ast.ClassDef) else [],
                    calls=_called_names(stmt),
                )
            )
            self._collect_nested_symbols(stmt, file_path, lines, qualified, symbols)
