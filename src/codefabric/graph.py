"""Code knowledge graph.

A lightweight directed multigraph (pure stdlib — no networkx dependency)
over files and symbols with typed edges:

    file   --contains-->  symbol
    file   --imports-->   file
    symbol --calls-->     symbol
    class  --inherits-->  class

This is the "fabric" that connects retrieval hits to their structural
neighborhood: callers, callees, subclasses, and importing modules. It
enables relational queries embeddings cannot answer (impact analysis,
reference walks) and lets search results carry graph context.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

from .models import Symbol


class KnowledgeGraph:
    def __init__(self) -> None:
        # node_id -> attrs
        self.nodes: dict[str, dict[str, Any]] = {}
        # edge_type -> src -> set(dst)
        self.out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.inc: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    # -- construction --------------------------------------------------

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.nodes.setdefault(node_id, {}).update(attrs)

    def add_edge(self, src: str, dst: str, edge_type: str) -> None:
        if src == dst:
            return
        self.add_node(src)
        self.add_node(dst)
        self.out[edge_type][src].add(dst)
        self.inc[edge_type][dst].add(src)

    def remove_file(self, file_path: str) -> None:
        """Drop a file node and every symbol it contains (for incremental
        re-indexing)."""
        file_id = f"file:{file_path}"
        doomed = {file_id}
        doomed.update(self.out["contains"].get(file_id, set()))
        for node_id in doomed:
            self.nodes.pop(node_id, None)
            for etype in list(self.out.keys()):
                for dst in self.out[etype].pop(node_id, set()):
                    self.inc[etype][dst].discard(node_id)
                for src in self.inc[etype].pop(node_id, set()):
                    self.out[etype][src].discard(node_id)

    # -- queries -------------------------------------------------------

    def neighbors(self, node_id: str, edge_type: str, reverse: bool = False) -> set[str]:
        table = self.inc if reverse else self.out
        return set(table[edge_type].get(node_id, set()))

    def callers_of(self, symbol_id: str) -> set[str]:
        return self.neighbors(symbol_id, "calls", reverse=True)

    def callees_of(self, symbol_id: str) -> set[str]:
        return self.neighbors(symbol_id, "calls")

    def subclasses_of(self, symbol_id: str) -> set[str]:
        return self.neighbors(symbol_id, "inherits", reverse=True)

    def importers_of(self, file_path: str) -> set[str]:
        return self.neighbors(f"file:{file_path}", "imports", reverse=True)

    def impact(self, node_id: str, max_depth: int = 4,
               edge_types: Iterable[str] = ("calls", "inherits", "imports"),
               limit: int = 200) -> list[tuple[str, int, str]]:
        """Transitive dependents of a node: everything that could break
        if it changes. Returns (node_id, depth, via_edge_type) tuples,
        breadth-first."""
        seen = {node_id}
        results: list[tuple[str, int, str]] = []
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        while queue and len(results) < limit:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for etype in edge_types:
                for dep in self.inc[etype].get(current, set()):
                    if dep in seen:
                        continue
                    seen.add(dep)
                    results.append((dep, depth + 1, etype))
                    queue.append((dep, depth + 1))
        return results

    # -- persistence ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": {
                etype: {src: sorted(dsts) for src, dsts in table.items() if dsts}
                for etype, table in self.out.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KnowledgeGraph":
        g = cls()
        for node_id, attrs in d.get("nodes", {}).items():
            g.add_node(node_id, **attrs)
        for etype, table in d.get("edges", {}).items():
            for src, dsts in table.items():
                for dst in dsts:
                    g.add_edge(src, dst, etype)
        return g


def symbol_node_id(sym: Symbol) -> str:
    return f"sym:{sym.file_path}::{sym.qualified_name}"


class GraphBuilder:
    """Builds the knowledge graph from per-file parse results.

    Call resolution is best-effort and name-based: a call edge is drawn
    when the callee name uniquely-ish matches a known symbol (same file
    preferred, then global name table). That trades a few false edges
    for zero external analyzer dependencies; precision improves when the
    tree-sitter extra provides better symbol tables.
    """

    def __init__(self) -> None:
        self.graph = KnowledgeGraph()
        # name -> [symbol node ids]
        self._by_name: dict[str, list[str]] = defaultdict(list)
        self._symbols: dict[str, Symbol] = {}
        # module path guesses: "pkg/mod.py" -> "pkg.mod"
        self._module_index: dict[str, str] = {}
        self._pending_imports: list[tuple[str, str]] = []
        self._pending_calls: list[tuple[str, str, str]] = []
        self._pending_bases: list[tuple[str, str, str]] = []

    def add_file(self, file_path: str, language: str,
                 symbols: list[Symbol], imports: list[str]) -> None:
        file_id = f"file:{file_path}"
        self.graph.add_node(file_id, type="file", path=file_path, language=language)
        module = _path_to_module(file_path)
        if module:
            self._module_index[module] = file_path

        for sym in symbols:
            sid = symbol_node_id(sym)
            self._symbols[sid] = sym
            self._by_name[sym.name].append(sid)
            self.graph.add_node(
                sid, type="symbol", name=sym.name, qualified=sym.qualified_name,
                kind=sym.kind, path=sym.file_path, line=sym.start_line,
            )
            self.graph.add_edge(file_id, sid, "contains")
            for callee in sym.calls:
                self._pending_calls.append((sid, file_path, callee))
            for base in sym.bases:
                self._pending_bases.append((sid, file_path, base))

        for imp in imports:
            self._pending_imports.append((file_path, imp))

    def finalize(self) -> KnowledgeGraph:
        # imports: resolve module strings to indexed files.
        for src_file, module in self._pending_imports:
            target = self._resolve_module(src_file, module)
            if target:
                self.graph.add_edge(f"file:{src_file}", f"file:{target}", "imports")

        # calls / inheritance: name-based resolution.
        for sid, src_file, name in self._pending_calls:
            target = self._resolve_name(name, src_file)
            if target and target != sid:
                self.graph.add_edge(sid, target, "calls")
        for sid, src_file, name in self._pending_bases:
            target = self._resolve_name(name, src_file, kinds=("class",))
            if target and target != sid:
                self.graph.add_edge(sid, target, "inherits")
        return self.graph

    # ------------------------------------------------------------------

    def _resolve_name(self, name: str, src_file: str,
                      kinds: tuple[str, ...] | None = None) -> str | None:
        candidates = self._by_name.get(name, [])
        if kinds:
            candidates = [c for c in candidates
                          if self._symbols[c].kind in kinds]
        if not candidates:
            return None
        same_file = [c for c in candidates if self._symbols[c].file_path == src_file]
        if same_file:
            return same_file[0]
        if len(candidates) <= 3:  # avoid wild edges on very common names
            return candidates[0]
        return None

    def _resolve_module(self, src_file: str, module: str) -> str | None:
        # Relative imports: "." prefixed (python) or "./" (js/ts).
        module = module.strip()
        if module.startswith("./") or module.startswith("../"):
            import posixpath

            base = posixpath.dirname(src_file)
            candidate = posixpath.normpath(posixpath.join(base, module))
            for suffix in ("", ".py", ".js", ".ts", ".jsx", ".tsx", "/index.js",
                           "/index.ts", "/__init__.py"):
                probe = candidate + suffix
                if f"file:{probe}" in self.graph.nodes:
                    return probe
            return None
        dotted = module.lstrip(".")
        # Longest-prefix match against indexed modules.
        parts = dotted.split(".")
        while parts:
            probe = ".".join(parts)
            if probe in self._module_index:
                return self._module_index[probe]
            parts.pop()
        return None


def _path_to_module(file_path: str) -> str | None:
    if not file_path.endswith(".py"):
        return None
    p = file_path[: -len(".py")]
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    parts = [seg for seg in p.split("/") if seg not in ("", ".", "src")]
    return ".".join(parts) if parts else None
