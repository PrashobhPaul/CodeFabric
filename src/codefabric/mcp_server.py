"""MCP server exposing CodeFabric to AI agents (Claude Code, etc.).

Requires the ``mcp`` extra (MIT-licensed reference SDK):

    pip install codefabric[mcp]
    codefabric mcp --path /repo

Registered tools give agents hybrid semantic search and impact
analysis on top of symbol-level navigation.
"""
from __future__ import annotations

import json

from .indexer import Indexer, IndexConfig
from .search import SearchEngine


def serve(root: str) -> None:  # pragma: no cover - needs mcp installed
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("codefabric")
    state: dict[str, SearchEngine] = {}

    def engine() -> SearchEngine:
        if "e" not in state:
            state["e"] = SearchEngine(root)
        return state["e"]

    @app.tool()
    def search_code(query: str, k: int = 8, language: str = "",
                    kind: str = "") -> str:
        """Hybrid (keyword + semantic + graph) search over the codebase.
        Returns ranked chunks with file:line provenance and related
        symbols (callers/callees). kind: function|method|class|block."""
        results = engine().search(query, k=k, language=language or None,
                                  kind=kind or None)
        return json.dumps([r.to_dict() for r in results], indent=2)

    @app.tool()
    def find_symbol(name: str) -> str:
        """Locate symbol definitions by exact or fuzzy name."""
        return json.dumps([s.to_dict() for s in engine().find_symbol(name)],
                          indent=2)

    @app.tool()
    def find_callers(name: str) -> str:
        """List symbols that call the named function/method."""
        return json.dumps(engine().callers(name), indent=2)

    @app.tool()
    def impact_analysis(name: str, max_depth: int = 4) -> str:
        """Transitive dependents of a symbol — what could break if it
        changes (via call, inheritance, and import edges)."""
        return json.dumps(engine().impact(name, max_depth=max_depth), indent=2)

    @app.tool()
    def file_outline(file_path: str) -> str:
        """Symbol outline (classes/functions with lines) for a file."""
        return json.dumps(engine().outline(file_path), indent=2)

    @app.tool()
    def reindex(full: bool = False) -> str:
        """Refresh the index after code changes (incremental by default)."""
        stats = Indexer(root, IndexConfig()).build(full=full)
        state.pop("e", None)
        return (f"indexed {stats.files_indexed} files, "
                f"{stats.chunks} chunks, {stats.symbols} symbols")

    @app.tool()
    def index_stats() -> str:
        """Index size, languages, and embedder info."""
        return json.dumps(engine().stats(), indent=2)

    app.run()
