"""codefabric CLI — argparse only, no third-party dependencies.

    codefabric index [PATH] [--full] [--embedder hash|st:MODEL]
    codefabric search QUERY [--path P] [--k N] [--lang L] [--kind K] [--json]
    codefabric symbol NAME [--path P]
    codefabric callers NAME [--path P]
    codefabric impact NAME [--path P] [--depth N]
    codefabric outline FILE [--path P]
    codefabric stats [--path P]
    codefabric mcp [--path P]
"""
from __future__ import annotations

import argparse
import json
import sys

from .indexer import Indexer, IndexConfig
from .search import SearchEngine


def _engine(path: str) -> SearchEngine:
    try:
        return SearchEngine(path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)


def cmd_index(args: argparse.Namespace) -> None:
    cfg = IndexConfig(embedder=args.embedder,
                      exclude_globs=args.exclude or [])
    stats = Indexer(args.path, cfg).build(full=args.full)
    print(
        f"indexed {stats.files_indexed} files "
        f"({stats.files_skipped} unchanged, {stats.files_removed} removed) "
        f"-> {stats.chunks} chunks, {stats.symbols} symbols "
        f"in {stats.seconds}s"
    )


def cmd_search(args: argparse.Namespace) -> None:
    engine = _engine(args.path)
    results = engine.search(
        args.query, k=args.k, language=args.lang, kind=args.kind,
        use_dense=not args.no_dense,
    )
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return
    if not results:
        print("no results")
        return
    for i, r in enumerate(results, 1):
        c = r.chunk
        head = f"{c.file_path}:{c.start_line}-{c.end_line}"
        sym = f"  [{c.kind}] {c.symbol}" if c.symbol else f"  [{c.kind}]"
        print(f"{i:2}. {head}{sym}  (score {r.score:.4f}, {'+'.join(r.sources)})")
        preview = c.text.strip().splitlines()
        for line in preview[: args.preview]:
            print(f"      {line[:160]}")
        for rel in r.related[:3]:
            print(f"      ~ {rel['relation']}: {rel['symbol']} "
                  f"({rel['file']}:{rel['line']})")
        print()


def cmd_symbol(args: argparse.Namespace) -> None:
    engine = _engine(args.path)
    syms = engine.find_symbol(args.name)
    if not syms:
        print("symbol not found")
        return
    for s in syms:
        print(f"[{s.kind}] {s.qualified_name}  {s.file_path}:{s.start_line}")
        if s.signature:
            print(f"    {s.signature}")
        if s.docstring:
            print(f"    \"{s.docstring[:140]}\"")


def cmd_callers(args: argparse.Namespace) -> None:
    engine = _engine(args.path)
    rows = engine.callers(args.name)
    if not rows:
        print("no callers found (or symbol unknown)")
        return
    for r in rows:
        print(f"{r['symbol']}  {r['file']}:{r['line']}")


def cmd_impact(args: argparse.Namespace) -> None:
    engine = _engine(args.path)
    rows = engine.impact(args.name, max_depth=args.depth)
    if not rows:
        print("no dependents found (or symbol unknown)")
        return
    for r in rows:
        indent = "  " * (r["depth"] - 1)
        loc = f"{r['file']}:{r['line']}" if r.get("line") else r.get("file", "")
        print(f"{indent}{r['name']}  ({r['via']})  {loc}")


def cmd_outline(args: argparse.Namespace) -> None:
    engine = _engine(args.path)
    rows = engine.outline(args.file)
    if not rows:
        print("no symbols found for file")
        return
    for r in rows:
        print(f"{r['line']:5}  [{r['kind']}] {r['name']}")
        if r["doc"]:
            print(f"        {r['doc']}")


def cmd_stats(args: argparse.Namespace) -> None:
    print(json.dumps(_engine(args.path).stats(), indent=2))


def cmd_mcp(args: argparse.Namespace) -> None:
    try:
        from .mcp_server import serve
    except ImportError:
        print("MCP support requires: pip install codefabric[mcp]", file=sys.stderr)
        raise SystemExit(2)
    serve(args.path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codefabric",
        description="Structure-aware RAG knowledge fabric for codebases",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_path(sp):
        sp.add_argument("--path", default=".", help="repository root")

    sp = sub.add_parser("index", help="build or update the index")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("--full", action="store_true", help="ignore cached state")
    sp.add_argument("--embedder", default="hash",
                    help="'hash' (default, offline) or 'st:<model>' (needs [ml])")
    sp.add_argument("--exclude", action="append", help="glob to exclude")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("search", help="hybrid search")
    sp.add_argument("query")
    add_path(sp)
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--lang", default=None)
    sp.add_argument("--kind", default=None,
                    choices=[None, "function", "method", "class", "block"])
    sp.add_argument("--no-dense", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--preview", type=int, default=4, help="preview lines")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("symbol", help="find a symbol definition")
    sp.add_argument("name")
    add_path(sp)
    sp.set_defaults(func=cmd_symbol)

    sp = sub.add_parser("callers", help="who calls this symbol")
    sp.add_argument("name")
    add_path(sp)
    sp.set_defaults(func=cmd_callers)

    sp = sub.add_parser("impact", help="transitive dependents of a symbol")
    sp.add_argument("name")
    add_path(sp)
    sp.add_argument("--depth", type=int, default=4)
    sp.set_defaults(func=cmd_impact)

    sp = sub.add_parser("outline", help="symbol outline of a file")
    sp.add_argument("file")
    add_path(sp)
    sp.set_defaults(func=cmd_outline)

    sp = sub.add_parser("stats", help="index statistics")
    add_path(sp)
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("mcp", help="serve tools over Model Context Protocol")
    add_path(sp)
    sp.set_defaults(func=cmd_mcp)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
