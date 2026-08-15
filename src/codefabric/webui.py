"""CodeFabric local web UI — zero dependencies, one command:

    codefabric ui --path /repo-or-workspace

Serves a single-page search console on http://localhost:8377 backed by
the same SearchEngine the CLI uses. Built for quick demos: hybrid
search, filters, graph context on every hit, and an Ask panel that uses
an LLM when one is configured and extractive answers otherwise.
"""
from __future__ import annotations

import html
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import llm as llm_mod
from .search import SearchEngine

DEFAULT_PORT = 8377

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CodeFabric</title>
<style>
  :root {
    --bg: #0e1116; --panel: #161b23; --panel2: #1c232e; --line: #2a3442;
    --text: #dce3ec; --dim: #8b98a9; --accent: #4cc38a; --accent2: #6cb2f5;
    --warn: #e5b567;
  }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--bg); color: var(--text);
         font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { display: flex; align-items: center; gap: 14px;
           padding: 18px 28px; border-bottom: 1px solid var(--line); }
  .logo { width: 34px; height: 34px; }
  h1 { font-size: 20px; letter-spacing: .3px; }
  h1 span { color: var(--accent); }
  .tagline { color: var(--dim); font-size: 13px; margin-left: auto; }
  main { max-width: 1060px; margin: 0 auto; padding: 26px 20px 60px; }
  .searchrow { display: flex; gap: 10px; }
  input[type=text] { flex: 1; padding: 12px 16px; font-size: 16px;
    background: var(--panel); color: var(--text);
    border: 1px solid var(--line); border-radius: 10px; outline: none; }
  input[type=text]:focus { border-color: var(--accent); }
  button { padding: 12px 20px; font-size: 15px; border: none; cursor: pointer;
    border-radius: 10px; background: var(--accent); color: #08251a;
    font-weight: 600; }
  button.alt { background: var(--panel2); color: var(--accent2);
    border: 1px solid var(--line); }
  .filters { display: flex; gap: 10px; margin: 12px 0 4px; align-items: center;
    color: var(--dim); font-size: 13px; flex-wrap: wrap; }
  select { background: var(--panel); color: var(--text); padding: 6px 10px;
    border: 1px solid var(--line); border-radius: 8px; }
  .stats { color: var(--dim); font-size: 13px; margin-left: auto; }
  .card { background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 14px 18px; margin-top: 14px; }
  .cardhead { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
  .path { color: var(--accent2); font-family: ui-monospace, monospace;
    font-size: 14px; }
  .kind { font-size: 11px; text-transform: uppercase; letter-spacing: .8px;
    background: var(--panel2); border: 1px solid var(--line);
    padding: 2px 8px; border-radius: 20px; color: var(--warn); }
  .score { color: var(--dim); font-size: 12px; margin-left: auto; }
  pre { background: #0a0d12; border: 1px solid var(--line); border-radius: 8px;
    padding: 12px; overflow-x: auto; margin-top: 10px;
    font: 13px/1.45 ui-monospace, "Cascadia Code", monospace; color: #c8d3e0; }
  .related { margin-top: 8px; font-size: 13px; color: var(--dim); }
  .related b { color: var(--text); font-weight: 500; }
  .answer { white-space: pre-wrap; background: var(--panel2);
    border-left: 3px solid var(--accent); padding: 14px 16px;
    border-radius: 8px; margin-top: 14px; }
  .mode { font-size: 12px; color: var(--warn); margin-top: 6px; }
  .empty { color: var(--dim); margin-top: 30px; text-align: center; }
  a { color: var(--accent2); }
</style>
</head>
<body>
<header>
  <svg class="logo" viewBox="0 0 64 64"><defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#4cc38a"/><stop offset="1" stop-color="#6cb2f5"/>
    </linearGradient></defs>
    <rect x="4" y="4" width="56" height="56" rx="14" fill="#161b23" stroke="url(#g)" stroke-width="2.5"/>
    <path d="M16 22h32M16 32h32M16 42h32" stroke="#2a3442" stroke-width="3" stroke-linecap="round"/>
    <path d="M24 14v36M40 14v36" stroke="url(#g)" stroke-width="3.5" stroke-linecap="round"/>
    <circle cx="24" cy="32" r="4.5" fill="#4cc38a"/><circle cx="40" cy="22" r="4.5" fill="#6cb2f5"/>
    <circle cx="40" cy="42" r="4.5" fill="#6cb2f5"/>
  </svg>
  <h1>Code<span>Fabric</span></h1>
  <div class="tagline">weave your code into a searchable knowledge fabric</div>
</header>
<main>
  <div class="searchrow">
    <input id="q" type="text" placeholder="Search code... e.g. &quot;where do we hash passwords&quot;" autofocus>
    <button onclick="doSearch()">Search</button>
    <button class="alt" onclick="doAsk()">Ask</button>
  </div>
  <div class="filters">
    <label>language <select id="lang"><option value="">any</option></select></label>
    <label>kind <select id="kind">
      <option value="">any</option><option>function</option>
      <option>method</option><option>class</option><option>block</option>
    </select></label>
    <span class="stats" id="stats"></span>
  </div>
  <div id="out"><div class="empty">Type a query and hit Search — or Ask a question
    about the codebase.</div></div>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => (s ?? "").toString().replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

fetch("api/stats").then(r => r.json()).then(s => {
  $("stats").textContent =
    `${s.chunks} chunks · ${s.symbols} symbols · ${s.graph_edges} graph edges`;
  for (const lang of Object.keys(s.chunks_by_language || {})) {
    const o = document.createElement("option"); o.textContent = lang;
    $("lang").appendChild(o);
  }
});

function busy(msg) { $("out").innerHTML = `<div class="empty">${esc(msg)}</div>`; }

async function doSearch() {
  const q = $("q").value.trim(); if (!q) return;
  busy("searching…");
  const params = new URLSearchParams({q, k: 10});
  if ($("lang").value) params.set("lang", $("lang").value);
  if ($("kind").value) params.set("kind", $("kind").value);
  const results = await (await fetch("api/search?" + params)).json();
  if (!results.length) { busy("no results"); return; }
  $("out").innerHTML = results.map(r => {
    const c = r.chunk;
    const rel = (r.related || []).slice(0, 4).map(x =>
      `<b>${esc(x.relation)}</b> ${esc(x.symbol)} <span>(${esc(x.file)}:${x.line})</span>`
    ).join(" &nbsp;·&nbsp; ");
    return `<div class="card">
      <div class="cardhead">
        <span class="path">${esc(c.file_path)}:${c.start_line}-${c.end_line}</span>
        <span class="kind">${esc(c.kind)}</span>
        ${c.symbol ? `<span>${esc(c.symbol)}</span>` : ""}
        <span class="score">${r.score.toFixed(4)} · ${r.sources.join("+")}</span>
      </div>
      <pre>${esc(c.text.slice(0, 1600))}</pre>
      ${rel ? `<div class="related">↳ ${rel}</div>` : ""}
    </div>`;
  }).join("");
}

async function doAsk() {
  const q = $("q").value.trim(); if (!q) return;
  busy("thinking…");
  const resp = await fetch("api/ask", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question: q})});
  const data = await resp.json();
  if (data.error) { busy("error: " + data.error); return; }
  const srcs = (data.sources || []).map(s =>
    `<div class="related">[${s.n}] <b>${esc(s.symbol) || "block"}</b> — ${esc(s.file)}:${esc(s.lines)}</div>`
  ).join("");
  $("out").innerHTML =
    `<div class="answer">${esc(data.answer)}</div>
     <div class="mode">mode: ${esc(data.mode)}</div>${srcs}`;
}

$("q").addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });
</script>
</body>
</html>
"""


def make_handler(engine: SearchEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)

            if parsed.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/search":
                q = (qs.get("q") or [""])[0]
                k = int((qs.get("k") or ["10"])[0])
                results = engine.search(
                    q, k=min(k, 50),
                    language=(qs.get("lang") or [None])[0] or None,
                    kind=(qs.get("kind") or [None])[0] or None,
                )
                self._json([r.to_dict() for r in results])
            elif parsed.path == "/api/stats":
                self._json(engine.stats())
            elif parsed.path == "/api/outline":
                f = (qs.get("file") or [""])[0]
                self._json(engine.outline(f))
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/api/ask":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                question = str(payload.get("question", "")).strip()
                if not question:
                    self._json({"error": "empty question"}, 400)
                    return
                self._json(llm_mod.ask(engine, question, k=8))
            except llm_mod.LLMUnavailable as e:
                self._json({"error": str(e)}, 502)
            except Exception as e:  # keep demo server alive
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def serve(root: str, port: int = DEFAULT_PORT, open_browser: bool = True,
          background: bool = False) -> ThreadingHTTPServer:
    engine = SearchEngine(root)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(engine))
    url = f"http://localhost:{server.server_address[1]}"
    print(f"CodeFabric UI serving {root}")
    print(f"  → {url}   (Ctrl+C to stop)")
    if open_browser:
        try:
            import webbrowser

            threading.Timer(0.4, webbrowser.open, args=(url,)).start()
        except Exception:
            pass
    if background:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return server
