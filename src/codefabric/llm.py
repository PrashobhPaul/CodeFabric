"""Optional LLM answer synthesis — CodeFabric works fully without it.

Retrieval never needs a model. When a provider is configured, ``ask()``
adds a grounded, cited answer on top of the retrieved snippets; when
none is configured it degrades to an extractive answer built from the
top-ranked chunks.

Providers (pure stdlib urllib, no SDK dependency, provider-neutral by
design):
  - anthropic  — env ANTHROPIC_API_KEY   (default model claude-opus-5)
  - openai     — env OPENAI_API_KEY      (any OpenAI-compatible server;
                 override endpoint via OPENAI_BASE_URL)
  - ollama     — env CODEFABRIC_LLM_BASE_URL=http://localhost:11434/v1
                 (or any other OpenAI-compatible base URL, no key needed)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder"

SYSTEM_PROMPT = (
    "You are CodeFabric, a code-search assistant. Answer the question using "
    "ONLY the provided code snippets. Cite snippets by their number like [1]. "
    "Reference code locations as file:line. If the snippets do not contain "
    "the answer, say so plainly instead of guessing."
)


class LLMUnavailable(RuntimeError):
    pass


def detect_provider() -> str | None:
    """Pick a provider from the environment; None = extractive mode."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("CODEFABRIC_LLM_BASE_URL"):
        return "ollama"
    return None


def _post_json(url: str, headers: dict[str, str], payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise LLMUnavailable(f"LLM request failed ({e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise LLMUnavailable(f"cannot reach LLM endpoint: {e.reason}") from e


def complete(prompt: str, system: str = SYSTEM_PROMPT,
             provider: str | None = None, model: str | None = None,
             max_tokens: int = 16000) -> str:
    """One-shot completion against the configured provider."""
    provider = provider or detect_provider()
    if provider is None:
        raise LLMUnavailable(
            "No LLM configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
            "CODEFABRIC_LLM_BASE_URL (e.g. http://localhost:11434/v1 for "
            "Ollama) — or use extractive mode, which needs no LLM."
        )

    if provider == "anthropic":
        body = _post_json(
            ANTHROPIC_URL,
            {
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": ANTHROPIC_VERSION,
            },
            {
                "model": model or DEFAULT_ANTHROPIC_MODEL,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if body.get("stop_reason") == "refusal":
            raise LLMUnavailable("the model declined to answer this request")
        return "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        ).strip()

    # OpenAI-compatible chat/completions (OpenAI itself, Ollama, vLLM, ...)
    if provider == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        key = os.environ.get("OPENAI_API_KEY", "")
        model = model or DEFAULT_OPENAI_MODEL
    elif provider == "ollama":
        base = os.environ.get("CODEFABRIC_LLM_BASE_URL",
                              "http://localhost:11434/v1")
        key = os.environ.get("CODEFABRIC_LLM_API_KEY", "ollama")
        model = model or os.environ.get("CODEFABRIC_LLM_MODEL",
                                        DEFAULT_OLLAMA_MODEL)
    else:
        raise LLMUnavailable(f"unknown provider: {provider}")

    body = _post_json(
        f"{base.rstrip('/')}/chat/completions",
        {"Authorization": f"Bearer {key}"},
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
    )
    choices = body.get("choices") or []
    if not choices:
        raise LLMUnavailable("LLM returned no choices")
    return (choices[0].get("message", {}).get("content") or "").strip()


# -- grounded Q&A ------------------------------------------------------


def build_prompt(question: str, results) -> str:
    parts = [f"Question: {question}", "", "Code snippets:"]
    for i, r in enumerate(results, 1):
        c = r.chunk
        header = f"[{i}] {c.file_path}:{c.start_line}-{c.end_line}"
        if c.symbol:
            header += f"  ({c.kind} {c.symbol})"
        parts.append(header)
        parts.append("```" + (c.language or ""))
        parts.append(c.text[:2000])
        parts.append("```")
        for rel in r.related[:3]:
            parts.append(f"    related {rel['relation']}: {rel['symbol']} "
                         f"({rel['file']}:{rel['line']})")
        parts.append("")
    parts.append("Answer the question, citing snippet numbers like [1].")
    return "\n".join(parts)


def extractive_answer(question: str, results) -> str:
    """No-LLM fallback: present the best-matching code directly."""
    if not results:
        return "No matching code found for this question."
    lines = [f"Top matches for: {question}", ""]
    for i, r in enumerate(results, 1):
        c = r.chunk
        what = f"{c.kind} `{c.symbol}`" if c.symbol else c.kind
        lines.append(f"[{i}] {c.file_path}:{c.start_line} — {what}")
        if c.docstring:
            lines.append(f"    \"{c.docstring.splitlines()[0][:120]}\"")
        snippet = c.text.strip().splitlines()
        for row in snippet[:6]:
            lines.append(f"    {row[:150]}")
        lines.append("")
    lines.append("(extractive mode — configure an LLM for synthesized answers)")
    return "\n".join(lines)


def ask(engine, question: str, k: int = 8, provider: str | None = "auto",
        model: str | None = None) -> dict:
    """Retrieve, then answer. provider: 'auto' | 'none' | explicit name."""
    results = engine.search(question, k=k)
    sources = [
        {
            "n": i + 1,
            "file": r.chunk.file_path,
            "lines": f"{r.chunk.start_line}-{r.chunk.end_line}",
            "symbol": r.chunk.symbol,
            "score": round(r.score, 4),
        }
        for i, r in enumerate(results)
    ]
    resolved = detect_provider() if provider == "auto" else (
        None if provider in (None, "none") else provider
    )
    if resolved is None:
        return {
            "mode": "extractive",
            "answer": extractive_answer(question, results),
            "sources": sources,
        }
    answer = complete(build_prompt(question, results), provider=resolved,
                      model=model)
    return {"mode": f"llm:{resolved}", "answer": answer, "sources": sources}
