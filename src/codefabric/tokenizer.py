"""Code-aware tokenization.

Identifiers carry most of the signal in code retrieval, but they arrive
camelCased, snake_cased, or dotted. We emit both the original identifier
and its subword splits so `getUserById` matches the queries "get user",
"user by id", and "getUserById" alike.
"""
from __future__ import annotations

import re

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)

# Ubiquitous keywords add noise, not signal.
_STOPWORDS = frozenset(
    """
    the a an and or not in of to for with is are was be this that it as at
    on by from if else elif return def class import function var let const
    public private protected static void int str bool true false none null
    self cls new pass raise try except finally while do switch case break
    continue
    """.split()
)


def split_identifier(ident: str) -> list[str]:
    """Split an identifier into lowercase subwords.

    handles snake_case, camelCase, PascalCase, SCREAMING_SNAKE and digits.
    """
    parts: list[str] = []
    for piece in ident.split("_"):
        if not piece:
            continue
        parts.extend(m.group(0) for m in _CAMEL_RE.finditer(piece))
    return [p.lower() for p in parts if p]


def tokenize(text: str) -> list[str]:
    """Tokenize source text (or a query) into search terms."""
    tokens: list[str] = []
    for m in _IDENT_RE.finditer(text):
        ident = m.group(0)
        lower = ident.lower()
        subs = split_identifier(ident)
        if lower not in _STOPWORDS:
            tokens.append(lower)
        for s in subs:
            if s != lower and s not in _STOPWORDS and len(s) > 1:
                tokens.append(s)
    return tokens
