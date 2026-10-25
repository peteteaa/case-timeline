"""Westlaw-style search (SPEC §8.5): translate a query subset to SQLite FTS5, plus semantic.

Supported syntax (Westlaw "terms and connectors"):

====================  =====================================  ===============================
Westlaw-style         Meaning                                FTS5
====================  =====================================  ===============================
``a b``               either term (space = OR)               ``"a" OR "b"``
``a & b``             both terms                             ``"a" AND "b"``
``"exact phrase"``    phrase                                 ``"exact phrase"``
``a /5 b``            within 5 words (also ``w/5``)          ``NEAR("a" "b", 5)``
``a /s b``            same sentence (≈ 20 words)             ``NEAR("a" "b", 20)``
``a /p b``            same paragraph (≈ 60 words)            ``NEAR("a" "b", 60)``
``a % b``             a but not b                            ``"a" NOT "b"``
``contract!``         root expander (``*`` works too)        ``"contract"*``
``( ... )``           grouping
====================  =====================================  ===============================

Connectors bind in Westlaw's order, tightest first: OR (space), ``/n``, ``/s``, ``/p``, ``&``,
``%``. FTS5's ``NEAR`` only takes phrases, so an OR-group under a proximity connector is
distributed: ``(a b) /5 c`` becomes ``NEAR("a" "c", 5) OR NEAR("b" "c", 5)``. Every term is
quoted, so words such as ``AND``/``NOT``/``NEAR`` and punctuation are always plain terms.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from itertools import product

from pipeline.store import Store

SYNTAX_HELP = (
    'space = OR · & = AND · "phrase" · /n = within n words · /s same sentence · '
    "/p same paragraph · % = but not · ! = root expander (contract! → contract*)"
)

SENTENCE_WORDS = 20  # /s approximation
PARAGRAPH_WORDS = 60  # /p approximation
RRF_K = 60
_MAX_ALTERNATIVES = 256  # cap on distributed NEAR groups
_MAX_PLAIN_TERMS = 64


class QueryError(ValueError):
    """The query can't be translated (unbalanced quotes, dangling connector, ...)."""


class _EmptyQuery(QueryError):
    """Nothing searchable in the query (blank or punctuation only)."""


# --------------------------------------------------------------------------- tokens

# token kinds: TERM, PHRASE, AND, NOT, NEAR, LPAREN, RPAREN
_PROX_RE = re.compile(r"(?:w)?/(\d+|s|p)(?=[\s()\"&%]|$)", re.IGNORECASE)
_TERM_STOP = set(' \t\r\n\f\v()"&%')
_WORD_RE = re.compile(r"[^\W_]+")


@dataclass(frozen=True)
class _Tok:
    kind: str
    text: str = ""
    prefix: bool = False
    n: int = 0
    label: str = ""  # how the connector was written, for error messages


def _searchable(text: str) -> bool:
    return bool(_WORD_RE.search(text))


def _term_token(raw: str) -> _Tok | None:
    """A bare term; ``!`` or ``*`` (anywhere) makes it a prefix up to that character."""
    cut = min((i for i in (raw.find("!"), raw.find("*")) if i >= 0), default=-1)
    prefix = cut >= 0
    text = raw[:cut] if prefix else raw
    if not _searchable(text):
        return None
    return _Tok("TERM", text, prefix=prefix)


def _tokenize(query: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        if ch.isspace():
            i += 1
        elif ch == '"':
            end = query.find('"', i + 1)
            if end < 0:
                raise QueryError('Unbalanced quotation mark: close the "exact phrase".')
            body = query[i + 1 : end]
            i = end + 1
            prefix = False
            stripped = body.rstrip()
            if stripped.endswith(("!", "*")):
                prefix, body = True, stripped.rstrip("!*")
            # a root expander written just after the closing quote: "breach of contr"!
            if i < n and query[i] in "!*":
                prefix = True
                while i < n and query[i] in "!*":
                    i += 1
            if _searchable(body):
                toks.append(_Tok("PHRASE", " ".join(body.split()), prefix=prefix))
        elif ch == "(":
            toks.append(_Tok("LPAREN", label="("))
            i += 1
        elif ch == ")":
            toks.append(_Tok("RPAREN", label=")"))
            i += 1
        elif ch == "&":
            toks.append(_Tok("AND", label="&"))
            i += 1
        elif ch == "%":
            toks.append(_Tok("NOT", label="%"))
            i += 1
        elif m := _PROX_RE.match(query, i):
            arg = m.group(1).lower()
            dist = {"s": SENTENCE_WORDS, "p": PARAGRAPH_WORDS}.get(arg)
            level = {"s": "S", "p": "P"}.get(arg, "N")
            toks.append(
                _Tok(
                    f"NEAR_{level}",
                    n=dist if dist is not None else int(arg),
                    label=m.group(0),
                )
            )
            i = m.end()
        else:
            j = i
            while j < n and query[j] not in _TERM_STOP:
                j += 1
            tok = _term_token(query[i:j])
            if tok is not None:
                toks.append(tok)
            i = j
    return toks


# --------------------------------------------------------------------------- AST


@dataclass(frozen=True)
class _Phrase:
    text: str
    prefix: bool = False

    def fts(self) -> str:
        return '"' + self.text.replace('"', '""') + '"' + ("*" if self.prefix else "")


@dataclass(frozen=True)
class _Or:
    items: tuple


@dataclass(frozen=True)
class _And:
    items: tuple


@dataclass(frozen=True)
class _Not:
    left: object
    right: object


@dataclass(frozen=True)
class _Near:
    left: object
    right: object
    n: int


class _Parser:
    """Recursive descent, loosest first: % < & < /p < /s < /n < space (OR) < atom."""

    _LEVELS = ("NOT", "AND", "NEAR_P", "NEAR_S", "NEAR_N")
    _CONNECTORS = {"AND", "NOT", "NEAR_P", "NEAR_S", "NEAR_N"}

    def __init__(self, toks: list[_Tok]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self) -> _Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def parse(self) -> object:
        if not self.toks:
            raise _EmptyQuery("Type at least one word to search for.")
        node = self.level(0)
        tok = self.peek()
        if tok is not None:
            if tok.kind == "RPAREN":
                raise QueryError("Unbalanced parentheses: there is a ')' without a '('.")
            raise QueryError(f"Unexpected {tok.label!r} in the query.")  # pragma: no cover
        return node

    def level(self, depth: int) -> object:
        if depth == len(self._LEVELS):
            return self.or_group()
        kind = self._LEVELS[depth]
        tok = self.peek()
        if tok is not None and tok.kind == kind:
            raise QueryError(self._leading_message(tok))
        node = self.level(depth + 1)
        while (tok := self.peek()) is not None and tok.kind == kind:
            self.i += 1
            nxt = self.peek()
            if nxt is None or nxt.kind in self._CONNECTORS or nxt.kind == "RPAREN":
                raise QueryError(
                    f"{tok.label!r} needs a search term after it"
                    + ("." if nxt is None or nxt.kind == "RPAREN" else f", not {nxt.label!r}.")
                )
            right = self.level(depth + 1)
            if kind == "NOT":
                node = _Not(node, right)
            elif kind == "AND":
                node = _And(_flat(_And, node, right))
            else:
                node = _Near(node, right, tok.n)
        return node

    @staticmethod
    def _leading_message(tok: _Tok) -> str:
        if tok.kind == "NOT":
            return "'%' (but not) needs a search term before it, e.g. contract % lease."
        return f"{tok.label!r} needs a search term before it."

    def or_group(self) -> object:
        items = []
        while (tok := self.peek()) is not None and tok.kind in ("TERM", "PHRASE", "LPAREN"):
            items.append(self.atom())
        if not items:
            tok = self.peek()
            if tok is None:  # pragma: no cover - guarded by callers
                raise QueryError("The query ends too early.")
            if tok.kind == "RPAREN":
                raise QueryError("Empty or unbalanced parentheses.")
            raise QueryError(self._leading_message(tok))
        return items[0] if len(items) == 1 else _Or(_flat(_Or, *items))

    def atom(self) -> object:
        tok = self.toks[self.i]
        self.i += 1
        if tok.kind in ("TERM", "PHRASE"):
            return _Phrase(tok.text, tok.prefix)
        # LPAREN
        if (nxt := self.peek()) is not None and nxt.kind == "RPAREN":
            raise QueryError("Empty parentheses: put search terms inside ( ).")
        if nxt is None:
            raise QueryError("Unbalanced parentheses: a '(' is never closed.")
        node = self.level(0)
        tok = self.peek()
        if tok is None or tok.kind != "RPAREN":
            raise QueryError("Unbalanced parentheses: a '(' is never closed.")
        self.i += 1
        return node


def _flat(cls: type, *nodes: object) -> tuple:
    out: list[object] = []
    for node in nodes:
        out.extend(node.items if isinstance(node, cls) else [node])  # type: ignore[attr-defined]
    return tuple(out)


# --------------------------------------------------------------------------- rendering

_PREC_OR, _PREC_AND, _PREC_NOT, _PREC_ATOM = 1, 2, 3, 4  # FTS5: NOT > AND > OR


@dataclass
class _Alt:
    """One distributed alternative of a proximity operand: its phrases + NEAR constraints."""

    phrases: list[_Phrase]
    constraints: list[str] = field(default_factory=list)


def _near(phrases: list[_Phrase], n: int) -> str:
    return f"NEAR({' '.join(p.fts() for p in phrases)}, {n})"


def _alternatives(node: object) -> list[_Alt]:
    if isinstance(node, _Phrase):
        return [_Alt([node])]
    if isinstance(node, _Or):
        out = [alt for item in node.items for alt in _alternatives(item)]
    elif isinstance(node, (_And, _Near)):
        parts = node.items if isinstance(node, _And) else (node.left, node.right)
        out = []
        for combo in product(*(_alternatives(p) for p in parts)):
            phrases = [p for alt in combo for p in alt.phrases]
            constraints = [c for alt in combo for c in alt.constraints]
            if isinstance(node, _Near):
                constraints.append(_near(phrases, node.n))
            out.append(_Alt(phrases, constraints))
            if len(out) > _MAX_ALTERNATIVES:
                break
    else:  # _Not
        raise QueryError(
            "'%' (but not) can't be used inside a proximity connector; put it at the end, "
            "e.g. (breach /5 contract) % lease."
        )
    if len(out) > _MAX_ALTERNATIVES:
        raise QueryError("The query expands to too many combinations; use fewer alternatives.")
    return out


def _wrap(text: str, prec: int) -> str:
    """Parenthesize compound operands so the output never leans on FTS5 precedence."""
    return text if prec == _PREC_ATOM else f"({text})"


def _render(node: object) -> tuple[str, int]:
    if isinstance(node, _Phrase):
        return node.fts(), _PREC_ATOM
    if isinstance(node, _Or):
        parts = [_render(i) for i in node.items]
        return " OR ".join(_wrap(t, p) for t, p in parts), _PREC_OR
    if isinstance(node, _And):
        parts = [_render(i) for i in node.items]
        return " AND ".join(_wrap(t, p) for t, p in parts), _PREC_AND
    if isinstance(node, _Not):
        lt, lp = _render(node.left)
        rt, rp = _render(node.right)
        return f"{_wrap(lt, lp)} NOT {_wrap(rt, rp)}", _PREC_NOT
    # _Near: OR over distributed alternatives, each an AND of NEAR constraints.
    alts = _alternatives(node)
    if len(alts) == 1:
        cs = alts[0].constraints
        return (cs[0], _PREC_ATOM) if len(cs) == 1 else (" AND ".join(cs), _PREC_AND)
    terms = []
    for alt in alts:
        cs = alt.constraints
        terms.append(cs[0] if len(cs) == 1 else "(" + " AND ".join(cs) + ")")
    return " OR ".join(terms), _PREC_OR


def to_fts5(query: str) -> str:
    """Translate a Westlaw-style query into an FTS5 MATCH expression.

    Raises :class:`QueryError` with a friendly message when the query can't be translated.
    """
    node = _Parser(_tokenize(query or "")).parse()
    return _render(node)[0]


def _plain_terms(query: str) -> list[str]:
    seen: list[str] = []
    for word in _WORD_RE.findall(query or ""):
        w = word.lower()
        if w not in seen:
            seen.append(w)
    return seen[:_MAX_PLAIN_TERMS]


def plain_fts5(query: str) -> str:
    """Every word of ``query`` OR-ed together (the fallback for non-Westlaw input); '' if none."""
    return " OR ".join(_Phrase(w).fts() for w in _plain_terms(query))


def _run(store: Store, case_id: str, match: str, limit: int) -> list[str]:
    try:
        return store.fts_fact_ids(case_id, match, limit)
    except sqlite3.OperationalError as exc:  # should not happen: every term is quoted
        raise QueryError(f"The search engine rejected the query ({exc}).") from exc


def keyword_fact_ids(store: Store, case_id: str, query: str, limit: int = 1000) -> list[str]:
    """Fact ids matching the Westlaw-style query, best first (rejected facts excluded)."""
    try:
        match = to_fts5(query)
    except _EmptyQuery:
        return []
    return _run(store, case_id, match, limit)


def _semantic_text(query: str) -> str:
    """The query's words for embedding: connectors dropped, the '% ...' (but not) part cut."""
    head = query.split("%", 1)[0]
    return " ".join(_WORD_RE.findall(_PROX_RE.sub(" ", head)))


def _rrf(*rankings: list[str], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, fact_id in enumerate(ranking, start=1):
            scores[fact_id] = scores.get(fact_id, 0.0) + 1.0 / (k + rank)
    order = {fid: i for i, fid in enumerate(dict.fromkeys(f for r in rankings for f in r))}
    return sorted(scores, key=lambda f: (-scores[f], order[f]))


def search_facts(
    store: Store, case_id: str, query: str, k: int = 20, *, semantic: bool = True
) -> list[str]:
    """Keyword hits fused with sqlite-vec similarity (reciprocal-rank fusion); fact ids.

    A query that isn't valid Westlaw syntax still works: its words are OR-ed for the keyword
    half. ``semantic=False`` returns keyword hits only.
    """
    if not query or not query.strip() or k <= 0:
        return []
    pool = max(k * 5, 100)
    try:
        keyword = keyword_fact_ids(store, case_id, query, limit=pool)
    except QueryError:
        match = plain_fts5(query)
        keyword = _run(store, case_id, match, pool) if match else []
    if not semantic:
        return keyword[:k]
    store.refresh_vectors(case_id)
    text = _semantic_text(query)
    vector = store.vector_fact_ids(case_id, text, k=max(k * 2, 20)) if text else []
    return _rrf(keyword, vector)[:k]
