"""Westlaw-style search (SPEC §8.5): translate a query subset to SQLite FTS5, plus semantic.

STUB — signatures are the contract; the search owner fills in the bodies.
"""

from __future__ import annotations

from pipeline.store import Store

SYNTAX_HELP = (
    'space = OR · & = AND · "phrase" · /n = within n words · /s same sentence · '
    "/p same paragraph · % = but not · ! = root expander (contract! → contract*)"
)


class QueryError(ValueError):
    """The query can't be translated (unbalanced quotes, dangling connector, ...)."""


def to_fts5(query: str) -> str:
    """Translate a Westlaw-style query into an FTS5 MATCH expression."""
    raise NotImplementedError


def keyword_fact_ids(store: Store, case_id: str, query: str, limit: int = 1000) -> list[str]:
    """Fact ids matching the Westlaw-style query, best first (rejected facts excluded)."""
    raise NotImplementedError


def search_facts(
    store: Store, case_id: str, query: str, k: int = 20, *, semantic: bool = True
) -> list[str]:
    """Keyword hits fused with sqlite-vec similarity (reciprocal-rank fusion); fact ids."""
    raise NotImplementedError
