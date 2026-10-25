"""Westlaw-style search: translation to FTS5, keyword search on a real store, RRF fusion."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pipeline import llm, search
from pipeline.schema import AGENT, SpanIn
from pipeline.search import QueryError, keyword_fact_ids, search_facts, to_fts5
from pipeline.store import Store

CASE = "c1"


def _embed(texts: list[str]) -> list[list[float]]:
    return [llm.hashed_embedding(t) for t in texts]


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "case.db", embedder=_embed)
    yield s
    s.close()


# --------------------------------------------------------------------------- translation

TRANSLATIONS = [
    # spec table (§8.5)
    ("term1 term2", '"term1" OR "term2"'),
    ("a & b", '"a" AND "b"'),
    ('"exact phrase"', '"exact phrase"'),
    ("a /5 b", 'NEAR("a" "b", 5)'),
    ("a /s b", 'NEAR("a" "b", 20)'),
    ("a /p b", 'NEAR("a" "b", 60)'),
    ("a % b", '"a" NOT "b"'),
    ("contract!", '"contract"*'),
    # extras
    ("contract*", '"contract"*'),
    ("contr*ct", '"contr"*'),
    ('"breach of contr!"', '"breach of contr"*'),
    ('"breach of contr"!', '"breach of contr"*'),
    ("a w/5 b", 'NEAR("a" "b", 5)'),
    ("a W/S b", 'NEAR("a" "b", 20)'),
    ("a /0 b", 'NEAR("a" "b", 0)'),
    ('"  spaced   out  "', '"spaced out"'),
    # precedence: OR binds tightest, then /n, /s, /p, then &, then %
    ("a b & c", '("a" OR "b") AND "c"'),
    ("a & b c", '"a" AND ("b" OR "c")'),
    ("a & b % c", '("a" AND "b") NOT "c"'),
    ("a % b & c", '"a" NOT ("b" AND "c")'),
    ("a & b /5 c", '"a" AND NEAR("b" "c", 5)'),
    ("a /5 b & c", 'NEAR("a" "b", 5) AND "c"'),
    ("a & b & c", '"a" AND "b" AND "c"'),
    ("a % b % c", '("a" NOT "b") NOT "c"'),
    ("(a % b) & c", '("a" NOT "b") AND "c"'),
    ("(a & b) c", '("a" AND "b") OR "c"'),
    # distribution of OR-groups under proximity connectors
    ("(a b) /5 c", 'NEAR("a" "c", 5) OR NEAR("b" "c", 5)'),
    ("a b /5 c", 'NEAR("a" "c", 5) OR NEAR("b" "c", 5)'),
    (
        "a b /5 c d",
        'NEAR("a" "c", 5) OR NEAR("a" "d", 5) OR NEAR("b" "c", 5) OR NEAR("b" "d", 5)',
    ),
    ('contract! /s "material breach"', 'NEAR("contract"* "material breach", 20)'),
    # nested proximity: /n binds tighter than /s, which binds tighter than /p
    ("a /5 b /s c", 'NEAR("a" "b", 5) AND NEAR("a" "b" "c", 20)'),
    ("a /s b /5 c", 'NEAR("b" "c", 5) AND NEAR("a" "b" "c", 20)'),
    ("a /p b /s c", 'NEAR("b" "c", 20) AND NEAR("a" "b" "c", 60)'),
    ("(a & b) /3 c", 'NEAR("a" "b" "c", 3)'),
    (
        "(a b) /5 c /s d",
        '(NEAR("a" "c", 5) AND NEAR("a" "c" "d", 20)) OR '
        '(NEAR("b" "c", 5) AND NEAR("b" "c" "d", 20))',
    ),
    ("(a /5 b) % c", 'NEAR("a" "b", 5) NOT "c"'),
    # keywords and punctuation are plain, quoted terms
    ("NOT AND NEAR OR", '"NOT" OR "AND" OR "NEAR" OR "OR"'),
    ("cats and dogs or birds", '"cats" OR "and" OR "dogs" OR "or" OR "birds"'),
    ("Musk's e-mail", '"Musk\'s" OR "e-mail"'),
    ("and/or 3/5/2020", '"and/or" OR "3/5/2020"'),
    ("??! hello", '"hello"'),
    ("When did Musk send the letter?", '"When" OR "did" OR "Musk" OR "send" OR "the" OR "letter?"'),
    ("((a))", '"a"'),
]


@pytest.mark.parametrize(("query", "expected"), TRANSLATIONS)
def test_translation(query: str, expected: str) -> None:
    assert to_fts5(query) == expected


@pytest.mark.parametrize(("query", "_"), TRANSLATIONS)
def test_every_translation_is_valid_fts5(query: str, _: str) -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(a, tokenize='porter unicode61')")
    conn.execute("INSERT INTO t VALUES ('a b c d contract material breach')")
    conn.execute("SELECT count(*) FROM t WHERE t MATCH ?", (to_fts5(query),)).fetchone()


@pytest.mark.parametrize(
    ("query", "fragment"),
    [
        ('"unclosed phrase', "quotation"),
        ("(a b", "parenthes"),
        ("a b)", "parenthes"),
        ("()", "parenthes"),
        ("a &", "&"),
        ("& a", "&"),
        ("a & & b", "&"),
        ("% a", "but not"),
        ("a %", "%"),
        ("a /5", "/5"),
        ("/s a", "/s"),
        ("a /5 & b", "/5"),
        ("(a % b) /5 c", "proximity"),
    ],
)
def test_query_errors_are_friendly(query: str, fragment: str) -> None:
    with pytest.raises(QueryError) as err:
        to_fts5(query)
    assert fragment in str(err.value)
    assert isinstance(err.value, ValueError)


@pytest.mark.parametrize("query", ["", "   ", "??!", "...", '""'])
def test_empty_queries(query: str) -> None:
    with pytest.raises(QueryError):
        to_fts5(query)


def test_distribution_is_capped() -> None:
    query = " ".join(f"a{i}" for i in range(20)) + " /5 " + " ".join(f"b{i}" for i in range(20))
    with pytest.raises(QueryError, match="too many"):
        to_fts5(query)


def test_syntax_help_mentions_every_connector() -> None:
    for token in ("OR", "&", '"', "/n", "/s", "/p", "%", "!"):
        assert token in search.SYNTAX_HELP


# --------------------------------------------------------------------------- keyword search


def _doc(store: Store, text: str, sha: str = "sha1") -> str:
    doc, _ = store.register_document(
        CASE, filename=f"{sha}.txt", rel_path=f"{sha}.txt", sha256=sha, location="evidence"
    )
    store.save_parsed_text(doc.doc_id, text)
    return doc.doc_id


def _span(doc_id: str, text: str, quote: str) -> SpanIn:
    start = text.index(quote)
    return SpanIn(doc_id=doc_id, char_start=start, char_end=start + len(quote), quote=quote)


@pytest.fixture
def facts(store: Store) -> dict[str, str]:
    text = "The landlord terminated the lease on March 3. Payment of rent was late again."
    doc = _doc(store, text)
    ids = {}
    ids["title"] = store.add_fact(CASE, "Board approves merger agreement", by="alice").fact_id
    ids["details"] = store.add_fact(
        CASE, "Meeting", by="alice", details="The parties discussed the indemnification clause."
    ).fact_id
    ids["note"] = store.add_fact(CASE, "Phone call", by="alice").fact_id
    store.add_note(ids["note"], author="alice", body="Witness recalls a confidential settlement.")
    ids["quote"] = store.add_fact(
        CASE, "Lease ended", by=AGENT, spans=[_span(doc, text, "terminated the lease")]
    ).fact_id
    ids["near"] = store.add_fact(
        CASE,
        "Contract breach alleged",
        by="alice",
        details="Acme said the contract was materially breached by the supplier",
    ).fact_id
    ids["far"] = store.add_fact(
        CASE,
        "Contract signed",
        by="alice",
        details="contract " + "filler " * 30 + "breached much later",
    ).fact_id
    ids["rejected"] = store.add_fact(CASE, "Rejected merger rumor", by=AGENT).fact_id
    store.review_fact(ids["rejected"], "reject", reviewer="alice")
    store.add_fact("other-case", "Merger in another case", by="alice")
    return ids


def test_keyword_search_covers_title_details_notes_quotes(
    store: Store, facts: dict[str, str]
) -> None:
    assert keyword_fact_ids(store, CASE, "merger") == [facts["title"]]
    assert keyword_fact_ids(store, CASE, "indemnification") == [facts["details"]]
    assert keyword_fact_ids(store, CASE, "settlement") == [facts["note"]]
    assert keyword_fact_ids(store, CASE, '"terminated the lease"') == [facts["quote"]]


def test_keyword_search_excludes_rejected_and_other_cases(
    store: Store, facts: dict[str, str]
) -> None:
    assert facts["rejected"] not in keyword_fact_ids(store, CASE, "rumor merger")
    assert len(keyword_fact_ids(store, CASE, "merger")) == 1


def test_keyword_connectors(store: Store, facts: dict[str, str]) -> None:
    assert set(keyword_fact_ids(store, CASE, "merger settlement")) == {
        facts["title"],
        facts["note"],
    }
    assert keyword_fact_ids(store, CASE, "board & merger") == [facts["title"]]
    assert keyword_fact_ids(store, CASE, "board & settlement") == []
    assert set(keyword_fact_ids(store, CASE, "contract")) == {facts["near"], facts["far"]}
    assert keyword_fact_ids(store, CASE, "contract % signed") == [facts["near"]]
    # porter stemming + root expander + proximity
    assert set(keyword_fact_ids(store, CASE, "contract /5 breach")) == {facts["near"]}
    assert set(keyword_fact_ids(store, CASE, "contract /s breach!")) == {facts["near"]}
    assert set(keyword_fact_ids(store, CASE, "contract /p breach!")) == {
        facts["near"],
        facts["far"],
    }
    assert keyword_fact_ids(store, CASE, "indemnif!") == [facts["details"]]
    assert keyword_fact_ids(store, CASE, '"merger agreement"') == [facts["title"]]
    assert keyword_fact_ids(store, CASE, '"agreement merger"') == []


def test_keyword_search_handles_keywords_and_empty(store: Store, facts: dict[str, str]) -> None:
    assert keyword_fact_ids(store, CASE, "") == []
    assert keyword_fact_ids(store, CASE, "  ?? ") == []
    assert keyword_fact_ids(store, CASE, "NEAR NOT AND") == []  # plain words, no FTS5 error
    with pytest.raises(QueryError):
        keyword_fact_ids(store, CASE, "merger &")


def test_keyword_limit(store: Store, facts: dict[str, str]) -> None:
    assert len(keyword_fact_ids(store, CASE, "contract", limit=1)) == 1


# --------------------------------------------------------------------------- fusion


def test_rrf_orders_by_fused_score() -> None:
    assert search._rrf(["a", "b", "c"], ["b"]) == ["b", "a", "c"]
    assert search._rrf(["a", "b", "c"], ["c", "b"]) == ["c", "b", "a"]  # 1/63+1/61 > 2/62
    assert search._rrf(["a", "b"], []) == ["a", "b"]
    assert search._rrf([], ["x", "y"]) == ["x", "y"]
    # a tie keeps the keyword list's order first
    assert search._rrf(["a"], ["b"]) == ["a", "b"]


def test_search_facts_fuses_keyword_and_vector(
    store: Store, facts: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real_refresh = store.refresh_vectors

    def refresh(case_id: str, batch: int = 64) -> int:
        calls.append(case_id)
        return real_refresh(case_id, batch)

    monkeypatch.setattr(store, "refresh_vectors", refresh)
    monkeypatch.setattr(
        store, "vector_fact_ids", lambda c, t, k=20: [facts["note"], facts["title"]]
    )
    hits = search_facts(store, CASE, "merger", k=5)
    assert calls == [CASE]
    # "title" is in both rankings so it wins; "note" only comes from the vector half
    assert hits == [facts["title"], facts["note"]]


def test_search_facts_semantic_off_is_keyword_only(
    store: Store, facts: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> list[str]:
        raise AssertionError("vector search must not run")

    monkeypatch.setattr(store, "vector_fact_ids", boom)
    monkeypatch.setattr(store, "refresh_vectors", boom)
    assert search_facts(store, CASE, "merger", semantic=False) == [facts["title"]]


def test_search_facts_falls_back_on_invalid_westlaw(store: Store, facts: dict[str, str]) -> None:
    hits = search_facts(store, CASE, "merger & (", semantic=False)
    assert hits == [facts["title"]]
    hits = search_facts(store, CASE, '"settlement', semantic=False)
    assert hits == [facts["note"]]


def test_search_facts_real_vectors(store: Store, facts: dict[str, str]) -> None:
    if not store.vector_enabled:
        pytest.skip("sqlite-vec not available")
    hits = search_facts(store, CASE, "what did the board approve about the merger", k=3)
    assert hits[0] == facts["title"]
    assert facts["rejected"] not in hits
    assert len(hits) <= 3


def test_search_facts_survives_zero_vectors(store: Store, facts: dict[str, str]) -> None:
    assert search_facts(store, CASE, "??!") == []
    assert search_facts(store, CASE, "") == []
    assert search_facts(store, CASE, "merger", k=0) == []


def test_semantic_text_drops_connectors_and_negation() -> None:
    assert search._semantic_text('contract! /5 "material breach" & acme % lease') == (
        "contract material breach acme"
    )
