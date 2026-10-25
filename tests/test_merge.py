"""Tests for pipeline.merge (no live model needed)."""

from __future__ import annotations

from datetime import date

import pytest

from pipeline import merge
from pipeline.schema import DraftFact, SpanIn
from tests.pipeline_helpers import make_fact


def draft(
    key: str,
    title: str,
    d: date | None = date(2024, 3, 3),
    *,
    end: date | None = None,
    date_text: str | None = "March 3, 2024",
    quote: str = "some quote",
    doc: str = "D-0002",
    at: int = 0,
) -> DraftFact:
    return DraftFact(
        key=key,
        title=title,
        date_text=date_text,
        date_start=d,
        date_end=end or d,
        date_certainty="exact" if d else "unknown",
        quotes=[quote],
        spans=[SpanIn(doc_id=doc, char_start=at, char_end=at + len(quote), quote=quote)],
    )


@pytest.fixture(autouse=True)
def no_embed(monkeypatch):
    calls: list[list[str]] = []

    def fake(texts):
        calls.append(list(texts))
        return None

    monkeypatch.setattr(merge, "_embed", fake)
    return calls


def test_fuzzy_duplicate_detected():
    old = make_fact("F-0001", "John Smith signed the lease agreement", witnesses=["W-1"])
    new = draft("n1", "Smith signed the lease agreement")
    assert merge.plan([new], [old], witness_ids={"n1": ["W-1"]}) == {"n1": "F-0001"}


def test_different_date_not_merged():
    old = make_fact("F-0001", "John Smith signed the lease agreement")
    new = draft("n1", "John Smith signed the lease agreement", d=date(2024, 3, 10))
    assert merge.plan([new], [old]) == {"n1": None}


def test_range_overlap_counts_as_same_date():
    old = make_fact(
        "F-0001",
        "Smith signed the lease",
        date(2024, 3, 1),
        end=date(2024, 3, 31),
        certainty="approximate",
    )
    new = draft("n1", "Smith signed the lease")
    assert merge.plan([new], [old]) == {"n1": "F-0001"}


def test_disjoint_people_not_merged():
    old = make_fact("F-0001", "signed the lease agreement", witnesses=["W-1"])
    new = draft("n1", "signed the lease agreement")
    assert merge.plan([new], [old], witness_ids={"n1": ["W-2"]}) == {"n1": None}


def test_missing_people_on_one_side_still_matches():
    old = make_fact("F-0001", "signed the lease agreement", witnesses=[])
    new = draft("n1", "signed the lease agreement")
    assert merge.plan([new], [old], witness_ids={"n1": ["W-2"]}) == {"n1": "F-0001"}


def test_undated_match_on_identical_or_missing_date_text():
    old = make_fact("F-0001", "meeting at the office", None, date_text="the next morning")
    same = draft("n1", "meeting at the office", None, date_text="The next  morning")
    other = draft("n2", "meeting at the office", None, date_text="later that week")
    none = draft("n3", "meeting at the office", None, date_text=None)
    out = merge.plan([same, other, none], [old])
    assert out == {"n1": "F-0001", "n2": None, "n3": "F-0001"}


def test_dated_and_undated_not_merged():
    old = make_fact("F-0001", "meeting at the office", None)
    assert merge.plan([draft("n1", "meeting at the office")], [old]) == {"n1": None}


def test_rejected_facts_are_never_targets():
    old = make_fact("F-0001", "Smith paid rent", review="rejected")
    assert merge.plan([draft("n1", "Smith paid rent")], [old]) == {"n1": None}


def test_best_scoring_fact_wins():
    a = make_fact("F-0001", "Smith paid the rent for March and April late")
    b = make_fact("F-0002", "Smith paid rent")
    assert merge.plan([draft("n1", "Smith paid rent")], [a, b]) == {"n1": "F-0002"}


def test_shared_quote_with_related_title_merges():
    q = "On March 3, 2024 Smith handed the keys to Jones."
    old = make_fact("F-0001", "Keys handed over to Jones", quotes=[q], docs=["D-0001"])
    new = draft("n1", "Smith gives Jones the keys", quote=q, doc="D-0009")
    assert merge.plan([new], [old]) == {"n1": "F-0001"}
    unrelated = draft("n2", "Rent dispute begins", quote=q, doc="D-0009")
    assert merge.plan([unrelated], [old]) == {"n2": None}


def test_embedding_used_only_for_candidate_pairs(monkeypatch):
    vecs = {"Smith paid rent": [1.0, 0.0], "rent payment made by Smith": [0.99, 0.05]}
    calls: list[list[str]] = []

    def fake(texts):
        calls.append(list(texts))
        return [vecs.get(t, [0.0, 1.0]) for t in texts]

    monkeypatch.setattr(merge, "_embed", fake)
    old = make_fact("F-0001", "Smith paid rent")
    far = make_fact("F-0002", "Smith paid rent", date(2025, 1, 1))  # wrong date: never embedded
    new = draft("n1", "rent payment made by Smith")
    assert merge.plan([new], [old, far]) == {"n1": "F-0001"}
    assert len(calls) == 1
    assert sorted(calls[0]) == sorted(["rent payment made by Smith", "Smith paid rent"])


def test_low_cosine_not_merged(monkeypatch):
    monkeypatch.setattr(merge, "_embed", lambda texts: [[1.0, 0.0], [0.0, 1.0]])
    old = make_fact("F-0001", "Smith paid rent")
    assert merge.plan([draft("n1", "Smith slipped on the stairs")], [old]) == {"n1": None}


def test_no_embed_call_when_fuzzy_already_matches(no_embed):
    old = make_fact("F-0001", "Smith paid rent")
    merge.plan([draft("n1", "Smith paid rent")], [old])
    assert no_embed == []


def test_no_existing_facts():
    assert merge.plan([draft("n1", "x"), draft("n2", "y")], []) == {"n1": None, "n2": None}
