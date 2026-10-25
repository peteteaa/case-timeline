"""Tests for pipeline.alerts (SPEC §8.4): no LLM, store-backed."""

from __future__ import annotations

from datetime import date

import pytest

from pipeline import alerts
from pipeline.schema import AGENT, SpanIn
from tests.pipeline_helpers import add_doc, new_store

TODAY = date(2024, 6, 1)
TEXT = (
    "Tom Becker was in the room. The answer is due no later than July 1, 2024. "
    "Due to a conflict the meeting moved. The parties met on May 2, 2024."
)


@pytest.fixture
def store(tmp_path):
    s = new_store(tmp_path)
    yield s
    s.close()


def _span(doc_id: str, quote: str) -> SpanIn:
    start = TEXT.index(quote)
    return SpanIn(doc_id=doc_id, char_start=start, char_end=start + len(quote), quote=quote)


def _fact(store, doc_id, title, quote, day=None, witness_ids=()):
    return store.add_fact(
        "demo",
        title,
        by=AGENT,
        date_start=day,
        date_end=day,
        date_certainty="exact" if day else "unknown",
        witness_ids=list(witness_ids),
        spans=[_span(doc_id, quote)],
    ).fact_id


def test_contradiction_alerts(store):
    d = add_doc(store, TEXT)
    f1 = _fact(store, d, "Parties meet", "The parties met on May 2, 2024.", date(2024, 5, 2))
    f2 = _fact(store, d, "Becker present", "Tom Becker was in the room.", date(2024, 5, 3))
    c, _ = store.add_contradiction("demo", [f1, f2], "date", "x")
    made = alerts.compute(
        store, "demo", job_id="J-0001", new_fact_ids=[], contradictions=[c], doc_ids=[], today=TODAY
    )
    (a,) = made
    assert a.type == "contradiction"
    assert a.fact_ids == c.fact_ids and a.doc_ids == [d]
    assert store.list_alerts("demo", pending_only=True) == [a]


def test_possible_deadline_from_language_or_future_date(store):
    d = add_doc(store, TEXT)
    due = _fact(store, d, "Answer due", "The answer is due no later than July 1, 2024.")
    future = _fact(store, d, "Parties meet", "The parties met on May 2, 2024.", date(2024, 7, 9))
    past = _fact(
        store, d, "Parties meet again", "The parties met on May 2, 2024.", date(2024, 5, 2)
    )
    due_to = _fact(store, d, "Meeting moved", "Due to a conflict the meeting moved.")
    made = alerts.compute(
        store,
        "demo",
        job_id=None,
        new_fact_ids=[due, future, past, due_to],
        contradictions=[],
        doc_ids=[],
        today=TODAY,
    )
    assert [(a.type, a.fact_ids) for a in made] == [
        ("possible_deadline", [due]),
        ("possible_deadline", [future]),
    ]
    assert all(a.doc_ids == [d] for a in made)


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("Responses are due by May 1.", True),
        ("The bar date is June 3.", True),
        ("Filing deadline passed.", True),
        ("Payment on or before March 1.", True),
        ("The appeal must be received by May 6, 2024.", True),
        ("Not later than Friday.", True),
        ("The statute of limitations runs in 2026.", True),
        ("Due to weather the hearing moved.", False),
        ("Due process was observed.", False),
        ("We met for lunch.", False),
    ],
)
def test_deadline_language(text, hit):
    assert alerts.has_deadline_language(text) is hit


def test_flagged_witness_by_mention_or_link(store):
    becker = store.add_witness("demo", "Tom Becker", by="alice", flagged=True)
    ortiz = store.add_witness(
        "demo", "Jamie Ortiz", by="alice", aliases=["Officer Ortiz"], flagged=True
    )
    store.add_witness("demo", "Mark Ellison", by="alice")  # not flagged
    nand = store.add_witness("demo", "Priya Nand", by="alice", flagged=True)
    d1 = add_doc(store, TEXT, filename="one.txt")
    d2 = add_doc(store, "Officer Ortiz took a statement.", filename="two.txt")
    d3 = add_doc(store, "Nothing about anyone here.", filename="three.txt")
    linked = _fact(
        store,
        d1,
        "Parties meet",
        "The parties met on May 2, 2024.",
        date(2024, 5, 2),
        witness_ids=[nand.witness_id],
    )
    made = alerts.compute(
        store,
        "demo",
        job_id="J-1",
        new_fact_ids=[],
        contradictions=[],
        doc_ids=[d1, d2, d3],
        today=TODAY,
    )
    got = sorted((a.type, tuple(a.doc_ids), tuple(a.fact_ids)) for a in made)
    assert got == sorted(
        [
            ("flagged_witness", (d1,), ()),  # Becker named in the text
            ("flagged_witness", (d2,), ()),  # Ortiz by alias
            ("flagged_witness", (d1,), (linked,)),  # Nand linked to a fact drafted from d1
        ]
    )
    assert becker and ortiz


def test_flagged_witness_only_for_processed_docs(store):
    store.add_witness("demo", "Tom Becker", by="alice", flagged=True)
    add_doc(store, TEXT)
    assert (
        alerts.compute(
            store, "demo", job_id=None, new_fact_ids=[], contradictions=[], doc_ids=[], today=TODAY
        )
        == []
    )


def test_rejected_new_fact_gets_no_alert(store):
    d = add_doc(store, TEXT)
    due = _fact(store, d, "Answer due", "The answer is due no later than July 1, 2024.")
    store.review_fact(due, "reject", reviewer="alice")
    assert (
        alerts.compute(
            store,
            "demo",
            job_id=None,
            new_fact_ids=[due],
            contradictions=[],
            doc_ids=[],
            today=TODAY,
        )
        == []
    )
