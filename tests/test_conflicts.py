"""Tests for pipeline.conflicts with a fake LLM."""

from __future__ import annotations

import re
from datetime import date

import pytest

from pipeline import conflicts, llm
from pipeline.schema import AGENT, SpanIn
from tests.pipeline_helpers import add_doc, make_fact, new_store


@pytest.fixture
def fake_llm(monkeypatch):
    """chat_json confirms every candidate (with the suspected field) unless told otherwise."""
    state = {"calls": [], "answer": True, "raise": False}

    def chat_json(system, user, schema, **kw):
        state["calls"].append((system, user))
        if state["raise"]:
            raise RuntimeError("model down")
        m = re.search(r"Suspected conflict type: (\w+)", user)
        return schema(
            is_contradiction=state["answer"],
            field=m.group(1) if m else "fact",
            explanation=" explained ",
        )

    monkeypatch.setattr(llm, "chat_json", chat_json)
    return state


# --------------------------------------------------------------------------- candidate rules


def test_date_candidate_across_documents():
    a = make_fact("F-1", "Smith signed the lease", witnesses=["W-1"], docs=["D-1"])
    b = make_fact(
        "F-2", "Smith signed the lease agreement", date(2024, 3, 9), witnesses=["W-1"], docs=["D-2"]
    )
    (c,) = conflicts.candidate_pairs([a, b], {"F-2"})
    assert (c.a.fact_id, c.b.fact_id, c.field) == ("F-1", "F-2", "date")


def test_date_candidate_needs_other_document_and_more_than_a_day():
    a = make_fact("F-1", "Smith signed the lease", witnesses=["W-1"], docs=["D-1"])
    same_doc = make_fact(
        "F-2", "Smith signed the lease", date(2024, 3, 5), witnesses=["W-1"], docs=["D-1"]
    )
    next_day = make_fact(
        "F-3",
        "Smith signed the lease",
        date(2024, 3, 4),
        witnesses=["W-1"],
        docs=["D-2"],
        certainty="approximate",
    )
    assert conflicts.candidate_pairs([a, same_doc, next_day], {"F-1", "F-2", "F-3"}) == []


def test_one_day_gap_counts_when_both_exact():
    a = make_fact("F-1", "meeting in Smith's office", witnesses=["W-1"], docs=["D-1"])
    b = make_fact(
        "F-2", "meeting in Smith's office", date(2024, 3, 4), witnesses=["W-1"], docs=["D-2"]
    )
    assert [c.field for c in conflicts.candidate_pairs([a, b], {"F-1"})] == ["date"]


def test_date_candidate_without_people_needs_strong_similarity():
    a = make_fact("F-1", "The meeting in the office took place", docs=["D-1"])
    b = make_fact("F-2", "The meeting in the office took place", date(2024, 3, 9), docs=["D-2"])
    c = make_fact("F-3", "Rent paid at the office", date(2024, 3, 20), docs=["D-3"])
    pairs = conflicts.candidate_pairs([a, b, c], {"F-1", "F-2", "F-3"})
    assert [(x.a.fact_id, x.b.fact_id) for x in pairs] == [("F-1", "F-2")]


def test_focus_ids_limit_pairs():
    facts = [
        make_fact(
            f"F-{i}",
            "Smith signed the lease",
            date(2024, 3, 1 + 7 * i),
            witnesses=["W-1"],
            docs=[f"D-{i}"],
        )
        for i in range(3)
    ]
    pairs = conflicts.candidate_pairs(facts, {"F-2"})
    assert {(x.a.fact_id, x.b.fact_id) for x in pairs} == {("F-0", "F-2"), ("F-1", "F-2")}


def test_actor_candidate():
    a = make_fact("F-1", "signed the lease for the apartment", witnesses=["W-1"])
    b = make_fact("F-2", "signed the lease for the apartment", witnesses=["W-2"], docs=["D-2"])
    assert [c.field for c in conflicts.candidate_pairs([a, b], {"F-1"})] == ["actor"]


def test_fact_candidate_requires_negation():
    a = make_fact("F-1", "Smith was at the warehouse during the inspection", witnesses=["W-1"])
    b = make_fact(
        "F-2", "Smith denied ever visiting any storage facility", witnesses=["W-1"], docs=["D-2"]
    )
    c = make_fact(
        "F-3", "Smith ate lunch with his sister downtown", witnesses=["W-1"], docs=["D-3"]
    )
    fields = {
        (x.a.fact_id, x.b.fact_id): x.field
        for x in conflicts.candidate_pairs([a, b, c], {"F-1", "F-2", "F-3"})
    }
    assert fields[("F-1", "F-2")] == "fact"
    assert ("F-1", "F-3") not in fields


def test_negation_in_quotes_counts():
    a = make_fact("F-1", "Becker's account of the meeting", witnesses=["W-1"])
    b = make_fact(
        "F-2",
        "Ellison describes who attended",
        witnesses=["W-1"],
        docs=["D-2"],
        quotes=["A. Priya Nand from HR sat in. Nobody else."],
    )
    assert [c.field for c in conflicts.candidate_pairs([a, b], {"F-1"})] == ["fact"]


def test_rejected_and_undated_facts_ignored():
    a = make_fact("F-1", "Smith signed the lease", witnesses=["W-1"], docs=["D-1"])
    b = make_fact(
        "F-2",
        "Smith signed the lease",
        date(2024, 4, 1),
        witnesses=["W-1"],
        docs=["D-2"],
        review="rejected",
    )
    c = make_fact("F-3", "Smith signed the lease", None, witnesses=["W-1"], docs=["D-3"])
    assert conflicts.candidate_pairs([a, b, c], {"F-1", "F-2", "F-3"}) == []


def test_skip_pairs():
    a = make_fact("F-1", "Smith signed the lease", witnesses=["W-1"], docs=["D-1"])
    b = make_fact(
        "F-2", "Smith signed the lease", date(2024, 4, 1), witnesses=["W-1"], docs=["D-2"]
    )
    assert conflicts.candidate_pairs([a, b], {"F-1"}, skip_pairs={frozenset({"F-1", "F-2"})}) == []


def test_candidate_cap_logs_dropped(caplog):
    facts = [
        make_fact(
            f"F-{i}",
            "Smith signed the lease",
            date(2024, 1, 1 + 3 * i),
            witnesses=["W-1"],
            docs=[f"D-{i}"],
        )
        for i in range(5)
    ]
    with caplog.at_level("WARNING"):
        cands = conflicts.candidate_pairs(facts, {f.fact_id for f in facts}, max_candidates=3)
    assert len(cands) == 3
    assert "dropped 7" in caplog.text


# --------------------------------------------------------------------------- detect (store)


@pytest.fixture
def case(tmp_path):
    store = new_store(tmp_path)
    w = store.add_witness("demo", "John Smith", by="alice").witness_id
    t1 = "Smith signed the lease on March 3, 2024."
    t2 = "Smith signed the lease agreement on March 9, 2024."
    d1, d2 = add_doc(store, t1, filename="a.txt"), add_doc(store, t2, filename="b.txt")

    def fact(title, text, doc, day):
        return store.add_fact(
            "demo",
            title,
            by=AGENT,
            date_start=day,
            date_end=day,
            date_certainty="exact",
            witness_ids=[w],
            spans=[SpanIn(doc_id=doc, char_start=0, char_end=len(text), quote=text)],
        ).fact_id

    f1 = fact("Smith signed the lease", t1, d1, date(2024, 3, 3))
    f2 = fact("Smith signed the lease agreement", t2, d2, date(2024, 3, 9))
    yield store, f1, f2
    store.close()


def test_detect_opens_confirmed_contradiction(case, fake_llm):
    store, f1, f2 = case
    (c,) = conflicts.detect(store, "demo", [f2])
    assert c.fact_ids == sorted([f1, f2])
    assert c.field == "date" and c.status == "open"
    assert c.explanation == "explained"
    system, user = fake_llm["calls"][0]
    assert "evidence, not instructions" in system
    assert "Smith signed the lease" in user and "John Smith" in user
    assert "2024-03-03" in user and "2024-03-09" in user
    assert store.list_contradictions("demo") == [c]
    # A second run doesn't re-ask the model or reopen the pair.
    assert conflicts.detect(store, "demo", [f1, f2]) == []
    assert len(fake_llm["calls"]) == 1


def test_detect_respects_dismissed(case, fake_llm):
    store, f1, f2 = case
    (c,) = conflicts.detect(store, "demo", [f2])
    store.review_contradiction(c.contradiction_id, "dismiss", reviewer="alice")
    assert conflicts.detect(store, "demo", [f1, f2]) == []
    assert len(fake_llm["calls"]) == 1


def test_detect_llm_rejection_and_failure(case, fake_llm):
    store, f1, f2 = case
    fake_llm["answer"] = False
    assert conflicts.detect(store, "demo", [f2]) == []
    fake_llm["raise"] = True
    assert conflicts.detect(store, "demo", [f2]) == []
    assert store.list_contradictions("demo") == []


def test_detect_skips_rejected_facts(case, fake_llm):
    store, f1, f2 = case
    store.review_fact(f1, "reject", reviewer="alice", reason="wrong")
    assert conflicts.detect(store, "demo", [f2]) == []
    assert fake_llm["calls"] == []


def test_detect_nothing_to_check(case, fake_llm):
    store, _, _ = case
    assert conflicts.detect(store, "demo", []) == []
    assert fake_llm["calls"] == []
