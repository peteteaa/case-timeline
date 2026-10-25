"""Tests for pipeline.conflicts with a fake LLM module."""

from __future__ import annotations

import sys
import types
from datetime import UTC, date, datetime

import pytest

import pipeline
from pipeline import conflicts
from pipeline.contracts import (
    AlertType,
    Certainty,
    ConflictField,
    Contradiction,
    ContradictionStatus,
    Event,
    EventStatus,
    SourceRef,
)


def ev(
    eid: str,
    desc: str,
    d: date | None = date(2024, 3, 3),
    actors: list[str] | None = None,
    doc: str = "doc1",
    quote: str = "quote",
    **kw,
) -> Event:
    return Event(
        id=eid,
        case_id="demo",
        date_text="some date",
        date_start=d,
        date_end=d,
        certainty=Certainty.EXACT if d else Certainty.UNKNOWN,
        actors=actors if actors is not None else ["John Smith"],
        description=desc,
        source=SourceRef(doc_id=doc, char_start=0, char_end=5, quote=quote),
        confidence=0.9,
        **kw,
    )


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a fake pipeline.llm whose chat_json confirms everything by default."""
    state = {"calls": [], "answer": True, "raise": False}

    def chat_json(system, user, schema, *, temperature=0.0):
        state["calls"].append((system, user))
        if state["raise"]:
            raise RuntimeError("model down")
        field = "date"
        for f in ("date", "actor", "fact"):
            if f"conflict type: {f}" in user:
                field = f
        return schema(is_contradiction=state["answer"], field=field, explanation=" explained ")

    mod = types.ModuleType("pipeline.llm")
    mod.chat_json = chat_json
    monkeypatch.setitem(sys.modules, "pipeline.llm", mod)
    monkeypatch.setattr(pipeline, "llm", mod, raising=False)
    return state


def test_date_conflict_across_documents(fake_llm):
    a = ev("e1", "Smith signed the lease", doc="d1")
    b = ev("e2", "Smith signed the lease agreement", d=date(2024, 3, 9), doc="d2")
    out = conflicts.find_contradictions([a, b], "demo")
    assert len(out) == 1
    c = out[0]
    assert c.field == ConflictField.DATE
    assert c.event_ids == ["e1", "e2"]
    assert c.explanation == "explained"
    assert c.id == conflicts.contradiction_id(["e2", "e1"], ConflictField.DATE)
    assert "Smith signed the lease" in fake_llm["calls"][0][1]


def test_date_conflict_same_document_or_one_day_ignored(fake_llm):
    a = ev("e1", "Smith signed the lease", doc="d1")
    same_doc = ev("e2", "Smith signed the lease", d=date(2024, 3, 5), doc="d1")
    next_day = ev("e3", "Smith signed the lease", d=date(2024, 3, 4), doc="d2")
    next_day = next_day.model_copy(update={"certainty": Certainty.APPROXIMATE})
    assert conflicts.candidate_pairs([a, same_doc]) == []
    assert conflicts.candidate_pairs([a, next_day]) == []
    assert conflicts.find_contradictions([a, same_doc, next_day], "demo") == []
    assert fake_llm["calls"] == []


def test_one_day_gap_counts_when_both_exact(fake_llm):
    a = ev("e1", "meeting in Smith's office", doc="d1")
    b = ev("e2", "meeting in Smith's office", d=date(2024, 3, 4), doc="d2")
    assert [c.field for c in conflicts.candidate_pairs([a, b])] == [ConflictField.DATE]


def test_focus_ids_limits_pairs(fake_llm):
    a = ev("e1", "Smith signed the lease", doc="d1")
    b = ev("e2", "Smith signed the lease agreement", d=date(2024, 3, 9), doc="d2")
    c = ev("e3", "Smith signed the lease", d=date(2024, 3, 20), doc="d3")
    pairs = conflicts.candidate_pairs([a, b, c], focus_ids={"e3"})
    assert {(x.a.id, x.b.id) for x in pairs} == {("e1", "e3"), ("e2", "e3")}


def test_actor_conflict(fake_llm):
    a = ev("e1", "signed the lease for the apartment", actors=["Alice"])
    b = ev("e2", "signed the lease for the apartment", actors=["Bob"], doc="d2")
    cands = conflicts.candidate_pairs([a, b])
    assert [c.field for c in cands] == [ConflictField.ACTOR]
    out = conflicts.find_contradictions([a, b], "demo")
    assert out[0].field == ConflictField.ACTOR


def test_fact_conflict_requires_negation(fake_llm):
    a = ev("e1", "Smith was at the warehouse during the inspection")
    b = ev("e2", "Smith denied ever visiting any storage facility", doc="d2")
    c = ev("e3", "Smith ate lunch with his sister downtown", doc="d3")
    fields = {(x.a.id, x.b.id): x.field for x in conflicts.candidate_pairs([a, b, c])}
    assert fields[("e1", "e2")] == ConflictField.FACT
    assert ("e1", "e3") not in fields


def test_llm_rejection_and_failure(fake_llm):
    a = ev("e1", "Smith signed the lease", doc="d1")
    b = ev("e2", "Smith signed the lease", d=date(2024, 4, 1), doc="d2")
    fake_llm["answer"] = False
    assert conflicts.find_contradictions([a, b], "demo") == []
    fake_llm["raise"] = True
    assert conflicts.find_contradictions([a, b], "demo") == []


def test_skip_ids_avoid_llm_calls(fake_llm):
    a = ev("e1", "Smith signed the lease", doc="d1")
    b = ev("e2", "Smith signed the lease", d=date(2024, 4, 1), doc="d2")
    known = {conflicts.contradiction_id(["e1", "e2"], ConflictField.DATE)}
    assert conflicts.find_contradictions([a, b], "demo", skip_ids=known) == []
    assert fake_llm["calls"] == []


def test_rejected_and_unresolved_events_ignored(fake_llm):
    a = ev("e1", "Smith signed the lease", doc="d1")
    b = ev("e2", "Smith signed the lease", d=date(2024, 4, 1), doc="d2")
    b = b.model_copy(update={"status": EventStatus.REJECTED})
    c = ev("e3", "Smith signed the lease", d=None, doc="d3")
    assert conflicts.candidate_pairs([a, b, c]) == []


def test_candidate_cap_logs_dropped(fake_llm, caplog):
    events = [
        ev(f"e{i}", "Smith signed the lease", d=date(2024, 1, 1 + 3 * i), doc=f"d{i}")
        for i in range(5)
    ]
    with caplog.at_level("WARNING"):
        cands = conflicts.candidate_pairs(events, max_candidates=3)
    assert len(cands) == 3
    assert "dropped 7" in caplog.text


def test_contradiction_id_deterministic():
    x = conflicts.contradiction_id(["b", "a"], ConflictField.FACT)
    assert x == conflicts.contradiction_id(["a", "b"], "fact")
    assert x != conflicts.contradiction_id(["a", "b"], ConflictField.DATE)


def test_alerts_have_no_case_content():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    events = [
        ev("e1", "Secret description", actors=["Zelda Quinn"]),
        ev("e2", "Another secret", actors=["Zelda Quinn"]),
        ev("e3", "Third", actors=["zelda  quinn", "Other"]),
        ev("e4", "Filing deadline", deadline_relevant=True),
    ]
    cs = [
        Contradiction(
            id="c1",
            case_id="demo",
            event_ids=["e1", "e2"],
            field="date",
            explanation="secret explanation",
        ),
        Contradiction(
            id="c2", case_id="demo", event_ids=["e2", "e3"], field="fact", explanation="secret"
        ),
        Contradiction(
            id="c3",
            case_id="demo",
            event_ids=["e1", "e3"],
            field="fact",
            explanation="x",
            status=ContradictionStatus.DISMISSED,
        ),
    ]
    alerts = conflicts.alerts_for(cs, events, "demo", "http://127.0.0.1:8080/", now=now)
    kinds = [a.type for a in alerts]
    assert kinds.count(AlertType.NEW_CONTRADICTION) == 2
    assert kinds.count(AlertType.DEADLINE_DATE) == 1
    witness = [a for a in alerts if a.type == AlertType.FLAGGED_WITNESS]
    assert len(witness) == 1 and witness[0].ref_ids == ["c1", "c2"]
    for a in alerts:
        assert a.dashboard_url == "http://127.0.0.1:8080/alerts"
        assert a.created_at == now
        dumped = a.model_dump_json().lower()
        assert "secret" not in dumped and "zelda" not in dumped
    again = conflicts.alerts_for(cs, events, "demo", "http://127.0.0.1:8080", now=now)
    assert [a.id for a in again] == [a.id for a in alerts]
