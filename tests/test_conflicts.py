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
    quote: str = "",
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
    state = {"calls": [], "answer": True, "raise": False, "extra": {}}

    def chat_json(system, user, schema, *, temperature=0.0, **kw):
        state["calls"].append((system, user))
        if state["raise"]:
            raise RuntimeError("model down")
        field = "date"
        for f in ("date", "actor", "fact"):
            if f"conflict type: {f}" in user:
                field = f
        return schema(
            is_contradiction=state["answer"],
            field=field,
            explanation=" explained ",
            **state["extra"],
        )

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
    c = ev("e3", "Smith paid the deposit", d=None, doc="d3")
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


# --------------------------------------------------------------------------- generalization


def test_honorific_and_partial_names_link_events(fake_llm):
    # "Mr. Ellison" and "Mark Ellison" are one person: the exact-string actor test missed this.
    a = ev("e1", "meeting in the manager's office", actors=["Mark Ellison"], doc="d1")
    b = ev("e2", "manager threatened her during a meeting in his office", actors=["Mr. Ellison"])
    b = b.model_copy(update={"date_start": date(2024, 3, 4), "date_end": date(2024, 3, 4)})
    b = b.model_copy(update={"source": b.source.model_copy(update={"doc_id": "d2"})})
    assert [c.field for c in conflicts.candidate_pairs([a, b])] == [ConflictField.DATE]


def test_close_dates_with_shared_topic_word_are_candidates(fake_llm):
    # Low description similarity, but same person, one shared topic word, one day apart.
    a = ev("e1", "A meeting took place in Lee's office", actors=["Ann Lee"], doc="d1")
    b = ev("e2", "Lee attended a meeting with Ortiz", d=date(2024, 3, 4), actors=["Ms. Lee"])
    b = b.model_copy(update={"source": b.source.model_copy(update={"doc_id": "d2"})})
    assert conflicts.description_similarity(a, b) < conflicts.SIMILAR_THRESHOLD
    assert [c.field for c in conflicts.candidate_pairs([a, b])] == [ConflictField.DATE]


def test_shared_employer_alone_does_not_link(fake_llm):
    a = ev("e1", "interview held", actors=["Ann Lee", "Acme Corp."], doc="d1")
    b = ev("e2", "interview scheduled", d=date(2024, 3, 5), actors=["Bo Chan", "Acme Corp."])
    b = b.model_copy(update={"source": b.source.model_copy(update={"doc_id": "d2"})})
    assert conflicts.candidate_pairs([a, b]) == []


def test_same_date_different_people_is_actor_candidate(fake_llm):
    a = ev("e1", "meeting in Lee's office", actors=["Ann Lee", "Bo Chan"], doc="d1")
    b = ev("e2", "Lee held a meeting", actors=["Ann Lee", "Cy Diaz"], doc="d2")
    assert [c.field for c in conflicts.candidate_pairs([a, b])] == [ConflictField.ACTOR]
    same_people = ev("e3", "Lee held a meeting", actors=["Ms. Lee", "Bo Chan"], doc="d2")
    assert conflicts.candidate_pairs([a, same_people]) == []


def test_unresolved_date_needs_strong_topic_overlap(fake_llm):
    a = ev("e1", "Lee reported the brake failure on truck 7", actors=["Ann Lee"], doc="d1")
    b = ev("e2", "Lee first learned about the truck 7 brake failure", d=None, actors=["Lee"])
    b = b.model_copy(update={"source": b.source.model_copy(update={"doc_id": "d2"})})
    weak = ev("e3", "Lee went on vacation", d=None, actors=["Lee"], doc="d3")
    assert {(c.a.id, c.b.id, c.field) for c in conflicts.candidate_pairs([a, b, weak])} == {
        ("e1", "e2", ConflictField.FACT)
    }


def test_topic_tokens_ignore_dates_and_names():
    e = ev("e1", "Ann Lee signed the lease on March 3, 2024", actors=["Ann Lee"])
    assert conflicts.topic_tokens(e) == {"sign", "lease"}


def test_quote_context_snaps_to_sentences():
    text = "Intro line.\nFirst fact here. The quoted part is this. Only two people attended.\nEnd."
    start = text.index("The quoted")
    e = ev("e1", "x", quote="The quoted part is this.")
    e = e.model_copy(
        update={
            "source": e.source.model_copy(
                update={"char_start": start, "char_end": start + len("The quoted part is this.")}
            )
        }
    )
    ctx = conflicts.quote_context(text, e)
    assert "The quoted part is this." in ctx and "Only two people attended." in ctx
    assert conflicts.quote_context(None, e) == ""
    # Offsets that do not match the quote fall back to a text search.
    bad = e.model_copy(update={"source": e.source.model_copy(update={"char_start": 0})})
    assert "The quoted part" in conflicts.quote_context(text, bad)


def test_context_and_doc_names_reach_the_prompt(fake_llm):
    a = ev("e1", "Smith signed the lease", doc="d1", quote="Smith signed.")
    b = ev("e2", "Smith signed the lease", d=date(2024, 3, 9), doc="d2", quote="He signed.")
    texts = {"d1": "Smith signed. Nobody else was there.", "d2": "He signed. It rained."}
    conflicts.find_contradictions(
        [a, b], "demo", source_text=texts.get, doc_name=lambda d: f"{d}.pdf"
    )
    user = fake_llm["calls"][0][1]
    assert "Nobody else was there." in user and "It rained." in user
    assert "d1.pdf" in user and "Suspected conflict type: date" in user


def test_multi_field_verdict_and_same_subject_gate(fake_llm):
    a = ev("e1", "Smith signed the lease", doc="d1")
    b = ev("e2", "Smith signed the lease", d=date(2024, 3, 9), doc="d2")
    fake_llm["extra"] = {"other_fields": ["actor", "fact"]}
    texts = {"d1": "", "d2": "Nobody else was there."}
    out = conflicts.find_contradictions([a, b], "demo", source_text=texts.get)
    # "fact" as a secondary field restates the primary disagreement and is dropped.
    assert sorted(c.field for c in out) == [ConflictField.ACTOR, ConflictField.DATE]
    assert len({c.id for c in out}) == 2
    fake_llm["extra"] = {"same_subject": False}
    assert conflicts.find_contradictions([a, b], "demo", source_text=lambda _: None) == []


def test_date_field_needs_two_resolved_dates(fake_llm):
    a = ev("e1", "Lee reported the brake failure on truck 7", actors=["Ann Lee"], doc="d1")
    b = ev("e2", "Lee first learned about the truck 7 brake failure", d=None, actors=["Lee"])
    b = b.model_copy(update={"source": b.source.model_copy(update={"doc_id": "d2"})})

    def chat_json(system, user, schema, **kw):
        return schema(is_contradiction=True, field="date", explanation="x")

    pipeline.llm.chat_json = chat_json
    out = conflicts.find_contradictions([a, b], "demo", source_text=lambda _: None)
    assert [c.field for c in out] == [ConflictField.FACT]


def test_redundant_event_vs_document_findings_dropped(fake_llm):
    # e1 conflicts with document d2; a weaker second d2 event repeating it adds nothing.
    a = ev("e1", "Smith signed the lease", doc="d1")
    b = ev("e2", "Smith signed the lease", d=date(2024, 3, 9), doc="d2")
    c = ev("e3", "Smith signed the lease papers late", d=date(2024, 3, 20), doc="d2")
    out = conflicts.find_contradictions([a, b, c], "demo", source_text=lambda _: None)
    pairs = {tuple(x.event_ids) for x in out}
    assert ("e1", "e2") in pairs or ("e1", "e3") in pairs
    assert not {("e1", "e2"), ("e1", "e3")} <= pairs


def test_verdict_schema_requires_every_field():
    schema = conflicts.ConflictVerdict.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    v = conflicts.ConflictVerdict(is_contradiction=True, field="date", explanation="e")
    assert v.same_subject and v.fields() == [ConflictField.DATE]


def test_linked_duplicates_join_the_pool_and_focus():
    a = ev("e1", "Smith signed the lease", doc="d1")
    b = ev("e2", "Smith signed the lease", doc="d2")
    dup_new = ev("e3", "Smith signed the lease", doc="d2")
    dup_old = ev("e4", "Smith signed the lease", doc="d9")
    pool, focus = conflicts.with_linked_duplicates([a, b], [dup_new, dup_old, a], {"e2"})
    assert [e.id for e in pool] == ["e1", "e2", "e3", "e4"]
    assert focus == {"e2", "e3"}
    assert conflicts.with_linked_duplicates([a], [], None) == ([a], None)


def test_actor_omission_without_exclusion_cue_dropped(fake_llm):
    a = ev("e1", "meeting in Lee's office", actors=["Ann Lee", "Bo Chan"], doc="d1")
    b = ev("e2", "Lee held a meeting", actors=["Ann Lee", "Cy Diaz"], doc="d2")
    texts = {"d1": "Lee and Chan met.", "d2": "Lee met Diaz."}
    assert conflicts.find_contradictions([a, b], "demo", source_text=texts.get) == []
    texts["d2"] = "Lee met Diaz. Only the two of them were present."
    out = conflicts.find_contradictions([a, b], "demo", source_text=texts.get)
    assert [c.field for c in out] == [ConflictField.ACTOR]


def test_overlapping_dates_cannot_be_a_date_conflict(fake_llm):
    a = ev("e1", "Lee was sent home without pay", actors=["Ann Lee"], doc="d1")
    b = ev("e2", "Lee was placed on unpaid leave and denied", actors=["Lee"], doc="d2")
    fake_llm["extra"] = {}

    def chat_json(system, user, schema, **kw):
        return schema(is_contradiction=True, field="date", explanation="x")

    pipeline.llm.chat_json = chat_json
    assert conflicts.find_contradictions([a, b], "demo", source_text=lambda _: None) == []


def test_plan_versus_outcome_is_not_a_date_candidate(fake_llm):
    plan = ev("e1", "Interview with Lee scheduled for March 3", actors=["Ann Lee"], doc="d1")
    held = ev("e2", "Interview with Lee held", d=date(2024, 3, 4), actors=["Lee"], doc="d2")
    assert conflicts.candidate_pairs([plan, held]) == []
    plan2 = ev("e3", "Interview with Lee set for March 4", d=date(2024, 3, 4), actors=["Lee"])
    plan2 = plan2.model_copy(update={"source": plan2.source.model_copy(update={"doc_id": "d3"})})
    assert [c.field for c in conflicts.candidate_pairs([plan, plan2])] == [ConflictField.DATE]


def test_without_is_not_a_negation():
    assert not conflicts._has_negation(ev("e1", "sent home without pay"))
    assert conflicts._has_negation(ev("e1", "he denied it"))


def test_knowledge_timing_conflict_is_fact_not_date(fake_llm):
    # "Lee first learned of the defect in mid-June" vs "Ann Lee was sent the defect report on
    # June 2": both dates resolve, but the dispute is about when Lee knew, i.e. notice.
    a = ev(
        "e1",
        "Ann Lee was sent a written report of the pump defect",
        d=date(2021, 6, 2),
        actors=["Ann Lee", "Ray Ono"],
        doc="d1",
    )
    b = ev(
        "e2",
        "Lee first learned about the pump defect",
        d=date(2021, 6, 11),
        actors=["Lee"],
        doc="d2",
    )
    [cand] = conflicts.candidate_pairs([a, b])
    assert cand.field == ConflictField.FACT
    # Even when the model answers "date", the stored field is FACT.
    verdict = conflicts.ConflictVerdict(is_contradiction=True, field=ConflictField.DATE)
    assert conflicts._final_fields(verdict, a, b) == [ConflictField.FACT]
    out = conflicts.find_contradictions([a, b], "demo", source_text=lambda _d: None)
    assert [c.field for c in out] == [ConflictField.FACT]
