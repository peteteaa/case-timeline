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
    alerts = conflicts.alerts_for(cs, events, "demo", "http://127.0.0.1:8090/", now=now)
    kinds = [a.type for a in alerts]
    assert kinds.count(AlertType.NEW_CONTRADICTION) == 2
    assert kinds.count(AlertType.DEADLINE_DATE) == 1
    witness = [a for a in alerts if a.type == AlertType.FLAGGED_WITNESS]
    assert len(witness) == 1 and witness[0].ref_ids == ["c1", "c2"]
    for a in alerts:
        assert a.dashboard_url == "http://127.0.0.1:8090/alerts"
        assert a.created_at == now
        dumped = a.model_dump_json().lower()
        assert "secret" not in dumped and "zelda" not in dumped
    again = conflicts.alerts_for(cs, events, "demo", "http://127.0.0.1:8090", now=now)
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
    a = ev("e1", "Smith signed the lease", actors=["John Smith", "Ann Lee"], doc="d1")
    b = ev("e2", "Smith signed the lease", d=date(2024, 3, 9), doc="d2")
    fake_llm["extra"] = {"other_fields": ["actor", "fact"]}
    # d2 names only Smith and excludes everyone else, so Lee's presence (d1) is disputed.
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


# --------------------------------------------------------------------------- precision rules


def dated(e: Event, start: date, end: date | None = None, **kw) -> Event:
    """Copy of ``e`` with another resolved date (a range when ``end`` differs)."""
    end = end or start
    certainty = kw.pop("certainty", Certainty.EXACT if end == start else Certainty.APPROXIMATE)
    return e.model_copy(update={"date_start": start, "date_end": end, "certainty": certainty, **kw})


def test_range_containment_and_granularity_are_not_date_conflicts(fake_llm):
    # "early March" (an approximate period) vs exact dates inside it or just past its edge.
    early = dated(
        ev("e1", "Lee signed the supply contract", actors=["Ann Lee"], doc="d1"),
        date(2024, 3, 1),
        date(2024, 3, 10),
        date_text="early March 2024",
    )
    inside = ev("e2", "Lee signed the supply contract", d=date(2024, 3, 5), actors=["Lee"])
    edge = ev("e3", "Lee signed the supply contract", d=date(2024, 3, 12), actors=["Lee"])
    far = ev("e4", "Lee signed the supply contract", d=date(2024, 3, 25), actors=["Lee"])
    inside, edge, far = (
        x.model_copy(update={"source": x.source.model_copy(update={"doc_id": "d2"})})
        for x in (inside, edge, far)
    )
    assert conflicts.date_relation(early, inside).compatible
    assert conflicts.date_relation(early, edge).compatible
    assert not conflicts.date_relation(early, far).compatible
    fields = {(c.a.id, c.b.id): c.field for c in conflicts.candidate_pairs([early, inside])}
    assert ConflictField.DATE not in fields.values()
    assert conflicts.candidate_pairs([early, edge]) == []
    assert [c.field for c in conflicts.candidate_pairs([early, far])] == [ConflictField.DATE]
    # Even if the model calls it a date conflict, a contained date is dropped.
    verdict = conflicts.ConflictVerdict(is_contradiction=True, field=ConflictField.DATE)
    assert conflicts._final_fields(verdict, early, inside) == []
    assert conflicts._final_fields(verdict, early, edge) == []


def test_contained_date_is_no_evidence_of_one_occurrence(fake_llm):
    # A period that merely contains a meeting's date does not make the events one occurrence,
    # so "different people named" is not an actor candidate.
    period = dated(
        ev("e1", "Lee reviewed the budget with Chan", actors=["Ann Lee", "Bo Chan"], doc="d1"),
        date(2024, 3, 11),
        date(2024, 3, 20),
    )
    meeting = ev("e2", "Lee held a budget meeting with Diaz", d=date(2024, 3, 12), doc="d2")
    meeting = meeting.model_copy(update={"actors": ["Ann Lee", "Cy Diaz"]})
    cands = conflicts.candidate_pairs([period, meeting])
    assert all(c.field == ConflictField.FACT for c in cands)  # never ACTOR or DATE


def test_knowledge_event_is_never_an_actor_conflict(fake_llm):
    learned = ev("e1", "Lee first learned about the pump defect", actors=["Ann Lee"], doc="d1")
    meeting = ev(
        "e2", "Lee discussed the pump defect with Ono", actors=["Ann Lee", "Ray Ono"], doc="d2"
    )
    assert all(
        c.field != ConflictField.ACTOR for c in conflicts.candidate_pairs([learned, meeting])
    )
    verdict = conflicts.ConflictVerdict(is_contradiction=True, field=ConflictField.ACTOR)
    local = ("Nobody else knew.", "Only Lee and Ono were present.")
    assert conflicts._final_fields(verdict, learned, meeting, local=local) == []


def test_plan_versus_outcome_is_no_conflict_of_any_kind(fake_llm):
    plan = ev(
        "e1", "Review meeting scheduled with Lee and Chan", actors=["Ann Lee", "Bo Chan"], doc="d1"
    )
    held = ev(
        "e2", "Review meeting held with Lee and Diaz", actors=["Ann Lee", "Cy Diaz"], doc="d2"
    )
    assert conflicts.candidate_pairs([plan, held]) == []
    verdict = conflicts.ConflictVerdict(is_contradiction=True, field=ConflictField.ACTOR)
    assert conflicts._final_fields(verdict, plan, held, local=("only them", "only them")) == []
    # Two plans for the same review on different dates are still compared.
    plan2 = ev("e3", "Review meeting with Lee set for March 9", d=date(2024, 3, 9), doc="d3")
    plan2 = plan2.model_copy(update={"actors": ["Ann Lee"]})
    assert [c.field for c in conflicts.candidate_pairs([plan, plan2])] == [ConflictField.DATE]


def test_message_date_is_not_the_date_of_the_reported_occurrence(fake_llm):
    # An email dated by its header (March 14) reports a meeting on March 12; another document
    # dates the meeting March 12. The header date must not be compared with the meeting date.
    email = ev(
        "e1",
        "Lee called Ortiz into her office about the audit",
        d=date(2024, 3, 14),
        actors=["Ann Lee", "Jo Ortiz"],
        doc="d1",
        quote="On March 12, 2024 she called me into her office about the audit.",
    ).model_copy(update={"date_text": "Thursday, March 14, 2024 9:02 AM"})
    report = ev(
        "e2",
        "Lee met Ortiz in her office about the audit",
        d=date(2024, 3, 12),
        actors=["Ann Lee", "Jo Ortiz"],
        doc="d2",
    )
    assert conflicts.carrier_dates(email) == [date(2024, 3, 12)]
    rel = conflicts.date_relation(email, report)
    assert rel.carrier and rel.gap == 0 and rel.compatible
    assert all(c.field != ConflictField.DATE for c in conflicts.candidate_pairs([email, report]))
    # The quoted date is what gets compared when it disagrees with the other source.
    early = dated(report, date(2024, 3, 9), id="e3")
    rel = conflicts.date_relation(email, early)
    assert rel.carrier and rel.gap == 3
    assert [c.field for c in conflicts.candidate_pairs([email, early])] == [ConflictField.DATE]
    # A quote that states the event's own date is not a message date.
    assert conflicts.carrier_dates(dated(email, date(2024, 3, 12))) == []


def test_actor_exclusion_must_come_from_the_side_missing_the_person(fake_llm):
    both = ev("e1", "meeting in Lee's office", actors=["Ann Lee", "Bo Chan"], doc="d1")
    lee_only = ev("e2", "Lee held a meeting", actors=["Ann Lee"], doc="d2")
    verdict = conflicts.ConflictVerdict(is_contradiction=True, field=ConflictField.ACTOR)
    # Only the side naming everyone has an exclusion cue: d2 merely omits Chan.
    local = ("Lee and Chan met. Nobody else came.", "Lee held a meeting.")
    assert conflicts._final_fields(verdict, both, lee_only, local=local) == []
    # d2 itself says no one else was there: a real disagreement about Chan.
    local = ("Lee and Chan met.", "Lee held a meeting. No one else was present.")
    assert conflicts._final_fields(verdict, both, lee_only, local=local) == [ConflictField.ACTOR]


def test_local_context_ignores_exclusions_about_earlier_occurrences():
    text = (
        "On May 1 Lee met Chan. No one else was present.\n"
        "On May 3 Lee was suspended.\n"
        "Q. Who attended? A. Lee and Ortiz. Nobody else."
    )
    suspended = ev("e1", "Lee was suspended", quote="On May 3 Lee was suspended.")
    suspended = suspended.model_copy(
        update={"source": suspended.source.model_copy(update={"char_start": 48, "char_end": 75})}
    )
    assert text[48:75] == "On May 3 Lee was suspended."
    local = conflicts.local_context(text, suspended)
    assert "No one else" not in local and local.startswith("On May 3")
    answer = ev("e2", "Lee and Ortiz attended", quote="Q. Who attended?")
    assert "Nobody else" in conflicts.local_context(text, answer)


def test_time_zone_shift_and_clock_precision_are_not_conflicts(fake_llm):
    local_day = ev("e1", "The aircraft crashed on landing", actors=["Lee"], doc="d1")
    utc_next = ev("e2", "The aircraft crashed on landing", d=date(2024, 3, 4), doc="d2")
    utc_next = utc_next.model_copy(
        update={"actors": ["Lee"], "date_text": "March 4, 2024 0216 UTC"}
    )
    assert conflicts.date_relation(local_day, utc_next).compatible
    assert conflicts.candidate_pairs([local_day, utc_next]) == []
    # Two transcripts timing one radio call a second apart.
    a = ev("e3", "Tower cleared Lee to land", d=None, actors=["Lee"], doc="d1")
    b = ev("e4", "Tower cleared Lee to land", d=None, actors=["Lee"], doc="d2")
    a = a.model_copy(update={"date_text": "1815:23"})
    b = b.model_copy(update={"date_text": "at about 6:15:22 PM"})
    rel = conflicts.date_relation(a, b)
    assert rel.clock_gap == 1 and rel.same_moment and rel.compatible
    verdict = conflicts.ConflictVerdict(is_contradiction=True, field=ConflictField.DATE)
    assert conflicts._final_fields(verdict, a, b) == []
    assert conflicts.clock_seconds("June 5, 2021") is None


def test_inferred_year_or_inherited_day_is_not_a_soft_date(fake_llm):
    # The date resolver marks a date APPROXIMATE when only its year was inferred ("March 9")
    # or when a bare clock time inherits its day ("6:15 PM"). Both are one definite day, so a
    # few days' gap is still a date conflict; only a hedged day ("around March 9") is soft.
    exact = ev("e1", "Lee met Chan about the safety audit", d=date(2024, 3, 12), doc="d1")
    exact = exact.model_copy(
        update={"actors": ["Ann Lee", "Bo Chan"], "date_text": "March 12, 2024"}
    )

    def other(eid: str, text: str, d: date) -> Event:
        e = ev(eid, "Lee met Chan about the safety audit", d=d, doc="d2")
        return e.model_copy(
            update={
                "actors": ["Ann Lee", "Bo Chan"],
                "date_text": text,
                "certainty": Certainty.APPROXIMATE,
            }
        )

    for text, d in (("March 9", date(2024, 3, 9)), ("6:15 PM", date(2024, 3, 15))):
        o = other("e2", text, d)
        assert not conflicts.date_relation(exact, o).compatible, text
        assert [c.field for c in conflicts.candidate_pairs([exact, o])] == [ConflictField.DATE]
    for text in ("around March 9", "early March", "a few days before"):
        o = other("e2", text, date(2024, 3, 9))
        assert conflicts.date_relation(exact, o).compatible, text
        assert conflicts.candidate_pairs([exact, o]) == []


def test_distinct_timed_moments_are_not_one_occurrence(fake_llm):
    # Same day, but timestamps given to the second minutes apart: two radio calls, not one
    # occurrence told two ways, so "different people named" is no actor candidate.
    a = ev("e1", "Tower told Lee and Diaz to go around", actors=["Ann Lee", "Cy Diaz"], doc="d1")
    b = ev("e2", "Tower told Lee and Ono to go around", actors=["Ann Lee", "Ray Ono"], doc="d2")
    a = a.model_copy(update={"date_text": "1813:31"})
    b = b.model_copy(update={"date_text": "18:16:22"})
    assert conflicts.distinct_moments(a, b)
    assert conflicts.candidate_pairs([a, b]) == []
    # Close timestamps are one moment, so the same pair is still worth an LLM check.
    b_close = b.model_copy(update={"date_text": "18:13:50"})
    assert not conflicts.distinct_moments(a, b_close)
    assert [c.field for c in conflicts.candidate_pairs([a, b_close])] == [ConflictField.ACTOR]
    # Loose times from two witnesses ("about 2:30" vs "2:47") may still be one occurrence.
    w1 = a.model_copy(update={"date_text": "about 2:30 PM"})
    w2 = b.model_copy(update={"date_text": "2:47 PM"})
    assert not conflicts.distinct_moments(w1, w2)
    assert [c.field for c in conflicts.candidate_pairs([w1, w2])] == [ConflictField.ACTOR]
    # One source's own timeline lists distinct moments, whatever the precision.
    s1 = w1.model_copy(update={"source": w2.source})
    assert conflicts.distinct_moments(s1, w2)


def test_confirm_prompt_carries_rules_and_computed_date_notes(fake_llm):
    email = ev(
        "e1",
        "Lee met Ortiz about the audit",
        d=date(2024, 3, 14),
        actors=["Ann Lee"],
        doc="d1",
        quote="On March 6, 2024 Lee met Ortiz about the audit.",
    )
    report = ev("e2", "Lee met Ortiz about the audit", d=date(2024, 3, 12), doc="d2")
    report = report.model_copy(update={"actors": ["Ann Lee"]})
    conflicts.find_contradictions([email, report], "demo", source_text=lambda _d: None)
    system, user = fake_llm["calls"][0]
    assert system.endswith(conflicts.CONFIRM_RULES) and "Plan vs outcome" in system
    assert "Notes from the date rules" in user and "2024-03-06" in user


def test_concurrent_confirmations_match_sequential_order(fake_llm, monkeypatch):
    import threading
    import time

    from pipeline.config import get_settings

    lock = threading.Lock()
    flight = {"now": 0, "max": 0}

    def slow_chat_json(system, user, schema, **kw):
        with lock:
            flight["now"] += 1
            flight["max"] = max(flight["max"], flight["now"])
        time.sleep(0.05)
        with lock:
            flight["now"] -= 1
        return schema(is_contradiction=True, field="date", explanation=user[-40:])

    pipeline.llm.chat_json = slow_chat_json
    events = [
        ev(f"e{i}", "Smith signed the lease", d=date(2024, 3, 3 + 2 * i), doc=f"d{i}")
        for i in range(6)
    ]

    def run(n: int) -> tuple[list[Contradiction], float]:
        monkeypatch.setenv("CT_LLM_CONCURRENCY", str(n))
        get_settings.cache_clear()
        flight["max"] = 0
        t0 = time.monotonic()
        out = conflicts.find_contradictions(events, "demo", source_text=lambda _d: None)
        return out, time.monotonic() - t0

    try:
        seq, t_seq = run(1)
        assert flight["max"] == 1
        par, t_par = run(8)
        assert flight["max"] > 1
    finally:
        get_settings.cache_clear()
    assert [c.model_dump() for c in par] == [c.model_dump() for c in seq]
    assert len(seq) > 3 and t_par < t_seq / 2
