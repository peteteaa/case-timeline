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
    # An undated fact is only a candidate with a strong topic overlap (see below).
    c = make_fact("F-3", "Smith paid the deposit", None, witnesses=["W-1"], docs=["D-3"])
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


# --------------------------------------------------------------------------- ported rules

NAMES = {
    "W-1": ["Mark Ellison"],
    "W-2": ["Mr. Ellison"],
    "W-3": ["Ann Lee"],
    "W-4": ["Bo Chan"],
    "W-5": ["Cy Diaz"],
    "W-9": ["Acme Corp."],
}


def test_honorific_and_partial_names_link_facts():
    a = make_fact("F-1", "meeting in the manager's office", witnesses=["W-1"], docs=["D-1"])
    b = make_fact(
        "F-2",
        "manager threatened her during a meeting in his office",
        date(2024, 3, 4),
        witnesses=["W-2"],
        docs=["D-2"],
    )
    assert conflicts.candidate_pairs([a, b], {"F-1"}) == []  # ids alone don't link them
    (c,) = conflicts.candidate_pairs([a, b], {"F-1"}, witness_names=NAMES)
    assert c.field == "date"


def test_shared_employer_alone_does_not_link():
    a = make_fact("F-1", "interview held", witnesses=["W-3", "W-9"], docs=["D-1"])
    b = make_fact("F-2", "interview held", date(2024, 3, 5), witnesses=["W-4", "W-9"], docs=["D-2"])
    assert conflicts.candidate_pairs([a, b], {"F-1"}, witness_names=NAMES) == []


def test_plan_versus_outcome_is_not_a_date_candidate():
    plan = make_fact(
        "F-1",
        "Investigation interview with Lee scheduled",
        date(2024, 3, 27),
        witnesses=["W-3"],
        docs=["D-1"],
        quotes=["HR scheduled an interview with you for March 27, 2024."],
    )
    held = make_fact(
        "F-2",
        "Investigation interview with Lee held",
        date(2024, 3, 28),
        witnesses=["W-3"],
        docs=["D-2"],
    )
    assert conflicts.candidate_pairs([plan, held], {"F-1", "F-2"}) == []


def test_same_source_never_checked_against_itself():
    a = make_fact("F-1", "Smith signed the lease", witnesses=["W-1"], docs=["D-1", "D-2"])
    b = make_fact(
        "F-2", "Smith signed the lease", date(2024, 3, 9), witnesses=["W-1"], docs=["D-1", "D-2"]
    )
    assert conflicts.candidate_pairs([a, b], {"F-1"}) == []
    c = b.model_copy(update={"fact_id": "F-3"})
    c.evidence[0].doc_id = "D-3"  # one of its sources differs: worth a check
    assert [x.field for x in conflicts.candidate_pairs([a, c], {"F-1"})] == ["date"]


def test_different_people_on_one_occurrence_is_actor_candidate():
    a = make_fact("F-1", "meeting in Lee's office", witnesses=["W-3", "W-4"], docs=["D-1"])
    b = make_fact("F-2", "Lee held a meeting", witnesses=["W-3", "W-5"], docs=["D-2"])
    assert [c.field for c in conflicts.candidate_pairs([a, b], {"F-1"})] == ["actor"]
    one_sided = make_fact("F-3", "Lee held a meeting", witnesses=["W-3"], docs=["D-2"])
    assert [c.field for c in conflicts.candidate_pairs([a, one_sided], {"F-1"})] == ["actor"]
    same = make_fact("F-4", "Lee held a meeting", witnesses=["W-3", "W-4"], docs=["D-2"])
    assert conflicts.candidate_pairs([a, same], {"F-1"}) == []


def test_undated_fact_needs_strong_topic_overlap_and_knowledge_is_fact():
    a = make_fact(
        "F-1", "Lee reported the brake failure on truck 7", witnesses=["W-3"], docs=["D-1"]
    )
    b = make_fact(
        "F-2",
        "Lee first learned about the truck 7 brake failure",
        None,
        witnesses=["W-3"],
        docs=["D-2"],
    )
    weak = make_fact("F-3", "Lee went on vacation", None, witnesses=["W-3"], docs=["D-3"])
    found = conflicts.candidate_pairs([a, b, weak], {"F-1", "F-2", "F-3"})
    assert [(c.a.fact_id, c.b.fact_id, c.field) for c in found] == [("F-1", "F-2", "fact")]
    later = b.model_copy(update={"date_start": date(2024, 3, 20), "date_end": date(2024, 3, 20)})
    assert [c.field for c in conflicts.candidate_pairs([a, later], {"F-1"})] == ["fact"]


def test_verdict_schema_requires_every_field():
    schema = conflicts.ConflictVerdict.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    v = conflicts.ConflictVerdict(is_contradiction=True, field="date", other_fields=["fact"])
    assert v.same_subject and v.fields() == ["date"]


def test_final_fields_sanity_checks():
    a = make_fact("F-1", "Lee met Chan", witnesses=["W-3", "W-4"], docs=["D-1"])
    b = make_fact("F-2", "Lee met Diaz", date(2024, 3, 9), witnesses=["W-3", "W-5"], docs=["D-2"])
    undated = b.model_copy(update={"date_start": None, "date_end": None})
    same_day = b.model_copy(update={"date_start": a.date_start, "date_end": a.date_end})

    def verdict(field, **kw):
        return conflicts.ConflictVerdict(is_contradiction=True, field=field, **kw)

    assert conflicts.final_fields(verdict("date"), a, b) == ["date"]
    assert conflicts.final_fields(verdict("date"), a, undated) == ["fact"]
    assert conflicts.final_fields(verdict("date"), a, same_day) == []
    # Linked facts: an actor conflict needs someone to exclude a person.
    assert conflicts.final_fields(verdict("actor"), a, same_day, "Lee and Diaz met.") == []
    assert conflicts.final_fields(
        verdict("actor"), a, same_day, "Lee met Diaz. Nobody else was there."
    ) == ["actor"]
    assert conflicts.final_fields(verdict("date", same_subject=False), a, b) == []
    assert conflicts.final_fields(verdict("date", other_fields=["actor"]), a, b, "only") == [
        "date",
        "actor",
    ]


def test_quote_context_snaps_to_sentences():
    text = "Intro line.\nFirst fact here. The quoted part is this. Only two people attended.\nEnd."
    q = "The quoted part is this."
    start = text.index(q)
    ctx = conflicts.quote_context(text, q, start, start + len(q))
    assert q in ctx and "Only two people attended." in ctx
    assert conflicts.quote_context(None, q) == ""
    assert q in conflicts.quote_context(text, q, 0, 5)  # wrong offsets: found by search


# --------------------------------------------------------------------------- detect, merged facts


@pytest.fixture
def meeting_case(tmp_path):
    """Becker dates the meeting March 11; two other documents say March 12 (as two facts)."""
    store = new_store(tmp_path)
    w = store.add_witness("demo", "Mark Ellison", by="alice").witness_id
    texts = {
        "becker.txt": "Intro. The meeting in Mark Ellison's office took place on March 11, 2024.",
        "memo.txt": "On March 12, 2024, I attended a meeting with Mark Ellison in his office.",
        "email.txt": "On March 12, 2024 Mark Ellison called me into his office for a meeting.",
        "depo.txt": "A. March 12, 2024. The meeting with Mark Ellison was in his office.",
    }
    docs = {name: add_doc(store, t, filename=name) for name, t in texts.items()}

    def span(name):
        t = texts[name]
        i = 7 if name == "becker.txt" else 0
        return SpanIn(doc_id=docs[name], char_start=i, char_end=len(t), quote=t[i:])

    def fact(title, day, *names):
        return store.add_fact(
            "demo",
            title,
            by=AGENT,
            date_start=day,
            date_end=day,
            date_certainty="exact",
            witness_ids=[w],
            spans=[span(n) for n in names],
        ).fact_id

    becker = fact("Meeting in Ellison's office", date(2024, 3, 11), "becker.txt")
    merged = fact("Meeting in Ellison's office", date(2024, 3, 12), "memo.txt", "email.txt")
    depo = fact("Ellison meets in his office", date(2024, 3, 12), "depo.txt")
    yield store, becker, merged, depo
    store.close()


def test_detect_shows_every_document_of_a_merged_fact(meeting_case, fake_llm):
    store, becker, merged, _ = meeting_case
    conflicts.detect(store, "demo", [becker])
    user = next(u for _, u in fake_llm["calls"] if merged in u)
    assert "memo.txt" in user and "email.txt" in user and "becker.txt" in user
    assert "called me into his office" in user and "I attended a meeting" in user
    assert "Intro." in user  # surrounding text of the quote


def test_detect_does_not_repeat_the_same_disagreement(meeting_case, fake_llm):
    store, becker, merged, depo = meeting_case
    opened = conflicts.detect(store, "demo", [becker, merged, depo])
    assert len(opened) == 1 and opened[0].field == "date"
    assert becker in opened[0].fact_ids
    # Confirmations run in parallel, so the restatement may be asked, but it never opens.
    asked = len(fake_llm["calls"])
    assert 1 <= asked <= 2
    # Across runs: a dismissed disagreement isn't reopened, or even re-asked, through the
    # other fact.
    store.review_contradiction(opened[0].contradiction_id, "dismiss", reviewer="alice")
    assert conflicts.detect(store, "demo", [becker, merged, depo]) == []
    assert len(fake_llm["calls"]) == asked


def test_detect_same_subject_false_opens_nothing(meeting_case, fake_llm, monkeypatch):
    store, becker, _, _ = meeting_case

    def chat_json(system, user, schema, **kw):
        fake_llm["calls"].append((system, user))
        return schema(is_contradiction=True, same_subject=False, field="date")

    monkeypatch.setattr(llm, "chat_json", chat_json)
    assert conflicts.detect(store, "demo", [becker]) == []
    assert fake_llm["calls"]


# --------------------------------------------------------------------------- precision rules


def verdict_for(field, **kw):
    return conflicts.ConflictVerdict(is_contradiction=True, field=field, **kw)


def test_range_containment_and_approximate_edges_are_not_date_conflicts():
    # "early March" (an approximate period) vs exact dates inside it or just past its edge.
    early = make_fact(
        "F-1",
        "Lee signed the supply contract",
        date(2024, 3, 1),
        end=date(2024, 3, 10),
        certainty="approximate",
        date_text="early March 2024",
        witnesses=["W-3"],
        docs=["D-1"],
    )

    def on(fact_id, day):
        title = "Lee signed the supply contract"
        return make_fact(fact_id, title, day, witnesses=["W-3"], docs=["D-2"])

    inside, edge = on("F-2", date(2024, 3, 5)), on("F-3", date(2024, 3, 12))
    far = on("F-4", date(2024, 3, 25))
    assert conflicts.date_relation(early, inside).compatible
    assert conflicts.date_relation(early, edge).compatible
    assert not conflicts.date_relation(early, far).compatible
    assert all(c.field != "date" for c in conflicts.candidate_pairs([early, inside], {"F-1"}))
    assert conflicts.candidate_pairs([early, edge], {"F-1"}) == []
    assert [c.field for c in conflicts.candidate_pairs([early, far], {"F-1"})] == ["date"]
    # Even if the model calls it a date conflict, a contained or edge date is dropped.
    assert conflicts.final_fields(verdict_for("date"), early, inside) == []
    assert conflicts.final_fields(verdict_for("date"), early, edge) == []
    assert conflicts.final_fields(verdict_for("date"), early, far) == ["date"]
    # A span stated to the day at both ends has hard edges: three days past it still counts.
    span = early.model_copy(
        update={"date_end": date(2024, 3, 9), "date_certainty": "exact", "date_text": "March 1-9"}
    )
    assert conflicts.date_relation(span, inside).compatible  # contained
    assert not conflicts.date_relation(span, edge).compatible
    assert [c.field for c in conflicts.candidate_pairs([span, edge], {"F-1"})] == ["date"]


def test_contained_date_is_no_evidence_of_one_occurrence():
    # A period that merely contains a meeting's date does not make the facts one occurrence,
    # so "different people named" is not an actor candidate.
    period = make_fact(
        "F-1",
        "Lee reviewed the budget with Chan",
        date(2024, 3, 11),
        end=date(2024, 3, 20),
        certainty="approximate",
        witnesses=["W-3", "W-4"],
        docs=["D-1"],
    )
    meeting = make_fact(
        "F-2",
        "Lee held a budget meeting with Diaz",
        date(2024, 3, 12),
        witnesses=["W-3", "W-5"],
        docs=["D-2"],
    )
    assert all(c.field == "fact" for c in conflicts.candidate_pairs([period, meeting], {"F-1"}))
    # On one single day the same pair is one occurrence told two ways: worth an actor check.
    day = {"date_start": date(2024, 3, 12), "date_end": date(2024, 3, 12)}
    one_day = period.model_copy(update={**day, "date_certainty": "exact"})
    assert [c.field for c in conflicts.candidate_pairs([one_day, meeting], {"F-1"})] == ["actor"]


def test_knowledge_fact_is_never_an_actor_conflict():
    learned = make_fact(
        "F-1", "Lee first learned about the pump defect", witnesses=["W-3"], docs=["D-1"]
    )
    meeting = make_fact(
        "F-2", "Lee discussed the pump defect with Ono", witnesses=["W-3", "W-6"], docs=["D-2"]
    )
    assert all(c.field != "actor" for c in conflicts.candidate_pairs([learned, meeting], {"F-1"}))
    local = ("Nobody else knew.", "Only Lee and Ono were present.")
    assert conflicts.final_fields(verdict_for("actor"), learned, meeting, local=local) == []
    # Without the knowledge wording the same pair is an actor candidate.
    told = learned.model_copy(update={"title": "Lee reported the pump defect"})
    assert [c.field for c in conflicts.candidate_pairs([told, meeting], {"F-1"})] == ["actor"]


def test_plan_versus_outcome_is_no_conflict_of_any_kind():
    plan = make_fact(
        "F-1", "Review meeting scheduled with Lee and Chan", witnesses=["W-3", "W-4"], docs=["D-1"]
    )
    held = make_fact(
        "F-2", "Review meeting held with Lee and Diaz", witnesses=["W-3", "W-5"], docs=["D-2"]
    )
    assert conflicts.candidate_pairs([plan, held], {"F-1"}) == []
    local = ("Only them.", "Only them.")
    assert conflicts.final_fields(verdict_for("actor"), plan, held, local=local) == []
    # Two plans for the same review on different dates are still compared.
    plan2 = make_fact(
        "F-3",
        "Review meeting with Lee set for March 9",
        date(2024, 3, 9),
        witnesses=["W-3"],
        docs=["D-3"],
    )
    assert [c.field for c in conflicts.candidate_pairs([plan, plan2], {"F-1"})] == ["date"]


def test_message_date_is_not_the_date_of_the_reported_occurrence():
    # An email dated by its header (March 14) reports a meeting on March 12; another document
    # dates the meeting March 12. The header date must not be compared with the meeting date.
    email = make_fact(
        "F-1",
        "Lee called Ortiz into her office about the audit",
        date(2024, 3, 14),
        date_text="Thursday, March 14, 2024 9:02 AM",
        witnesses=["W-3", "W-7"],
        docs=["D-1"],
        quotes=["On March 12, 2024 she called me into her office about the audit."],
    )
    report = make_fact(
        "F-2",
        "Lee met Ortiz in her office about the audit",
        date(2024, 3, 12),
        witnesses=["W-3", "W-7"],
        docs=["D-2"],
    )
    assert conflicts.carrier_dates(email) == [date(2024, 3, 12)]
    rel = conflicts.date_relation(email, report)
    assert rel.carrier and rel.gap == 0 and rel.compatible
    assert all(c.field != "date" for c in conflicts.candidate_pairs([email, report], {"F-1"}))
    assert conflicts.final_fields(verdict_for("date"), email, report) == []
    # The quoted date is what gets compared when it disagrees with the other source.
    march_9 = {"date_start": date(2024, 3, 9), "date_end": date(2024, 3, 9)}
    early = report.model_copy(update={"fact_id": "F-3", **march_9})
    rel = conflicts.date_relation(email, early)
    assert rel.carrier and rel.gap == 3
    assert [c.field for c in conflicts.candidate_pairs([email, early], {"F-1"})] == ["date"]
    # A quote that states the fact's own date is not a message date ...
    march_12 = {"date_start": date(2024, 3, 12), "date_end": date(2024, 3, 12)}
    assert conflicts.carrier_dates(email.model_copy(update=march_12)) == []
    # ... nor is a merged fact whose other document's quote states it.
    merged = make_fact(
        "F-4",
        "Lee called Ortiz into her office about the audit",
        date(2024, 3, 14),
        witnesses=["W-3", "W-7"],
        docs=["D-1", "D-4"],
        quotes=[
            "On March 12, 2024 she called me into her office about the audit.",
            "The audit meeting in her office was on March 14.",
        ],
    )
    assert conflicts.carrier_dates(merged) == []


def test_actor_exclusion_must_come_from_the_side_missing_the_person():
    both = make_fact("F-1", "meeting in Lee's office", witnesses=["W-3", "W-4"], docs=["D-1"])
    lee_only = make_fact("F-2", "Lee held a meeting", witnesses=["W-3"], docs=["D-2"])
    # Only the side naming everyone has an exclusion cue: D-2 merely omits Chan.
    local = ("Lee and Chan met. Nobody else came.", "Lee held a meeting.")
    assert conflicts.final_fields(verdict_for("actor"), both, lee_only, local=local) == []
    # D-2 itself says no one else was there: a real disagreement about Chan.
    local = ("Lee and Chan met.", "Lee held a meeting. No one else was present.")
    assert conflicts.final_fields(verdict_for("actor"), both, lee_only, local=local) == ["actor"]


def test_local_context_ignores_exclusions_about_earlier_occurrences():
    text = (
        "On May 1 Lee met Chan. No one else was present.\n"
        "On May 3 Lee was suspended.\n"
        "Q. Who attended? A. Lee and Ortiz. Nobody else."
    )
    quote = "On May 3 Lee was suspended."
    assert text[48:75] == quote
    local = conflicts.local_context(text, quote, 48, 75)
    assert "No one else" not in local and local.startswith("On May 3")
    assert "No one else" in conflicts.quote_context(text, quote, 48, 75)  # the wider window
    assert "Nobody else" in conflicts.local_context(text, "Q. Who attended?")
    assert conflicts.local_context(None, " A  quote ") == "A quote"


@pytest.fixture
def office_case(tmp_path):
    """One document says Lee and Chan met in Lee's office (and nobody else came)."""
    store = new_store(tmp_path)
    lee = store.add_witness("demo", "Ann Lee", by="alice").witness_id
    chan = store.add_witness("demo", "Bo Chan", by="alice").witness_id

    def fact(title, text, quote, witnesses):
        doc = add_doc(store, text, filename=f"{len(store.list_facts('demo'))}.txt")
        i = text.index(quote)
        return store.add_fact(
            "demo",
            title,
            by=AGENT,
            date_start=date(2024, 3, 4),
            date_end=date(2024, 3, 4),
            date_certainty="exact",
            witness_ids=witnesses,
            spans=[SpanIn(doc_id=doc, char_start=i, char_end=i + len(quote), quote=quote)],
        ).fact_id

    both = "Ann Lee and Bo Chan met in Lee's office on March 4, 2024."
    first = fact(
        "Lee and Chan meet in Lee's office", both + " Nobody else came.", both, [lee, chan]
    )
    yield store, fact, first, lee
    store.close()


def test_detect_needs_the_omitting_side_to_exclude_people(office_case, fake_llm):
    store, fact, first, lee = office_case
    quote = "Ann Lee held a meeting in her office on March 4, 2024."
    # The earlier sentence excludes people from another occurrence, not from this meeting.
    text = "On March 1 she met the auditors. No one else was present. " + quote
    second = fact("Lee holds a meeting in her office", text, quote, [lee])
    assert conflicts.detect(store, "demo", [second]) == []
    assert len(fake_llm["calls"]) == 1  # the model was asked (and said yes)
    excluded = quote + " No one else was present."
    third = fact("Lee holds a meeting in her office alone", excluded, quote, [lee])
    opened = conflicts.detect(store, "demo", [third])
    assert [(c.field, c.fact_ids) for c in opened] == [("actor", sorted([first, third]))]


def test_time_zone_shift_and_clock_precision_are_not_conflicts():
    local_day = make_fact("F-1", "The aircraft crashed on landing", witnesses=["W-3"], docs=["D-1"])
    utc_next = make_fact(
        "F-2",
        "The aircraft crashed on landing",
        date(2024, 3, 4),
        date_text="March 4, 2024 0216 UTC",
        witnesses=["W-3"],
        docs=["D-2"],
    )
    rel = conflicts.date_relation(local_day, utc_next)
    assert rel.zone_shift and rel.compatible
    assert conflicts.candidate_pairs([local_day, utc_next], {"F-1"}) == []
    # Without the zone the exact one-day gap is a candidate.
    plain = utc_next.model_copy(update={"date_text": "March 4, 2024"})
    assert [c.field for c in conflicts.candidate_pairs([local_day, plain], {"F-1"})] == ["date"]
    # Two transcripts timing one radio call a second apart.
    a = make_fact(
        "F-3",
        "Tower cleared Lee to land",
        None,
        date_text="1815:23",
        witnesses=["W-3"],
        docs=["D-1"],
    )
    b = make_fact(
        "F-4",
        "Tower cleared Lee to land",
        None,
        date_text="at about 6:15:22 PM",
        witnesses=["W-3"],
        docs=["D-2"],
    )
    rel = conflicts.date_relation(a, b)
    assert rel.clock_gap == 1 and rel.same_moment and rel.compatible
    assert conflicts.final_fields(verdict_for("date"), a, b) == []
    # A minute and a half is the limit.
    later = b.model_copy(update={"date_text": "6:17:00 PM"})
    assert conflicts.date_relation(a, later).clock_gap == 97
    assert conflicts.final_fields(verdict_for("date"), a, later) == ["fact"]  # both undated
    assert conflicts.clock_seconds("June 5, 2021") is None
    assert conflicts.clock_seconds("12:05 a.m.") == 300


def test_inferred_year_or_inherited_day_is_not_a_soft_date():
    # The date resolver marks a date approximate when only its year was inferred ("March 9")
    # or when a bare clock time inherits its day ("6:15 PM"), and relative when it is an
    # offset ("three days earlier"). Each is one definite day, so a few days' gap is still a
    # date conflict; only a hedged day ("around March 9") or a period is soft.
    exact = make_fact(
        "F-1",
        "Lee met Chan about the safety audit",
        date(2024, 3, 12),
        date_text="March 12, 2024",
        witnesses=["W-3", "W-4"],
        docs=["D-1"],
    )

    def other(text, day=date(2024, 3, 9), certainty="approximate"):
        return make_fact(
            "F-2",
            "Lee met Chan about the safety audit",
            day,
            date_text=text,
            certainty=certainty,
            witnesses=["W-3", "W-4"],
            docs=["D-2"],
        )

    hard = [
        other("March 9"),
        other("6:15 PM", date(2024, 3, 15)),
        other("three days earlier", certainty="relative"),
        other("approximately 3:40 PM"),  # a soft time on a definite day
    ]
    for o in hard:
        assert not conflicts.date_relation(exact, o).compatible, o.date_text
        assert [c.field for c in conflicts.candidate_pairs([exact, o], {"F-1"})] == ["date"]
    soft = [
        other("around March 9"),
        other("early March"),
        other("a few days before", certainty="relative"),
    ]
    for o in soft:
        assert conflicts.date_relation(exact, o).compatible, o.date_text
        assert conflicts.candidate_pairs([exact, o], {"F-1"}) == []


def test_distinct_timed_moments_are_not_one_occurrence():
    # Same day, but timestamps given to the second minutes apart: two radio calls, not one
    # occurrence told two ways, so "different people named" is no actor candidate.
    def call(fact_id, title, text, witnesses, docs):
        return make_fact(fact_id, title, date_text=text, witnesses=witnesses, docs=docs)

    a = call("F-1", "Tower told Lee and Diaz to go around", "1813:31", ["W-3", "W-5"], ["D-1"])
    b = call("F-2", "Tower told Lee and Ono to go around", "18:16:22", ["W-3", "W-6"], ["D-2"])
    assert conflicts.distinct_moments(a, b)
    assert conflicts.candidate_pairs([a, b], {"F-1"}) == []
    # Close timestamps are one moment, so the same pair is still worth an LLM check.
    b_close = b.model_copy(update={"date_text": "18:13:50"})
    assert not conflicts.distinct_moments(a, b_close)
    assert [c.field for c in conflicts.candidate_pairs([a, b_close], {"F-1"})] == ["actor"]
    # Loose times from two witnesses ("about 2:30" vs "2:47") may still be one occurrence.
    w1 = a.model_copy(update={"date_text": "about 2:30 PM"})
    w2 = b.model_copy(update={"date_text": "2:47 PM"})
    assert not conflicts.distinct_moments(w1, w2)
    assert [c.field for c in conflicts.candidate_pairs([w1, w2], {"F-1"})] == ["actor"]
    # One document's own timeline lists distinct moments, whatever the precision.
    s1 = call("F-3", w1.title, "about 2:30 PM", ["W-3", "W-5"], ["D-1", "D-2"])
    assert conflicts.distinct_moments(s1, w2)


def test_confirm_prompt_carries_rules_and_computed_date_notes(fake_llm):
    email = make_fact(
        "F-1",
        "Lee met Ortiz about the audit",
        date(2024, 3, 14),
        witnesses=["W-3"],
        docs=["D-1"],
        quotes=["On March 6, 2024 Lee met Ortiz about the audit."],
    )
    report = make_fact(
        "F-2", "Lee met Ortiz about the audit", date(2024, 3, 12), witnesses=["W-3"], docs=["D-2"]
    )
    (cand,) = conflicts.candidate_pairs([email, report], {"F-1"})
    conflicts.confirm(cand, {"W-3": "Ann Lee"})
    system, user = fake_llm["calls"][-1]
    assert "precision is not disagreement" in system and "never from an inference" in system
    assert 'knowledge is never an "actor" conflict' in system
    assert "Notes from the date rules" in user and "2024-03-06" in user
    assert "date of the message or document" in user
    # Plain dates need no notes.
    plain = make_fact(
        "F-3", "Lee met Ortiz about the audit", date(2024, 3, 14), witnesses=["W-3"], docs=["D-1"]
    )
    conflicts.confirm(conflicts.Candidate(plain, report, "date", 1.0), {})
    assert "Notes from the date rules" not in fake_llm["calls"][-1][1]


def test_date_notes_explain_compatible_dates_zones_and_plans():
    plan = make_fact(
        "F-1",
        "Interview with Lee scheduled",
        date(2024, 3, 1),
        end=date(2024, 3, 10),
        certainty="approximate",
        witnesses=["W-3"],
        docs=["D-1"],
    )
    held = make_fact(
        "F-2", "Interview with Lee", date(2024, 3, 12), witnesses=["W-3"], docs=["D-2"]
    )
    notes = conflicts.date_notes(plan, held)
    assert any("compatible" in n for n in notes)
    assert any(n.startswith("Fact A is about a plan") for n in notes)
    utc = make_fact(
        "F-3",
        "Interview with Lee",
        date(2024, 3, 13),
        date_text="March 13, 2024 0216 UTC",
        witnesses=["W-3"],
        docs=["D-3"],
    )
    assert any("time zones" in n for n in conflicts.date_notes(held, utc))


def test_detect_sends_confirmations_through_the_parallel_limiter(
    meeting_case, fake_llm, monkeypatch
):
    store, becker, merged, depo = meeting_case
    batches: list[int] = []
    real = llm.map_ordered

    def recording(fn, items, **kw):
        batches.append(len(items))
        return real(fn, items, **kw)

    monkeypatch.setattr(llm, "map_ordered", recording)
    conflicts.detect(store, "demo", [becker, merged, depo])
    assert batches and batches[0] == len(fake_llm["calls"]) >= 1
