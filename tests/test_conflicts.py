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
    assert len(fake_llm["calls"]) == 1  # the restatement was not even sent to the model
    # Also across runs: a dismissed disagreement isn't reopened through the other fact.
    store.review_contradiction(opened[0].contradiction_id, "dismiss", reviewer="alice")
    assert conflicts.detect(store, "demo", [becker, merged, depo]) == []
    assert len(fake_llm["calls"]) == 1


def test_detect_same_subject_false_opens_nothing(meeting_case, fake_llm, monkeypatch):
    store, becker, _, _ = meeting_case

    def chat_json(system, user, schema, **kw):
        fake_llm["calls"].append((system, user))
        return schema(is_contradiction=True, same_subject=False, field="date")

    monkeypatch.setattr(llm, "chat_json", chat_json)
    assert conflicts.detect(store, "demo", [becker]) == []
    assert fake_llm["calls"]
