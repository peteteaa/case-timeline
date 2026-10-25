"""Tests for pipeline.merge (no live model needed)."""

from __future__ import annotations

from datetime import date

import pytest

from pipeline import merge
from pipeline.contracts import Certainty, Event, SourceRef


def ev(
    eid: str,
    desc: str,
    d: date | None = date(2024, 3, 3),
    actors: list[str] | None = None,
    doc: str = "doc1",
    date_text: str = "March 3, 2024",
    end: date | None = None,
) -> Event:
    return Event(
        id=eid,
        case_id="demo",
        date_text=date_text,
        date_start=d,
        date_end=end or d,
        certainty=Certainty.EXACT if d else Certainty.UNKNOWN,
        actors=actors if actors is not None else ["John Smith"],
        description=desc,
        source=SourceRef(doc_id=doc, char_start=0, char_end=5, quote="quote"),
        confidence=0.9,
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
    old = ev("e1", "John Smith signed the lease agreement")
    new = ev("n1", "Smith signed the lease agreement", actors=["john  smith"], doc="doc2")
    to_insert, dups = merge.merge_events([new], [old])
    assert to_insert == []
    assert [d.id for d in dups] == ["n1"]
    assert merge.duplicate_pairs([new], [old]) == [(new, "e1")]


def test_different_date_not_duplicate():
    old = ev("e1", "John Smith signed the lease agreement")
    new = ev("n1", "John Smith signed the lease agreement", d=date(2024, 3, 10))
    to_insert, dups = merge.merge_events([new], [old])
    assert [e.id for e in to_insert] == ["n1"] and dups == []


def test_range_overlap_counts_as_same_date():
    old = ev("e1", "Smith signed the lease", d=date(2024, 3, 1), end=date(2024, 3, 31))
    new = ev("n1", "Smith signed the lease")
    assert merge.duplicate_pairs([new], [old]) == [(new, "e1")]


def test_disjoint_actors_not_duplicate():
    old = ev("e1", "signed the lease agreement", actors=["Alice"])
    new = ev("n1", "signed the lease agreement", actors=["Bob"])
    to_insert, dups = merge.merge_events([new], [old])
    assert len(to_insert) == 1 and dups == []


def test_missing_actors_on_one_side_still_matches():
    old = ev("e1", "signed the lease agreement", actors=[])
    new = ev("n1", "signed the lease agreement", actors=["Bob"])
    assert merge.duplicate_pairs([new], [old]) == [(new, "e1")]


def test_unknown_dates_match_on_identical_text():
    old = ev("e1", "meeting at the office", d=None, date_text="the next morning")
    new = ev("n1", "meeting at the office", d=None, date_text="The next  morning")
    other = ev("n2", "meeting at the office", d=None, date_text="later that week")
    to_insert, dups = merge.merge_events([new, other], [old])
    assert [d.id for d in dups] == ["n1"]
    assert [e.id for e in to_insert] == ["n2"]


def test_embedding_used_only_for_candidate_pairs(monkeypatch, no_embed):
    vecs = {"Smith paid rent": [1.0, 0.0], "rent payment made by Smith": [0.99, 0.05]}
    calls: list[list[str]] = []

    def fake(texts):
        calls.append(list(texts))
        return [vecs.get(t, [0.0, 1.0]) for t in texts]

    monkeypatch.setattr(merge, "_embed", fake)
    old = ev("e1", "Smith paid rent")
    far = ev("e2", "Smith paid rent", d=date(2025, 1, 1))  # wrong date: never embedded
    new = ev("n1", "rent payment made by Smith")
    pairs = merge.duplicate_pairs([new], [old, far])
    assert pairs == [(new, "e1")]
    assert len(calls) == 1
    assert sorted(calls[0]) == sorted(["rent payment made by Smith", "Smith paid rent"])


def test_low_cosine_not_duplicate(monkeypatch):
    monkeypatch.setattr(merge, "_embed", lambda texts: [[1.0, 0.0], [0.0, 1.0]])
    old = ev("e1", "Smith paid rent")
    new = ev("n1", "Smith slipped on the stairs")
    assert merge.duplicate_pairs([new], [old]) == []


def test_no_embed_call_when_fuzzy_already_matches(no_embed):
    old = ev("e1", "Smith paid rent")
    new = ev("n1", "Smith paid rent")
    merge.merge_events([new], [old])
    assert no_embed == []


def test_in_batch_duplicates_link_to_canonical():
    a = ev("n1", "Smith paid rent", doc="d1")
    b = ev("n2", "Smith paid rent", doc="d2")
    c = ev("n3", "Smith paid the rent", doc="d3")
    to_insert, pairs = merge.merge_events_linked([a, b, c], [])
    assert [e.id for e in to_insert] == ["n1"]
    assert [(e.id, t) for e, t in pairs] == [("n2", "n1"), ("n3", "n1")]


def test_already_stored_events_are_not_reinserted():
    old = ev("e1", "Smith paid rent")
    to_insert, dups = merge.merge_events([old], [old])
    assert to_insert == [] and dups == []


def test_names_match_handles_titles_and_partial_names():
    assert merge.names_match("Mr. Ellison", "Mark Ellison")
    assert merge.names_match("Ms. Rivera", "Dana Rivera")
    assert merge.names_match("Officer Ortiz", "Officer Jamie Ortiz")
    assert merge.names_match("Dana", "Dana Rivera")
    assert not merge.names_match("John Smith", "Mary Smith")
    assert not merge.names_match("Alice", "Bob")
    assert merge.is_organization("Halvorsen Freight Co.") and not merge.is_organization("Dana")


def test_title_variants_count_as_same_actor():
    old = ev("e1", "Rivera submitted a formal complaint", actors=["Ms. Rivera"])
    new = ev("n1", "Rivera submitted a formal complaint", actors=["Dana Rivera"], doc="doc2")
    assert merge.duplicate_pairs([new], [old]) == [(new, "e1")]


def test_subset_description_not_enough_for_duplicate():
    # token_set_ratio is 100 for a subset; the generic one must not swallow the detailed one.
    old = ev("e1", "Meeting held")
    new = ev("n1", "Meeting held with the board to approve the merger and the new budget")
    assert merge.description_similarity(old, new) == 100
    assert not merge.fuzzy_duplicate(old, new)
    assert merge.duplicate_pairs([new], [old]) == []


def test_cross_document_people_must_agree():
    # Linked duplicates are hidden from contradiction checks, so a cross-document merge must
    # not hide a disagreement about who was there.
    old = ev("e1", "Meeting in Smith's office", actors=["John Smith", "Ann Lee"])
    new = ev("n1", "Meeting in Smith's office", actors=["John Smith", "Bo Chan"], doc="doc2")
    assert merge.duplicate_pairs([new], [old]) == []
    same_doc = ev("n2", "Meeting in Smith's office", actors=["John Smith", "Bo Chan"])
    assert merge.duplicate_pairs([same_doc], [old]) == [(same_doc, "e1")]
    org_only_extra = ev(
        "n3", "Meeting in Smith's office", actors=["Smith", "Lee", "Acme Corp."], doc="doc3"
    )
    assert merge.duplicate_pairs([org_only_extra], [old]) == [(org_only_extra, "e1")]


def test_conflicting_dates_never_merge_across_documents():
    old = ev("e1", "Meeting in Smith's office", d=date(2024, 3, 11))
    new = ev("n1", "Meeting in Smith's office", d=date(2024, 3, 12), doc="doc2")
    assert merge.duplicate_pairs([new], [old]) == []


# --- cross-document dedupe with clock times ------------------------------------------------


def test_clock_of_reads_common_time_formats():
    def clock(text):
        return merge.clock_of(ev("x", "d", date_text=text))

    assert clock("0942:15") == (9 * 3600 + 42 * 60 + 15, True, "")
    assert clock("about 1405:07") == (14 * 3600 + 5 * 60 + 7, True, "")
    assert clock("3:40 p.m.") == (15 * 3600 + 40 * 60, False, "")
    assert clock("12:05 a.m.") == (5 * 60, False, "")
    assert clock("June 5, 2021, 0940 central daylight time") == (9 * 3600 + 40 * 60, False, "cdt")
    assert clock("1405 UTC") == (14 * 3600 + 5 * 60, False, "utc")
    assert clock("at 0940") == (9 * 3600 + 40 * 60, False, "")
    assert clock("07/28/10 2 p.m.") == (14 * 3600, False, "")
    assert clock("07/28/10 0915 HOURS") == (9 * 3600 + 15 * 60, False, "")
    assert clock("March 3, 2024") is None
    assert clock("in 2021") is None


def test_same_moment_in_two_documents_links_despite_role_names():
    # A transcript and a log describe one instruction with different role names for the
    # same participants and timestamps a second apart.
    log = ev(
        "e1",
        "Dispatch instructed unit 12 to hold at the north gate.",
        actors=["County dispatcher", "Unit 12"],
        doc="log",
        date_text="1415:22",
    )
    transcript = ev(
        "n1",
        "The dispatcher instructed the unit to hold at the north gate.",
        actors=["Central Dispatch"],
        doc="transcript",
        date_text="about 1415:23",
    )
    assert merge.clocks_agree(log, transcript)
    assert merge.duplicate_pairs([transcript], [log]) == [(transcript, "e1")]


def test_timed_pairs_use_embeddings_at_a_lower_bar(monkeypatch):
    monkeypatch.setattr(merge, "_embed", lambda texts: [[1.0, 0.0], [0.8, 0.6]])  # cos 0.8
    a = ev("e1", "Officer Diaz radioed that the car stopped", doc="d1", date_text="2210:05")
    b = ev("n1", "Patrol reported the vehicle came to a halt", doc="d2", date_text="2210:06")
    assert merge.duplicate_pairs([b], [a]) == [(b, "e1")]
    untimed = b.model_copy(update={"date_text": "March 3, 2024"})
    assert merge.duplicate_pairs([untimed], [a]) == []  # 0.8 < the untimed bar


def test_different_clock_times_never_merge_even_in_one_document():
    a = ev("e1", 'The alarm announced "low pressure" five times.', date_text="1816:35")
    b = ev("n1", 'The alarm announced "low pressure" two times.', date_text="1816:44")
    assert merge.clocks_agree(a, b) is False
    assert merge.duplicate_pairs([b], [a]) == []
    minute = ev("n2", "Smith signed the lease agreement", date_text="3:40 PM", doc="d2")
    later = ev("e2", "Smith signed the lease agreement", date_text="4:30 PM")
    assert merge.duplicate_pairs([minute], [later]) == []  # 50 minutes apart: a conflict
    close = later.model_copy(update={"id": "n3", "date_text": "about 4:31 PM"})
    close = close.model_copy(update={"source": close.source.model_copy(update={"doc_id": "d3"})})
    assert merge.duplicate_pairs([close], [later]) == [(close, "e2")]


def test_utc_and_local_times_are_not_compared():
    a = ev("e1", "Smith signed the lease agreement", date_text="2316 UTC")
    b = ev("n1", "Smith signed the lease agreement", date_text="1816 CDT", doc="d2")
    assert merge.clocks_agree(a, b) is None
    assert merge.duplicate_pairs([b], [a]) == [(b, "e1")]  # still the same day and text


def test_cross_document_dates_must_be_identical_not_just_overlapping():
    month = ev("e1", "Smith signed the lease", d=date(2024, 3, 1), end=date(2024, 3, 31))
    day = ev("n1", "Smith signed the lease", doc="doc2")
    assert merge.duplicate_pairs([day], [month]) == []


def test_named_people_conflict_but_roles_do_not():
    assert merge.is_named_person("Ana Lopez") and merge.is_named_person("Mr. Keech")
    for role in ("the pilot", "North Tower", "Unit 12", "NTSB", "Acme Corp.", "Gate camera"):
        assert not merge.is_named_person(role), role
    a = ev("e1", "Meeting held", actors=["Ana Lopez", "front desk"])
    b = ev("n1", "Meeting held", actors=["Ben Ortiz"], doc="d2")
    c = ev("n2", "Meeting held", actors=["the receptionist", "Ana Lopez"], doc="d3")
    assert merge.people_conflict(a, b)
    assert not merge.people_conflict(a, c)
    assert merge.duplicate_pairs([b], [a]) == []
    assert merge.duplicate_pairs([c], [a]) == [(c, "e1")]


def test_relaxed_bars_apply_only_across_documents(monkeypatch):
    # A transcript lists distinct utterances under one timestamp; they must not collapse.
    monkeypatch.setattr(merge, "_embed", lambda texts: [[1.0, 0.0], [0.8, 0.6]])  # cos 0.8
    a = ev("e1", "Dispatch asked the unit which gate it wanted.", date_text="about 1415:22")
    b = ev("n1", "Dispatch told the unit to expect the north gate.", date_text="about 1415:22")
    assert merge.clocks_agree(a, b) and not merge.timed_pair(a, b)
    assert merge.duplicate_pairs([b], [a]) == []
    other_doc = b.model_copy(update={"source": b.source.model_copy(update={"doc_id": "d2"})})
    assert merge.timed_pair(a, other_doc)
    assert merge.duplicate_pairs([other_doc], [a]) == [(other_doc, "e1")]
