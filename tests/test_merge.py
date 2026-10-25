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
