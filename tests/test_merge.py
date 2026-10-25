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


# --------------------------------------------------------------------------- loose names, gates


NAMES = {
    "W-1": ["Mark Ellison"],
    "W-2": ["Dana Rivera"],
    "W-3": ["Priya Nand"],
    "W-4": ["Tom Becker"],
    "W-5": ["Ms. Rivera"],  # a second record for Rivera (e.g. an ambiguous suggestion)
    "W-9": ["Halvorsen Freight Co."],
}


def test_names_match_handles_titles_and_partial_names():
    assert merge.names_match("Mr. Ellison", "Mark Ellison")
    assert merge.names_match("Ms. Rivera", "Dana Rivera")
    assert merge.names_match("Officer Ortiz", "Officer Jamie Ortiz")
    assert merge.names_match("Dana", "Dana Rivera")
    assert not merge.names_match("John Smith", "Mary Smith")
    assert not merge.names_match("Alice", "Bob")
    assert merge.is_organization("Halvorsen Freight Co.") and not merge.is_organization("Dana")


def test_subset_title_is_not_a_fuzzy_duplicate():
    # token_set_ratio is 100 for a subset; the generic title must not swallow the detailed one.
    short, long = "Meeting held", "Meeting held with the board to approve the merger and budget"
    assert merge.similarity(short, long) == 100
    assert not merge.fuzzy_duplicate(short, None, long, None)
    old = make_fact("F-0001", short)
    assert merge.plan([draft("n1", long)], [old]) == {"n1": None}


def test_cross_document_different_casts_never_merge():
    title = "Meeting in Ellison's office"
    old = make_fact("F-0001", title, witnesses=["W-1", "W-2"], docs=["D-0001"])
    other_cast = draft("n1", title, doc="D-0002")
    ids = {"n1": ["W-1", "W-4"]}  # each side names someone the other doesn't
    assert merge.plan([other_cast], [old], witness_ids=ids, witness_names=NAMES) == {"n1": None}
    # Within one document sharing one person is enough.
    same_doc = draft("n2", title, doc="D-0001")
    out = merge.plan([same_doc], [old], witness_ids={"n2": ["W-1", "W-4"]}, witness_names=NAMES)
    assert out == {"n2": "F-0001"}


def test_one_sided_extra_person_merges_unless_someone_is_excluded():
    title = "Meeting in Ellison's office"
    old = make_fact("F-0001", title, witnesses=["W-1", "W-2"], docs=["D-0001"])
    ids = {"n1": ["W-1", "W-2", "W-3"]}
    silent = draft("n1", title, doc="D-0002", quote="Nand attended the meeting.")
    assert merge.plan([silent], [old], witness_ids=ids) == {"n1": "F-0001"}
    excluding = make_fact(
        "F-0002",
        title,
        witnesses=["W-1", "W-2"],
        docs=["D-0003"],
        quotes=["Only he and Ms. Rivera were present in the office."],
    )
    assert merge.plan([silent], [excluding], witness_ids=ids) == {"n1": None}


def test_loose_names_and_organizations_with_witness_names():
    title = "Rivera placed on unpaid administrative leave"
    old = make_fact("F-0001", title, witnesses=["W-2"], docs=["D-0001"])
    # Another record for the same person ("Ms. Rivera") plus an organization.
    new = draft("n1", title, doc="D-0002")
    ids = {"n1": ["W-5", "W-9"]}
    assert merge.plan([new], [old], witness_ids=ids, witness_names=NAMES) == {"n1": "F-0001"}
    # Without names the ids are compared exactly, and disjoint ids block the merge.
    assert merge.plan([new], [old], witness_ids=ids) == {"n1": None}


def test_same_day_paraphrase_merges_on_topic_words():
    old = make_fact(
        "F-0001",
        "Rivera files a formal complaint with HR alleging safety retaliation",
        date(2024, 3, 18),
        witnesses=["W-2"],
        docs=["D-0001"],
    )
    new = draft(
        "n1",
        "Ms. Rivera submits formal complaint alleging retaliation",
        date(2024, 3, 18),
        doc="D-0002",
    )
    other = draft("n2", "Police report approved and case closed", date(2024, 3, 18), doc="D-0002")
    ids = {"n1": ["W-2"], "n2": []}
    out = merge.plan([new, other], [old], witness_ids=ids, witness_names=NAMES)
    assert out == {"n1": "F-0001", "n2": None}


def test_topic_rule_needs_a_precise_date():
    old = make_fact(
        "F-0001",
        "Rivera files formal complaint alleging retaliation",
        date(2024, 3, 1),
        end=date(2024, 3, 31),
        certainty="approximate",
    )
    new = draft("n1", "Rivera submits a formal retaliation complaint to HR", date(2024, 3, 18))
    assert merge.plan([new], [old]) == {"n1": None}


def test_topic_tokens_ignore_dates_names_and_stopwords():
    tokens = merge.topic_tokens("Ann Lee signed the lease on March 3, 2024", ["Ann Lee"])
    assert tokens == {"sign", "lease"}
    assert merge.proper_words("Smith met Officer Ortiz in Lakeview") == {
        "officer",
        "ortiz",
        "lakeview",
    }
