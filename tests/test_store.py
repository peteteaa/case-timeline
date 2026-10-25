"""Store tests (SPEC §6, §8.1): ids, documents, evidence, permissions, review, filters, sorts,
witnesses, tags, contradictions, alerts, jobs, reprocessing, WAL and multi-process access."""

from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path

import pytest

from pipeline import llm
from pipeline.schema import AGENT, FactQuery, Line, Page, SpanIn
from pipeline.store import Store

CASE = "c1"
TEXT = (
    "On March 3, 2022 the board met in Austin. "
    "John Smith signed the merger agreement. "
    "The complaint alleges Acme hid the losses."
)


def _embed(texts: list[str]) -> list[list[float]]:
    return [llm.hashed_embedding(t) for t in texts]


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "case.db"


@pytest.fixture
def store(db: Path) -> Store:
    s = Store(db, embedder=_embed)
    yield s
    s.close()


def _doc(
    store: Store,
    text: str = TEXT,
    sha: str = "sha-1",
    kind: str = "other",
    case_id: str = CASE,
    pages: list[Page] | None = None,
    lines: list[Line] | None = None,
) -> str:
    doc, created = store.register_document(
        case_id,
        filename=f"{sha}.pdf",
        rel_path=f"dir/{sha}.pdf",
        sha256=sha,
        location="evidence",
        kind=kind,
    )
    assert created
    store.save_parsed_text(doc.doc_id, text, pages or [], lines or [])
    store.set_parse_result(doc.doc_id, "parsed", pages=len(pages or []) or 1)
    return doc.doc_id


def _span(doc_id: str, quote: str, text: str = TEXT) -> SpanIn:
    start = text.index(quote)
    return SpanIn(doc_id=doc_id, char_start=start, char_end=start + len(quote), quote=quote)


# --------------------------------------------------------------------------- basics


def test_wal_mode(store: Store) -> None:
    assert store._q1("PRAGMA journal_mode")[0].lower() == "wal"


def test_two_instances_share_writes(db: Path) -> None:
    a, b = Store(db, embedder=_embed), Store(db, embedder=_embed)
    try:
        fact = a.add_fact(CASE, "Written by A", by="alice")
        assert b.get_fact(fact.fact_id).title == "Written by A"
        b.review_fact(fact.fact_id, "reject", reviewer="bob")
        assert a.get_fact(fact.fact_id).review == "rejected"
        job, _ = b.request_job(CASE, "ui")
        assert a.claim_next_job().job_id == job.job_id
    finally:
        a.close()
        b.close()


def test_short_id_formats(store: Store) -> None:
    doc = _doc(store)
    fact = store.add_fact(CASE, "A fact", by="alice", spans=[_span(doc, "the board met")])
    other = store.add_fact(CASE, "Another", by="alice")
    witness = store.add_witness(CASE, "John Smith", by="alice")
    tag = store.add_tag(CASE, "Fraud", by="alice")
    note = store.add_note(fact.fact_id, author="alice", body="hello")
    contradiction, _ = store.add_contradiction(CASE, [fact.fact_id, other.fact_id], "date", "x")
    alert = store.add_alert(CASE, "contradiction", fact_ids=[fact.fact_id])
    job, _ = store.request_job(CASE, "agent")
    for value, prefix in (
        (doc, "D"),
        (fact.fact_id, "F"),
        (fact.evidence[0].evidence_id, "E"),
        (witness.witness_id, "W"),
        (tag.tag_id, "T"),
        (note.note_id, "N"),
        (contradiction.contradiction_id, "C"),
        (alert.alert_id, "A"),
        (job.job_id, "J"),
    ):
        assert re.fullmatch(rf"{prefix}-\d{{4}}", value), value
    assert (fact.fact_id, other.fact_id) == ("F-0001", "F-0002")


# --------------------------------------------------------------------------- documents


def test_register_document_dedupes_by_sha(store: Store) -> None:
    doc, created = store.register_document(
        CASE, filename="a.pdf", rel_path="a.pdf", sha256="abc", location="evidence"
    )
    assert created and doc.parse_status == "pending" and doc.kind == "other"
    again, created = store.register_document(
        CASE,
        filename="renamed.pdf",
        rel_path="inbox/renamed.pdf",
        sha256="abc",
        location="inbox",
    )
    assert not created and again.doc_id == doc.doc_id and again.filename == "a.pdf"
    other_case, created = store.register_document(
        "c2", filename="a.pdf", rel_path="a.pdf", sha256="abc", location="evidence"
    )
    assert created and other_case.doc_id != doc.doc_id
    with pytest.raises(ValueError):
        store.register_document(
            CASE, filename="b", rel_path="b", sha256="x", location="evidence", kind="memo"
        )
    assert store.get_document_by_sha(CASE, "abc").doc_id == doc.doc_id
    assert [d.doc_id for d in store.documents_to_process(CASE)] == [doc.doc_id]


def test_document_kind_set_by_person_wins(store: Store) -> None:
    doc = _doc(store)
    store.set_document_kind(doc, "pleading", by="alice")
    store.set_parse_result(doc, "parsed", kind="email")
    assert store.get_document(doc).kind == "pleading"
    with pytest.raises(PermissionError):
        store.set_document_kind(doc, "email", by=AGENT)


# --------------------------------------------------------------------------- evidence


def test_span_evidence_verbatim_check(store: Store) -> None:
    doc = _doc(store)
    fact = store.add_fact(CASE, "Board met", by="alice")
    bad = SpanIn(doc_id=doc, char_start=0, char_end=5, quote="Later")
    with pytest.raises(ValueError, match="verbatim"):
        store.add_evidence(fact.fact_id, kind="span", by="alice", span=bad)
    with pytest.raises(ValueError, match="range"):
        store.add_evidence(
            fact.fact_id,
            kind="span",
            by="alice",
            span=SpanIn(doc_id=doc, char_start=0, char_end=10_000, quote="x"),
        )
    # a fact whose span fails is not created at all
    with pytest.raises(ValueError):
        store.add_fact(CASE, "Draft", by=AGENT, spans=[bad])
    assert [f.title for f in store.list_facts(CASE)] == ["Board met"]


def test_span_fills_page_and_lines(store: Store) -> None:
    split = TEXT.index("John")
    pages = [
        Page(number=1, char_start=0, char_end=split),
        Page(number=2, char_start=split, char_end=len(TEXT)),
    ]
    sentence_end = TEXT.index("The complaint")
    lines = [
        Line(page=2, number=7, char_start=split, char_end=split + 10),
        Line(page=2, number=8, char_start=split + 10, char_end=sentence_end),
    ]
    doc = _doc(store, pages=pages, lines=lines)
    fact = store.add_fact(
        CASE, "Signed", by="alice", spans=[_span(doc, "John Smith signed the merger agreement")]
    )
    ev = fact.evidence[0]
    assert (ev.page, ev.line_start, ev.line_end) == (2, 7, 8)
    assert ev.quote == "John Smith signed the merger agreement"
    first = store.add_evidence(
        fact.fact_id, kind="span", by="alice", span=_span(doc, "the board met")
    )
    assert first.page == 1 and first.line_start is None


def test_evidence_kinds_and_dedupe(store: Store) -> None:
    doc = _doc(store)
    fact = store.add_fact(CASE, "F", by="alice")
    d1 = store.add_evidence(fact.fact_id, kind="document", by="alice", doc_id=doc)
    d2 = store.add_evidence(fact.fact_id, kind="document", by="alice", doc_id=doc)
    assert d1.evidence_id == d2.evidence_id
    ext = store.add_evidence(
        fact.fact_id, kind="external", by="alice", reference=" https://example.com/x "
    )
    assert ext.reference == "https://example.com/x" and ext.kind == "external"
    with pytest.raises(ValueError):
        store.add_evidence(fact.fact_id, kind="external", by="alice", reference="  ")
    with pytest.raises(ValueError):
        store.add_evidence(fact.fact_id, kind="document", by="alice", doc_id="D-9999")
    assert len(store.get_fact(fact.fact_id).evidence) == 2


# --------------------------------------------------------------------------- permissions (§6.2)


def test_agent_facts_are_drafts_people_facts_accepted(store: Store) -> None:
    doc = _doc(store)
    draft = store.add_fact(
        CASE, "Draft", by=AGENT, confidence=0.7, spans=[_span(doc, "the board met")]
    )
    assert (draft.origin, draft.review, draft.confidence) == ("ai", "draft", 0.7)
    assert draft.evidence[0].origin == "ai" and draft.evidence[0].review == "draft"
    mine = store.add_fact(
        CASE, "Mine", by="alice", confidence=0.7, spans=[_span(doc, "the board met")]
    )
    assert (mine.origin, mine.review, mine.confidence) == ("user", "accepted", None)
    assert mine.evidence[0].review == "accepted"
    with pytest.raises(ValueError):
        store.add_fact(CASE, "Nobody", by="  ")
    with pytest.raises(ValueError):
        store.add_fact(CASE, "   ", by="alice")


def test_agent_permission_rules(store: Store) -> None:
    tag = store.add_tag(CASE, "Fraud", by="alice")
    with pytest.raises(PermissionError):
        store.add_fact(CASE, "x", by=AGENT, statuses=["key_fact"])
    with pytest.raises(PermissionError):
        store.add_fact(CASE, "x", by=AGENT, tag_ids=[tag.tag_id])
    draft = store.add_fact(CASE, "Draft", by=AGENT)
    with pytest.raises(PermissionError):
        store.update_fact(draft.fact_id, {"statuses": ["disputed"]}, by=AGENT)
    with pytest.raises(PermissionError):
        store.update_fact(draft.fact_id, {"tag_ids": [tag.tag_id]}, by=AGENT)
    # the agent may still edit its own unreviewed draft (stays a draft)
    edited = store.update_fact(draft.fact_id, {"details": "more"}, by=AGENT)
    assert edited.review == "draft" and edited.details == "more"
    store.review_fact(draft.fact_id, "accept", reviewer="alice")
    with pytest.raises(PermissionError):
        store.update_fact(draft.fact_id, {"title": "changed"}, by=AGENT)
    with pytest.raises(PermissionError):
        store.review_fact(draft.fact_id, "reject", reviewer=AGENT)
    with pytest.raises(PermissionError):
        store.add_note(draft.fact_id, author=AGENT, body="hi")
    with pytest.raises(PermissionError):
        store.add_tag(CASE, "New", by=AGENT)
    with pytest.raises(PermissionError):
        store.review_contradiction("C-0001", "confirm", reviewer=AGENT)
    with pytest.raises(PermissionError):
        store.update_witness("W-0001", {"name": "x"}, by=AGENT)
    # but it may attach draft evidence to any fact and suggest witnesses
    doc = _doc(store)
    ev = store.add_evidence(draft.fact_id, kind="document", by=AGENT, doc_id=doc)
    assert ev.review == "draft" and ev.origin == "ai"
    suggested = store.add_witness(CASE, "Jane Roe", by=AGENT, flagged=True)
    assert suggested.confirmed is False and suggested.flagged is False
    with pytest.raises(PermissionError):
        store.review_evidence(ev.evidence_id, "accept", reviewer=AGENT)


# --------------------------------------------------------------------------- review


def test_editing_a_draft_accepts_it(store: Store) -> None:
    doc = _doc(store)
    draft = store.add_fact(CASE, "Board met", by=AGENT, spans=[_span(doc, "the board met")])
    fact = store.update_fact(draft.fact_id, {"title": "Board met in Austin"}, by="alice")
    assert fact.review == "accepted" and fact.title == "Board met in Austin"
    assert fact.updated_by == "alice"
    assert fact.evidence[0].review == "accepted"
    events = store.list_review_events(CASE)
    assert [(e.action, e.reviewer, e.fact_id) for e in events] == [("edit", "alice", draft.fact_id)]
    history = store.fact_history(draft.fact_id)
    assert {"field": "title", "old": "Board met", "new": "Board met in Austin"}.items() <= (
        history[0].items()
    )
    assert any(h["field"] == "review" and h["new"] == "accepted" for h in history)


def test_review_edit_with_reason(store: Store) -> None:
    draft = store.add_fact(CASE, "Board met", by=AGENT)
    store.review_fact(
        draft.fact_id, "edit", reviewer="alice", reason="typo", changes={"title": "Board meets"}
    )
    (event,) = store.list_review_events(CASE)
    assert (event.action, event.reason) == ("edit", "typo")
    with pytest.raises(ValueError):
        store.review_fact(draft.fact_id, "edit", reviewer="alice")


def test_review_accept_and_reject(store: Store) -> None:
    doc = _doc(store)
    a = store.add_fact(CASE, "A", by=AGENT, spans=[_span(doc, "the board met")])
    b = store.add_fact(CASE, "B", by=AGENT)
    accepted = store.review_fact(a.fact_id, "accept", reviewer="alice")
    assert accepted.review == "accepted"
    assert accepted.evidence[0].review == "accepted"
    rejected = store.review_fact(b.fact_id, "reject", reviewer="bob", reason="not a fact")
    assert rejected.review == "rejected"
    events = store.list_review_events(CASE)
    assert [(e.action, e.reason) for e in events] == [("accept", None), ("reject", "not a fact")]
    assert [f.fact_id for f in store.list_facts(CASE)] == [a.fact_id]
    assert store.review_counts(CASE) == {"draft": 0, "accepted": 1, "rejected": 1}
    with pytest.raises(ValueError):
        store.review_fact(a.fact_id, "maybe", reviewer="alice")
    with pytest.raises(KeyError):
        store.review_fact("F-9999", "accept", reviewer="alice")


def test_evidence_review(store: Store) -> None:
    doc = _doc(store)
    fact = store.add_fact(CASE, "Mine", by="alice")
    ev = store.add_evidence(fact.fact_id, kind="span", by=AGENT, span=_span(doc, "the board met"))
    assert ev.review == "draft"
    rejected = store.review_evidence(ev.evidence_id, "reject", reviewer="alice", reason="wrong")
    assert rejected.review == "rejected"
    assert store.get_fact(fact.fact_id).evidence == []
    assert len(store.get_fact(fact.fact_id, rejected_evidence=True).evidence) == 1
    (event,) = store.list_review_events(CASE)
    assert (event.evidence_id, event.action, event.reason) == (ev.evidence_id, "reject", "wrong")
    # a rejected quote no longer matches keyword search
    assert store.fts_fact_ids(CASE, '"board"') == []


# --------------------------------------------------------------------------- list_facts


@pytest.fixture
def world(store: Store) -> dict[str, str]:
    pleading = _doc(store, sha="p", kind="pleading")
    email = _doc(store, sha="e", kind="email")
    fraud = store.add_tag(CASE, "Fraud", by="alice")
    damages = store.add_tag(CASE, "Damages", by="alice")
    smith = store.add_witness(CASE, "John Smith", by="alice")
    jones = store.add_witness(CASE, "Mary Jones", by="alice")
    ids: dict[str, str] = {
        "pleading": pleading,
        "email": email,
        "fraud": fraud.tag_id,
        "damages": damages.tag_id,
        "smith": smith.witness_id,
        "jones": jones.witness_id,
    }
    ids["jan"] = store.add_fact(
        CASE,
        "beta January",
        by="alice",
        date_start=date(2022, 1, 10),
        statuses=["key_fact"],
        tag_ids=[fraud.tag_id],
        witness_ids=[smith.witness_id],
        spans=[_span(pleading, "the board met")],
    ).fact_id
    time.sleep(0.002)
    ids["range"] = store.add_fact(
        CASE,
        "alpha range",
        by="alice",
        date_start=date(2022, 2, 1),
        date_end=date(2022, 4, 30),
        tag_ids=[fraud.tag_id, damages.tag_id],
        doc_ids=[email],
    ).fact_id
    time.sleep(0.002)
    ids["draft"] = store.add_fact(
        CASE,
        "Gamma draft",
        by=AGENT,
        date_start=date(2022, 6, 1),
        confidence=0.4,
        witness_ids=[jones.witness_id],
        spans=[_span(email, "John Smith signed")],
    ).fact_id
    time.sleep(0.002)
    ids["undated"] = store.add_fact(
        CASE, "delta undated", by="alice", statuses=["disputed", "case_impact"]
    ).fact_id
    store.add_note(ids["undated"], author="alice", body="n1")
    store.add_note(ids["undated"], author="alice", body="n2")
    time.sleep(0.002)
    ids["rejected"] = store.add_fact(
        CASE, "epsilon rejected", by=AGENT, date_start=date(2022, 1, 1)
    ).fact_id
    store.review_fact(ids["rejected"], "reject", reviewer="alice")
    store.add_fact("other", "other case", by="alice")
    return ids


def _ids(store: Store, **kw: object) -> list[str]:
    return [f.fact_id for f in store.list_facts(CASE, FactQuery(**kw))]


def test_list_default_is_date_order_undated_last_no_rejected(
    store: Store, world: dict[str, str]
) -> None:
    w = world
    assert _ids(store) == [w["jan"], w["range"], w["draft"], w["undated"]]
    assert _ids(store, descending=True) == [w["draft"], w["range"], w["jan"], w["undated"]]
    assert _ids(store, include_undated=False) == [w["jan"], w["range"], w["draft"]]


def test_list_filters(store: Store, world: dict[str, str]) -> None:
    w = world
    # date range overlaps ranges and drops undated facts
    assert _ids(store, date_from=date(2022, 3, 1), date_to=date(2022, 5, 1)) == [w["range"]]
    assert _ids(store, date_from=date(2022, 5, 1)) == [w["draft"]]
    assert _ids(store, date_to=date(2022, 1, 31)) == [w["jan"]]
    assert _ids(store, statuses=["disputed"]) == [w["undated"]]
    assert _ids(store, statuses=["key_fact", "case_impact"]) == [w["jan"], w["undated"]]
    assert _ids(store, tag_ids=[w["damages"]]) == [w["range"]]
    assert _ids(store, tag_ids=[w["fraud"]]) == [w["jan"], w["range"]]
    assert _ids(store, witness_ids=[w["jones"]]) == [w["draft"]]
    assert _ids(store, witness_ids=[w["smith"], w["jones"]]) == [w["jan"], w["draft"]]
    assert _ids(store, doc_ids=[w["email"]]) == [w["range"], w["draft"]]
    assert _ids(store, doc_kinds=["pleading"]) == [w["jan"]]
    assert _ids(store, doc_kinds=["email", "pleading"]) == [w["jan"], w["range"], w["draft"]]
    assert _ids(store, reviews=["draft"]) == [w["draft"]]
    assert _ids(store, reviews=["rejected"]) == [w["rejected"]]
    assert _ids(store, reviews=[]) == [
        w["rejected"],
        w["jan"],
        w["range"],
        w["draft"],
        w["undated"],
    ]
    assert _ids(store, origins=["ai"]) == [w["draft"]]
    assert _ids(store, origins=["user"]) == [w["jan"], w["range"], w["undated"]]
    assert _ids(store, fact_ids=[w["undated"], w["jan"], w["rejected"]]) == [w["jan"], w["undated"]]
    assert _ids(store, fact_ids=[]) == []
    assert _ids(store, limit=2) == [w["jan"], w["range"]]
    # rejected evidence no longer counts for document filters
    ev = store.get_fact(w["draft"]).evidence[0]
    store.review_evidence(ev.evidence_id, "reject", reviewer="alice")
    assert _ids(store, doc_ids=[w["email"]]) == [w["range"]]


def test_list_sorts(store: Store, world: dict[str, str]) -> None:
    w = world
    assert _ids(store, sort="title") == [w["range"], w["jan"], w["undated"], w["draft"]]
    assert _ids(store, sort="title", descending=True) == [
        w["draft"],
        w["undated"],
        w["jan"],
        w["range"],
    ]
    assert _ids(store, sort="tags", descending=True)[0] == w["range"]
    assert _ids(store, sort="notes", descending=True)[0] == w["undated"]
    assert _ids(store, sort="evidence")[0] == w["undated"]
    assert _ids(store, sort="witnesses", descending=True)[:2] == [w["jan"], w["draft"]]
    assert _ids(store, sort="created") == [w["jan"], w["range"], w["draft"], w["undated"]]
    assert _ids(store, sort="created", descending=True) == [
        w["undated"],
        w["draft"],
        w["range"],
        w["jan"],
    ]
    time.sleep(0.002)
    store.update_fact(w["jan"], {"details": "touched"}, by="alice")
    assert _ids(store, sort="updated", descending=True)[0] == w["jan"]
    assert _ids(store, sort="confidence")[0] == w["draft"]


def test_counts_and_alleged(store: Store, world: dict[str, str]) -> None:
    w = world
    counts = store.fact_counts([w["jan"], w["range"], w["undated"]])
    assert counts[w["jan"]] == {"evidence": 1, "tags": 1, "witnesses": 1, "notes": 0}
    assert counts[w["range"]] == {"evidence": 1, "tags": 2, "witnesses": 0, "notes": 0}
    assert counts[w["undated"]]["notes"] == 2
    # only pleading evidence -> alleged; no evidence -> not alleged
    assert store.alleged_fact_ids(CASE) == {w["jan"]}
    store.add_evidence(w["jan"], kind="external", by="alice", reference="Ex. 4")
    assert store.alleged_fact_ids(CASE) == set()


# --------------------------------------------------------------------------- witnesses & tags


def test_witness_find_merge_delete(store: Store) -> None:
    smith = store.add_witness(CASE, "John  Smith", by="alice", aliases=["J. Smith"])
    assert smith.name == "John Smith"
    guess = store.add_witness(CASE, "Mr. Smith", by=AGENT)
    assert store.find_witness(CASE, "j.  smith").witness_id == smith.witness_id
    assert store.find_witness(CASE, "mr. smith").witness_id == guess.witness_id
    assert store.find_witness(CASE, "Nobody") is None
    a = store.add_fact(CASE, "A", by="alice", witness_ids=[smith.witness_id])
    b = store.add_fact(CASE, "B", by="alice", witness_ids=[guess.witness_id])
    merged = store.merge_witnesses(guess.witness_id, smith.witness_id, by="alice")
    assert set(merged.aliases) == {"J. Smith", "Mr. Smith"}
    assert store.get_witness(guess.witness_id) is None
    assert store.get_fact(b.fact_id).witness_ids == [smith.witness_id]
    assert [f.fact_id for f in store.witness_facts(smith.witness_id)] == [a.fact_id, b.fact_id]
    assert store.find_witness(CASE, "Mr. Smith").witness_id == smith.witness_id
    with pytest.raises(ValueError):
        store.merge_witnesses(smith.witness_id, smith.witness_id, by="alice")
    store.delete_witness(smith.witness_id, by="alice")
    assert store.get_fact(a.fact_id).witness_ids == []
    assert store.list_witnesses(CASE) == []


def test_tags_unique_case_insensitive(store: Store) -> None:
    fraud = store.add_tag(CASE, "Fraud", by="alice", color="#f00")
    again = store.add_tag(CASE, "  fraud ", by="bob")
    assert again.tag_id == fraud.tag_id
    other_case = store.add_tag("c2", "Fraud", by="alice")
    assert other_case.tag_id != fraud.tag_id
    assert [t.name for t in store.list_tags(CASE)] == ["Fraud"]
    store.add_tag(CASE, "Damages", by="alice")
    assert [t.name for t in store.list_tags(CASE)] == ["Damages", "Fraud"]
    with pytest.raises(ValueError):
        store.add_fact(CASE, "x", by="alice", tag_ids=[other_case.tag_id])  # wrong case


# --------------------------------------------------------------------------- contradictions


def test_contradictions_dedupe_and_confirm(store: Store) -> None:
    a = store.add_fact(CASE, "Met on March 3", by=AGENT, date_start=date(2022, 3, 3))
    b = store.add_fact(
        CASE, "Met on March 5", by="alice", date_start=date(2022, 3, 5), statuses=["key_fact"]
    )
    c1, created = store.add_contradiction(CASE, [a.fact_id, b.fact_id], "date", "3 vs 5")
    assert created and c1.status == "open"
    c2, created = store.add_contradiction(CASE, [b.fact_id, a.fact_id], "date", "again")
    assert not created and c2.contradiction_id == c1.contradiction_id
    _, created = store.add_contradiction(CASE, [a.fact_id, b.fact_id], "actor", "who")
    assert created
    with pytest.raises(ValueError):
        store.add_contradiction(CASE, [a.fact_id], "date", "one fact")
    assert store.conflicted_fact_ids(CASE) == {a.fact_id, b.fact_id}
    confirmed = store.review_contradiction(c1.contradiction_id, "confirm", reviewer="alice")
    assert confirmed.status == "confirmed"
    assert store.get_fact(a.fact_id).statuses == ["disputed"]
    assert store.get_fact(a.fact_id).review == "draft"  # confirming a dispute isn't a review
    assert store.list_review_events(CASE) == []
    assert store.get_fact(b.fact_id).statuses == ["key_fact", "disputed"]
    assert [c.contradiction_id for c in store.list_contradictions(CASE, "confirmed")] == [
        c1.contradiction_id
    ]
    assert len(store.list_contradictions(CASE, fact_id=a.fact_id)) == 2


# --------------------------------------------------------------------------- alerts & jobs


def test_alerts_pending_and_delivered(store: Store) -> None:
    a1 = store.add_alert(CASE, "contradiction", fact_ids=["F-0001", "F-0001"], doc_ids=["D-1"])
    a2 = store.add_alert(CASE, "possible_deadline", fact_ids=["F-0002"])
    assert a1.fact_ids == ["F-0001"]
    with pytest.raises(ValueError):
        store.add_alert(CASE, "gossip")
    assert {a.alert_id for a in store.list_alerts(CASE, pending_only=True)} == {
        a1.alert_id,
        a2.alert_id,
    }
    assert store.mark_alerts_delivered([a1.alert_id, a1.alert_id]) == 1
    assert store.mark_alerts_delivered([a1.alert_id]) == 0
    assert store.mark_alerts_delivered([]) == 0
    assert [a.alert_id for a in store.list_alerts(CASE, pending_only=True)] == [a2.alert_id]
    assert store.get_alert(a1.alert_id).delivered_at is not None
    assert len(store.list_alerts(CASE)) == 2


def test_jobs_lifecycle(store: Store) -> None:
    j1, created = store.request_job(CASE, "agent")
    assert created and j1.status == "queued"
    same, created = store.request_job(CASE, "ui")
    assert not created and same.job_id == j1.job_id
    with pytest.raises(ValueError):
        store.request_job(CASE, "cron")
    other, _ = store.request_job("c2", "upload")

    running = store.claim_next_job()
    assert running.job_id == j1.job_id and running.status == "running"
    assert running.started_at is not None
    # a new job for the busy case can be queued but isn't claimed while one runs
    j2, created = store.request_job(CASE, "ui")
    assert created
    assert store.claim_next_job().job_id == other.job_id
    assert store.claim_next_job() is None

    # crash recovery: running jobs go back to the queue
    assert store.requeue_running() == 2
    assert store.get_job(j1.job_id).status == "queued"
    assert store.get_job(j1.job_id).started_at is None

    again = store.claim_next_job()
    assert again.job_id == j1.job_id
    done = store.finish_job(j1.job_id, "done", counts={"docs_seen": 3, "drafts": 2})
    assert done.status == "done" and done.counts == {"docs_seen": 3, "drafts": 2}
    assert done.finished_at is not None
    with pytest.raises(ValueError):
        store.finish_job(j1.job_id, "running")
    failed = store.finish_job(store.claim_next_job().job_id, "failed", error="boom")
    assert (failed.status, failed.error) == ("failed", "boom")
    assert store.latest_job(CASE).job_id == j2.job_id


# --------------------------------------------------------------------------- reprocessing


def test_reset_document_removes_only_unreviewed_ai_drafts(store: Store) -> None:
    doc = _doc(store)
    other_doc = _doc(store, sha="sha-2")
    only_here = store.add_fact(CASE, "draft here", by=AGENT, spans=[_span(doc, "the board met")])
    reviewed = store.add_fact(CASE, "accepted", by=AGENT, spans=[_span(doc, "board met")])
    store.review_fact(reviewed.fact_id, "accept", reviewer="alice")
    mixed = store.add_fact(
        CASE, "two docs", by=AGENT, spans=[_span(doc, "John Smith"), _span(other_doc, "John Smith")]
    )
    person = store.add_fact(CASE, "mine", by="alice", spans=[_span(doc, "Austin")])
    store.add_evidence(person.fact_id, kind="span", by=AGENT, span=_span(doc, "met in Austin"))
    survivor = store.add_fact(CASE, "survivor", by=AGENT, spans=[_span(other_doc, "Acme")])
    c, _ = store.add_contradiction(CASE, [only_here.fact_id, survivor.fact_id], "date", "x")
    store.mark_drafted(doc)

    assert store.reset_document(doc) == 1
    assert store.get_fact(only_here.fact_id) is None
    assert store.get_contradiction(c.contradiction_id) is None
    assert store.get_fact(reviewed.fact_id).review == "accepted"
    assert len(store.get_fact(reviewed.fact_id).evidence) == 1
    assert [e.doc_id for e in store.get_fact(mixed.fact_id).evidence] == [other_doc]
    assert [e.origin for e in store.get_fact(person.fact_id).evidence] == ["user"]
    assert store.get_fact(survivor.fact_id) is not None
    assert store.get_document(doc).parse_status == "pending"
    assert doc in [d.doc_id for d in store.documents_to_process(CASE)]
    assert store.fts_fact_ids(CASE, '"draft"') == []


# --------------------------------------------------------------------------- search indexes


def test_fts_index_and_vectors(store: Store) -> None:
    doc = _doc(store)
    a = store.add_fact(
        CASE,
        "Board meeting",
        by="alice",
        details="held in Austin",
        spans=[_span(doc, "merger agreement")],
    )
    b = store.add_fact(CASE, "Something happened", by="alice")
    store.add_note(b.fact_id, author="alice", body="ask about the losses")
    assert store.fts_fact_ids(CASE, '"austin"') == [a.fact_id]
    assert store.fts_fact_ids(CASE, '"merger"') == [a.fact_id]
    assert store.fts_fact_ids(CASE, '"losses"') == [b.fact_id]
    store.update_fact(a.fact_id, {"details": "held in Dallas"}, by="alice")
    assert store.fts_fact_ids(CASE, '"austin"') == []
    if store.vector_enabled:
        assert store.refresh_vectors(CASE) == 2
        assert store.refresh_vectors(CASE) == 0
        assert store.vector_fact_ids(CASE, "something happened", 5)[0] == b.fact_id


def test_search_survives_zero_vectors(store: Store) -> None:
    store.add_fact(CASE, "Something happened", by="alice")
    store.add_fact(CASE, "!!!", by="alice")  # embeds to a zero vector
    store.refresh_vectors(CASE)
    assert store.vector_fact_ids(CASE, "??!", 5) == []
    assert store.vector_fact_ids(CASE, "   ", 5) == []
    if store.vector_enabled:
        assert store.vector_fact_ids(CASE, "something happened", 5)[0] == "F-0001"
