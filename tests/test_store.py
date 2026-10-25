"""Store tests: schema, upserts, review-status preservation, hybrid search, change feed."""

from __future__ import annotations

import threading
from datetime import UTC, date, datetime, timedelta

import pytest

from pipeline.contracts import (
    Alert,
    AlertType,
    Certainty,
    ConflictField,
    Contradiction,
    ContradictionStatus,
    Document,
    Event,
    EventStatus,
    JobState,
    Page,
    SourceRef,
)
from pipeline.llm import hashed_embedding
from pipeline.store import Store

CASE = "demo"


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [hashed_embedding(t) for t in texts]


def make_doc(doc_id: str = "d1", case_id: str = CASE) -> Document:
    return Document(
        id=doc_id,
        case_id=case_id,
        path=f"/evidence/{doc_id}.txt",
        filename=f"{doc_id}.txt",
        mime="text/plain",
        sha256=doc_id * 8,
    )


def make_event(
    eid: str,
    description: str,
    *,
    actors: list[str] | None = None,
    quote: str | None = None,
    day: date | None = None,
    case_id: str = CASE,
    status: EventStatus = EventStatus.PROPOSED,
) -> Event:
    quote = quote or description
    return Event(
        id=eid,
        case_id=case_id,
        date_text=day.isoformat() if day else "unknown",
        date_start=day,
        date_end=day,
        certainty=Certainty.EXACT if day else Certainty.UNKNOWN,
        actors=actors or [],
        description=description,
        source=SourceRef(doc_id="d1", page=1, char_start=0, char_end=len(quote), quote=quote),
        confidence=0.9,
        status=status,
    )


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "case.db", embedder=fake_embed)
    yield s
    s.close()


def test_init_is_idempotent_and_wal(tmp_path):
    s = Store(tmp_path / "x.db", embedder=fake_embed)
    s.init()
    mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    s.close()
    s2 = Store(tmp_path / "x.db", embedder=fake_embed)  # reopen existing db
    assert s2.list_events(CASE) == []
    s2.close()


def test_documents_and_parsed_text(store):
    doc = make_doc()
    store.upsert_document(doc)
    store.upsert_document(doc)  # idempotent
    assert [d.id for d in store.list_documents(CASE)] == ["d1"]
    assert store.get_document("d1").ingested_at is not None
    assert store.processed_doc_ids(CASE) == set()
    assert store.get_parsed_text("d1") is None

    store.add_parsed_text("d1", "hello world", [Page(number=1, char_start=0, char_end=11)])
    parsed = store.get_parsed_text("d1")
    assert parsed.text == "hello world"
    assert parsed.pages[0].char_end == 11
    assert parsed.doc.filename == "d1.txt"
    assert store.processed_doc_ids(CASE) == {"d1"}

    store.upsert_document(make_doc("d2"))
    store.mark_processed("d2")
    assert store.processed_doc_ids(CASE) == {"d1", "d2"}
    assert store.processed_doc_ids("other") == set()


def test_events_roundtrip_and_order(store):
    events = [
        make_event("e2", "Second", day=date(2024, 3, 5), actors=["Bob"]),
        make_event("e0", "Undated thing"),
        make_event("e1", "First", day=date(2024, 3, 1), actors=["Alice", "Bob"]),
    ]
    assert store.add_events(events) == 3
    listed = store.list_events(CASE)
    assert [e.id for e in listed] == ["e1", "e2", "e0"]
    assert store.get_event("e1") == events[2]
    assert store.get_event("missing") is None
    assert store.list_events("other") == []


def test_human_status_preserved_on_reupsert(store):
    ev = make_event("e1", "Meeting held", day=date(2024, 1, 2))
    store.add_events([ev])
    assert store.set_event_status("e1", EventStatus.ACCEPTED)
    assert not store.set_event_status("nope", "accepted")
    store.add_events([ev.model_copy(update={"description": "Meeting was held"})])
    got = store.get_event("e1")
    assert got.status == EventStatus.ACCEPTED
    assert got.description == "Meeting was held"
    assert [e.id for e in store.list_events(CASE, status="accepted")] == ["e1"]
    assert store.list_events(CASE, status=EventStatus.PROPOSED) == []


def test_duplicates_hidden_by_default(store):
    store.add_events(
        [
            make_event("a", "Contract signed", day=date(2024, 2, 1)),
            make_event("b", "Contract was signed", day=date(2024, 2, 1)),
        ]
    )
    store.link_duplicate("b", "a")
    assert [e.id for e in store.list_events(CASE)] == ["a"]
    assert {e.id for e in store.list_events(CASE, include_duplicates=True)} == {"a", "b"}
    assert store.duplicates_of("a") == ["b"]
    with pytest.raises(ValueError):
        store.link_duplicate("a", "a")


def test_fts_sync_and_search(store):
    store.add_events(
        [
            make_event("e1", "Alice emailed the landlord about the leak", actors=["Alice Smith"]),
            make_event("e2", "Bob signed the lease agreement", actors=["Bob Jones"]),
            make_event("e3", "Inspection of the roof", quote="the roof was inspected by Carol"),
        ]
    )
    assert [e.id for e in store.search_events(CASE, "leak", k=5)][0] == "e1"
    assert store.search_events(CASE, "Jones", k=5)[0].id == "e2"  # actors indexed
    assert store.search_events(CASE, "Carol", k=5)[0].id == "e3"  # quote indexed
    assert store.search_events("other", "leak") == []
    assert store.search_events(CASE, "   ") == []
    # Punctuation / FTS syntax in user queries must not raise.
    assert store.search_events(CASE, 'leak" OR (NEAR', k=5)[0].id == "e1"

    # Updating the description keeps FTS in sync.
    old = store.get_event("e1")
    new_src = old.source.model_copy(update={"quote": "Alice phoned about mould"})
    ev = old.model_copy(update={"description": "Alice phoned about mould", "source": new_src})
    store.add_events([ev])
    assert all(e.id != "e1" for e in store.search_events(CASE, "leak landlord", k=5))
    assert store.search_events(CASE, "mould", k=5)[0].id == "e1"


def test_vector_search_contributes(store):
    if not store.vector_enabled:
        pytest.skip("sqlite-vec not loadable here")
    store.add_events([make_event("e1", "Payment of rent overdue"), make_event("e2", "Fire alarm")])
    rows = store._conn.execute("SELECT count(*) FROM event_vec").fetchone()[0]
    assert rows == 2
    # Prefix-free stemmed FTS would match "rent"; vector match on identical tokens ranks e1 top.
    assert store.search_events(CASE, "rent overdue payment", k=1)[0].id == "e1"


def test_search_fts_only_when_vec_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(Store, "_load_vec", lambda self: False)
    s = Store(tmp_path / "novec.db", embedder=fake_embed)
    assert not s.vector_enabled
    s.add_events([make_event("e1", "Alice called Bob")])
    assert s.search_events(CASE, "Bob")[0].id == "e1"
    s.close()


def test_embedder_failure_degrades(tmp_path):
    def broken(texts):
        raise RuntimeError("no model")

    s = Store(tmp_path / "b.db", embedder=broken)
    s.add_events([make_event("e1", "Alice called Bob")])
    assert s.search_events(CASE, "Alice")[0].id == "e1"
    s.close()


def test_unchanged_events_not_reembedded(tmp_path):
    calls: list[int] = []

    def counting(texts):
        calls.append(len(texts))
        return fake_embed(texts)

    s = Store(tmp_path / "c.db", embedder=counting)
    if not s.vector_enabled:
        pytest.skip("sqlite-vec not loadable here")
    ev = make_event("e1", "Alice called Bob")
    s.add_events([ev])
    s.add_events([ev])
    assert calls == [1]
    s.close()


def test_contradictions_and_alerts(store):
    c = Contradiction(
        id="c1", case_id=CASE, event_ids=["e1", "e2"], field=ConflictField.DATE, explanation="x"
    )
    store.add_contradictions([c])
    assert store.set_contradiction_status("c1", ContradictionStatus.DISMISSED)
    store.add_contradictions([c.model_copy(update={"explanation": "y"})])  # re-detected
    [got] = store.list_contradictions(CASE)
    assert got.status == ContradictionStatus.DISMISSED and got.explanation == "y"
    assert store.list_contradictions(CASE, status="open") == []

    a = Alert(
        id="a1",
        case_id=CASE,
        type=AlertType.NEW_CONTRADICTION,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        dashboard_url="http://127.0.0.1:8090/cases/demo/alerts/a1",
        ref_ids=["c1"],
    )
    assert store.add_alerts([a]) == 1
    assert store.add_alerts([a]) == 0
    assert store.list_alerts(CASE) == [a]


def test_jobs(store):
    job = store.create_job(CASE, docs_total=3)
    assert job.state == JobState.QUEUED
    now = datetime.now(UTC)
    job = store.update_job(job.id, state=JobState.RUNNING, started_at=now)
    job = store.update_job(job, docs_done=3, events_added=7, state=JobState.DONE)
    got = store.get_job(job.id)
    assert got == job
    assert got.started_at == now and got.events_added == 7
    assert store.latest_job(CASE).id == job.id
    assert store.get_job("missing") is None
    with pytest.raises(ValueError):
        store.update_job(job.id, bogus=1)
    with pytest.raises(KeyError):
        store.update_job("missing", docs_done=1)


def test_changes_since_and_cursor(store):
    t0 = datetime.now(UTC)
    store.add_events([make_event("e1", "One")])
    full = store.changes_since(CASE, None)
    assert [e.id for e in full.new_events] == ["e1"]
    assert [e.id for e in store.changes_since(CASE, t0).new_events] == ["e1"]
    assert store.changes_since(CASE, datetime.now(UTC) + timedelta(seconds=1)).new_events == []
    # Naive datetimes are treated as UTC.
    assert len(store.changes_since(CASE, t0.replace(tzinfo=None)).new_events) == 1

    assert store.get_cursor("mcp", CASE) is None
    first = store.get_changes(CASE, "mcp")
    assert [e.id for e in first.new_events] == ["e1"] and first.since is None
    assert store.get_changes(CASE, "mcp").new_events == []

    store.add_events([make_event("e1", "One (re-extracted)"), make_event("e2", "Two")])
    store.add_contradictions(
        [
            Contradiction(
                id="c1", case_id=CASE, event_ids=["e1", "e2"], field="fact", explanation="z"
            )
        ]
    )
    nxt = store.get_changes(CASE, "mcp")
    assert [e.id for e in nxt.new_events] == ["e2"]  # updates are not "new"
    assert [c.id for c in nxt.new_contradictions] == ["c1"]
    # Independent consumers have independent cursors.
    assert len(store.get_changes(CASE, "ui").new_events) == 2
    stamp = store.advance_cursor("ui", CASE, t0)
    assert store.get_cursor("ui", CASE) == stamp
    store.set_cursor(case_id=CASE, consumer="ui", ts=t0)
    assert store.get_cursor(case_id=CASE, consumer="ui") == t0


def test_thread_safety(store):
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(10):
                store.add_events([make_event(f"t{n}-{i}", f"thread {n} event {i}")])
                store.list_events(CASE)
                store.search_events(CASE, f"thread {n}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(store.list_events(CASE)) == 40
