"""Tests for pipeline.digest."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from pipeline.digest import daily_digest
from pipeline.schema import AGENT, SpanIn
from tests.pipeline_helpers import add_doc, new_store

TEXT = "Smith signed the lease on March 3, 2024. Jones paid rent on March 9, 2024."


@pytest.fixture
def store(tmp_path):
    s = new_store(tmp_path)
    yield s
    s.close()


def _draft(store, doc_id, title, quote):
    start = TEXT.index(quote)
    return store.add_fact(
        "demo",
        title,
        by=AGENT,
        date_start=date(2024, 3, 3),
        date_certainty="exact",
        spans=[SpanIn(doc_id=doc_id, char_start=start, char_end=start + len(quote), quote=quote)],
    ).fact_id


def test_digest_summarizes_the_day_with_fact_ids(store):
    d = add_doc(store, TEXT, filename="lease.txt")
    f1 = _draft(store, d, "Smith signs lease", "Smith signed the lease")
    f2 = _draft(store, d, "Jones pays rent", "Jones paid rent")
    f3 = _draft(store, d, "Lease signed in March", "on March 3, 2024")
    f4 = _draft(store, d, "Rent paid in March", "on March 9, 2024")
    mine = store.add_fact("demo", "Call opposing counsel", by="alice").fact_id
    store.review_fact(f1, "accept", reviewer="alice")
    store.review_fact(f2, "reject", reviewer="alice", reason="duplicate")
    store.review_fact(f3, "edit", reviewer="bob", changes={"title": "Lease signed"})
    c, _ = store.add_contradiction("demo", [f1, f4], "date", "x")
    store.add_alert("demo", "possible_deadline", fact_ids=[f4], doc_ids=[d])
    store.add_alert("demo", "contradiction", fact_ids=c.fact_ids)

    out = daily_digest(store, "demo")
    assert out["case_id"] == "demo"
    since = datetime.fromisoformat(out["since"])
    assert timedelta(hours=23) < datetime.now(UTC) - since < timedelta(hours=25)
    counts = out["counts"]
    assert counts["new_drafts"] == 1  # f4: f1 and f3 were accepted, f2 rejected
    assert counts["new_user_facts"] == 1
    assert (counts["accepted"], counts["edited"], counts["rejected"]) == (1, 1, 1)
    assert counts["contradictions"] == 1 and counts["alerts"] == 2
    assert counts["documents"] == 1 and counts["drafts_pending"] == 1
    text = out["text"]
    for fid in (f1, f2, f3, f4, mine, c.contradiction_id, d):
        assert fid in text
    assert "awaiting review" in text and "possible deadline" in text
    assert "lease.txt" in text
    assert set(out["fact_ids"]) == {f1, f2, f3, f4, mine}


def test_digest_with_no_activity(store):
    later = datetime.now(UTC) + timedelta(days=3)
    out = daily_digest(store, "demo", now=later)
    assert "No changes" in out["text"]
    assert out["fact_ids"] == []
    assert all(v == 0 for v in out["counts"].values())


def test_digest_window_excludes_old_activity(store):
    d = add_doc(store, TEXT)
    _draft(store, d, "Smith signs lease", "Smith signed the lease")
    out = daily_digest(store, "demo", now=datetime.now(UTC) + timedelta(days=2))
    assert out["counts"]["new_drafts"] == 0
    assert out["counts"]["drafts_pending"] == 1  # still waiting, though not new today
    assert "1 draft in total awaiting review" in out["text"]


def test_digest_accepts_naive_now(store):
    out = daily_digest(store, "demo", now=datetime(2026, 1, 1, 12, 0))
    assert out["since"].startswith("2025-12-31T12:00")
