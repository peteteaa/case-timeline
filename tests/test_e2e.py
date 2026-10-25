"""End-to-end: run the full pipeline on the demo fixtures with a deterministic fake LLM."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline import llm, runner
from pipeline.config import get_settings
from pipeline.contracts import EventStatus, JobState
from pipeline.store import Store

FIXTURES = Path(__file__).parent / "fixtures" / "case_demo"

_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
)
_DATE = re.compile(rf"{_MONTH} \d{{1,2}}, \d{{4}}|\b\w+ days after the meeting", re.IGNORECASE)
_FRAGMENT = re.compile(r"[^\n.!?]+[.!?]?")
_PEOPLE = {
    "Rivera": "Dana Rivera",
    "Dana": "Dana Rivera",
    "Ellison": "Mark Ellison",
    "Mark": "Mark Ellison",
    "Becker": "Tom Becker",
    "Nand": "Priya Nand",
    "Ortiz": "Jamie Ortiz",
}


def _fake_extract(user: str, schema):
    """Every sentence fragment with a date becomes an event whose quote is copied verbatim."""
    excerpt = user.split("<excerpt>\n", 1)[1].rsplit("\n</excerpt>", 1)[0]
    events = []
    for frag in _FRAGMENT.findall(excerpt):
        quote = frag.strip()
        m = _DATE.search(quote)
        if not m or len(quote) < 12:
            continue
        actors = []
        for key, name in _PEOPLE.items():
            if re.search(rf"\b{key}\b", quote) and name not in actors:
                actors.append(name)
        lowered = quote.lower()
        if "meeting" in lowered and "ellison" in lowered and "days after" not in lowered:
            description = "Meeting in Mark Ellison's office with Dana Rivera"
        else:
            description = quote
        events.append(
            {
                "quote": quote,
                "date_text": m.group(0),
                "actors": actors,
                "description": description,
                "deadline_relevant": "appeal" in lowered or "due" in lowered,
                "confidence": 0.8,
            }
        )
    return schema.model_validate({"events": events})


def fake_chat_json(system, user, schema, *, temperature=0.0, **kw):
    if schema.__name__ == "ConflictVerdict":
        m = re.search(r"Suspected conflict type: (\w+)", user)
        return schema(
            is_contradiction=True, field=m.group(1) if m else "fact", explanation="fake verdict"
        )
    return _fake_extract(user, schema)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated settings (temp data dir, fixture evidence) and a fake, offline LLM."""
    monkeypatch.setenv("CT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CT_EVIDENCE_DIR", str(FIXTURES))
    monkeypatch.setenv("CT_CASE_ID", "demo")
    monkeypatch.setenv("CT_DISABLE_DOCLING", "1")
    monkeypatch.delenv("CT_DB_PATH", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(llm, "embed", lambda texts: [llm.hashed_embedding(t) for t in texts])
    settings = get_settings()
    yield settings
    get_settings.cache_clear()


def _evidence_snapshot() -> dict[str, tuple[int, int]]:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in FIXTURES.iterdir()}


def test_full_pipeline(env):
    before = _evidence_snapshot()
    job = runner.scan_case_sync("demo")
    assert job.state == JobState.DONE, job.error
    assert job.error is None
    assert job.docs_total == job.docs_done == 5
    assert job.events_added > 0

    with Store(env.db_path) as store:
        events = store.list_events("demo")
        assert events, "timeline should not be empty"
        assert len(events) == job.events_added
        assert any(e.date_start is not None for e in events)

        # Every event points at a verbatim slice of the stored normalized text.
        for e in events:
            assert e.status != EventStatus.UNVERIFIED
            parsed = store.get_parsed_text(e.source.doc_id)
            assert parsed is not None
            assert parsed.text[e.source.char_start : e.source.char_end] == e.source.quote
        pdf_events = [e for e in events if e.source.page is not None]
        assert pdf_events and all(e.source.page in (1, 2) for e in pdf_events)

        # Relative date resolved against the anchor meeting (Becker: March 11 + 2 days).
        rel = [e for e in events if "days after the meeting" in e.date_text.lower()]
        assert rel and rel[0].date_start is not None

        # The police report repeats the HR memo's March 12 meeting: linked as a duplicate.
        all_events = store.list_events("demo", include_duplicates=True)
        assert len(all_events) > len(events)

        # March 11 (Becker) vs March 12 meeting is flagged; alerts exist and carry no content.
        contradictions = store.list_contradictions("demo")
        assert job.contradictions_added == len(contradictions) >= 1
        assert any(c.field == "date" for c in contradictions)
        alerts = store.list_alerts("demo")
        assert alerts
        assert all(a.dashboard_url.startswith("http://127.0.0.1:") for a in alerts)
        assert store.processed_doc_ids("demo") == {d.id for d in store.list_documents("demo")}

    # Second scan is a no-op.
    again = runner.scan_case_sync("demo")
    assert again.state == JobState.DONE
    assert again.docs_total == 0 and again.events_added == 0
    with Store(env.db_path) as store:
        assert len(store.list_events("demo", include_duplicates=True)) == len(all_events)
        assert len(store.list_contradictions("demo")) == len(contradictions)

    digest = runner.daily_digest("demo")
    assert digest["events_total"] == len(events)
    assert digest["contradictions_open"] == len(contradictions)
    assert digest["documents"] == 5
    assert "description" not in json.dumps(digest)

    assert _evidence_snapshot() == before, "evidence folder must never be modified"
    assert not (FIXTURES / "scan.lock").exists()


def test_background_scan_and_consumers(env):
    job = runner.scan_case("demo")
    assert job.state == JobState.QUEUED
    with Store(env.db_path) as store:
        for _ in range(600):
            current = store.get_job(job.id)
            if current and current.state in (JobState.DONE, JobState.FAILED):
                break
            time.sleep(0.05)
        assert current is not None and current.state == JobState.DONE, current
        event = next(e for e in store.list_events("demo") if e.source.page is not None)

    # UI renders the timeline and the highlighted source.
    from ui.app import create_app

    client = TestClient(create_app(env), base_url="http://127.0.0.1:8080")
    r = client.get("/timeline")
    assert r.status_code == 200
    r = client.get(
        f"/source/{event.source.doc_id}",
        params={"start": event.source.char_start, "end": event.source.char_end, "event": event.id},
    )
    assert r.status_code == 200
    assert 'id="hl"' in r.text
    r = client.get("/api/events")
    assert r.status_code == 200 and len(r.json()) > 0

    # MCP get_timeline over the real store.
    from mcp_server import server as mcp_server

    with Store(env.db_path) as store:
        mcp_server.set_backend(
            mcp_server.Backend(
                store=store,
                scan_case=runner.scan_case,
                daily_digest=runner.daily_digest,
                case_id="demo",
                data_dir=Path(env.data_dir),
            )
        )
        try:
            srv = mcp_server.build_server()
            result = asyncio.run(srv.call_tool("get_timeline", {}))
            payload = result.structured_content or json.loads(result.content[0].text)
            assert payload["total"] > 0
            assert all("source" in e for e in payload["events"])
            digest = asyncio.run(srv.call_tool("daily_digest", {}))
            assert not digest.is_error
        finally:
            mcp_server.set_backend(None)


def test_failed_document_does_not_stop_scan(env, monkeypatch):
    from pipeline import extract

    real = extract.extract_events

    def flaky(parsed):
        if parsed.doc.filename.endswith(".eml"):
            raise RuntimeError("model unavailable")
        return real(parsed)

    monkeypatch.setattr(extract, "extract_events", flaky)
    job = runner.scan_case_sync("demo")
    assert job.state == JobState.DONE
    assert job.docs_done == 5
    assert job.error and "email_thread.eml" in job.error

    # The failed document is retried next time and succeeds once the model is back.
    monkeypatch.setattr(extract, "extract_events", real)
    retry = runner.scan_case_sync("demo")
    assert retry.docs_total == 1 and retry.error is None and retry.events_added > 0


def test_refuses_data_dir_inside_evidence(env, monkeypatch, tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setenv("CT_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("CT_DATA_DIR", str(evidence / "data"))
    from pipeline.config import Settings

    with pytest.raises(RuntimeError):
        runner.scan_case_sync("demo", settings=Settings())
    assert not (evidence / "data").exists()
