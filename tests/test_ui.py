"""UI tests with a fake store/runner/llm; no database, model or OpenClaw needed."""

from __future__ import annotations

import os
import stat
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.config import Settings
from pipeline.contracts import (
    Alert,
    AlertType,
    Certainty,
    ConflictField,
    Contradiction,
    ContradictionStatus,
    Event,
    EventStatus,
    Job,
    JobState,
    SourceRef,
)
from ui.app import create_app, host_is_loopback, sanitize_filename

DOC_TEXT = "Intro line.\nOn March 3, 2024 <script>alert(1)</script> Smith signed the lease.\nEnd."
QUOTE = "<script>alert(1)</script> Smith signed"
Q_START = DOC_TEXT.index(QUOTE)
Q_END = Q_START + len(QUOTE)


def _event(eid: str, **kw) -> Event:
    base = dict(
        id=eid,
        case_id="demo",
        date_text="March 3, 2024",
        date_start=date(2024, 3, 3),
        date_end=date(2024, 3, 3),
        certainty=Certainty.EXACT,
        actors=["Smith"],
        description=f"Smith signed the lease ({eid})",
        source=SourceRef(doc_id="doc1", char_start=Q_START, char_end=Q_END, quote=QUOTE),
        confidence=0.9,
    )
    base.update(kw)
    return Event(**base)


class FakeStore:
    def __init__(self) -> None:
        self.events = {
            "e1": _event("e1", deadline_relevant=True, importance=0.9),
            "e2": _event(
                "e2",
                date_text="early March 2024",
                date_start=date(2024, 3, 1),
                date_end=date(2024, 3, 10),
                certainty=Certainty.RANGE,
                status=EventStatus.UNVERIFIED,
                description="<b>bold</b> claim",
            ),
            "e3": _event("e3", date_start=None, date_end=None, certainty=Certainty.UNKNOWN),
        }
        self.contradictions = [
            Contradiction(
                id="c1",
                case_id="demo",
                event_ids=["e1", "e2"],
                field=ConflictField.DATE,
                explanation="Different signing dates",
            )
        ]
        self.alerts = [
            Alert(
                id="a1",
                case_id="demo",
                type=AlertType.NEW_CONTRADICTION,
                created_at=datetime(2024, 3, 4, tzinfo=UTC),
                dashboard_url="http://127.0.0.1:8090/cases/demo/alerts/a1",
                ref_ids=["c1"],
            )
        ]
        self.jobs: dict[str, Job] = {}
        self.status_calls: list[tuple[str, EventStatus]] = []

    def list_events(self, case_id, status=None):
        return [e for e in self.events.values() if status is None or e.status == status]

    def get_event(self, eid):
        return self.events.get(eid)

    def set_event_status(self, eid, status):
        self.status_calls.append((eid, status))
        self.events[eid] = self.events[eid].model_copy(update={"status": status})

    def list_contradictions(self, case_id, status=None):
        return list(self.contradictions)

    def set_contradiction_status(self, cid, status):
        self.contradictions = [
            c.model_copy(update={"status": status}) if c.id == cid else c
            for c in self.contradictions
        ]

    def list_alerts(self, case_id):
        return list(self.alerts)

    def processed_doc_ids(self, case_id):
        return {"doc1"}

    def get_parsed_text(self, doc_id):
        return DOC_TEXT if doc_id == "doc1" else None

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def search_events(self, case_id, query, k):
        return [self.events["e1"]]


class FakeRunner:
    def __init__(self, store: FakeStore) -> None:
        self.store = store

    def scan_case(self, case_id):
        job = Job(id="job1", case_id=case_id, state=JobState.RUNNING, docs_total=2, docs_done=1)
        self.store.jobs[job.id] = job
        return job


class FakeLLM:
    def __init__(self, reply: str = "Smith signed on March 3, 2024 [E1].") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def env(tmp_path: Path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "orig.txt").write_text("original")
    settings = Settings(evidence_dir=evidence, data_dir=tmp_path / "data")
    app = create_app(settings)
    store = FakeStore()
    deps = app.state.deps
    deps.store, deps.runner, deps.llm = store, FakeRunner(store), FakeLLM()
    deps.openclaw_bin = None
    client = TestClient(app, base_url="http://127.0.0.1:8090")
    return client, store, settings, deps


def test_dashboard_counts(env):
    client, *_ = env
    r = client.get("/")
    assert r.status_code == 200
    assert "Process now" in r.text and 'hx-post="/process"' in r.text
    assert "Content-Security-Policy" in r.headers
    assert "/static/vendor/htmx.min.js" in r.text and "cdn" not in r.text.lower()


def test_process_returns_polling_fragment(env):
    client, store, _, deps = env
    r = client.post("/process")
    assert r.status_code == 200
    assert 'hx-get="/jobs/job1"' in r.text and "every 2s" in r.text
    assert deps.last_job_id == "job1"
    store.jobs["job1"] = store.jobs["job1"].model_copy(update={"state": JobState.DONE})
    r = client.get("/jobs/job1")
    assert "every 2s" not in r.text and "done" in r.text
    assert client.get("/jobs/nope").status_code == 404


def test_timeline_and_api_events(env):
    client, *_ = env
    r = client.get("/timeline?view=all")
    assert r.status_code == 200
    assert "Needs attorney review" in r.text
    assert "<b>bold</b>" not in r.text and "&lt;b&gt;bold&lt;/b&gt;" in r.text
    data = client.get("/api/events?view=all").json()
    items = {i["id"]: i for i in data["items"]}
    assert "cert-exact" in items["e1"]["class_name"] and "conflict" in items["e1"]["class_name"]
    assert "deadline" in items["e1"]["class_name"]
    assert "status-unverified" in items["e2"]["class_name"]
    assert items["e2"]["end"] == "2024-03-11"  # inclusive range -> exclusive vis end
    assert items["e3"]["start"] is None
    assert items["e1"]["source_url"].startswith("/source/doc1?start=")


def test_key_events_are_the_default_view(env):
    client, *_ = env
    r = client.get("/timeline")
    assert r.status_code == 200
    assert "Key events" in r.text and "(1 of 3)" in r.text
    assert "Smith signed the lease (e1)" in r.text and "Key event</span>" in r.text
    assert "&lt;b&gt;bold" not in r.text and "(e3)" not in r.text  # importance 0.5: hidden
    assert 'href="/timeline?view=all"' in r.text and "2 less important events hidden" in r.text
    assert 'data-src="/api/events?view=key"' in r.text
    data = client.get("/api/events").json()
    assert data["view"] == "key" and data["total"] == 3
    assert [i["id"] for i in data["items"]] == ["e1"]
    assert data["items"][0]["key"] is True and data["items"][0]["importance"] == 0.9
    assert "key" in data["items"][0]["class_name"].split()


def test_show_all_is_remembered_in_the_query(env):
    client, *_ = env
    r = client.get("/timeline", params={"view": "all"})
    assert "All events" in r.text and "(e3)" in r.text and "&lt;b&gt;bold" in r.text
    assert 'data-src="/api/events?view=all"' in r.text
    assert 'aria-current="page">Show all' in r.text
    assert 'name="view" value="all"' in r.text  # review forms keep the view
    data = client.get("/api/events", params={"view": "all"}).json()
    assert data["view"] == "all" and len(data["items"]) == 3
    assert {i["id"]: i["key"] for i in data["items"]} == {"e1": True, "e2": False, "e3": False}
    assert client.get("/api/events", params={"view": "bogus"}).json()["view"] == "key"


def test_key_view_with_no_key_events_offers_show_all(env):
    client, store, *_ = env
    store.events = {k: v.model_copy(update={"importance": 0.2}) for k, v in store.events.items()}
    r = client.get("/timeline")
    assert "No key events" in r.text and "Show all 3 events" in r.text


def test_review_redirect_keeps_view(env):
    client, *_ = env
    r = client.post(
        "/events/e2/status",
        data={"status": "accepted", "view": "all"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"] == "/timeline?view=all"
    r = client.post("/events/e2/status", data={"status": "rejected"}, follow_redirects=False)
    assert r.headers["location"] == "/timeline"


def test_source_escapes_and_highlights(env):
    client, *_ = env
    r = client.get(f"/source/doc1?start={Q_START}&end={Q_END}&event=e1")
    assert r.status_code == 200
    assert "<script>alert(1)</script>" not in r.text
    assert '<mark id="hl">&lt;script&gt;alert(1)&lt;/script&gt; Smith signed</mark>' in r.text
    assert "/static/source.js" in r.text


def test_source_bad_span_and_missing_doc(env):
    client, *_ = env
    r = client.get("/source/doc1?start=5&end=99999")
    assert r.status_code == 200 and "<mark" not in r.text and "could not be located" in r.text
    assert client.get("/source/missing").status_code == 404
    assert client.get("/source/..%2Fetc").status_code == 404


def test_upload_saves_to_inbox_and_sanitizes(env):
    client, _, settings, _ = env
    files = [
        ("files", ("../../evil.txt", b"a", "text/plain")),
        ("files", ("..\\..\\win.md", b"b", "text/markdown")),
        ("files", ("payload.sh", b"c", "text/x-sh")),
        ("files", ("../../evidence/orig.txt", b"d", "text/plain")),
    ]
    r = client.post("/upload", files=files)
    assert r.status_code == 200
    inbox = settings.data_dir / "inbox"
    saved = sorted(p.name for p in inbox.iterdir())
    assert saved == ["evil.txt", "orig.txt", "win.md"]
    assert "unsupported file type" in r.text
    assert (settings.evidence_dir / "orig.txt").read_text() == "original"
    assert sorted(p.name for p in settings.evidence_dir.iterdir()) == ["orig.txt"]
    assert not (settings.data_dir.parent / "evil.txt").exists()


def test_upload_does_not_overwrite(env):
    client, _, settings, _ = env
    for body in (b"one", b"two"):
        client.post("/upload", files=[("files", ("note.txt", body, "text/plain"))])
    inbox = settings.data_dir / "inbox"
    assert (inbox / "note.txt").read_bytes() == b"one"
    assert (inbox / "note-1.txt").read_bytes() == b"two"


def test_upload_refuses_inbox_inside_evidence(tmp_path):
    evidence = tmp_path / "ev"
    evidence.mkdir()
    app = create_app(Settings(evidence_dir=evidence, data_dir=evidence / "data"))
    client = TestClient(app, base_url="http://127.0.0.1:8090", raise_server_exceptions=False)
    r = client.post("/upload", files=[("files", ("a.txt", b"x", "text/plain"))])
    assert r.status_code == 500
    assert not (evidence / "data").exists()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\x\\..\\report.pdf", "report.pdf"),
        ("..", None),
        ("", None),
        ("a<b>.txt", "a_b_.txt"),
        ("...hidden.txt", "hidden.txt"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_event_review_status(env):
    client, store, *_ = env
    hx = {"HX-Request": "true"}
    r = client.post("/events/e1/status", data={"status": "accepted"}, headers=hx)
    assert r.status_code == 200 and 'id="event-e1"' in r.text and "accepted" in r.text
    assert store.status_calls == [("e1", EventStatus.ACCEPTED)]
    r = client.post("/events/e1/status", data={"status": "rejected"}, follow_redirects=False)
    assert r.status_code == 303
    assert client.post("/events/e1/status", data={"status": "unverified"}).status_code == 422
    assert client.post("/events/zzz/status", data={"status": "accepted"}).status_code == 404


def test_contradictions_page_and_status(env):
    client, store, *_ = env
    r = client.get("/contradictions")
    assert r.status_code == 200
    assert "Different signing dates" in r.text and "/source/doc1?start=" in r.text
    assert "&lt;script&gt;" in r.text and "<script>alert" not in r.text
    r = client.post(
        "/contradictions/c1/status", data={"status": "dismissed"}, headers={"HX-Request": "true"}
    )
    assert r.status_code == 200 and "dismissed" in r.text
    assert store.contradictions[0].status == ContradictionStatus.DISMISSED


def test_alerts_and_alert_link(env):
    client, *_ = env
    r = client.get("/alerts")
    assert r.status_code == 200 and 'id="alert-a1"' in r.text
    r = client.get("/cases/demo/alerts/a1", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/alerts#alert-a1"
    assert client.get("/cases/other/alerts/a1", follow_redirects=False).status_code == 404


def test_chat_local_fallback_with_citations(env):
    client, _, _, deps = env
    r = client.post("/chat", data={"question": "When was the lease signed?"})
    assert r.status_code == 200
    assert '<a href="/source/doc1?start=' in r.text and "[E1]</a>" in r.text
    assert "local model" in r.text
    assert "When was the lease signed?" in deps.llm.calls[0][1]["content"]


def test_chat_escapes_model_output(env):
    client, _, _, deps = env
    deps.llm = FakeLLM("<img src=x onerror=alert(1)> [E1]")
    r = client.post("/chat", data={"question": "q"}, headers={"HX-Request": "true"})
    assert "<img src=x" not in r.text and "&lt;img src=x" in r.text


def test_chat_llm_down_still_lists_evidence(env):
    client, _, _, deps = env

    class Down:
        def chat(self, messages, **kw):
            raise ConnectionError("vLLM not running")

    deps.llm = Down()
    r = client.post("/chat", data={"question": "q"})
    assert r.status_code == 200 and "local model is unavailable" in r.text
    assert "Cited events" in r.text


def test_chat_uses_openclaw_when_available(env, tmp_path):
    client, _, _, deps = env
    script = tmp_path / "openclaw"
    reply = '{"reply": "From agent: see /source/doc1?start=1&end=5"}'
    script.write_text(f"#!/bin/sh\necho '{reply}'\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    deps.openclaw_bin = script
    r = client.post("/chat", data={"question": "--help me"})
    assert r.status_code == 200
    assert "From agent" in r.text and "OpenClaw agent" in r.text
    assert '<a href="/source/doc1?start=1&amp;end=5">' in r.text
    assert deps.llm.calls == []


def test_chat_openclaw_failure_falls_back(env, tmp_path):
    client, _, _, deps = env
    script = tmp_path / "openclaw"
    script.write_text("#!/bin/sh\nexit 3\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    deps.openclaw_bin = script
    r = client.post("/chat", data={"question": "q"})
    assert "local model" in r.text and len(deps.llm.calls) == 1


def test_rejects_foreign_host_and_cross_origin_post(env):
    client, *_ = env
    assert client.get("/", headers={"host": "evil.example"}).status_code == 400
    r = client.post("/process", headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    r = client.post("/process", headers={"sec-fetch-site": "cross-site"})
    assert r.status_code == 403


def test_vendored_assets_served(env):
    client, *_ = env
    for path in (
        "/static/vendor/htmx.min.js",
        "/static/vendor/vis-timeline-graph2d.min.js",
        "/static/vendor/vis-timeline-graph2d.min.css",
        "/static/app.css",
        "/static/timeline.js",
    ):
        assert client.get(path).status_code == 200, path


def test_loopback_check():
    assert host_is_loopback("127.0.0.1") and host_is_loopback("::1")
    assert host_is_loopback("localhost") and host_is_loopback("127.0.0.2")
    assert not host_is_loopback("0.0.0.0") and not host_is_loopback("192.168.1.5")


def test_main_refuses_non_loopback(monkeypatch):
    import ui.__main__ as main_mod

    monkeypatch.setattr(main_mod, "get_settings", lambda: Settings(ui_host="0.0.0.0"))
    monkeypatch.setattr(main_mod.uvicorn, "run", lambda *a, **k: pytest.fail("must not bind"))
    assert main_mod.main() == 2


def test_no_writes_into_evidence(env):
    client, _, settings, _ = env
    before = {p: p.stat().st_mtime_ns for p in settings.evidence_dir.rglob("*")}
    client.get("/")
    client.post("/upload", files=[("files", ("x.txt", b"x", "text/plain"))])
    after = {p: p.stat().st_mtime_ns for p in settings.evidence_dir.rglob("*")}
    assert before == after
    assert os.listdir(settings.evidence_dir) == ["orig.txt"]
