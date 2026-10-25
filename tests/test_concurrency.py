"""Bounded LLM concurrency: same output as a sequential run, much less wall time.

The fake model sits behind the real ``llm.chat_json`` -> ``llm.chat`` path (only the OpenAI
client is replaced), so every request goes through the shared limiter in ``pipeline.llm``.
Each request sleeps 0.2 s, like a slow but batching vLLM server.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import extract, llm, runner
from pipeline.config import get_settings
from pipeline.contracts import Document, JobState, ParsedDocument
from pipeline.store import Store
from tests.test_e2e import FIXTURES, _fake_extract

DELAY = 0.2


class SlowModel:
    """Fake OpenAI client: answers extraction and verdict requests after ``DELAY`` seconds."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, messages, response_format, **kw):
        with self._lock:
            self.calls += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(DELAY)
            name = response_format["json_schema"]["name"]
            user = messages[1]["content"]
            if name == "ConflictVerdict":
                field = "fact"
                for f in ("date", "actor", "fact"):
                    if f"Suspected conflict type: {f}" in user:
                        field = f
                content = json.dumps(
                    {
                        "comparison": "",
                        "same_subject": True,
                        "reconciliation": "none",
                        "is_contradiction": True,
                        "field": field,
                        "other_fields": [],
                        "explanation": "fake verdict",
                    }
                )
            else:
                content = _fake_extract(user, extract.ExtractionResult).model_dump_json()
        finally:
            with self._lock:
                self.in_flight -= 1
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def slow_model(monkeypatch):
    model = SlowModel()
    monkeypatch.setattr(llm, "_client", lambda: model)
    monkeypatch.setattr(llm, "embed", lambda texts: [llm.hashed_embedding(t) for t in texts])
    return model


def _configure(monkeypatch, data_dir: Path, concurrency: int) -> None:
    monkeypatch.setenv("CT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CT_EVIDENCE_DIR", str(FIXTURES))
    monkeypatch.setenv("CT_CASE_ID", "demo")
    monkeypatch.setenv("CT_DISABLE_DOCLING", "1")
    monkeypatch.setenv("CT_LLM_CONCURRENCY", str(concurrency))
    monkeypatch.delenv("CT_DB_PATH", raising=False)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clear_settings():
    yield
    get_settings.cache_clear()


def _scan(monkeypatch, tmp_path: Path, model: SlowModel, concurrency: int) -> dict:
    _configure(monkeypatch, tmp_path / f"data-{concurrency}", concurrency)
    model.max_in_flight = 0
    t0 = time.monotonic()
    job = runner.scan_case_sync("demo")
    elapsed = time.monotonic() - t0
    assert job.state == JobState.DONE, job.error
    with Store(get_settings().db_path) as store:
        events = store.list_events("demo", include_duplicates=True)
        return {
            "elapsed": elapsed,
            "max_in_flight": model.max_in_flight,
            "job": (job.docs_total, job.docs_done, job.events_added, job.contradictions_added),
            "events": [e.model_dump(mode="json") for e in events],
            "contradictions": [
                c.model_dump(mode="json") for c in store.list_contradictions("demo")
            ],
            "processed": sorted(store.processed_doc_ids("demo")),
        }


def test_scan_concurrency_is_faster_and_output_identical(slow_model, monkeypatch, tmp_path):
    # Small chunks so each document needs several extraction calls (chunk + document overlap).
    monkeypatch.setattr(extract, "CHUNK_SIZE", 900)
    monkeypatch.setattr(extract, "CHUNK_OVERLAP", 150)
    seq = _scan(monkeypatch, tmp_path, slow_model, 1)
    calls = slow_model.calls
    par = _scan(monkeypatch, tmp_path, slow_model, 8)
    assert slow_model.calls == 2 * calls  # same requests, no extra or lost calls

    assert seq["max_in_flight"] == 1
    assert 1 < par["max_in_flight"] <= 8
    assert par["job"] == seq["job"] and seq["job"][0] == seq["job"][1] == 5
    assert par["events"] == seq["events"]  # same ids, fields, order
    assert par["contradictions"] == seq["contradictions"]
    assert par["processed"] == seq["processed"]
    assert seq["contradictions"], "the demo case has planted contradictions"
    # Sequential time is ~calls x 0.2 s; concurrency must cut it by well over half.
    assert seq["elapsed"] >= calls * DELAY * 0.9
    assert par["elapsed"] < seq["elapsed"] / 2.5, (seq["elapsed"], par["elapsed"])


def test_limiter_bounds_nested_fan_out(slow_model, monkeypatch, tmp_path):
    # Documents x chunks fan out on nested thread pools; the shared limiter keeps the number
    # of requests in flight at the configured bound.
    monkeypatch.setattr(extract, "CHUNK_SIZE", 600)
    monkeypatch.setattr(extract, "CHUNK_OVERLAP", 100)
    llm.reset_limiter_stats()
    out = _scan(monkeypatch, tmp_path, slow_model, 3)
    assert out["max_in_flight"] == 3
    assert llm.limiter_stats()["peak"] == 3 and llm.limiter_stats()["in_flight"] == 0


def test_chunk_extraction_runs_concurrently_in_chunk_order(slow_model, monkeypatch, tmp_path):
    text = "\n\n".join(
        f"On March {day}, 2024 Dana Rivera filed report number {day} about forklift {day}. "
        + "Nothing else of note happened that day. " * 12
        for day in range(1, 13)
    )
    doc = Document(
        id="d" * 16,
        case_id="demo",
        path="/evidence/log.txt",
        filename="log.txt",
        mime="text/plain",
        sha256="0" * 64,
    )
    parsed = ParsedDocument(doc=doc, text=text)
    monkeypatch.setattr(extract, "CHUNK_SIZE", 900)
    monkeypatch.setattr(extract, "CHUNK_OVERLAP", 150)
    n_chunks = len(extract.chunk_text(text, 900, 150))
    assert n_chunks >= 6

    results = {}
    for n in (1, 8):
        _configure(monkeypatch, tmp_path / "data", n)
        slow_model.max_in_flight = 0
        t0 = time.monotonic()
        events = extract.extract_events(parsed)
        results[n] = (time.monotonic() - t0, slow_model.max_in_flight, events)
    (t1, peak1, ev1), (t8, peak8, ev8) = results[1], results[8]
    assert peak1 == 1 and peak8 > 1
    assert [e.model_dump() for e in ev8] == [e.model_dump() for e in ev1]
    assert len(ev1) == 12  # one per dated sentence, overlap duplicates merged
    assert t8 < t1 / 2.5, (t1, t8)


def test_one_failed_document_does_not_stop_a_concurrent_scan(slow_model, monkeypatch, tmp_path):
    real = extract.extract_events

    def flaky(parsed):
        if parsed.doc.filename.endswith(".eml"):
            raise RuntimeError("model unavailable")
        return real(parsed)

    monkeypatch.setattr(extract, "extract_events", flaky)
    _configure(monkeypatch, tmp_path / "data", 8)
    job = runner.scan_case_sync("demo")
    assert job.state == JobState.DONE
    assert job.docs_total == job.docs_done == 5
    assert job.error and "email_thread.eml" in job.error
    with Store(get_settings().db_path) as store:
        assert len(store.processed_doc_ids("demo")) == 4
        assert job.events_added == len(store.list_events("demo"))

    monkeypatch.setattr(extract, "extract_events", real)
    retry = runner.scan_case_sync("demo")
    assert retry.docs_total == 1 and retry.error is None and retry.events_added > 0
    again = runner.scan_case_sync("demo")  # idempotent: nothing left to do
    assert again.docs_total == 0 and again.events_added == 0
