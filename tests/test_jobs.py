"""Tests for pipeline.jobs: queue, worker, and scan robustness (fake LLM, temp folders)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pipeline import jobs, llm
from pipeline.config import Settings
from pipeline.extract import ExtractedFact, ExtractionResult
from pipeline.store import Store
from tests.pipeline_helpers import fake_embedder, new_store

GOOD = "On March 3, 2024 Smith signed the lease.\nOn March 9, 2024 Jones paid the rent.\n"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    ev = tmp_path / "evidence"
    ev.mkdir()
    return Settings(evidence_dir=ev, data_dir=tmp_path / "data")


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch):
    state = {"down": False, "calls": 0}

    def chat_json(system, user, schema, **kw):
        state["calls"] += 1
        if state["down"]:
            raise ConnectionError("vLLM down")
        if schema is ExtractionResult:
            facts = []
            for line in user.split("\n"):
                if line.startswith("On March"):
                    facts.append(
                        ExtractedFact(
                            title=line[17:],
                            date_text=line[3:16],
                            quotes=[line],
                            people=[line.split()[4]],
                        )
                    )
            return ExtractionResult(facts=facts)
        return schema(is_contradiction=False, field="fact", explanation="no")

    monkeypatch.setattr(llm, "chat_json", chat_json)
    monkeypatch.setattr(llm, "embed", fake_embedder)
    return state


def test_request_scan_reuses_the_queued_job(tmp_path: Path) -> None:
    with new_store(tmp_path) as store:
        a = jobs.request_scan(store, "demo", "agent")
        b = jobs.request_scan(store, "demo", "ui")
        assert a.job_id == b.job_id and a.status == "queued"
        with pytest.raises(ValueError):
            jobs.request_scan(store, "../etc", "ui")


def test_one_running_job_per_case(tmp_path: Path) -> None:
    with new_store(tmp_path) as store:
        first = jobs.request_scan(store, "a", "ui")
        assert store.claim_next_job().job_id == first.job_id
        second = jobs.request_scan(store, "a", "ui")  # queued behind the running one
        assert second.job_id != first.job_id
        other = jobs.request_scan(store, "b", "ui")
        assert store.claim_next_job().job_id == other.job_id  # case a is busy
        assert store.claim_next_job() is None
        store.finish_job(first.job_id, "done")
        assert store.claim_next_job().job_id == second.job_id


def test_worker_requeues_jobs_left_running(settings: Settings, monkeypatch) -> None:
    ran: list[str] = []

    def fake_run_job(store, job, settings=None):
        ran.append(job.job_id)
        return store.finish_job(job.job_id, "done", counts={})

    monkeypatch.setattr(jobs, "run_job", fake_run_job)
    with Store(settings.db_path, embedder=fake_embedder) as store:
        job = jobs.request_scan(store, "demo", "agent")
        store.claim_next_job()  # "crash" while running
        assert store.get_job(job.job_id).status == "running"
        worker = jobs.Worker(settings, store=store, poll_seconds=0.05)
        worker.start()
        try:
            deadline = time.monotonic() + 5
            while store.get_job(job.job_id).status != "done" and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            worker.stop()
        assert ran == [job.job_id]
        assert store.get_job(job.job_id).status == "done"


def test_scan_survives_a_corrupt_document(settings: Settings, fake_llm) -> None:
    (settings.evidence_dir / "good.txt").write_text(GOOD)
    (settings.evidence_dir / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    job = jobs.run_now("demo", settings=settings)
    assert job.status == "done"
    c = job.counts
    assert (c["docs_seen"], c["docs_new"], c["docs_parsed"], c["docs_failed"]) == (2, 2, 1, 1)
    assert c["drafts"] == 2 and c["facts_created"] == 2
    assert "broken.pdf" in (job.error or "")
    with Store(settings.db_path) as store:
        docs = {d.filename: d for d in store.list_documents("demo")}
        assert docs["broken.pdf"].parse_status == "failed"
        assert store.document_info(docs["broken.pdf"].doc_id)["parse_error"]
        assert docs["good.txt"].parse_status == "parsed"
        assert len(store.list_facts("demo")) == 2
    again = jobs.run_now("demo", settings=settings)  # failed docs wait for a reprocess
    assert again.counts["docs_failed"] == 0 and again.counts["drafts"] == 0


def test_model_outage_fails_job_and_retries_later(settings: Settings, fake_llm) -> None:
    (settings.evidence_dir / "good.txt").write_text(GOOD)
    fake_llm["down"] = True
    job = jobs.run_now("demo", settings=settings)
    assert job.status == "failed"
    assert job.counts["docs_parsed"] == 1 and job.counts["facts_created"] == 0
    assert "vLLM down" in (job.error or "")
    fake_llm["down"] = False
    retry = jobs.run_now("demo", settings=settings)
    assert retry.status == "done"
    assert retry.counts["docs_new"] == 0 and retry.counts["docs_parsed"] == 0  # parsed text kept
    assert retry.counts["facts_created"] == 2


def test_inbox_files_are_scanned(settings: Settings, fake_llm) -> None:
    (settings.inbox / "upload.txt").write_text(GOOD)
    job = jobs.run_now("demo", settings=settings)
    assert job.counts["docs_new"] == 1 and job.counts["facts_created"] == 2
    with Store(settings.db_path) as store:
        (doc,) = store.list_documents("demo")
        assert doc.location == "inbox"


def test_empty_evidence_folder(settings: Settings, fake_llm) -> None:
    job = jobs.run_now("demo", settings=settings)
    assert job.status == "done"
    assert job.counts["docs_seen"] == 0 and fake_llm["calls"] == 0


def test_main_cli(settings: Settings, fake_llm, monkeypatch, capsys) -> None:
    monkeypatch.setattr(jobs, "get_settings", lambda: settings)
    (settings.evidence_dir / "good.txt").write_text(GOOD)
    assert jobs.main(["scan", "--case", "demo"]) == 0
    assert '"status": "done"' in capsys.readouterr().out
