"""Orchestrate a case scan: discover → parse → extract → verify → dates → merge → conflicts.

Only new or changed documents are processed (a changed file has a new sha256, hence a new
document id). Evidence files are only read; everything is written to ``data_dir``.
``scan_case`` returns immediately and runs in a background thread; ``scan_case_sync`` runs
inline. A file lock in ``data_dir`` ensures only one scan runs at a time across processes.

Documents are parsed one at a time, and each one's extraction starts as soon as it is
parsed; up to ``settings.llm_concurrency`` of them are extracted concurrently (the model
requests share one limiter in ``pipeline.llm``). Once every document is parsed, the case's
incident date is taken as the consensus of the documents' own incident dates, so that a
document that never states it (an interview held weeks later) dates its account against it.
Verify, dates, merge and every store write stay on the scan thread, in document order.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from pipeline import conflicts, dates, extract, merge, parse, verify
from pipeline.config import Settings, get_settings, valid_case_id
from pipeline.contracts import (
    ContradictionStatus,
    Document,
    Event,
    EventStatus,
    Job,
    JobState,
    ParsedDocument,
)
from pipeline.store import Store

log = logging.getLogger(__name__)

LOCK_TIMEOUT = 6 * 3600.0  # seconds a queued scan waits for a running one to finish
_MAX_ERROR_CHARS = 2000

_active_lock = threading.Lock()
_active: dict[tuple[str, str], str] = {}  # (db_path, case_id) -> job id running in-process


@dataclass
class _ScanState:
    """Mutable per-scan bookkeeping."""

    existing: list[Event]
    case_incident: date | None = None  # the case's incident date (consensus of its documents)
    case_tz: str | None = None  # IANA zone that explicit UTC times are converted to
    new_ids: set[str] = field(default_factory=set)
    events_added: int = 0
    docs_done: int = 0
    failures: list[str] = field(default_factory=list)


def _now() -> datetime:
    return datetime.now(UTC)


def _check_dirs(settings: Settings) -> None:
    """Refuse to run if anything we write (data_dir, db, inbox) is inside the evidence dir."""
    settings.check_layout()


def _lock_path(settings: Settings) -> Path:
    data = Path(settings.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    return data / "scan.lock"


# --------------------------------------------------------------------------- per-document


def _parse_document(store: Store, doc: Document) -> ParsedDocument:
    """Register and parse one document (scan thread; parsing stays sequential)."""
    store.upsert_document(doc.model_copy(update={"ingested_at": doc.ingested_at or _now()}))
    parsed = parse.parse(doc)
    if parsed.doc.num_pages != doc.num_pages:
        store.upsert_document(parsed.doc)
    return parsed


def _store_document(
    store: Store, parsed: ParsedDocument, events: list[Event], state: _ScanState
) -> int:
    """Verify, date, merge and store one document's extracted events; returns events inserted.

    Runs on the scan thread, in document order, so merging against ``state.existing`` and all
    store writes stay serialized and the result does not depend on which extraction finished
    first.
    """
    doc = parsed.doc
    events = verify.verify_events(parsed, events)
    events = dates.resolve_dates(
        events,
        parse.doc_date(parsed),
        parsed=parsed,
        case_incident_date=state.case_incident,
        tz=state.case_tz,
    )
    to_insert, pairs = merge.merge_events_linked(events, state.existing)
    duplicates = [dup for dup, _ in pairs]
    store.add_events(to_insert + duplicates)
    for dup, canonical in pairs:
        store.link_duplicate(dup.id, canonical)
    # Text is stored (and the doc marked processed) only after events are safely written,
    # so a failed document is retried on the next scan.
    store.add_parsed_text(doc.id, parsed.text, parsed.pages)
    store.mark_processed(doc.id)
    state.existing.extend(to_insert)
    state.new_ids.update(e.id for e in to_insert)
    log.info(
        "doc %s: %d events (%d new, %d duplicates)",
        doc.id,
        len(events),
        len(to_insert),
        len(duplicates),
    )
    return len(to_insert)


def case_incident_date(parsed_docs: Iterable[ParsedDocument]) -> date | None:
    """The case's incident date: the date most of its documents give as their incident date.

    Each document votes with ``parse.incident_date`` (an incident form field, a where/when
    header or an opening sentence); documents that name none do not vote. None when no
    document names one or when two dates tie for the most votes.
    """
    votes: Counter[date] = Counter()
    for parsed in parsed_docs:
        try:
            found = parse.incident_date(parsed)
        except Exception:  # a heuristic must never fail the scan
            log.debug("incident date of %s failed", parsed.doc.id, exc_info=True)
            continue
        if found is not None:
            votes[found] += 1
    ranked = votes.most_common(2)
    if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None
    return ranked[0][0]


def _stored_parsed(store: Store, docs: Iterable[Document]) -> Iterable[ParsedDocument]:
    """The stored parsed text of already-processed documents (lazily, one at a time)."""
    for doc in docs:
        parsed = store.get_parsed_text(doc.id)
        if parsed is not None:
            yield parsed


@dataclass
class _Pending:
    """A parsed document whose extraction is running (or a document that failed to parse)."""

    doc: Document
    parsed: ParsedDocument | None = None
    extraction: Future[list[Event]] | None = None
    error: Exception | None = None


def _process_documents(
    store: Store,
    job: Job,
    todo: list[Document],
    state: _ScanState,
    settings: Settings,
    processed: list[Document] | None = None,
) -> Job:
    """Parse every document while their extractions run, then date and store them in order.

    Each document's extraction (the LLM-bound step) is submitted as soon as it is parsed and
    runs on a pool of ``settings.llm_concurrency`` threads; its model requests share the
    process-wide limiter in ``pipeline.llm``. When all documents are parsed, the case
    incident date is computed from them and from the case's already-processed documents
    (``processed``), unless ``settings.case_incident_date`` sets it. Everything else
    (verify, dates, merge, store writes, job progress) happens on this thread in document
    order, so the stored result is the same as a sequential scan. A document that fails is
    recorded and the scan goes on.
    """
    workers = max(1, settings.llm_concurrency)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="extract")
    items: list[_Pending] = []
    try:
        for doc in todo:
            item = _Pending(doc)
            try:
                item.parsed = _parse_document(store, doc)
                item.extraction = pool.submit(extract.extract_events, item.parsed)
            except Exception as exc:  # recorded when this document's turn comes
                item.error = exc
            items.append(item)

        state.case_tz = settings.case_tz
        state.case_incident = settings.case_incident_date or case_incident_date(
            chain(
                (i.parsed for i in items if i.parsed is not None),
                _stored_parsed(store, processed or []),
            )
        )
        log.info("case %s: incident date %s", job.case_id, state.case_incident)

        for item in items:
            try:
                if item.error is not None:
                    raise item.error
                assert item.parsed is not None and item.extraction is not None
                events = item.extraction.result()
                state.events_added += _store_document(store, item.parsed, events, state)
            except Exception as exc:  # one bad document must not kill the scan
                log.exception("document %s (%s) failed", item.doc.id, item.doc.filename)
                state.failures.append(f"{item.doc.filename}: {type(exc).__name__}: {exc}")
            item.parsed = item.extraction = None  # done with this document's text
            state.docs_done += 1
            job = store.update_job(job, docs_done=state.docs_done, events_added=state.events_added)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return job


def _contradictions_and_alerts(
    store: Store, case_id: str, new_ids: set[str], settings: Settings
) -> int:
    """Check pairs involving new events for contradictions, store them and the alerts."""
    events = store.list_events(case_id)
    known = {c.id for c in store.list_contradictions(case_id)}
    found = conflicts.find_contradictions(events, case_id, skip_ids=known, focus_ids=new_ids)
    fresh = [c for c in found if c.id not in known]
    if fresh:
        store.add_contradictions(fresh)
    all_contradictions = store.list_contradictions(case_id)
    alerts = conflicts.alerts_for(all_contradictions, events, case_id, settings.ui_base_url)
    inserted = store.add_alerts(alerts)
    log.info("case %s: %d new contradictions, %d new alerts", case_id, len(fresh), inserted)
    return len(fresh)


# --------------------------------------------------------------------------- scan


def _run_scan(store: Store, job: Job, settings: Settings) -> Job:
    """Do the work for ``job`` (lock already held). Never raises; failures land in the job."""
    case_id = job.case_id
    try:
        docs = parse.discover(Path(settings.evidence_dir), case_id)
        processed = store.processed_doc_ids(case_id)
        todo = [d for d in docs if d.id not in processed]
        job = store.update_job(
            job, state=JobState.RUNNING, started_at=_now(), docs_total=len(todo), docs_done=0
        )
        log.info("scan %s case=%s: %d documents, %d new", job.id, case_id, len(docs), len(todo))
        state = _ScanState(existing=store.list_events(case_id))
        done = [d for d in docs if d.id in processed]
        job = _process_documents(store, job, todo, state, settings, done)

        added = 0
        if state.new_ids:
            try:
                added = _contradictions_and_alerts(store, case_id, state.new_ids, settings)
            except Exception as exc:
                log.exception("contradiction check failed for case %s", case_id)
                state.failures.append(f"contradiction check: {type(exc).__name__}: {exc}")

        all_failed = bool(todo) and len(state.failures) >= len(todo)
        error = None
        if state.failures:
            error = f"{len(state.failures)} problem(s): " + "; ".join(state.failures)
            error = error[:_MAX_ERROR_CHARS]
        return store.update_job(
            job,
            state=JobState.FAILED if all_failed else JobState.DONE,
            finished_at=_now(),
            contradictions_added=added,
            error=error,
        )
    except Exception as exc:
        log.exception("scan %s failed", job.id)
        return store.update_job(
            job,
            state=JobState.FAILED,
            finished_at=_now(),
            error=f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS],
        )


def _locked_scan(store: Store, job: Job, settings: Settings, timeout: float) -> Job:
    try:
        with FileLock(str(_lock_path(settings)), timeout=timeout):
            return _run_scan(store, job, settings)
    except Timeout:
        log.warning("scan %s: another scan holds the lock", job.id)
        return store.update_job(
            job,
            state=JobState.FAILED,
            finished_at=_now(),
            error="another scan is already running; try again later",
        )


def _resolve(case_id: str | None, settings: Settings | None) -> tuple[str, Settings]:
    settings = settings or get_settings()
    _check_dirs(settings)
    return valid_case_id(case_id or settings.case_id), settings


def scan_case_sync(
    case_id: str | None = None,
    *,
    settings: Settings | None = None,
    store: Store | None = None,
    lock_timeout: float = LOCK_TIMEOUT,
) -> Job:
    """Scan inline and return the finished job (DONE or FAILED)."""
    cid, settings = _resolve(case_id, settings)
    own = store is None
    st = store or Store(settings.db_path)
    try:
        job = st.create_job(cid)
        return _locked_scan(st, job, settings, lock_timeout)
    finally:
        if own:
            st.close()


def scan_case(
    case_id: str | None = None,
    *,
    settings: Settings | None = None,
    store: Store | None = None,
    lock_timeout: float = LOCK_TIMEOUT,
) -> Job:
    """Start a background scan and return its job immediately (QUEUED).

    If a scan of the same case is already running in this process, its job is returned
    instead of starting another one. Poll ``Store.get_job(job.id)`` for progress.
    """
    cid, settings = _resolve(case_id, settings)
    key = (str(settings.db_path), cid)
    own = store is None
    st = store or Store(settings.db_path)
    with _active_lock:
        running_id = _active.get(key)
        running = st.get_job(running_id) if running_id else None
        if running is not None and running.state in (JobState.QUEUED, JobState.RUNNING):
            if own:
                st.close()
            return running
        job = st.create_job(cid)
        _active[key] = job.id

    def work() -> None:
        try:
            _locked_scan(st, job, settings, lock_timeout)
        except Exception:  # pragma: no cover - _locked_scan already records failures
            log.exception("background scan %s crashed", job.id)
        finally:
            with _active_lock:
                if _active.get(key) == job.id:
                    del _active[key]
            if own:
                st.close()

    threading.Thread(target=work, name=f"scan-{cid}-{job.id[:8]}", daemon=True).start()
    return job


# --------------------------------------------------------------------------- digest


def daily_digest(
    case_id: str | None = None,
    *,
    settings: Settings | None = None,
    store: Store | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Counts only (no case content), safe to relay to a messaging channel."""
    settings = settings or get_settings()
    cid = case_id or settings.case_id
    own = store is None
    st = store or Store(settings.db_path)
    try:
        events = st.list_events(cid)
        contradictions = st.list_contradictions(cid)
        current = now or _now()
        recent = st.changes_since(cid, current - timedelta(days=1), until=current)
        last = st.latest_job(cid)
        by_status = {s.value: 0 for s in EventStatus}
        for e in events:
            by_status[e.status.value] += 1
        return {
            "case_id": cid,
            "generated_at": current.isoformat(),
            "documents": len(st.list_documents(cid)),
            "events_total": len(events),
            "events_by_status": by_status,
            "events_needing_review": by_status[EventStatus.PROPOSED.value]
            + by_status[EventStatus.UNVERIFIED.value],
            "deadline_events": sum(
                1 for e in events if e.deadline_relevant and e.status != EventStatus.REJECTED
            ),
            "undated_events": sum(1 for e in events if e.date_start is None),
            "contradictions_open": sum(
                1 for c in contradictions if c.status == ContradictionStatus.OPEN
            ),
            "contradictions_total": len(contradictions),
            "alerts_total": len(st.list_alerts(cid)),
            "last_24h": {
                "events": len(recent.new_events),
                "contradictions": len(recent.new_contradictions),
                "alerts": len(recent.new_alerts),
            },
            "last_job": None
            if last is None
            else {
                "id": last.id,
                "state": last.state.value,
                "finished_at": last.finished_at.isoformat() if last.finished_at else None,
                "docs_done": last.docs_done,
                "docs_total": last.docs_total,
            },
            "dashboard_url": f"{settings.ui_base_url}/alerts",
        }
    finally:
        if own:
            st.close()


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """``python -m pipeline.runner scan|digest [--case ID]``; prints JSON to stdout."""
    import json

    parser = argparse.ArgumentParser(prog="python -m pipeline.runner")
    parser.add_argument("command", choices=["scan", "digest"])
    parser.add_argument("--case", dest="case_id", default=None, help="case id (default: settings)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "scan":
        job = scan_case_sync(args.case_id)
        sys.stdout.write(job.model_dump_json(indent=2) + "\n")
        return 1 if job.state == JobState.FAILED else 0
    sys.stdout.write(json.dumps(daily_digest(args.case_id), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
