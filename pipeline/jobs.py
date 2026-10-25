"""Job queue and worker (SPEC §5.1).

Anything may *request* a scan (:func:`request_scan` only adds a row to the jobs table); the
app server's :class:`Worker` thread *runs* jobs, one at a time per case. Job state lives in
SQLite, so it survives restarts, and the worker re-queues jobs left running by a crash.

A scan: discover files in the evidence folder and the inbox (hash dedupe) → parse new
documents → draft facts (extract → verify → dates → witnesses → merge) → contradictions →
alerts. Every stage lives in its own module; this file only orchestrates.

CLI: ``python -m pipeline.jobs scan [--case ID]`` queues a scan and runs it inline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.config import Settings, get_settings, valid_case_id
from pipeline.schema import AGENT, Contradiction, DraftFact, Job, ParsedDocument
from pipeline.store import Store

log = logging.getLogger(__name__)

_MAX_ERROR_CHARS = 2000
COUNT_KEYS = (
    "docs_seen",  # supported files found in the evidence folder and inbox
    "docs_new",  # of those, content not seen before
    "docs_parsed",
    "docs_failed",
    "drafts",  # drafts proposed by the model
    "quotes_checked",
    "quotes_rejected",  # quotes that failed the verbatim check
    "drafts_discarded",  # drafts left with no verified quote
    "facts_created",
    "evidence_attached",  # merged into an existing fact instead of a new one
    "contradictions",
    "alerts",
)


def request_scan(store: Store, case_id: str, trigger: str) -> Job:
    """Queue a scan (or return the one already queued for this case). Never blocks."""
    job, created = store.request_job(valid_case_id(case_id), trigger)
    if created:
        log.info("job %s queued for case %s (%s)", job.job_id, case_id, trigger)
    return job


@dataclass
class _Run:
    """Per-job bookkeeping."""

    job: Job
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNT_KEYS, 0))
    new_fact_ids: list[str] = field(default_factory=list)
    touched_fact_ids: list[str] = field(default_factory=list)  # gained evidence
    processed_doc_ids: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _doc_path(store: Store, settings: Settings, doc_id: str) -> Path:
    loc = store.document_source(doc_id)
    if loc is None:
        raise KeyError(doc_id)
    location, rel = loc
    root = Path(settings.evidence_dir) if location == "evidence" else settings.inbox
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"document path escapes its folder: {rel}")
    return path


def _discover(store: Store, settings: Settings, run: _Run) -> None:
    """Register every supported file; identical content is registered once."""
    from pipeline import parse

    files = parse.discover(Path(settings.evidence_dir), settings.inbox)
    run.counts["docs_seen"] = len(files)
    for f in files:
        _, created = store.register_document(
            run.job.case_id,
            filename=f.path.name,
            rel_path=f.rel_path,
            sha256=f.sha256,
            location=f.location,
        )
        run.counts["docs_new"] += int(created)


def _parse_doc(store: Store, settings: Settings, doc_id: str, run: _Run) -> ParsedDocument | None:
    from pipeline import parse

    doc = store.get_document(doc_id)
    assert doc is not None
    if doc.parse_status == "parsed":
        return store.get_parsed(doc_id)
    try:
        parsed = parse.parse_file(_doc_path(store, settings, doc_id), doc)
    except Exception as exc:  # one bad document must not sink the scan
        log.exception("parse failed for %s (%s)", doc_id, doc.filename)
        store.set_parse_result(doc_id, "failed", error=f"{type(exc).__name__}: {exc}"[:500])
        run.counts["docs_failed"] += 1
        run.problems.append(f"{doc.filename}: {type(exc).__name__}: {exc}")
        return None
    store.save_parsed_text(doc_id, parsed.text, parsed.pages, parsed.lines)
    kind = parse.detect_kind(doc.filename, parsed.text)
    store.set_parse_result(doc_id, "parsed", pages=len(parsed.pages) or None, kind=kind)
    run.counts["docs_parsed"] += 1
    return store.get_parsed(doc_id)


def _store_drafts(store: Store, parsed: ParsedDocument, drafts: list[DraftFact], run: _Run) -> None:
    """Witnesses, merge and storage for one document's verified, dated drafts."""
    from pipeline import merge, witnesses

    case_id = parsed.doc.case_id
    # Witnesses first (SPEC §6.1 step 3): merge needs them to tell whether people agree.
    people = {d.key: witnesses.match_people(store, case_id, d.people) for d in drafts}
    existing = store.list_facts(case_id)
    names = {w.witness_id: [w.name, *w.aliases] for w in store.list_witnesses(case_id)}
    # draft key -> fact id, or None
    plan = merge.plan(drafts, existing, witness_ids=people, witness_names=names)
    key_to_fact: dict[str, str] = {}
    created: set[str] = set()
    for draft in drafts:
        witness_ids = people[draft.key]
        target = plan.get(draft.key)
        try:
            if target is not None:
                for span in draft.spans:
                    store.add_evidence(target, kind="span", by=AGENT, span=span)
                fact = store.get_fact(target)
                if fact is not None and fact.review == "draft" and witness_ids:
                    store.link_witnesses(target, witness_ids, by=AGENT)
                key_to_fact[draft.key] = target
                run.touched_fact_ids.append(target)
                run.counts["evidence_attached"] += 1
                continue
            fact = store.add_fact(
                case_id,
                draft.title,
                by=AGENT,
                details=draft.details,
                date_text=draft.date_text,
                date_start=draft.date_start,
                date_end=draft.date_end,
                date_certainty=draft.date_certainty,
                anchor_fact_id=draft.anchor_fact_id,
                witness_ids=witness_ids,
                confidence=draft.confidence,
                spans=draft.spans,
            )
        except (ValueError, KeyError, PermissionError) as exc:  # one bad draft: skip it
            log.warning("could not store draft %r from %s: %s", draft.title, parsed.doc.doc_id, exc)
            run.problems.append(f"{parsed.doc.filename}: draft not stored: {exc}")
            continue
        key_to_fact[draft.key] = fact.fact_id
        created.add(draft.key)
        run.new_fact_ids.append(fact.fact_id)
        run.counts["facts_created"] += 1
    # Relative dates anchored on another draft of this document: point at the stored fact.
    for draft in drafts:
        if draft.anchor_key and draft.anchor_key in key_to_fact and draft.key in created:
            fid = key_to_fact[draft.key]
            anchor = key_to_fact[draft.anchor_key]
            fact = store.get_fact(fid)
            if fact and fact.review == "draft" and fid != anchor and not fact.anchor_fact_id:
                store.update_fact(fid, {"anchor_fact_id": anchor}, by=AGENT)


def _draft_doc(store: Store, settings: Settings, parsed: ParsedDocument, run: _Run) -> None:
    from pipeline import dates, extract, parse, verify

    drafts = extract.draft_facts(parsed)
    run.counts["drafts"] += len(drafts)
    drafts, vcounts = verify.verify_drafts(parsed, drafts)
    for k in ("quotes_checked", "quotes_rejected", "drafts_discarded"):
        run.counts[k] += vcounts.get(k, 0)
    cap = settings.max_drafts_per_doc
    if cap and len(drafts) > cap:
        drafts = sorted(drafts, key=lambda d: -d.confidence)[:cap]
    drafts = dates.resolve_drafts(drafts, parse.doc_date(parsed), store.list_facts(run.job.case_id))
    _store_drafts(store, parsed, drafts, run)
    store.mark_drafted(parsed.doc.doc_id)
    run.processed_doc_ids.append(parsed.doc.doc_id)


def run_job(store: Store, job: Job, settings: Settings | None = None) -> Job:
    """Run a claimed job to completion. Never raises; failures land in the job row."""
    from pipeline import alerts, conflicts

    settings = settings or get_settings()
    run = _Run(job=job)
    case_id = job.case_id
    try:
        settings.check_layout()
        _discover(store, settings, run)
        todo = store.documents_to_process(case_id)
        store.update_job_counts(job.job_id, run.counts)
        for doc in todo:
            parsed = _parse_doc(store, settings, doc.doc_id, run)
            if parsed is not None:
                try:
                    _draft_doc(store, settings, parsed, run)
                except Exception as exc:
                    log.exception("drafting failed for %s", doc.doc_id)
                    run.problems.append(f"{doc.filename}: {type(exc).__name__}: {exc}")
            store.update_job_counts(job.job_id, run.counts)

        store.refresh_vectors(case_id)
        opened: list[Contradiction] = []
        changed = list(dict.fromkeys(run.new_fact_ids + run.touched_fact_ids))
        if changed:
            try:
                opened = conflicts.detect(store, case_id, changed)
            except Exception as exc:
                log.exception("contradiction check failed for case %s", case_id)
                run.problems.append(f"contradiction check: {type(exc).__name__}: {exc}")
        run.counts["contradictions"] = len(opened)
        made = alerts.compute(
            store,
            case_id,
            job_id=job.job_id,
            new_fact_ids=run.new_fact_ids,
            contradictions=opened,
            doc_ids=run.processed_doc_ids,
        )
        run.counts["alerts"] = len(made)

        error = None
        if run.problems:
            error = f"{len(run.problems)} problem(s): " + "; ".join(run.problems)
            error = error[:_MAX_ERROR_CHARS]
        failed = bool(todo) and not run.processed_doc_ids
        return store.finish_job(
            job.job_id, "failed" if failed else "done", error=error, counts=run.counts
        )
    except Exception as exc:
        log.exception("job %s failed", job.job_id)
        return store.finish_job(
            job.job_id,
            "failed",
            error=f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS],
            counts=run.counts,
        )


class Worker:
    """Background thread that runs queued jobs (started by the app server)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: Store | None = None,
        poll_seconds: float | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._own_store = store is None
        self.store = store or Store(self.settings.db_path)
        self.poll_seconds = (
            poll_seconds if poll_seconds is not None else (self.settings.worker_poll_seconds)
        )
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        n = self.store.requeue_running()
        if n:
            log.warning("re-queued %d job(s) left running by a previous process", n)
        self._thread = threading.Thread(target=self._loop, name="job-worker", daemon=True)
        self._thread.start()

    def wake(self) -> None:
        """Check the queue now instead of waiting for the next poll."""
        self._wake.set()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        if self._own_store:
            self.store.close()

    def run_pending(self) -> int:
        """Run queued jobs until none can be claimed; returns how many ran."""
        n = 0
        while not self._stop.is_set():
            job = self.store.claim_next_job()
            if job is None:
                return n
            log.info("job %s running (case %s)", job.job_id, job.case_id)
            done = run_job(self.store, job, self.settings)
            log.info("job %s %s: %s", done.job_id, done.status, done.counts)
            n += 1
        return n

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_pending()
            except Exception:  # pragma: no cover - keep the worker alive
                log.exception("worker loop error")
            self._wake.wait(self.poll_seconds)
            self._wake.clear()


def run_now(
    case_id: str | None = None, *, trigger: str = "ui", settings: Settings | None = None
) -> Job:
    """Queue a scan and run the queue inline (CLI, tests, ``make scan``)."""
    settings = settings or get_settings()
    cid = valid_case_id(case_id or settings.case_id)
    with Store(settings.db_path) as store:
        job = request_scan(store, cid, trigger)
        Worker(settings, store=store).run_pending()
        return store.get_job(job.job_id) or job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline.jobs")
    parser.add_argument("command", choices=["scan"])
    parser.add_argument("--case", dest="case_id", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    job = run_now(args.case_id)
    sys.stdout.write(job.model_dump_json(indent=2) + "\n")
    return 1 if job.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
