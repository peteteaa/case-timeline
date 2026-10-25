"""Case Timeline MCP server (stdio), SPEC §8.2.

Thin wrappers over ``pipeline.store`` and friends for the OpenClaw agent. The server never runs
the pipeline: ``scan_case`` only adds a row to the jobs table, and the app server's worker runs
it. ``scan_case`` and ``mark_alerts_delivered`` are the only tools that write, and they write
only job and alert rows. No tool returns a rejected fact.

Stdout carries JSON-RPC only: logging goes to stderr and ``data_dir/mcp.log``, and
``sys.stdout`` is redirected to stderr while a tool runs.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import io
import logging
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("mcp_server")

SERVER_NAME = "case-timeline"
SERVER_VERSION = "0.2.0"
BACKEND_ENV = "CT_MCP_BACKEND"  # "pkg.module:factory" returning a Backend (tests/alt setups)
CONTEXT_CHARS = 300  # surrounding text on each side of a quote in get_fact
MAX_LIMIT = 1000
STALE_QUEUE_SECONDS = 120  # a job still queued after this suggests no worker is running
FUZZY_WITNESS_SCORE = 85.0

ALERT_LABELS = {
    "contradiction": "possible conflict between sources",
    "flagged_witness": "flagged witness mentioned in a new document",
    "possible_deadline": "possible deadline (needs attorney review)",
}
RELAY_POLICY = (
    "Outside the UI chat, relay only each alert's `message` (case name, alert type, local "
    "dashboard link). Never relay anything under `internal`: fact and document ids, names, "
    "dates, quotes or other case content."
)


# --------------------------------------------------------------------------- logging


def _data_dir() -> Path:
    """Resolve the data dir without touching the database (used before logging exists)."""
    env = os.environ.get("CT_DATA_DIR")
    if env:
        return Path(env).resolve()
    try:
        from pipeline.config import get_settings

        return Path(get_settings().data_dir).resolve()
    except Exception:  # noqa: BLE001 - logging is not configured yet; fall back quietly
        return Path("./data").resolve()


def _inside(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def configure_logging() -> None:
    """Send all logging to stderr and ``data_dir/mcp.log``; never stdout."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    err = logging.StreamHandler(sys.stderr)
    err.setFormatter(fmt)
    root.addHandler(err)
    root.setLevel(os.environ.get("CT_LOG_LEVEL", "INFO").upper())

    data_dir = _data_dir()
    evidence = os.environ.get("CT_EVIDENCE_DIR")
    if evidence and _inside(data_dir, Path(evidence)):
        log.warning("data_dir is inside the evidence dir; not writing mcp.log")
        return
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(data_dir / "mcp.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError as exc:
        log.warning("cannot open %s/mcp.log: %s", data_dir, exc)


# --------------------------------------------------------------------------- backend


@dataclass
class Backend:
    """What the tools need: the store, the default case, and the local dashboard URL."""

    store: Any  # pipeline.store.Store
    case_id: str
    dashboard: str = "http://127.0.0.1:8080"


_backend: Backend | None = None


def _default_backend() -> Backend:
    from pipeline.config import get_settings
    from pipeline.store import Store

    s = get_settings()
    return Backend(store=Store(s.db_path), case_id=s.case_id, dashboard=s.ui_base_url)


def get_backend() -> Backend:
    """Build the backend once; ``CT_MCP_BACKEND=module:factory`` overrides the default."""
    global _backend
    if _backend is None:
        spec = os.environ.get(BACKEND_ENV)
        if spec:
            mod_name, _, attr = spec.partition(":")
            factory = getattr(importlib.import_module(mod_name), attr or "make_backend")
            _backend = factory()
            log.info("using backend override %s", spec)
        else:
            _backend = _default_backend()
    return _backend


def set_backend(backend: Backend | None) -> None:
    """Test hook: install (or clear) the backend used by the tools."""
    global _backend
    _backend = backend


# --------------------------------------------------------------------------- helpers


def _guarded[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Redirect stray prints to stderr and log failures for the duration of a tool call.

    A ``ValueError`` is a bad argument (case id, date, status): it becomes a ``ToolError`` so
    the agent sees the message. Anything else is a crash, and the SDK hides its text.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                return fn(*args, **kwargs)
            except ValueError as exc:
                from mcp.server.mcpserver.exceptions import ToolError

                log.info("tool %s rejected its arguments: %s", fn.__name__, exc)
                raise ToolError(str(exc)) from exc
            except Exception:
                log.exception("tool %s failed", fn.__name__)
                raise

    return wrapper


def _case(case_id: str | None) -> str:
    """The requested (or default) case id; rejects ids that could act as a path."""
    from pipeline.config import valid_case_id

    return valid_case_id(case_id or get_backend().case_id)


def _parse_date(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


def _iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _live(facts: Iterable[Any]) -> list[Any]:
    """Drop rejected facts (SPEC §8.2: tools never return them)."""
    return [f for f in facts if f.review != "rejected"]


def _cite(filename: str | None, ev: Any) -> str:
    """Human citation: ``depo.txt 12:4-9`` for transcript lines, ``memo.pdf p. 3`` otherwise."""
    name = filename or ev.doc_id or "?"
    if ev.kind == "external":
        return ev.reference or ""
    if ev.kind == "document":
        return name
    if ev.line_start is not None and ev.page is not None:
        lines = f"{ev.line_start}" + (
            f"-{ev.line_end}" if ev.line_end not in (None, ev.line_start) else ""
        )
        return f"{name} {ev.page}:{lines}"
    return f"{name} p. {ev.page}" if ev.page is not None else name


@dataclass
class _View:
    """Lookups shared by every fact rendered in one tool call."""

    store: Any
    case_id: str
    dashboard: str
    docs: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    witnesses: dict[str, str] = field(default_factory=dict)
    alleged: set[str] = field(default_factory=set)
    conflicted: set[str] = field(default_factory=set)

    @classmethod
    def build(cls, b: Backend, case_id: str) -> _View:
        s = b.store
        return cls(
            store=s,
            case_id=case_id,
            dashboard=b.dashboard,
            docs={d.doc_id: d for d in s.list_documents(case_id)},
            tags={t.tag_id: t.name for t in s.list_tags(case_id)},
            witnesses={w.witness_id: w.name for w in s.list_witnesses(case_id)},
            alleged=s.alleged_fact_ids(case_id),
            conflicted=s.conflicted_fact_ids(case_id),
        )

    def doc(self, doc_id: str | None) -> Any:
        if doc_id is None:
            return None
        if doc_id not in self.docs:
            self.docs[doc_id] = self.store.get_document(doc_id)
        return self.docs[doc_id]

    def evidence(self, ev: Any) -> dict[str, Any]:
        doc = self.doc(ev.doc_id)
        filename = doc.filename if doc else None
        out: dict[str, Any] = {
            "evidence_id": ev.evidence_id,
            "kind": ev.kind,
            "review": ev.review,
            "origin": ev.origin,
            "cite": _cite(filename, ev),
        }
        if ev.kind == "external":
            out["reference"] = ev.reference
            return out
        out.update(
            doc_id=ev.doc_id,
            filename=filename,
            doc_kind=doc.kind if doc else None,
        )
        if ev.kind == "span":
            out.update(
                page=ev.page,
                line_start=ev.line_start,
                line_end=ev.line_end,
                quote=ev.quote,
            )
        out["source_link"] = f"{self.dashboard}/documents/{ev.doc_id}/view" + (
            f"?evidence={ev.evidence_id}" if ev.kind == "span" else ""
        )
        return out

    def fact(self, f: Any) -> dict[str, Any]:
        return {
            "fact_id": f.fact_id,
            "title": f.title,
            "details": f.details,
            "review": f.review,
            "origin": f.origin,
            "confidence": f.confidence,
            "date_text": f.date_text,
            "date_start": _iso(f.date_start),
            "date_end": _iso(f.date_end),
            "date_certainty": f.date_certainty,
            "anchor_fact_id": f.anchor_fact_id,
            "statuses": list(f.statuses),
            "disputed": "disputed" in f.statuses,
            "tags": [self.tags.get(t, t) for t in f.tag_ids],
            "witnesses": [self.witnesses.get(w, w) for w in f.witness_ids],
            "alleged": f.fact_id in self.alleged,
            "possible_conflict": f.fact_id in self.conflicted,
            "evidence": [self.evidence(e) for e in f.evidence],
        }


def _resolve_witness(store: Any, case_id: str, name: str) -> Any:
    """Witness by id, exact name/alias (``store.find_witness``), then a fuzzy match."""
    name = (name or "").strip()
    if not name:
        return None
    w = store.get_witness(name)
    if w is not None and w.case_id == case_id:
        return w
    w = store.find_witness(case_id, name)
    if w is not None:
        return w
    from rapidfuzz import fuzz

    best, best_score = None, 0.0
    for cand in store.list_witnesses(case_id):
        score = max(
            fuzz.token_set_ratio(name.casefold(), n.casefold()) for n in [cand.name, *cand.aliases]
        )
        score += 0.5 * cand.confirmed  # prefer confirmed witnesses on ties
        if score > best_score:
            best, best_score = cand, score
    return best if best_score >= FUZZY_WITNESS_SCORE else None


def _resolve_tag(store: Any, case_id: str, value: str) -> Any:
    value = (value or "").strip()
    for t in store.list_tags(case_id):
        if t.tag_id == value or t.name.casefold() == value.casefold():
            return t
    return None


def _context(parsed: Any, start: int | None, end: int | None) -> dict[str, Any]:
    """Text around ``[start, end)``: ~CONTEXT_CHARS before and after, plus a highlighted copy."""
    if parsed is None or start is None or end is None:
        return {}
    text = parsed.text
    lo, hi = max(0, start - CONTEXT_CHARS), min(len(text), end + CONTEXT_CHARS)
    before, quote, after = text[lo:start], text[start:end], text[end:hi]
    return {
        "before": before,
        "after": after,
        "highlighted": f"{'…' if lo > 0 else ''}{before}[[{quote}]]{after}"
        f"{'…' if hi < len(text) else ''}",
    }


def _alert_view(alert: Any, case_id: str, dashboard: str, live_ids: set[str]) -> dict[str, Any]:
    url = f"{dashboard}/cases/{case_id}/alerts"
    label = ALERT_LABELS.get(alert.type, alert.type)
    return {
        "alert_id": alert.alert_id,
        "type": alert.type,
        "label": label,
        "needs_attorney_review": alert.type == "possible_deadline",
        "created_at": _iso(alert.created_at),
        "dashboard_url": url,
        "message": f"Case {case_id}: {label}. Review: {url}",
        "internal": {
            "do_not_relay": True,
            "fact_ids": [i for i in alert.fact_ids if i in live_ids],
            "doc_ids": list(alert.doc_ids),
        },
    }


# --------------------------------------------------------------------------- server + tools

INSTRUCTIONS = """\
Case Timeline: a local, source-cited chronology (facts, not documents) for a legal case.
- scan_case only queues a job; the app server runs it. Poll job_status until done or failed.
- Facts carry `review` (draft = unreviewed AI output, accepted = a person kept it). Prefer
  accepted facts and say so when you cite a draft. Cite fact IDs plus document and page
  (transcripts: page:line) from `evidence[].cite`. If no fact supports an answer, say so.
- Call a fact disputed only if `disputed` is true. `possible_conflict` (an open
  contradiction) is a "possible conflict", not a dispute. `alleged` facts rest only on
  pleadings: call them allegations.
- Alerts: outside the UI chat relay only each alert's `message` (case name, alert type,
  local dashboard link). Never relay `internal` ids or any case content. Possible deadlines
  always need attorney review.
- You cannot create, edit or review facts; people do that in the web UI.
"""

mcp = None  # created lazily in build_server() so importing this module has no side effects


def build_server() -> Any:
    """Create the MCPServer with the SPEC §8.2 tools."""
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS, version=SERVER_VERSION)
    ro = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
    writes_rows = ToolAnnotations(
        read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False
    )

    @server.tool(annotations=writes_rows)
    @_guarded
    def scan_case(case_id: str | None = None) -> dict[str, Any]:
        """Queue a scan of the case's evidence folder and upload inbox; returns a job_id at once.

        Never blocks and never processes documents itself: the app server's job worker runs
        the job (new or changed documents only). If a scan is already queued for the case, its
        job_id is returned. Poll job_status(job_id) until status is "done" or "failed", then
        call get_pending_alerts. Writes only a job row; never touches evidence.
        """
        from pipeline import jobs

        b = get_backend()
        cid = _case(case_id)
        job = jobs.request_scan(b.store, cid, "agent")
        log.info("scan_case case=%s job=%s", cid, job.job_id)
        return {
            "job_id": job.job_id,
            "case_id": cid,
            "status": job.status,
            "queued_at": _iso(job.queued_at),
        }

    @server.tool(annotations=ro)
    @_guarded
    def job_status(job_id: str) -> dict[str, Any]:
        """Return a job's status (queued/running/done/failed), progress counts and error.

        counts include docs_seen, docs_new, docs_parsed, docs_failed, drafts, quotes_checked,
        quotes_rejected, facts_created, evidence_attached, contradictions and alerts. Poll every
        few seconds until status is "done" or "failed".
        """
        job = get_backend().store.get_job(job_id)
        if job is None:
            return {"job_id": job_id, "found": False, "error": "unknown job id"}
        out = {"found": True, **job.model_dump(mode="json")}
        if job.status == "queued":
            waited = (datetime.now(UTC) - job.queued_at).total_seconds()
            if waited > STALE_QUEUE_SECONDS:
                out["hint"] = (
                    f"queued for {int(waited)}s: the app server's job worker may not be running"
                )
        return out

    @server.tool(annotations=ro)
    @_guarded
    def get_timeline(
        case_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        witness: str | None = None,
        tag: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Facts in date order (undated last), each with its review state and evidence.

        start/end: ISO dates (YYYY-MM-DD); facts overlapping the window are kept and undated
        facts are dropped when a window is given. witness: a witness name, alias or id. tag: an
        issue tag name or id. status: key_fact | case_impact | disputed. Each fact has review
        (draft/accepted), date_text/date_start/date_end/date_certainty, statuses, tags,
        witnesses, alleged, possible_conflict, and evidence (document filename, page, line,
        quote, `cite`).
        """
        from pipeline.schema import STATUSES, FactQuery

        b = get_backend()
        cid = _case(case_id)
        q = FactQuery(
            date_from=_parse_date(start, "start"),
            date_to=_parse_date(end, "end"),
            reviews=["draft", "accepted"],
        )
        filters: dict[str, Any] = {}
        if status:
            if status not in STATUSES:
                raise ValueError(f"status must be one of {', '.join(STATUSES)}; got {status!r}")
            q.statuses = [status]  # type: ignore[list-item]
            filters["status"] = status
        if witness:
            w = _resolve_witness(b.store, cid, witness)
            if w is None:
                return {
                    "case_id": cid,
                    "total": 0,
                    "facts": [],
                    "error": f"no witness matches {witness!r}",
                }
            q.witness_ids = [w.witness_id]
            filters["witness"] = w.name
        if tag:
            t = _resolve_tag(b.store, cid, tag)
            if t is None:
                return {
                    "case_id": cid,
                    "total": 0,
                    "facts": [],
                    "error": f"no issue tag matches {tag!r}",
                }
            q.tag_ids = [t.tag_id]
            filters["tag"] = t.name
        facts = _live(b.store.list_facts(cid, q))
        limit = max(1, min(int(limit), MAX_LIMIT))
        view = _View.build(b, cid)
        return {
            "case_id": cid,
            "filters": filters,
            "total": len(facts),
            "truncated": len(facts) > limit,
            "facts": [view.fact(f) for f in facts[:limit]],
        }

    @server.tool(annotations=ro)
    @_guarded
    def search_facts(query: str, case_id: str | None = None, k: int = 20) -> dict[str, Any]:
        """Westlaw-style keyword search plus semantic search over fact titles, details, notes
        and evidence quotes; returns the best facts first.

        Syntax: space = OR, & = AND, "exact phrase", /n = within n words, /s same sentence,
        /p same paragraph, % = but not, ! = root expander (contract!). Plain questions work too
        (semantic match). An empty result means no fact supports the question: say so.
        """
        from pipeline import search

        b = get_backend()
        cid = _case(case_id)
        k = max(1, min(int(k), 100))
        try:
            ids = search.search_facts(b.store, cid, query, k)
        except search.QueryError as exc:
            return {
                "case_id": cid,
                "query": query,
                "facts": [],
                "error": str(exc),
                "syntax_help": search.SYNTAX_HELP,
            }
        facts = [f for f in _live(b.store.get_facts(ids)) if f.case_id == cid]
        view = _View.build(b, cid)
        return {"case_id": cid, "query": query, "facts": [view.fact(f) for f in facts]}

    @server.tool(annotations=ro)
    @_guarded
    def get_fact(fact_id: str) -> dict[str, Any]:
        """One fact with every quote in context: document filename, page (page:line for
        transcripts), the verbatim quote, ~300 characters before and after, and `highlighted`
        with the quote wrapped in [[ ]]. Also returns notes, open conflicts and the anchor fact
        for relative dates. Use it to check a quote really supports a claim before citing it.
        """
        b = get_backend()
        fact = b.store.get_fact(fact_id)
        if fact is None or fact.review == "rejected":
            return {"fact_id": fact_id, "found": False, "error": "unknown fact id"}
        view = _View.build(b, fact.case_id)
        out = {"found": True, "case_id": fact.case_id, **view.fact(fact)}
        parsed_cache: dict[str, Any] = {}
        for ev_out, ev in zip(out["evidence"], fact.evidence, strict=True):
            if ev.kind != "span" or ev.doc_id is None:
                continue
            if ev.doc_id not in parsed_cache:
                parsed_cache[ev.doc_id] = b.store.get_parsed(ev.doc_id)
            ev_out.update(_context(parsed_cache[ev.doc_id], ev.char_start, ev.char_end))
        out["notes"] = [
            {"author": n.author, "body": n.body, "created_at": _iso(n.created_at)}
            for n in b.store.list_notes(fact_id)
        ]
        conflicts = []
        for c in b.store.list_contradictions(fact.case_id, "open", fact_id=fact_id):
            others = [
                f.fact_id for f in _live(b.store.get_facts(c.fact_ids)) if f.fact_id != fact_id
            ]
            if others:
                conflicts.append(
                    {
                        "contradiction_id": c.contradiction_id,
                        "field": c.field,
                        "explanation": c.explanation,
                        "other_fact_ids": others,
                    }
                )
        out["possible_conflicts"] = conflicts
        if fact.anchor_fact_id:
            anchor = b.store.get_fact(fact.anchor_fact_id)
            if anchor is not None and anchor.review != "rejected":
                out["anchor"] = {
                    "fact_id": anchor.fact_id,
                    "title": anchor.title,
                    "date_start": _iso(anchor.date_start),
                }
        return out

    @server.tool(annotations=ro)
    @_guarded
    def get_witness(name: str, case_id: str | None = None) -> dict[str, Any]:
        """A witness profile matched by name or alias (fuzzy fallback, e.g. "Smith" finds
        "John Smith"): name, aliases, role, notes, flagged, confirmed (false = AI suggestion),
        the witness's facts in date order and the documents that mention them.
        """
        b = get_backend()
        cid = _case(case_id)
        w = _resolve_witness(b.store, cid, name)
        if w is None:
            return {
                "case_id": cid,
                "found": False,
                "error": f"no witness matches {name!r}",
                "witnesses": [x.name for x in b.store.list_witnesses(cid)][:50],
            }
        view = _View.build(b, cid)
        return {
            "case_id": cid,
            "found": True,
            "witness": w.model_dump(mode="json", exclude={"case_id"}),
            "facts": [view.fact(f) for f in _live(b.store.witness_facts(w.witness_id))],
            "documents": [
                {"doc_id": d.doc_id, "filename": d.filename, "kind": d.kind}
                for d in b.store.witness_documents(w.witness_id)
            ],
        }

    @server.tool(annotations=ro)
    @_guarded
    def list_contradictions(case_id: str | None = None, open_only: bool = True) -> dict[str, Any]:
        """Conflicts between sources (date, actor or fact), each with both facts and their quotes.

        open_only (default true) lists only open conflicts: call those "possible conflicts".
        status "confirmed" means a reviewer confirmed it and the facts are disputed;
        "dismissed" means a reviewer rejected the conflict.
        """
        b = get_backend()
        cid = _case(case_id)
        view = _View.build(b, cid)
        items = []
        for c in b.store.list_contradictions(cid, "open" if open_only else None):
            facts = _live(b.store.get_facts(c.fact_ids))
            if len(facts) < 2:  # one side was rejected: nothing left to conflict with
                continue
            items.append(
                {
                    "contradiction_id": c.contradiction_id,
                    "field": c.field,
                    "status": c.status,
                    "label": {
                        "open": "possible conflict",
                        "confirmed": "disputed",
                        "dismissed": "dismissed",
                    }[c.status],
                    "explanation": c.explanation,
                    "facts": [view.fact(f) for f in facts],
                }
            )
        return {"case_id": cid, "count": len(items), "contradictions": items}

    @server.tool(annotations=ro)
    @_guarded
    def get_pending_alerts(case_id: str | None = None) -> dict[str, Any]:
        """Alerts not yet delivered (newest first), each with a ready-to-send, content-free
        `message` (case name, alert type, local dashboard link). Types: contradiction,
        flagged_witness, possible_deadline (always "needs attorney review"). Fields under
        `internal` are for the UI chat only and must never be relayed. After announcing alerts
        on a messaging channel, call mark_alerts_delivered with their alert_ids.
        """
        b = get_backend()
        cid = _case(case_id)
        pending = b.store.list_alerts(cid, pending_only=True)
        all_ids = sorted({i for a in pending for i in a.fact_ids})
        live_ids = {f.fact_id for f in _live(b.store.get_facts(all_ids))}
        alerts = []
        for a in pending:
            stale = a.fact_ids and not any(i in live_ids for i in a.fact_ids)
            if stale and a.type != "flagged_witness":
                continue  # a reviewer rejected every fact behind it
            alerts.append(_alert_view(a, cid, b.dashboard, live_ids))
        return {
            "case_id": cid,
            "count": len(alerts),
            "dashboard_url": f"{b.dashboard}/cases/{cid}/alerts",
            "relay_policy": RELAY_POLICY,
            "alerts": alerts,
        }

    @server.tool(annotations=writes_rows)
    @_guarded
    def mark_alerts_delivered(alert_ids: list[str]) -> dict[str, Any]:
        """Record that these alerts were announced, so they are not announced again.

        Idempotent: alerts already marked are left alone. Returns how many changed.
        Writes only alert rows.
        """
        ids = [str(i) for i in alert_ids if str(i).strip()]
        n = get_backend().store.mark_alerts_delivered(ids)
        log.info("mark_alerts_delivered %d of %d", n, len(ids))
        return {"requested": len(ids), "marked": n}

    @server.tool(annotations=ro)
    @_guarded
    def daily_digest(case_id: str | None = None) -> dict[str, Any]:
        """Summary of the last 24 hours (new facts, reviews, conflicts, alerts, documents) as
        text with fact IDs; the same function behind the UI's "Today" panel.

        The text contains case content: use it in the UI chat only. On a messaging channel send
        only `message` (case name, counts, local dashboard link).
        """
        from pipeline import digest as digest_mod

        b = get_backend()
        cid = _case(case_id)
        d = dict(digest_mod.daily_digest(b.store, cid) or {})
        ids = list(d.get("fact_ids") or [])
        live = {f.fact_id for f in _live(b.store.get_facts(ids))}
        d["fact_ids"] = [i for i in ids if i in live]
        d["case_id"] = cid
        url = f"{b.dashboard}/cases/{cid}"
        d["dashboard_url"] = url
        counts = d.get("counts") or {}
        summary = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items() if v)
        d["message"] = f"Case {cid} daily digest: {summary or 'no changes'}. Review: {url}"
        d["relay_policy"] = "Relay only `message` outside the UI chat; `text` has case content."
        return d

    return server


def _unbuffered_stdout() -> io.TextIOWrapper:
    """A ``sys.stdout`` that writes straight to fd 1, with no Python-level buffer.

    While serving, the SDK points fd 1 at stderr and keeps a private duplicate for the
    wire. A buffered ``sys.stdout`` could hold stray prints and flush them at exit, after
    fd 1 is restored to the wire; writing through avoids that, so stray output always lands
    on stderr.
    """
    raw = io.FileIO(1, "w", closefd=False)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace", write_through=True)


def main() -> None:
    """Entry point: configure logging first, then serve over stdio."""
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # anything printed during startup goes to stderr
    try:
        configure_logging()
        global mcp
        mcp = build_server()
        log.info("starting %s MCP server (stdio)", SERVER_NAME)
        real_stdout.flush()
        sys.stdout = _unbuffered_stdout()
        mcp.run("stdio")
    finally:
        sys.stdout = sys.stderr  # never let late output reach the (restored) wire
