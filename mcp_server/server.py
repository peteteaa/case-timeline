"""Case Timeline MCP server (stdio).

Thin, read-mostly wrappers over ``pipeline.runner`` and ``pipeline.store`` for agents such as
OpenClaw. Stdout carries JSON-RPC only: logging goes to stderr and ``data_dir/mcp.log``, and
``sys.stdout`` is redirected to stderr while a tool runs. ``scan_case`` is the only tool that
writes, and it writes only to the data dir (never to the evidence folder).
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import io
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

log = logging.getLogger("mcp_server")

SERVER_NAME = "case-timeline"
DASHBOARD_BASE = "http://127.0.0.1:8090"
BACKEND_ENV = "CT_MCP_BACKEND"  # "pkg.module:factory" returning a Backend (tests/alt setups)
MAX_CONTEXT = 2000


# --------------------------------------------------------------------------- settings / logging


def _settings() -> Any:
    """Return pipeline settings, or an env-derived stand-in if ``pipeline.config`` is missing."""
    try:
        from pipeline.config import get_settings

        return get_settings()
    except ImportError:
        data_dir = Path(os.environ.get("CT_DATA_DIR", "./data"))
        return SimpleNamespace(
            data_dir=data_dir,
            db_path=Path(os.environ.get("CT_DB_PATH", data_dir / "case.db")),
            evidence_dir=Path(os.environ.get("CT_EVIDENCE_DIR", "./tests/fixtures/case_demo")),
            case_id=os.environ.get("CT_CASE_ID", "demo"),
        )


def _data_dir() -> Path:
    """Resolve the data dir without importing heavy modules (used before logging exists)."""
    env = os.environ.get("CT_DATA_DIR")
    if env:
        return Path(env).resolve()
    try:
        return Path(_settings().data_dir).resolve()
    except Exception:  # noqa: BLE001 - logging is not configured yet; fall back quietly
        return Path("./data").resolve()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def configure_logging() -> None:
    """Send all logging to stderr and ``data_dir/mcp.log``; never stdout."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    err = logging.StreamHandler(sys.stderr)
    err.setFormatter(fmt)
    root.addHandler(err)
    level = os.environ.get("CT_LOG_LEVEL", "INFO").upper()
    root.setLevel(level)

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
    """What the tools need: a Store-like object plus the two runner entry points."""

    store: Any
    scan_case: Callable[[str], Any]
    daily_digest: Callable[[str], dict]
    case_id: str
    data_dir: Path


_backend: Backend | None = None


def _default_backend() -> Backend:
    from pipeline import runner
    from pipeline.store import Store

    s = _settings()
    store = Store(s.db_path)
    store.init()
    return Backend(
        store=store,
        scan_case=runner.scan_case,
        daily_digest=runner.daily_digest,
        case_id=s.case_id,
        data_dir=Path(s.data_dir),
    )


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
    """Redirect stray prints to stderr and log failures for the duration of a tool call."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                return fn(*args, **kwargs)
            except Exception:
                log.exception("tool %s failed", fn.__name__)
                raise

    return wrapper


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _case(case_id: str | None) -> str:
    """The requested (or default) case id; rejects ids that could act as a path."""
    from pipeline.config import valid_case_id

    return valid_case_id(case_id or get_backend().case_id)


def _source_link(src: dict[str, Any]) -> str:
    return (
        f"{DASHBOARD_BASE}/source/{src['doc_id']}?start={src['char_start']}&end={src['char_end']}"
    )


def _event_view(event: Any) -> dict[str, Any]:
    d = _dump(event)
    d.pop("case_id", None)
    d["needs_attorney_review"] = bool(d.get("deadline_relevant"))
    src = d.get("source") or {}
    if src:
        src["link"] = _source_link(src)
    return d


def _parse_date(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


def _cursor_path(b: Backend) -> Path:
    return Path(b.data_dir) / "mcp_cursors.json"


def _get_cursor(b: Backend, case_id: str, consumer: str) -> datetime | None:
    getter = getattr(b.store, "get_cursor", None)
    if callable(getter):
        return getter(case_id=case_id, consumer=consumer)
    path = _cursor_path(b)
    try:
        raw = json.loads(path.read_text(encoding="utf-8")).get(f"{case_id}:{consumer}")
    except (OSError, ValueError):
        return None
    return datetime.fromisoformat(raw) if raw else None


def _set_cursor(b: Backend, case_id: str, consumer: str, ts: datetime) -> None:
    setter = getattr(b.store, "set_cursor", None)
    if callable(setter):
        setter(case_id=case_id, consumer=consumer, ts=ts)
        return
    from filelock import FileLock

    path = _cursor_path(b)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data[f"{case_id}:{consumer}"] = ts.isoformat()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)


def _changes_until(store: Any, case_id: str, since: datetime | None, until: datetime) -> Any:
    """Bound the window at ``until`` (the new cursor) when the store supports it.

    Without the upper bound, items stored between ``until`` and the query would be
    reported now and again on the next call.
    """
    try:
        accepts_until = "until" in inspect.signature(store.changes_since).parameters
    except (TypeError, ValueError):
        accepts_until = False
    if accepts_until:
        return store.changes_since(case_id, since, until=until)
    return store.changes_since(case_id, since)


def _split_parsed(parsed: Any) -> tuple[str, list[Any], str | None]:
    """Accept a ParsedDocument, ``(text, pages)`` tuple, or plain text."""
    if hasattr(parsed, "text"):
        doc = getattr(parsed, "doc", None)
        return parsed.text, list(getattr(parsed, "pages", []) or []), getattr(doc, "filename", None)
    if isinstance(parsed, tuple):
        text, pages = parsed[0], (parsed[1] if len(parsed) > 1 else [])
        return text, list(pages or []), None
    return str(parsed), [], None


def _page_for(pages: list[Any], offset: int) -> int | None:
    for p in pages:
        get = p.get if isinstance(p, dict) else lambda k, _p=p: getattr(_p, k)
        if get("char_start") <= offset < get("char_end"):
            return int(get("number"))
    return None


# --------------------------------------------------------------------------- server + tools

INSTRUCTIONS = """\
Case Timeline: a local, source-cited chronology of one legal case's evidence folder.
- Every event carries `source` (doc_id, page, char_start, char_end, quote). Cite it when stating
  facts; use get_source to show surrounding text. If no source exists, say so.
- Events with needs_attorney_review=true are deadline-relevant: always flag them for attorney
  review. Dates have a `certainty` (exact/approximate/relative/range/unknown); never add precision.
- Status 'proposed' means not yet reviewed by a human; 'unverified' means the quote was not found.
- scan_case is the only write tool. Nothing here edits or deletes evidence.
"""

mcp = None  # created lazily in build_server() so importing this module has no side effects


def build_server() -> Any:
    """Create the MCPServer with exactly the eight case-timeline tools."""
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS, version="0.1.0")
    ro = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    @_guarded
    def scan_case(case_id: str | None = None) -> dict[str, Any]:
        """Start a background scan of the case evidence folder; returns a job id immediately.

        Only new or changed documents are processed (parse, extract, verify, date, merge,
        contradiction check). Idempotent; only one scan runs at a time. Poll job_status until
        state is "done" or "failed", then call get_changes. Never modifies evidence files.
        """
        b = get_backend()
        cid = _case(case_id)
        job = _dump(b.scan_case(cid))
        log.info("scan_case case=%s job=%s", cid, job.get("id"))
        return {"job_id": job.get("id"), "case_id": cid, "state": job.get("state")}

    @server.tool(annotations=ro)
    @_guarded
    def job_status(job_id: str) -> dict[str, Any]:
        """Return a scan job's state (queued/running/done/failed), progress counts and error.

        Fields: state, docs_total, docs_done, events_added, contradictions_added, started_at,
        finished_at, error. Poll every few seconds until state is "done" or "failed".
        """
        job = get_backend().store.get_job(job_id)
        if job is None:
            return {"job_id": job_id, "found": False, "error": "unknown job id"}
        d = _dump(job)
        d["job_id"] = d.pop("id", job_id)
        d["found"] = True
        return d

    @server.tool()
    @_guarded
    def get_changes(case_id: str | None = None, consumer: str = "openclaw") -> dict[str, Any]:
        """Return events, contradictions and alerts added since this consumer last asked.

        Advances the consumer's cursor, so each item is reported once per consumer. Alerts
        contain no case content (type + local dashboard_url only) and are safe to relay to
        messaging. Every event includes a `source` citation; deadline-relevant events have
        needs_attorney_review=true.
        """
        b = get_backend()
        cid = _case(case_id)
        atomic = getattr(b.store, "get_changes", None)
        if callable(atomic):
            # Read + cursor advance in one write-locked transaction (no lost/double items).
            changes = atomic(cid, consumer)
            now = _get_cursor(b, cid, consumer) or datetime.now(UTC)
        else:
            since = _get_cursor(b, cid, consumer)
            now = datetime.now(UTC)
            changes = _changes_until(b.store, cid, since, now)
            _set_cursor(b, cid, consumer, now)
        d = _dump(changes)
        events = [_event_view(e) for e in changes.new_events]
        return {
            "case_id": cid,
            "consumer": consumer,
            "since": d.get("since"),
            "until": now.isoformat(),
            "counts": {
                "events": len(events),
                "contradictions": len(d.get("new_contradictions", [])),
                "alerts": len(d.get("new_alerts", [])),
                "needs_attorney_review": sum(e["needs_attorney_review"] for e in events),
            },
            "new_events": events,
            "new_contradictions": d.get("new_contradictions", []),
            "new_alerts": d.get("new_alerts", []),
        }

    @server.tool(annotations=ro)
    @_guarded
    def get_timeline(
        case_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Return the case chronology, sorted by date (undated events last).

        start/end: optional ISO dates (YYYY-MM-DD); events overlapping the window are kept and
        undated events are dropped when a window is given. status: proposed | accepted |
        rejected | unverified. Each event has date_text (as written), date_start/date_end,
        certainty, actors, description, confidence, status, needs_attorney_review, and a
        `source` citation (doc_id, page, char offsets, quote, dashboard link).
        """
        from pipeline.contracts import EventStatus

        cid = _case(case_id)
        lo, hi = _parse_date(start, "start"), _parse_date(end, "end")
        st = EventStatus(status) if status else None
        events = list(get_backend().store.list_events(cid, status=st))
        if lo or hi:
            kept = []
            for e in events:
                if e.date_start is None:
                    continue
                e_end = e.date_end or e.date_start
                if (lo and e_end < lo) or (hi and e.date_start > hi):
                    continue
                kept.append(e)
            events = kept
        events.sort(key=lambda e: (e.date_start is None, e.date_start or date.max, e.id))
        limit = max(1, min(int(limit), 1000))
        return {
            "case_id": cid,
            "total": len(events),
            "truncated": len(events) > limit,
            "events": [_event_view(e) for e in events[:limit]],
        }

    @server.tool(annotations=ro)
    @_guarded
    def search_events(query: str, k: int = 10, case_id: str | None = None) -> dict[str, Any]:
        """Hybrid keyword + semantic search over extracted events; returns the top k with sources.

        Use this to find evidence for a question, then cite each result's `source` (doc, page,
        quote) or call get_source for context. An empty result means no supporting source was
        found: say so rather than guessing.
        """
        cid = _case(case_id)
        k = max(1, min(int(k), 50))
        hits = get_backend().store.search_events(cid, query, k)
        return {"case_id": cid, "query": query, "events": [_event_view(e) for e in hits]}

    @server.tool(annotations=ro)
    @_guarded
    def get_source(
        doc_id: str, char_start: int, char_end: int, context: int = 300
    ) -> dict[str, Any]:
        """Return the exact quoted span from a source document plus surrounding text.

        Pass a SourceRef's doc_id/char_start/char_end. Returns quote (verbatim), before/after
        context (`context` chars each side, max 2000), page number, and `highlighted` with the
        quote wrapped in [[ ]]. Use this to verify and cite facts: doc + page + quote.
        """
        parsed = get_backend().store.get_parsed_text(doc_id)
        if parsed is None:
            return {"doc_id": doc_id, "found": False, "error": "unknown document id"}
        text, pages, filename = _split_parsed(parsed)
        if not (0 <= char_start < char_end <= len(text)):
            return {
                "doc_id": doc_id,
                "found": False,
                "error": f"offsets out of range (document length {len(text)})",
            }
        ctx = max(0, min(int(context), MAX_CONTEXT))
        lo, hi = max(0, char_start - ctx), min(len(text), char_end + ctx)
        before, quote, after = text[lo:char_start], text[char_start:char_end], text[char_end:hi]
        return {
            "doc_id": doc_id,
            "found": True,
            "filename": filename,
            "page": _page_for(pages, char_start),
            "char_start": char_start,
            "char_end": char_end,
            "quote": quote,
            "before": before,
            "after": after,
            "highlighted": f"{'…' if lo > 0 else ''}{before}[[{quote}]]{after}"
            f"{'…' if hi < len(text) else ''}",
            "link": _source_link(
                {"doc_id": doc_id, "char_start": char_start, "char_end": char_end}
            ),
        }

    @server.tool(annotations=ro)
    @_guarded
    def list_contradictions(
        status: str | None = None, case_id: str | None = None
    ) -> dict[str, Any]:
        """List contradictions between events (conflicting date, actor, or fact).

        status: open | dismissed | confirmed (default: all). Each item includes the explanation
        and the conflicting events with their source citations. Contradictions are flags for
        human review, not findings.
        """
        from pipeline.contracts import ContradictionStatus

        b = get_backend()
        cid = _case(case_id)
        st = ContradictionStatus(status) if status else None
        items = []
        for c in b.store.list_contradictions(cid, status=st):
            d = _dump(c)
            d.pop("case_id", None)
            events = [b.store.get_event(eid) for eid in c.event_ids]
            d["events"] = [_event_view(e) for e in events if e is not None]
            items.append(d)
        return {"case_id": cid, "count": len(items), "contradictions": items}

    @server.tool(annotations=ro)
    @_guarded
    def daily_digest(case_id: str | None = None) -> dict[str, Any]:
        """Return counts-only summary for the case (no case content) plus the dashboard link.

        Safe to relay to messaging channels: send only the case name, the counts/alert types,
        and the local dashboard link. Details stay on the local dashboard.
        """
        cid = _case(case_id)
        digest = dict(get_backend().daily_digest(cid) or {})
        digest.setdefault("case_id", cid)
        digest.setdefault("dashboard_url", f"{DASHBOARD_BASE}/alerts")
        return digest

    return server


def _unbuffered_stdout() -> io.TextIOWrapper:
    """A ``sys.stdout`` that writes straight to fd 1, with no Python-level buffer.

    While serving, the SDK points fd 1 at stderr and keeps a private duplicate for the
    wire. A buffered ``sys.stdout`` could hold stray prints (e.g. from the background
    scan thread) and flush them at exit, after fd 1 is restored to the wire; writing
    through avoids that, so stray output always lands on stderr.
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
