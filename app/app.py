"""Web UI and job-worker host (SPEC §5.2, §6, §7): FastAPI + Jinja2 + HTMX + vis-timeline.

Every read and write goes through ``pipeline.store``; search, export and the daily digest come
from their ``pipeline`` modules. All assets are vendored, the app binds 127.0.0.1 only, and a
display name kept in the ``ct_user`` cookie is the author of every write.

Uploads are saved to the writable inbox (``settings.inbox``), never the evidence folder.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.helpers import (
    ALERT_LABELS,
    FIELD_LABELS,
    LOOPBACK_HOSTS,
    SORT_KEYS,
    SORT_LABELS,
    STATUS_LABELS,
    USER_COOKIE,
    ParamGetter,
    build_query,
    clean_display_name,
    decode_user_cookie,
    encode_user_cookie,
    evidence_cite,
    fact_changes,
    fact_date,
    fact_ids_in,
    filter_pairs,
    filters_active,
    fmt_date,
    fmt_dt,
    history_value,
    host_is_loopback,
    hostname,
    linkify_fact_ids,
    parse_filters,
    parse_int,
    pct,
    query_string,
    safe_next,
    sanitize_filename,
    source_segments,
    valid_color,
)
from pipeline import digest as digest_mod
from pipeline import export as export_mod
from pipeline import search as search_mod
from pipeline.jobs import request_scan
from pipeline.schema import (
    CERTAINTIES,
    DOC_KINDS,
    STATUSES,
    Document,
    Fact,
    FactQuery,
    Job,
    SpanIn,
    Witness,
)

if TYPE_CHECKING:
    from pipeline.config import Settings
    from pipeline.store import Store

__all__ = ["create_app", "host_is_loopback", "sanitize_filename"]

log = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".eml"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_SPAN_CHARS = 20_000
OPENCLAW_DEFAULT = Path("~/.openclaw/bin/openclaw").expanduser()
OPENCLAW_TIMEOUT_S = 180
CHAT_TOP_K = 8

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'self'"
)
EXPORT_CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; sandbox"

EXPORT_TYPES = {
    "csv": ("text/csv; charset=utf-8", "csv"),
    "xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
    "svg": ("image/svg+xml", "svg"),
    "png": ("image/png", "png"),
}
COUNT_LABELS = {
    "docs_seen": "Files seen",
    "docs_new": "New documents",
    "docs_parsed": "Parsed",
    "docs_failed": "Failed to parse",
    "drafts": "Drafts proposed",
    "quotes_checked": "Quotes checked",
    "quotes_rejected": "Quotes rejected",
    "drafts_discarded": "Drafts discarded",
    "facts_created": "Facts created",
    "evidence_attached": "Evidence merged",
    "contradictions": "Contradictions",
    "alerts": "Alerts",
}

CHAT_SYSTEM = (
    "You are a careful litigation-support assistant. Answer ONLY from the case facts provided. "
    "Cite every factual statement with its fact ID in square brackets, e.g. [F-0012]. Prefer "
    "accepted facts; when you rely on an unreviewed draft, say so. Call a fact disputed only if "
    "it has the disputed status; an open contradiction is only a possible conflict. If the facts "
    "do not answer the question, say so plainly. Never invent dates. Flag any deadline or "
    "limitation date as 'needs attorney review'."
)


# --------------------------------------------------------------------------- dependencies


@dataclass
class Deps:
    """Collaborators, created lazily; tests replace these attributes with fakes."""

    settings: Any
    store: Any = None
    llm: Any = None
    worker: Any = None
    openclaw_bin: Path | None = None
    chat_backend: str = "auto"  # auto | openclaw | local
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_store(self) -> Store:
        with self._lock:
            if self.store is None:
                from pipeline.store import Store

                self.store = Store(self.settings.db_path)
            return self.store

    def get_llm(self) -> Any:
        if self.llm is None:
            from pipeline import llm

            self.llm = llm
        return self.llm

    def wake_worker(self) -> None:
        if self.worker is not None:
            self.worker.wake()


class NeedName(Exception):
    """A write was attempted without a display name."""


# --------------------------------------------------------------------------- small helpers


def _deps(request: Request) -> Deps:
    return request.app.state.deps


def _store(request: Request) -> Store:
    return _deps(request).get_store()


def _is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request", "").lower() == "true"


def _wants_json(request: Request) -> bool:
    ctype = request.headers.get("content-type", "")
    return ctype.startswith("application/json")


def current_user(request: Request) -> str | None:
    return decode_user_cookie(request.cookies.get(USER_COOKIE))


def require_user(request: Request) -> str:
    user = current_user(request)
    if user is None:
        raise NeedName()
    return user


@contextmanager
def store_errors() -> Iterator[None]:
    """Map the store's exceptions to HTTP errors: 403, 404, 422."""
    try:
        yield
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, f"not found: {exc.args[0] if exc.args else ''}") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


async def _params(request: Request) -> ParamGetter:
    """Form or JSON body as a ParamGetter."""
    if _wants_json(request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(422, "invalid JSON body") from exc
        if not isinstance(body, dict):
            raise HTTPException(422, "expected a JSON object")
        return ParamGetter(body)
    return ParamGetter(await request.form())


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def _unique_path(directory: Path, name: str) -> Path:
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate = directory / name
    n = 1
    while candidate.exists():
        candidate = directory / f"{stem}-{n}{suffix}"
        n += 1
    return candidate


def _extract_openclaw_text(stdout: str) -> str:
    """Best-effort reply extraction from ``openclaw agent --json`` output."""
    stdout = stdout.strip()
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout
    keys = ("reply", "text", "message", "content", "output", "result", "response", "payloads")

    def walk(node: Any) -> str | None:
        if isinstance(node, str):
            return node.strip() or None
        if isinstance(node, dict):
            for k in keys:
                if k in node and (found := walk(node[k])):
                    return found
        if isinstance(node, list):
            parts = [t for t in (walk(x) for x in node) if t]
            return "\n\n".join(parts) or None
        return None

    return walk(data) or stdout


def doc_view_url(doc_id: str, evidence_id: str | None = None) -> str:
    url = f"/documents/{quote(doc_id, safe='')}/view"
    return url + (f"?evidence={quote(evidence_id, safe='')}" if evidence_id else "")


def fact_url(fact_id: str) -> str:
    return f"/facts/{quote(fact_id, safe='')}"


# --------------------------------------------------------------------------- app factory


def create_app(settings: Settings | None = None, *, store: Store | None = None) -> FastAPI:
    """Build the app. Starts one job worker on startup when ``settings.run_worker`` is true."""
    if settings is None:
        from pipeline.config import get_settings

        settings = get_settings()
    case = settings.case_id

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        deps: Deps = app.state.deps
        if settings.run_worker and deps.worker is None:
            from pipeline.jobs import Worker

            deps.worker = Worker(settings)
            deps.worker.start()
            log.info("job worker started")
        try:
            yield
        finally:
            if deps.worker is not None:
                deps.worker.stop()
                deps.worker = None

    app = FastAPI(
        title="Case Timeline", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    backend = os.environ.get("CT_CHAT_BACKEND", "auto").lower()
    app.state.deps = Deps(
        settings=settings,
        store=store,
        openclaw_bin=OPENCLAW_DEFAULT if os.access(OPENCLAW_DEFAULT, os.X_OK) else None,
        chat_backend=backend if backend in {"auto", "openclaw", "local"} else "auto",
    )
    allowed_hosts = LOOPBACK_HOSTS | {str(settings.ui_host).lower()}

    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(default=True)
    )
    env.globals.update(
        case_id=case,
        doc_view_url=doc_view_url,
        fact_url=fact_url,
        fact_date=fact_date,
        fmt_date=fmt_date,
        fmt_dt=fmt_dt,
        pct=pct,
        evidence_cite=evidence_cite,
        valid_color=valid_color,
        STATUSES=STATUSES,
        STATUS_LABELS=STATUS_LABELS,
        DOC_KINDS=DOC_KINDS,
        CERTAINTIES=CERTAINTIES,
        ALERT_LABELS=ALERT_LABELS,
        FIELD_LABELS=FIELD_LABELS,
        COUNT_LABELS=COUNT_LABELS,
        SORT_KEYS=SORT_KEYS,
        SORT_LABELS=SORT_LABELS,
        query_string=query_string,
        quote=lambda s: quote(str(s), safe=""),
    )
    templates = Jinja2Templates(env=env)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def nav_counts(store: Store) -> dict[str, int]:
        return {
            "drafts": store.review_counts(case)["draft"],
            "conflicts": len(store.list_contradictions(case, "open")),
            "alerts": len(store.list_alerts(case, pending_only=True)),
        }

    def render(request: Request, name: str, status_code: int = 200, **ctx: Any) -> HTMLResponse:
        if not name.startswith("_"):
            ctx.setdefault("nav", nav_counts(_store(request)))
        ctx.setdefault("user", current_user(request))
        ctx.setdefault(
            "here", request.url.path + (f"?{request.url.query}" if request.url.query else "")
        )
        return templates.TemplateResponse(request, name, ctx, status_code=status_code)

    # ------------------------------------------------------------------ middleware / errors

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Response:
        # DNS-rebinding and cross-site request protection for a loopback-only app.
        if hostname(request.headers.get("host", "")) not in allowed_hosts:
            return Response("Invalid Host header", status_code=400)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and (urlsplit(origin).hostname or "") not in allowed_hosts:
                return Response("Cross-origin request refused", status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return Response("Cross-site request refused", status_code=403)
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", CSP)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        # Not "no-referrer": with it, browsers send "Origin: null" on our own form POSTs and
        # the origin check above refuses every write. "same-origin" still leaks nothing.
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if not request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(NeedName)
    async def need_name(request: Request, exc: NeedName) -> Response:
        if _is_htmx(request):
            return PlainTextResponse(
                "Enter your display name first.", status_code=401, headers={"HX-Redirect": "/name"}
            )
        if request.method == "POST" and not _wants_json(request):
            return _redirect("/name?next=" + quote(f"/cases/{case}/facts", safe=""))
        return JSONResponse({"detail": "a display name is required"}, status_code=401)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
        headers = getattr(exc, "headers", None)
        if _is_htmx(request):
            return PlainTextResponse(detail, status_code=exc.status_code, headers=headers)
        if "text/html" in request.headers.get("accept", ""):
            try:
                return render(
                    request,
                    "error.html",
                    status_code=exc.status_code,
                    code=exc.status_code,
                    detail=detail,
                )
            except Exception:  # noqa: BLE001 - never fail while reporting a failure
                log.exception("error page failed")
        return JSONResponse({"detail": detail}, status_code=exc.status_code, headers=headers)

    # ------------------------------------------------------------------ lookups

    def check_case(case_id: str) -> None:
        if case_id != case:
            raise HTTPException(404, "unknown case")

    def get_fact_or_404(store: Store, fact_id: str) -> Fact:
        fact = store.get_fact(fact_id)
        if fact is None or fact.case_id != case:
            raise HTTPException(404, "fact not found")
        return fact

    def get_doc_or_404(store: Store, doc_id: str) -> Document:
        doc = store.get_document(doc_id)
        if doc is None or doc.case_id != case:
            raise HTTPException(404, "document not found")
        return doc

    def get_witness_or_404(store: Store, witness_id: str) -> Witness:
        w = store.get_witness(witness_id)
        if w is None or w.case_id != case:
            raise HTTPException(404, "witness not found")
        return w

    def get_job_or_404(store: Store, job_id: str) -> Job:
        job = store.get_job(job_id)
        if job is None or job.case_id != case:
            raise HTTPException(404, "job not found")
        return job

    def make_span(store: Store, params: ParamGetter) -> SpanIn:
        """A span from viewer offsets; the quote is re-derived from the stored text."""
        doc_id = params.get("doc_id") or ""
        get_doc_or_404(store, doc_id)
        parsed = store.get_parsed(doc_id)
        if parsed is None:
            raise HTTPException(422, "this document has no parsed text yet")
        try:
            start = parse_int(params.get("char_start"), "char_start")
            end = parse_int(params.get("char_end"), "char_end")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not (0 <= start < end <= len(parsed.text)):
            raise HTTPException(422, "the selection is outside the document text")
        if end - start > MAX_SPAN_CHARS:
            raise HTTPException(422, f"select at most {MAX_SPAN_CHARS} characters")
        quote_text = parsed.text[start:end]
        claimed = params.get("quote")
        if claimed is not None and claimed != "" and claimed != quote_text:
            raise HTTPException(422, "the selection does not match the document text")
        if not quote_text.strip():
            raise HTTPException(422, "the selection is empty")
        return SpanIn(doc_id=doc_id, char_start=start, char_end=end, quote=quote_text)

    def queue_scan(request: Request, trigger: str) -> Job:
        job = request_scan(_store(request), case, trigger)
        _deps(request).wake_worker()
        return job

    def lookups(store: Store) -> dict[str, Any]:
        tags = store.list_tags(case)
        witnesses = store.list_witnesses(case)
        docs = store.list_documents(case)
        return {
            "tags": tags,
            "tag_map": {t.tag_id: t for t in tags},
            "witnesses": witnesses,
            "witness_map": {w.witness_id: w for w in witnesses},
            "docs": docs,
            "doc_map": {d.doc_id: d for d in docs},
        }

    def known_fact(store: Store) -> Any:
        def known(fid: str) -> bool:
            f = store.get_fact(fid)
            return f is not None and f.case_id == case

        return known

    # ------------------------------------------------------------------ name cookie

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return _redirect(f"/cases/{quote(case, safe='')}/facts")

    @app.get("/name", response_class=HTMLResponse)
    def name_form(request: Request, next: str | None = None) -> HTMLResponse:
        return render(
            request, "name.html", next=safe_next(next, f"/cases/{case}/facts"), error=None
        )

    @app.post("/name", response_model=None)
    async def set_name(request: Request) -> Response:
        form = ParamGetter(await request.form())
        target = safe_next(form.get("next"), f"/cases/{case}/facts")
        name = clean_display_name(form.get("name"))
        if name is None:
            return render(
                request,
                "name.html",
                status_code=422,
                next=target,
                error="Enter a display name (not “agent”).",
            )
        response = _redirect(target)
        response.set_cookie(
            USER_COOKIE,
            encode_user_cookie(name),
            max_age=365 * 24 * 3600,
            httponly=True,
            samesite="strict",
            path="/",
        )
        return response

    # ------------------------------------------------------------------ today / jobs

    @app.get("/cases/{case_id}", response_class=HTMLResponse)
    def today(request: Request, case_id: str) -> HTMLResponse:
        check_case(case_id)
        store = _store(request)
        digest, digest_html = None, None
        try:
            digest = digest_mod.daily_digest(store, case)
            digest_html = linkify_fact_ids(str(digest.get("text") or ""), known_fact(store))
        except Exception as exc:  # noqa: BLE001 - the panel degrades, the page still works
            log.warning("daily digest unavailable: %s", exc)
        return render(
            request,
            "today.html",
            counts=store.review_counts(case),
            open_conflicts=len(store.list_contradictions(case, "open")),
            pending_alerts=len(store.list_alerts(case, pending_only=True)),
            documents=len(store.list_documents(case)),
            job=store.latest_job(case),
            jobs=store.list_jobs(case, limit=8),
            digest=digest,
            digest_html=digest_html,
        )

    @app.post("/cases/{case_id}/process", response_model=None)
    def process(request: Request, case_id: str) -> Response:
        check_case(case_id)
        require_user(request)
        job = queue_scan(request, "ui")
        if _is_htmx(request):
            return render(request, "_job.html", job=job)
        return _redirect(f"/jobs/{quote(job.job_id, safe='')}")

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_status(request: Request, job_id: str) -> HTMLResponse:
        job = get_job_or_404(_store(request), job_id)
        if _is_htmx(request):
            return render(request, "_job.html", job=job)
        return render(request, "job.html", job=job)

    # ------------------------------------------------------------------ timeline (facts)

    def run_filters(store: Store, f: dict[str, Any]) -> tuple[list[Fact], str | None]:
        fact_ids: list[str] | None = None
        if f["q"]:
            try:
                fact_ids = search_mod.keyword_fact_ids(store, case, f["q"])
            except search_mod.QueryError as exc:
                return [], f"Search not understood: {exc}"
            except NotImplementedError:
                return [], "Search is not available yet."
        return store.list_facts(case, build_query(f, fact_ids)), None

    def visual_item(f: Fact, alleged: set[str], conflicts: set[str]) -> dict[str, Any]:
        end = None
        if f.date_start and f.date_end and f.date_end > f.date_start:
            end = (f.date_end + timedelta(days=1)).isoformat()  # vis end is exclusive
        classes = [f"review-{f.review}", f"cert-{f.date_certainty}"]
        classes += [f"st-{s}" for s in f.statuses]
        if f.fact_id in conflicts:
            classes.append("conflict")
        if f.fact_id in alleged:
            classes.append("alleged")
        return {
            "id": f.fact_id,
            "title": f.title,
            "start": f.date_start.isoformat() if f.date_start else None,
            "end": end,
            "type": "range" if end else "point",
            "undated": f.date_start is None,
            "date_label": fact_date(f),
            "date_text": f.date_text,
            "certainty": f.date_certainty,
            "statuses": list(f.statuses),
            "key_fact": "key_fact" in f.statuses,
            "review": f.review,
            "origin": f.origin,
            "alleged": f.fact_id in alleged,
            "conflict": f.fact_id in conflicts,
            "url": fact_url(f.fact_id),
            "class_name": " ".join(classes),
        }

    @app.get("/cases/{case_id}/facts", response_model=None)
    def timeline(request: Request, case_id: str, format: str | None = None) -> Response:
        check_case(case_id)
        store = _store(request)
        f, errors = parse_filters(ParamGetter(request.query_params))
        facts, search_error = run_filters(store, f)
        alleged = store.alleged_fact_ids(case)
        conflicts = store.conflicted_fact_ids(case)
        if format == "json":
            return JSONResponse(
                {
                    "case_id": case,
                    "error": search_error,
                    "items": [visual_item(x, alleged, conflicts) for x in facts],
                }
            )
        pairs = filter_pairs(f)
        return render(
            request,
            "timeline.html",
            f=f,
            errors=errors,
            search_error=search_error,
            syntax_help=search_mod.SYNTAX_HELP,
            dated=[x for x in facts if x.date_start is not None],
            undated=[x for x in facts if x.date_start is None],
            total=len(facts),
            counts=store.fact_counts([x.fact_id for x in facts]),
            alleged=alleged,
            conflicts=conflicts,
            pairs=pairs,
            filtered=filters_active(f),
            export_zooms=export_mod.ZOOMS,
            **lookups(store),
        )

    @app.post("/cases/{case_id}/facts", response_model=None)
    async def create_fact(request: Request, case_id: str) -> Response:
        check_case(case_id)
        user = require_user(request)
        store = _store(request)
        params = await _params(request)
        spans = [make_span(store, params)] if params.get("char_start") else []
        with store_errors():
            changes = fact_changes(params)
            title = changes.pop("title", "") or ""
            fact = store.add_fact(
                case,
                title,
                by=user,
                details=changes.get("details"),
                date_text=changes.get("date_text"),
                date_start=changes.get("date_start"),
                date_end=changes.get("date_end"),
                date_certainty=changes.get("date_certainty")
                or ("exact" if changes.get("date_start") else "unknown"),
                statuses=changes.get("statuses", ()),
                tag_ids=changes.get("tag_ids", ()),
                witness_ids=changes.get("witness_ids", ()),
                spans=spans,
            )
        if _wants_json(request):
            return JSONResponse(fact.model_dump(mode="json"), status_code=201)
        default = fact_url(fact.fact_id)
        if spans and fact.evidence:
            default = doc_view_url(spans[0].doc_id, fact.evidence[0].evidence_id)
        return _redirect(safe_next(params.get("next"), default))

    # ------------------------------------------------------------------ needs review / bulk

    @app.get("/cases/{case_id}/review", response_class=HTMLResponse)
    def needs_review(
        request: Request, case_id: str, order: str = "low", doc: str | None = None
    ) -> HTMLResponse:
        check_case(case_id)
        store = _store(request)
        high = order == "high"
        query = FactQuery(
            reviews=["draft"], sort="confidence", descending=high, doc_ids=[doc] if doc else []
        )
        facts = store.list_facts(case, query)
        return render(
            request,
            "review.html",
            facts=facts,
            order="high" if high else "low",
            doc_filter=doc,
            alleged=store.alleged_fact_ids(case),
            conflicts=store.conflicted_fact_ids(case),
            **lookups(store),
        )

    @app.post("/cases/{case_id}/review", response_model=None)
    async def bulk_review(request: Request, case_id: str) -> Response:
        check_case(case_id)
        user = require_user(request)
        store = _store(request)
        params = await _params(request)
        action = params.get("action")
        if action not in ("accept", "reject"):
            raise HTTPException(422, "action must be accept or reject")
        ids = params.getlist("fact_ids")
        if not ids:
            raise HTTPException(422, "select at least one fact")
        reason = (params.get("reason") or "").strip() or None
        facts = [get_fact_or_404(store, fid) for fid in ids]
        with store_errors():
            for fact in facts:
                store.review_fact(fact.fact_id, action, reviewer=user, reason=reason)
        if _wants_json(request):
            return JSONResponse({"reviewed": [f.fact_id for f in facts], "action": action})
        return _redirect(safe_next(params.get("next"), f"/cases/{case}/review"))

    # ------------------------------------------------------------------ fact card

    @app.get("/facts/{fact_id}", response_class=HTMLResponse)
    def fact_card(request: Request, fact_id: str, partial: int = 0) -> HTMLResponse:
        store = _store(request)
        fact = get_fact_or_404(store, fact_id)
        lk = lookups(store)
        ctx: dict[str, Any] = {
            "fact": fact,
            "alleged": fact.fact_id in store.alleged_fact_ids(case),
            "conflict": fact.fact_id in store.conflicted_fact_ids(case),
            "notes": store.list_notes(fact_id),
            **lk,
        }
        if partial:
            return render(request, "_fact_summary.html", **ctx)
        names = {t.tag_id: t.name for t in lk["tags"]}
        names.update({w.witness_id: w.name for w in lk["witnesses"]})
        history = [
            {
                **h,
                "old_text": history_value(h["old"], names),
                "new_text": history_value(h["new"], names),
            }
            for h in store.fact_history(fact_id)
        ]
        contradictions = []
        for c in store.list_contradictions(case, fact_id=fact_id):
            others = [x for x in store.get_facts(c.fact_ids) if x.fact_id != fact_id]
            contradictions.append({"c": c, "others": others})
        contradictions.sort(key=lambda v: v["c"].status != "open")
        anchor = store.get_fact(fact.anchor_fact_id) if fact.anchor_fact_id else None
        return render(
            request,
            "fact.html",
            history=history,
            contradictions=contradictions,
            anchor=anchor,
            witness_choices=[
                w for w in lk["witnesses"] if w.confirmed or w.witness_id in fact.witness_ids
            ],
            **ctx,
        )

    async def _edit_fact(request: Request, fact_id: str) -> Response:
        user = require_user(request)
        store = _store(request)
        get_fact_or_404(store, fact_id)
        params = await _params(request)
        with store_errors():
            changes = fact_changes(params)
            new_tag = (params.get("new_tag") or "").strip()
            if new_tag:
                tag = store.add_tag(case, new_tag, by=user)
                current = changes.get("tag_ids")
                if current is None:
                    current = list(store.get_fact(fact_id).tag_ids)  # type: ignore[union-attr]
                changes["tag_ids"] = [*current, tag.tag_id]
            fact = store.update_fact(fact_id, changes, by=user)
        if request.method == "PATCH" or _wants_json(request):
            return JSONResponse(fact.model_dump(mode="json"))
        return _redirect(safe_next(params.get("next"), fact_url(fact_id)))

    @app.patch("/facts/{fact_id}", response_model=None)
    async def patch_fact(request: Request, fact_id: str) -> Response:
        return await _edit_fact(request, fact_id)

    @app.post("/facts/{fact_id}", response_model=None)
    async def post_fact(request: Request, fact_id: str) -> Response:
        return await _edit_fact(request, fact_id)

    @app.post("/facts/{fact_id}/review", response_model=None)
    async def review_fact(request: Request, fact_id: str) -> Response:
        user = require_user(request)
        store = _store(request)
        get_fact_or_404(store, fact_id)
        params = await _params(request)
        action = params.get("action")
        if action not in ("accept", "reject", "edit"):
            raise HTTPException(422, "action must be accept, edit or reject")
        reason = (params.get("reason") or "").strip() or None
        with store_errors():
            changes = fact_changes(params) if action == "edit" else None
            fact = store.review_fact(fact_id, action, reviewer=user, reason=reason, changes=changes)
        if _is_htmx(request):
            return render(request, "_review_state.html", fact=fact, reason=reason)
        if _wants_json(request):
            return JSONResponse(fact.model_dump(mode="json"))
        return _redirect(safe_next(params.get("next"), fact_url(fact_id)))

    @app.post("/facts/{fact_id}/evidence", response_model=None)
    async def add_evidence(request: Request, fact_id: str) -> Response:
        user = require_user(request)
        store = _store(request)
        get_fact_or_404(store, fact_id)
        params = await _params(request)
        kind = params.get("kind") or ("span" if params.get("char_start") else "")
        with store_errors():
            if kind == "span":
                span = make_span(store, params)
                ev = store.add_evidence(fact_id, kind="span", by=user, span=span)
            elif kind == "document":
                doc_id = params.get("doc_id") or ""
                get_doc_or_404(store, doc_id)
                ev = store.add_evidence(fact_id, kind="document", by=user, doc_id=doc_id)
            elif kind == "external":
                ev = store.add_evidence(
                    fact_id,
                    kind="external",
                    by=user,
                    reference=(params.get("reference") or "")[:2000],
                )
            else:
                raise HTTPException(422, "kind must be span, document or external")
        if _wants_json(request):
            return JSONResponse(ev.model_dump(mode="json"), status_code=201)
        default = fact_url(fact_id) + "#evidence"
        if kind == "span":
            default = doc_view_url(ev.doc_id or "", ev.evidence_id)
        return _redirect(safe_next(params.get("next"), default))

    @app.post("/evidence/{evidence_id}/review", response_model=None)
    async def review_evidence(request: Request, evidence_id: str) -> Response:
        user = require_user(request)
        store = _store(request)
        found = store.get_evidence(evidence_id)
        if found is None:
            raise HTTPException(404, "evidence not found")
        fact_id, _ = found
        get_fact_or_404(store, fact_id)
        params = await _params(request)
        action = params.get("action")
        if action not in ("accept", "reject"):
            raise HTTPException(422, "action must be accept or reject")
        reason = (params.get("reason") or "").strip() or None
        with store_errors():
            ev = store.review_evidence(evidence_id, action, reviewer=user, reason=reason)
        if _is_htmx(request):
            return render(request, "_evidence_state.html", ev=ev)
        if _wants_json(request):
            return JSONResponse(ev.model_dump(mode="json"))
        return _redirect(safe_next(params.get("next"), fact_url(fact_id) + "#evidence"))

    @app.post("/facts/{fact_id}/notes", response_model=None)
    async def add_note(request: Request, fact_id: str) -> Response:
        user = require_user(request)
        store = _store(request)
        get_fact_or_404(store, fact_id)
        params = await _params(request)
        with store_errors():
            note = store.add_note(fact_id, author=user, body=(params.get("body") or "")[:20000])
        if _wants_json(request):
            return JSONResponse(note.model_dump(mode="json"), status_code=201)
        return _redirect(safe_next(params.get("next"), fact_url(fact_id) + "#notes"))

    # ------------------------------------------------------------------ documents

    def document_rows(store: Store) -> list[dict[str, Any]]:
        cites: Counter[str] = Counter()
        drafts: Counter[str] = Counter()
        for fact in store.list_facts(case):
            for doc_id in {e.doc_id for e in fact.evidence if e.doc_id}:
                cites[doc_id] += 1
                drafts[doc_id] += fact.review == "draft"
        rows = []
        for d in store.list_documents(case):
            info = store.document_info(d.doc_id) or {}
            rows.append(
                {"doc": d, "info": info, "facts": cites[d.doc_id], "drafts": drafts[d.doc_id]}
            )
        return rows

    @app.get("/cases/{case_id}/documents", response_class=HTMLResponse)
    def documents(request: Request, case_id: str) -> HTMLResponse:
        check_case(case_id)
        store = _store(request)
        return render(
            request,
            "documents.html",
            rows=document_rows(store),
            results=None,
            job=store.latest_job(case),
        )

    @app.post("/cases/{case_id}/documents", response_class=HTMLResponse)
    async def upload(
        request: Request, case_id: str, files: Annotated[list[UploadFile], File()]
    ) -> HTMLResponse:
        check_case(case_id)
        require_user(request)
        try:
            inbox = settings.inbox.resolve()
        except RuntimeError as exc:
            log.error("upload refused: %s", exc)
            raise HTTPException(500, str(exc)) from exc
        results: list[dict[str, Any]] = []
        for up in files:
            original = up.filename or ""
            name = sanitize_filename(original)
            if name is None or Path(name).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
                results.append({"name": original, "ok": False, "msg": "unsupported file type"})
                await up.close()
                continue
            dest = _unique_path(inbox, name)
            if dest.resolve().parent != inbox:
                results.append({"name": original, "ok": False, "msg": "invalid filename"})
                await up.close()
                continue
            size = 0
            try:
                with open(dest, "xb") as fh:
                    while chunk := await up.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise ValueError("file too large")
                        fh.write(chunk)
            except (ValueError, OSError) as exc:
                dest.unlink(missing_ok=True)
                results.append({"name": original, "ok": False, "msg": str(exc)})
                continue
            finally:
                await up.close()
            log.info("uploaded %s (%d bytes) to inbox", dest.name, size)
            results.append({"name": original, "saved_as": dest.name, "ok": True, "size": size})
        job = queue_scan(request, "upload") if any(r["ok"] for r in results) else None
        store = _store(request)
        return render(
            request,
            "documents.html",
            rows=document_rows(store),
            results=results,
            job=job or store.latest_job(case),
        )

    @app.get("/documents/{doc_id}/view", response_class=HTMLResponse)
    def source_view(request: Request, doc_id: str, evidence: str | None = None) -> HTMLResponse:
        store = _store(request)
        doc = get_doc_or_404(store, doc_id)
        parsed = store.get_parsed(doc_id)
        ev, ev_fact = None, None
        if evidence:
            found = store.get_evidence(evidence)
            if found is not None and found[1].doc_id == doc_id:
                ev_fact = store.get_fact(found[0])
                ev = found[1]
        segments = []
        if parsed is not None:
            hl = (ev.char_start, ev.char_end) if ev and ev.kind == "span" else (None, None)
            segments = source_segments(parsed, *hl)
        citing = store.facts_citing_document(doc_id)
        return render(
            request,
            "source.html",
            doc=doc,
            info=store.document_info(doc_id) or {},
            parsed=parsed,
            segments=segments,
            ev=ev,
            ev_fact=ev_fact,
            requested=evidence,
            citing=citing,
            alleged=store.alleged_fact_ids(case),
            conflicts=store.conflicted_fact_ids(case),
            all_facts=store.list_facts(case, FactQuery(sort="title")),
        )

    @app.post("/documents/{doc_id}/kind", response_model=None)
    async def set_kind(request: Request, doc_id: str) -> Response:
        user = require_user(request)
        store = _store(request)
        get_doc_or_404(store, doc_id)
        params = await _params(request)
        with store_errors():
            doc = store.set_document_kind(doc_id, params.get("kind") or "", by=user)
        if _wants_json(request):
            return JSONResponse(doc.model_dump(mode="json"))
        return _redirect(safe_next(params.get("next"), doc_view_url(doc_id)))

    @app.post("/documents/{doc_id}/reprocess", response_model=None)
    async def reprocess(request: Request, doc_id: str) -> Response:
        require_user(request)
        store = _store(request)
        get_doc_or_404(store, doc_id)
        with store_errors():
            removed = store.reset_document(doc_id)
        job = queue_scan(request, "ui")
        log.info("document %s reset for reprocessing (%d drafts removed)", doc_id, removed)
        if _wants_json(request):
            return JSONResponse({"job_id": job.job_id, "drafts_removed": removed})
        return _redirect(f"/jobs/{quote(job.job_id, safe='')}")

    # ------------------------------------------------------------------ witnesses

    def witness_changes(params: ParamGetter) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        declared = set(params.getlist("has"))
        for key in ("name", "role", "notes"):
            if params.has(key):
                changes[key] = (params.get(key) or "").strip()
        if params.has("aliases") or "aliases" in declared:
            changes["aliases"] = params.getlist("aliases")
        for key in ("flagged", "confirmed"):
            if params.has(key) or key in declared:
                raw = (params.get(key) or "").lower()
                changes[key] = raw in ("1", "true", "on", "yes")
        return changes

    @app.get("/cases/{case_id}/witnesses", response_class=HTMLResponse)
    def witnesses(request: Request, case_id: str) -> HTMLResponse:
        check_case(case_id)
        store = _store(request)
        everyone = store.list_witnesses(case)
        counts = {w.witness_id: len(store.witness_facts(w.witness_id)) for w in everyone}
        return render(
            request,
            "witnesses.html",
            confirmed=[w for w in everyone if w.confirmed],
            suggested=[w for w in everyone if not w.confirmed],
            fact_counts=counts,
        )

    @app.post("/cases/{case_id}/witnesses", response_model=None)
    async def create_witness(request: Request, case_id: str) -> Response:
        check_case(case_id)
        user = require_user(request)
        params = await _params(request)
        ch = witness_changes(params)
        with store_errors():
            w = _store(request).add_witness(
                case,
                ch.get("name", ""),
                by=user,
                aliases=ch.get("aliases", ()),
                role=ch.get("role") or None,
                notes=ch.get("notes") or None,
                flagged=bool(ch.get("flagged")),
            )
        if _wants_json(request):
            return JSONResponse(w.model_dump(mode="json"), status_code=201)
        return _redirect(f"/witnesses/{quote(w.witness_id, safe='')}")

    @app.get("/witnesses/{witness_id}", response_class=HTMLResponse)
    def witness_profile(request: Request, witness_id: str) -> HTMLResponse:
        store = _store(request)
        w = get_witness_or_404(store, witness_id)
        return render(
            request,
            "witness.html",
            w=w,
            facts=store.witness_facts(witness_id),
            documents=store.witness_documents(witness_id),
            others=[x for x in store.list_witnesses(case) if x.witness_id != witness_id],
            alleged=store.alleged_fact_ids(case),
            conflicts=store.conflicted_fact_ids(case),
        )

    async def _change_witness(request: Request, witness_id: str) -> Response:
        user = require_user(request)
        store = _store(request)
        get_witness_or_404(store, witness_id)
        params = await _params(request)
        action = params.get("action") or "update"
        target = f"/witnesses/{quote(witness_id, safe='')}"
        with store_errors():
            if action == "dismiss":
                store.delete_witness(witness_id, by=user)
                result: Witness | None = None
                target = f"/cases/{case}/witnesses"
            elif action == "merge":
                into = params.get("into") or ""
                get_witness_or_404(store, into)
                result = store.merge_witnesses(witness_id, into, by=user)
                target = f"/witnesses/{quote(into, safe='')}"
            elif action == "confirm":
                result = store.update_witness(witness_id, {"confirmed": True}, by=user)
            elif action == "update":
                result = store.update_witness(witness_id, witness_changes(params), by=user)
            else:
                raise HTTPException(422, "unknown witness action")
        if request.method == "PATCH" or _wants_json(request):
            return JSONResponse(result.model_dump(mode="json") if result else {"deleted": True})
        return _redirect(safe_next(params.get("next"), target))

    @app.patch("/witnesses/{witness_id}", response_model=None)
    async def patch_witness(request: Request, witness_id: str) -> Response:
        return await _change_witness(request, witness_id)

    @app.post("/witnesses/{witness_id}", response_model=None)
    async def post_witness(request: Request, witness_id: str) -> Response:
        return await _change_witness(request, witness_id)

    # ------------------------------------------------------------------ tags

    @app.get("/cases/{case_id}/tags", response_class=HTMLResponse)
    def tags(request: Request, case_id: str) -> HTMLResponse:
        check_case(case_id)
        store = _store(request)
        counts: Counter[str] = Counter()
        for fact in store.list_facts(case):
            counts.update(fact.tag_ids)
        return render(request, "tags.html", tags=store.list_tags(case), tag_counts=counts)

    @app.post("/cases/{case_id}/tags", response_model=None)
    async def change_tags(request: Request, case_id: str) -> Response:
        check_case(case_id)
        user = require_user(request)
        store = _store(request)
        params = await _params(request)
        action = params.get("action") or "create"
        color_raw = params.get("color")
        color = valid_color(color_raw)
        if color_raw and color is None:
            raise HTTPException(422, "color must look like #1a2b3c")
        result: Any = None
        with store_errors():
            if action == "create":
                result = store.add_tag(case, params.get("name") or "", by=user, color=color)
            else:
                tag_id = params.get("tag_id") or ""
                tag = store.get_tag(tag_id)
                if tag is None or tag.case_id != case:
                    raise HTTPException(404, "tag not found")
                if action == "update":
                    result = store.update_tag(tag_id, by=user, name=params.get("name"), color=color)
                elif action == "delete":
                    store.delete_tag(tag_id, by=user)
                else:
                    raise HTTPException(422, "unknown tag action")
        if _wants_json(request):
            return JSONResponse(result.model_dump(mode="json") if result else {"deleted": True})
        return _redirect(safe_next(params.get("next"), f"/cases/{case}/tags"))

    # ------------------------------------------------------------------ contradictions / alerts

    @app.get("/cases/{case_id}/contradictions", response_class=HTMLResponse)
    def contradictions(request: Request, case_id: str) -> HTMLResponse:
        check_case(case_id)
        store = _store(request)
        order = {"open": 0, "confirmed": 1, "dismissed": 2}
        items = []
        for c in store.list_contradictions(case):
            items.append({"c": c, "facts": store.get_facts(c.fact_ids)})
        items.sort(key=lambda v: order.get(v["c"].status, 3))
        return render(request, "contradictions.html", items=items, **lookups(store))

    @app.post("/contradictions/{contradiction_id}", response_model=None)
    async def review_contradiction(request: Request, contradiction_id: str) -> Response:
        user = require_user(request)
        store = _store(request)
        if store.contradiction_case(contradiction_id) != case:
            raise HTTPException(404, "contradiction not found")
        params = await _params(request)
        action = params.get("action")
        if action not in ("confirm", "dismiss", "reopen"):
            raise HTTPException(422, "action must be confirm, dismiss or reopen")
        with store_errors():
            c = store.review_contradiction(contradiction_id, action, reviewer=user)
        if _wants_json(request):
            return JSONResponse(c.model_dump(mode="json"))
        return _redirect(
            safe_next(
                params.get("next"),
                f"/cases/{case}/contradictions#{quote(contradiction_id, safe='')}",
            )
        )

    @app.get("/cases/{case_id}/alerts", response_class=HTMLResponse)
    def alerts(request: Request, case_id: str) -> HTMLResponse:
        check_case(case_id)
        store = _store(request)
        items = store.list_alerts(case)
        fact_ids = list(dict.fromkeys(fid for a in items for fid in a.fact_ids))
        facts = {f.fact_id: f for f in store.get_facts(fact_ids)}
        return render(
            request,
            "alerts.html",
            alerts=items,
            facts=facts,
            doc_map={d.doc_id: d for d in store.list_documents(case)},
        )

    @app.get("/cases/{case_id}/alerts/{alert_id}", include_in_schema=False)
    def alert_link(case_id: str, alert_id: str) -> RedirectResponse:
        """Content-free link target for alerts announced outside the UI."""
        check_case(case_id)
        return _redirect(f"/cases/{quote(case, safe='')}/alerts#{quote(alert_id, safe='')}")

    # ------------------------------------------------------------------ export

    @app.get("/cases/{case_id}/export", response_model=None)
    def export(
        request: Request, case_id: str, format: str = "xlsx", zoom: str = "month"
    ) -> Response:
        check_case(case_id)
        if format not in EXPORT_TYPES:
            raise HTTPException(422, "format must be xlsx, csv, svg or png")
        store = _store(request)
        params = ParamGetter(request.query_params)
        f, errors = parse_filters(params)
        if errors:
            raise HTTPException(422, "; ".join(errors))
        facts, search_error = run_filters(store, f)
        if search_error:
            raise HTTPException(422, search_error)
        columns = [c for c in params.getlist("columns") if c in export_mod.COLUMNS]
        columns = columns or list(export_mod.DEFAULT_COLUMNS)
        if zoom not in export_mod.ZOOMS:
            raise HTTPException(422, f"zoom must be one of {', '.join(export_mod.ZOOMS)}")
        try:
            if format == "csv":
                body: bytes | str = export_mod.to_csv(store, case, facts, columns)
            elif format == "xlsx":
                body = export_mod.to_xlsx(store, case, facts, columns)
            elif format == "svg":
                body = export_mod.to_svg(store, case, facts, zoom)
            else:
                body = export_mod.to_png(store, case, facts, zoom)
        except NotImplementedError as exc:
            raise HTTPException(501, f"{format.upper()} export is not available yet") from exc
        media, ext = EXPORT_TYPES[format]
        return Response(
            body.encode("utf-8") if isinstance(body, str) else body,
            media_type=media,
            headers={
                "Content-Disposition": f'attachment; filename="{case}-timeline.{ext}"',
                "Content-Security-Policy": EXPORT_CSP,
            },
        )

    # ------------------------------------------------------------------ chat

    def ask_openclaw(deps: Deps, question: str) -> str | None:
        exe = deps.openclaw_bin
        if deps.chat_backend == "local" or exe is None or shutil.which(str(exe)) is None:
            return None
        message = (
            f"[case-timeline case_id={case}] Answer using the case-timeline tools only. Cite "
            "every fact with its fact ID like [F-0012] and its document and page; say when a "
            "cited fact is an unreviewed draft, and flag deadlines as 'needs attorney "
            f"review'.\n\nQuestion: {question}"
        )
        cmd = [str(exe), "agent", f"--message={message}", "--json"]
        if agent := os.environ.get("CT_OPENCLAW_AGENT"):
            cmd.append(f"--agent={agent}")
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                cmd, capture_output=True, text=True, timeout=OPENCLAW_TIMEOUT_S, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("openclaw agent unavailable: %s", exc)
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            log.warning("openclaw agent exited %s: %s", proc.returncode, proc.stderr[-500:])
            return None
        return _extract_openclaw_text(proc.stdout)

    def chat_hits(store: Store, question: str) -> list[Fact]:
        try:
            ids = search_mod.search_facts(store, case, question, CHAT_TOP_K)
        except Exception as exc:  # noqa: BLE001 - fall back to vectors, then nothing
            log.warning("search_facts unavailable for chat (%s)", exc)
            try:
                ids = store.vector_fact_ids(case, question, CHAT_TOP_K)
            except Exception:  # noqa: BLE001
                ids = []
        facts = [f for f in store.get_facts(ids) if f.case_id == case and f.review != "rejected"]
        return facts[:CHAT_TOP_K]

    def fact_context(store: Store, f: Fact, doc_map: dict[str, Document]) -> str:
        lines = [
            f"[{f.fact_id}] {f.title}",
            f"date: {fact_date(f) or 'undated'}"
            + (f" (as written: {f.date_text})" if f.date_text else ""),
            f"review: {f.review}; statuses: {', '.join(f.statuses) or 'none'}",
        ]
        if f.details:
            lines.append(f"details: {f.details}")
        for e in f.evidence[:3]:
            if e.kind == "span" and e.quote:
                doc = doc_map.get(e.doc_id or "")
                where = f"{doc.filename if doc else e.doc_id}, {evidence_cite(e)}".strip(", ")
                lines.append(f'quote ({where}): "{e.quote}"')
        return "\n".join(lines)

    def ask_local(deps: Deps, question: str) -> tuple[str, list[Fact], str | None]:
        store = deps.get_store()
        facts = chat_hits(store, question)
        if not facts:
            return "No facts in this case match the question.", [], None
        doc_map = {d.doc_id: d for d in store.list_documents(case)}
        context = "\n\n".join(fact_context(store, f, doc_map) for f in facts)
        messages = [
            {"role": "system", "content": CHAT_SYSTEM},
            {"role": "user", "content": f"Case facts:\n\n{context}\n\nQuestion: {question}"},
        ]
        try:
            answer = deps.get_llm().chat(messages)
        except Exception as exc:  # noqa: BLE001 - model may be offline; still show the facts
            log.warning("local LLM unavailable for chat: %s", exc)
            return "", facts, "The local model is unavailable; the matching facts are listed."
        return answer, facts, None

    @app.get("/cases/{case_id}/chat", response_class=HTMLResponse)
    def chat_form(request: Request, case_id: str) -> HTMLResponse:
        check_case(case_id)
        return render(request, "chat.html", result=None)

    @app.post("/cases/{case_id}/chat", response_class=HTMLResponse)
    async def chat(request: Request, case_id: str) -> HTMLResponse:
        check_case(case_id)
        params = await _params(request)
        question = (params.get("question") or "").strip()[:4000]
        if not question:
            raise HTTPException(422, "empty question")
        deps = _deps(request)
        store = deps.get_store()
        facts: list[Fact] = []
        notice = None
        backend = "openclaw"
        answer = ask_openclaw(deps, question)
        if answer is None:
            if deps.chat_backend == "openclaw":
                notice, answer = "The OpenClaw agent is unavailable.", ""
            else:
                backend = "local"
                answer, facts, notice = ask_local(deps, question)
        known = known_fact(store)
        cited = [fid for fid in fact_ids_in(answer) if known(fid)]
        if not facts and cited:
            facts = store.get_facts(cited)
        result = {
            "question": question,
            "answer_html": linkify_fact_ids(answer, known),
            "facts": facts,
            "cited": set(cited),
            "notice": notice,
            "backend": backend,
        }
        name = "_chat_answer.html" if _is_htmx(request) else "chat.html"
        return render(request, name, result=result)

    return app
