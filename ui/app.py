"""Local review UI: FastAPI + Jinja2 + HTMX + vis-timeline (all assets vendored).

The UI never talks to SQL or the model directly; it goes through ``pipeline.store``,
``pipeline.runner`` and ``pipeline.llm``. Uploads land in ``data_dir/inbox`` only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import quote, urlencode, urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

from pipeline.contracts import (
    Contradiction,
    ContradictionStatus,
    Event,
    EventStatus,
    Job,
)
from pipeline.store import KEY_EVENT_THRESHOLD, is_key_event

if TYPE_CHECKING:
    from pipeline.config import Settings

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"

ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".eml"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
REVIEW_STATUSES = {EventStatus.ACCEPTED, EventStatus.REJECTED}
# Timeline views: key events only (the default) or every event; chosen with ?view=.
VIEW_KEY = "key"
VIEW_ALL = "all"
OPENCLAW_DEFAULT = Path("~/.openclaw/bin/openclaw").expanduser()
OPENCLAW_TIMEOUT_S = 180
CHAT_TOP_K = 8

_DOC_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_CITE_RE = re.compile(r"\[E(\d+)\]")
_SOURCE_LINK_RE = re.compile(
    r"/source/[A-Za-z0-9_-]+\?start=\d+&amp;end=\d+(?:&amp;event=[A-Za-z0-9_-]+)?"
)

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'self'"
)

CHAT_SYSTEM = (
    "You are a careful litigation-support assistant. Answer ONLY from the numbered "
    "evidence excerpts provided. Cite every factual statement with its tag, e.g. [E1]. "
    "If the excerpts do not answer the question, say so plainly. Never invent dates. "
    "Any deadline or limitation date must be flagged as 'needs attorney review'."
)


# --------------------------------------------------------------------------- dependencies


@dataclass
class Deps:
    """Lazily-created collaborators; tests replace these attributes with fakes."""

    settings: Any
    store: Any = None
    runner: Any = None
    llm: Any = None
    openclaw_bin: Path | None = None
    chat_backend: str = "auto"  # auto | openclaw | local
    last_job_id: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_store(self) -> Any:
        with self._lock:
            if self.store is None:
                from pipeline.store import Store

                store = Store(self.settings.db_path)
                store.init()
                self.store = store
            return self.store

    def get_runner(self) -> Any:
        if self.runner is None:
            from pipeline import runner

            self.runner = runner
        return self.runner

    def get_llm(self) -> Any:
        if self.llm is None:
            from pipeline import llm

            self.llm = llm
        return self.llm


def _deps(request: Request) -> Deps:
    return request.app.state.deps


# --------------------------------------------------------------------------- helpers


def host_is_loopback(host: str) -> bool:
    """True for 127.0.0.0/8, ::1 and localhost."""
    import ipaddress

    host = host.strip().strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _hostname(host_header: str) -> str:
    host_header = host_header.strip().lower()
    if host_header.startswith("["):
        return host_header[1 : host_header.find("]")] if "]" in host_header else host_header
    return host_header.rsplit(":", 1)[0] if host_header.count(":") == 1 else host_header


def sanitize_filename(name: str) -> str | None:
    """Reduce an uploaded filename to a safe basename, or None if nothing usable remains."""
    name = unicodedata.normalize("NFKC", name or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    name = re.sub(r"\.{2,}", ".", name)
    if not name or name in {".", ".."}:
        return None
    stem, dot, suffix = name.rpartition(".")
    if dot and len(name) > 150:
        name = stem[: 150 - len(suffix) - 1] + "." + suffix
    return name[:150]


def inbox_dir(settings: Any) -> Path:
    """Upload folder; refuses any location inside the read-only evidence dir."""
    inbox = (Path(settings.data_dir) / "inbox").resolve()
    evidence = Path(settings.evidence_dir).resolve()
    if inbox == evidence or inbox.is_relative_to(evidence):
        raise RuntimeError("inbox would be inside the evidence folder; refusing to write")
    inbox.mkdir(parents=True, exist_ok=True)
    return inbox


def _unique_path(directory: Path, name: str) -> Path:
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate = directory / name
    n = 1
    while candidate.exists():
        candidate = directory / f"{stem}-{n}{suffix}"
        n += 1
    return candidate


def source_url(ev: Event) -> str:
    params = {"start": ev.source.char_start, "end": ev.source.char_end, "event": ev.id}
    return f"/source/{quote(ev.source.doc_id, safe='')}?{urlencode(params)}"


def timeline_view(value: str | None) -> str:
    """The timeline view named by a ``view`` query value; key events unless "all"."""
    return VIEW_ALL if (value or "").strip().lower() == VIEW_ALL else VIEW_KEY


def filter_view(events: list[Event], view: str) -> list[Event]:
    """Events shown in ``view``: everything, or only key events (importance >= threshold)."""
    return list(events) if view == VIEW_ALL else [e for e in events if is_key_event(e)]


def _sort_key(ev: Event) -> tuple:
    return (ev.date_start is None, ev.date_start or "", ev.id)


def _conflict_ids(contradictions: list[Contradiction]) -> set[str]:
    return {
        eid
        for c in contradictions
        if c.status != ContradictionStatus.DISMISSED
        for eid in c.event_ids
    }


def _parsed_text(raw: Any) -> tuple[str | None, list[Any]]:
    """Accept str, ParsedDocument-like, (text, pages) or dict from ``Store.get_parsed_text``."""
    if raw is None:
        return None, []
    if isinstance(raw, str):
        return raw, []
    if isinstance(raw, tuple):
        return raw[0], list(raw[1]) if len(raw) > 1 and raw[1] else []
    if isinstance(raw, dict):
        return raw.get("text"), list(raw.get("pages") or [])
    return getattr(raw, "text", None), list(getattr(raw, "pages", None) or [])


def _page_for(pages: list[Any], offset: int) -> int | None:
    for p in pages:
        get = p.get if isinstance(p, dict) else lambda k, _p=p: getattr(_p, k)
        if get("char_start") <= offset < get("char_end"):
            return get("number")
    return None


def highlight(text: str, start: int | None, end: int | None) -> tuple[Markup, bool]:
    """HTML-escape ``text`` and wrap ``text[start:end]`` in ``<mark id=hl>``."""
    if start is None or end is None or not (0 <= start < end <= len(text)):
        return Markup(escape(text)), False
    html = (
        str(escape(text[:start]))
        + '<mark id="hl">'
        + str(escape(text[start:end]))
        + "</mark>"
        + str(escape(text[end:]))
    )
    return Markup(html), True


def linkify_answer(answer: str, citations: list[dict[str, Any]]) -> Markup:
    """Escape a model answer, then turn [E#] tags and local /source links into anchors."""
    html = str(escape(answer))
    by_tag = {c["tag"]: c["url"] for c in citations}

    def cite(m: re.Match[str]) -> str:
        url = by_tag.get(f"E{m.group(1)}")
        return f'<a href="{escape(url)}">{m.group(0)}</a>' if url else m.group(0)

    html = _CITE_RE.sub(cite, html)
    html = _SOURCE_LINK_RE.sub(lambda m: f'<a href="{m.group(0)}">{m.group(0)}</a>', html)
    return Markup(html)


def _is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request", "").lower() == "true"


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


# --------------------------------------------------------------------------- app factory


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the UI app. Collaborators live on ``app.state.deps`` and are created lazily."""
    if settings is None:
        from pipeline.config import get_settings

        settings = get_settings()

    app = FastAPI(title="Case Timeline", docs_url=None, redoc_url=None, openapi_url=None)
    backend = os.environ.get("CT_CHAT_BACKEND", "auto").lower()
    app.state.deps = Deps(
        settings=settings,
        openclaw_bin=OPENCLAW_DEFAULT if os.access(OPENCLAW_DEFAULT, os.X_OK) else None,
        chat_backend=backend if backend in {"auto", "openclaw", "local"} else "auto",
    )
    allowed_hosts = LOOPBACK_HOSTS | {str(settings.ui_host).lower()}

    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(default=True)
    )
    templates = Jinja2Templates(env=env)
    env.globals.update(
        source_url=source_url, case_id=settings.case_id, key_threshold=KEY_EVENT_THRESHOLD
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def render(request: Request, name: str, status_code: int = 200, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, ctx, status_code=status_code)

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Response:
        # DNS-rebinding and cross-site request protection for a loopback-only app.
        if _hostname(request.headers.get("host", "")) not in allowed_hosts:
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
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if not request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    # ------------------------------------------------------------------ dashboard / jobs

    def last_job(deps: Deps) -> Job | None:
        store = deps.get_store()
        if deps.last_job_id:
            return store.get_job(deps.last_job_id)
        latest = getattr(store, "latest_job", None)
        return latest(settings.case_id) if callable(latest) else None

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        deps = _deps(request)
        store = deps.get_store()
        events: list[Event] = store.list_events(settings.case_id)
        contradictions: list[Contradiction] = store.list_contradictions(settings.case_id)
        alerts = store.list_alerts(settings.case_id)
        by_status = {s.value: 0 for s in EventStatus}
        for ev in events:
            by_status[ev.status.value] += 1
        try:
            inbox_count = sum(1 for p in inbox_dir(settings).iterdir() if p.is_file())
        except (OSError, RuntimeError):
            inbox_count = 0
        counts = {
            "documents": len(store.processed_doc_ids(settings.case_id)),
            "events": len(events),
            "by_status": by_status,
            "deadline": sum(1 for e in events if e.deadline_relevant),
            "contradictions_open": sum(
                1 for c in contradictions if c.status == ContradictionStatus.OPEN
            ),
            "alerts": len(alerts),
            "inbox": inbox_count,
        }
        return render(request, "dashboard.html", counts=counts, job=last_job(deps))

    @app.post("/process", response_class=HTMLResponse)
    def process(request: Request) -> HTMLResponse:
        deps = _deps(request)
        try:
            job: Job = deps.get_runner().scan_case(settings.case_id)
        except Exception as exc:  # noqa: BLE001 - surfaced to the reviewer, not swallowed
            log.exception("scan_case failed to start")
            return render(request, "_job.html", job=None, error=str(exc))
        deps.last_job_id = job.id
        return render(request, "_job.html", job=job, error=None)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_status(request: Request, job_id: str) -> HTMLResponse:
        job = _deps(request).get_store().get_job(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return render(request, "_job.html", job=job, error=None)

    # ------------------------------------------------------------------ upload

    @app.get("/upload", response_class=HTMLResponse)
    def upload_form(request: Request) -> HTMLResponse:
        return render(request, "upload.html", results=None)

    @app.post("/upload", response_class=HTMLResponse)
    async def upload(request: Request, files: Annotated[list[UploadFile], File()]) -> HTMLResponse:
        try:
            inbox = inbox_dir(settings)
        except RuntimeError as exc:
            log.error("upload refused: %s", exc)
            raise HTTPException(500, str(exc)) from exc
        results: list[dict[str, Any]] = []
        for up in files:
            original = up.filename or ""
            name = sanitize_filename(original)
            if name is None or Path(name).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
                results.append({"name": original, "ok": False, "msg": "unsupported file type"})
                continue
            dest = _unique_path(inbox, name)
            if dest.resolve().parent != inbox:
                results.append({"name": original, "ok": False, "msg": "invalid filename"})
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
        return render(request, "upload.html", results=results)

    # ------------------------------------------------------------------ timeline / events

    @app.get("/timeline", response_class=HTMLResponse)
    def timeline(request: Request, view: str | None = None) -> HTMLResponse:
        store = _deps(request).get_store()
        view = timeline_view(view)
        every = sorted(store.list_events(settings.case_id), key=_sort_key)
        events = filter_view(every, view)
        conflicts = _conflict_ids(store.list_contradictions(settings.case_id))
        return render(
            request,
            "timeline.html",
            events=events,
            conflicts=conflicts,
            view=view,
            total=len(every),
            hidden=len(every) - len(events),
        )

    @app.get("/api/events")
    def api_events(request: Request, view: str | None = None) -> JSONResponse:
        store = _deps(request).get_store()
        view = timeline_view(view)
        every = sorted(store.list_events(settings.case_id), key=_sort_key)
        events = filter_view(every, view)
        conflicts = _conflict_ids(store.list_contradictions(settings.case_id))
        items = []
        for ev in events:
            end = None
            if ev.date_start and ev.date_end and ev.date_end > ev.date_start:
                end = (ev.date_end + timedelta(days=1)).isoformat()  # vis end is exclusive
            classes = [f"cert-{ev.certainty.value}", f"status-{ev.status.value}"]
            if ev.id in conflicts:
                classes.append("conflict")
            if ev.deadline_relevant:
                classes.append("deadline")
            if is_key_event(ev):
                classes.append("key")
            items.append(
                {
                    "id": ev.id,
                    "label": ev.description,
                    "start": ev.date_start.isoformat() if ev.date_start else None,
                    "end": end,
                    "date_text": ev.date_text,
                    "certainty": ev.certainty.value,
                    "status": ev.status.value,
                    "actors": ev.actors,
                    "quote": ev.source.quote,
                    "confidence": ev.confidence,
                    "deadline_relevant": ev.deadline_relevant,
                    "importance": ev.importance,
                    "key": is_key_event(ev),
                    "contradiction": ev.id in conflicts,
                    "class_name": " ".join(classes),
                    "source_url": source_url(ev),
                }
            )
        return JSONResponse(
            {
                "case_id": settings.case_id,
                "view": view,
                "key_threshold": KEY_EVENT_THRESHOLD,
                "total": len(every),
                "items": items,
            }
        )

    @app.post("/events/{event_id}/status", response_model=None)
    def event_status(
        request: Request,
        event_id: str,
        status: Annotated[str, Form()],
        view: Annotated[str | None, Form()] = None,
    ) -> HTMLResponse | RedirectResponse:
        try:
            new = EventStatus(status)
        except ValueError as exc:
            raise HTTPException(422, "status must be accepted or rejected") from exc
        if new not in REVIEW_STATUSES:
            raise HTTPException(422, "status must be accepted or rejected")
        store = _deps(request).get_store()
        if store.get_event(event_id) is None:
            raise HTTPException(404, "event not found")
        store.set_event_status(event_id, new)
        log.info("event %s marked %s by reviewer", event_id, new.value)
        view = timeline_view(view)
        if not _is_htmx(request):
            target = "/timeline?view=all" if view == VIEW_ALL else "/timeline"
            return RedirectResponse(target, status_code=303)
        ev = store.get_event(event_id)
        conflicts = _conflict_ids(store.list_contradictions(settings.case_id))
        return render(request, "_event_row.html", e=ev, conflicts=conflicts, view=view)

    # ------------------------------------------------------------------ source viewer

    @app.get("/source/{doc_id}", response_class=HTMLResponse)
    def source(
        request: Request,
        doc_id: str,
        start: int | None = None,
        end: int | None = None,
        event: str | None = None,
    ) -> HTMLResponse:
        if not _DOC_ID_RE.match(doc_id):
            raise HTTPException(404, "document not found")
        store = _deps(request).get_store()
        text, pages = _parsed_text(store.get_parsed_text(doc_id))
        if text is None:
            raise HTTPException(404, "document not found")
        body, highlighted = highlight(text, start, end)
        get_doc = getattr(store, "get_document", None)
        doc = get_doc(doc_id) if callable(get_doc) else None
        ev = store.get_event(event) if event else None
        if ev is not None and ev.source.doc_id != doc_id:
            ev = None
        page = _page_for(pages, start) if highlighted and start is not None else None
        conflicts = _conflict_ids(store.list_contradictions(settings.case_id)) if ev else set()
        return render(
            request,
            "source.html",
            doc_id=doc_id,
            doc=doc,
            body=body,
            highlighted=highlighted,
            requested=start is not None or end is not None,
            page=page,
            e=ev,
            conflicts=conflicts,
        )

    # ------------------------------------------------------------------ contradictions / alerts

    def contradiction_view(store: Any, c: Contradiction) -> dict[str, Any]:
        events = [ev for ev in (store.get_event(i) for i in c.event_ids) if ev is not None]
        return {"c": c, "events": events}

    @app.get("/contradictions", response_class=HTMLResponse)
    def contradictions(request: Request) -> HTMLResponse:
        store = _deps(request).get_store()
        items = [contradiction_view(store, c) for c in store.list_contradictions(settings.case_id)]
        order = {ContradictionStatus.OPEN: 0, ContradictionStatus.CONFIRMED: 1}
        items.sort(key=lambda v: order.get(v["c"].status, 2))
        return render(request, "contradictions.html", items=items)

    @app.post("/contradictions/{cid}/status", response_model=None)
    def contradiction_status(
        request: Request, cid: str, status: Annotated[str, Form()]
    ) -> HTMLResponse | RedirectResponse:
        try:
            new = ContradictionStatus(status)
        except ValueError as exc:
            raise HTTPException(422, "invalid status") from exc
        store = _deps(request).get_store()
        setter = getattr(store, "set_contradiction_status", None)
        if not callable(setter):
            raise HTTPException(501, "store does not support contradiction review yet")
        current = {c.id: c for c in store.list_contradictions(settings.case_id)}
        if cid not in current:
            raise HTTPException(404, "contradiction not found")
        setter(cid, new)
        log.info("contradiction %s marked %s by reviewer", cid, new.value)
        if not _is_htmx(request):
            return RedirectResponse("/contradictions", status_code=303)
        c = current[cid].model_copy(update={"status": new})
        return render(request, "_contradiction_status.html", c=c)

    @app.get("/alerts", response_class=HTMLResponse)
    def alerts(request: Request) -> HTMLResponse:
        items = _deps(request).get_store().list_alerts(settings.case_id)
        items = sorted(items, key=lambda a: a.created_at, reverse=True)
        return render(request, "alerts.html", alerts=items)

    @app.get("/cases/{case}/alerts/{alert_id}")
    def alert_link(case: str, alert_id: str) -> RedirectResponse:
        """Target of ``Alert.dashboard_url`` links sent over messaging."""
        if case != settings.case_id:
            raise HTTPException(404, "unknown case")
        return RedirectResponse(f"/alerts#alert-{quote(alert_id, safe='')}", status_code=303)

    # ------------------------------------------------------------------ chat

    def ask_openclaw(deps: Deps, question: str) -> str | None:
        exe = deps.openclaw_bin
        if deps.chat_backend == "local" or exe is None or shutil.which(str(exe)) is None:
            return None
        message = (
            f"[case-timeline case_id={settings.case_id}] Answer using the case-timeline "
            "tools only, cite each fact with its source link, and flag deadlines as "
            f"'needs attorney review'.\n\nQuestion: {question}"
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

    def ask_local(deps: Deps, question: str) -> tuple[str, list[dict[str, Any]], str | None]:
        events: list[Event] = deps.get_store().search_events(settings.case_id, question, CHAT_TOP_K)
        citations = [
            {"tag": f"E{i}", "event": ev, "url": source_url(ev)}
            for i, ev in enumerate(events, start=1)
        ]
        if not events:
            return "No matching events were found in this case.", [], None
        context = "\n\n".join(
            f"[{c['tag']}] date: {c['event'].date_text} "
            f"({c['event'].date_start or 'unresolved'}, {c['event'].certainty.value})\n"
            f"actors: {', '.join(c['event'].actors) or '-'}\n"
            f"event: {c['event'].description}\n"
            f'quote: "{c["event"].source.quote}"'
            for c in citations
        )
        messages = [
            {"role": "system", "content": CHAT_SYSTEM},
            {"role": "user", "content": f"Evidence excerpts:\n\n{context}\n\nQuestion: {question}"},
        ]
        try:
            answer = deps.get_llm().chat(messages)
        except Exception as exc:  # noqa: BLE001 - model may be offline; still show evidence
            log.warning("local LLM unavailable for chat: %s", exc)
            return "", citations, "The local model is unavailable; matching events are listed."
        return answer, citations, None

    @app.get("/chat", response_class=HTMLResponse)
    def chat_form(request: Request) -> HTMLResponse:
        return render(request, "chat.html", result=None)

    @app.post("/chat", response_class=HTMLResponse)
    def chat(request: Request, question: Annotated[str, Form()]) -> HTMLResponse:
        question = question.strip()[:4000]
        if not question:
            raise HTTPException(422, "empty question")
        deps = _deps(request)
        citations: list[dict[str, Any]] = []
        notice = None
        backend = "openclaw"
        answer = ask_openclaw(deps, question)
        if answer is None:
            if deps.chat_backend == "openclaw":
                notice = "OpenClaw agent unavailable."
                answer = ""
            else:
                backend = "local"
                answer, citations, notice = ask_local(deps, question)
        result = {
            "question": question,
            "answer_html": linkify_answer(answer, citations),
            "citations": citations,
            "notice": notice,
            "backend": backend,
        }
        name = "_chat_answer.html" if _is_htmx(request) else "chat.html"
        return render(request, name, result=result)

    return app
