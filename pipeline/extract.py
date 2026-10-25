"""Turn a parsed document into proposed timeline events via the local LLM."""

from __future__ import annotations

import hashlib
import importlib
import logging
import re

from pydantic import BaseModel, Field

from pipeline import prompts
from pipeline.contracts import Event, EventStatus, ParsedDocument, SourceRef
from pipeline.llm import map_ordered

logger = logging.getLogger(__name__)

CHUNK_SIZE = 6000
CHUNK_OVERLAP = 500
HEAD_CHARS = 600  # start of the document shown as context for later chunks
# Second "did you miss anything?" pass per chunk. Off by default: it roughly doubles
# extraction latency; enable when recall matters more than scan time.
GLEAN_PASSES = 0
UNCOPIED_DATE_PENALTY = 0.8  # date_text not found in the excerpt (model computed it)

_WS = re.compile(r"\s+")


class ExtractedEvent(BaseModel):
    """One event as returned by the model (offsets are computed later, not by the LLM)."""

    quote: str = Field(description="Verbatim sentence or clause copied from the excerpt.")
    date_text: str = Field(
        description="This event's time reference copied verbatim from the excerpt "
        "(never computed), e.g. 'June 5, 2021' or 'three weeks later'."
    )
    actors: list[str] = Field(default_factory=list, description="Named people/organizations.")
    description: str = Field(
        description="One short neutral sentence: who did what to whom (no date or location)."
    )
    deadline_relevant: bool = False
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Structured-output schema for one chunk."""

    events: list[ExtractedEvent] = Field(default_factory=list)


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[tuple[int, int]]:
    """Split text into overlapping (start, end) spans, preferring paragraph boundaries."""
    n = len(text)
    if n == 0:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            end = _break_before(text, start + size // 2, end)
        spans.append((start, end))
        if end >= n:
            break
        nxt = _break_after(text, max(end - overlap, start + 1), end)
        start = nxt if nxt > start else end
    return spans


def _break_before(text: str, lo: int, hi: int) -> int:
    """Best boundary in text[lo:hi]: paragraph, then line, then sentence, then whitespace."""
    window = text[lo:hi]
    for pattern in (r"\n\s*\n", r"\n", r"[.!?][\"')\]]?\s", r"\s"):
        matches = list(re.finditer(pattern, window))
        if matches:
            return lo + matches[-1].end()
    return hi


def _break_after(text: str, lo: int, hi: int) -> int:
    """Earliest good boundary in text[lo:hi] to start the next (overlapping) chunk."""
    window = text[lo:hi]
    for pattern in (r"\n\s*\n", r"\n", r"[.!?][\"')\]]?\s", r"\s"):
        m = re.search(pattern, window)
        if m and lo + m.end() < hi:
            return lo + m.end()
    return lo


def event_id(doc_id: str, quote: str, date_text: str) -> str:
    """Deterministic 16-hex id from doc id, quote and date text (whitespace-normalized)."""
    key = "\x1f".join((doc_id, _norm(quote), _norm(date_text)))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _chat_json(system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    """Resolve pipeline.llm lazily so this module imports even before llm.py exists."""
    llm = importlib.import_module("pipeline.llm")
    return llm.chat_json(system, user, schema)


def _squash(s: str) -> str:
    """Lowercase, unify quotes/dashes and drop whitespace for lenient containment checks."""
    s = s.lower().translate(str.maketrans("\u2018\u2019\u201c\u201d\u2013\u2014", "''\"\"--"))
    return _WS.sub("", s)


def _head(text: str, chunk_start: int, limit: int = HEAD_CHARS) -> str:
    """The document's opening (title, date, parties) for chunks that do not include it."""
    if chunk_start <= 0:
        return ""
    end = min(limit, chunk_start, len(text))
    cut = text.rfind("\n", 0, end)
    return text[: cut if cut > limit // 2 else end]


def _page_for(parsed: ParsedDocument, offset: int) -> int | None:
    for page in parsed.pages:
        if page.char_start <= offset < page.char_end:
            return page.number
    return None


def _to_event(
    parsed: ParsedDocument, item: ExtractedEvent, chunk_start: int, chunk_end: int
) -> Event | None:
    quote = item.quote.strip()
    date_text = item.date_text.strip()
    if not quote or not item.description.strip():
        return None
    chunk = parsed.text[chunk_start:chunk_end]
    idx = chunk.find(quote)
    if idx >= 0:
        # Provisional document offsets; verify.locate_quote is authoritative.
        char_start = chunk_start + idx
        char_end = char_start + len(quote)
        page = _page_for(parsed, char_start)
    else:
        # Not found verbatim: point at the chunk span as a search hint for verify.
        char_start, char_end, page = chunk_start, chunk_end, None
    actors: list[str] = []
    for actor in item.actors:
        actor = _norm(actor)
        if actor and actor not in actors:
            actors.append(actor)
    confidence = min(1.0, max(0.0, item.confidence))
    if date_text and _squash(date_text) not in _squash(parsed.text):
        # The model computed or reworded the date instead of copying it; keep the event
        # (the date is often right) but do not let it look as solid as a copied one.
        logger.debug("date_text %r not found verbatim in %s", date_text, parsed.doc.id)
        confidence *= UNCOPIED_DATE_PENALTY
    doc_id = parsed.doc.id
    return Event(
        id=event_id(doc_id, quote, date_text),
        case_id=parsed.doc.case_id,
        date_text=date_text,
        actors=actors,
        description=_norm(item.description),
        source=SourceRef(
            doc_id=doc_id, page=page, char_start=char_start, char_end=char_end, quote=quote
        ),
        confidence=confidence,
        status=EventStatus.PROPOSED,
        deadline_relevant=item.deadline_relevant,
    )


def _merge_dup(a: Event, b: Event) -> Event:
    """Combine two extractions of the same event (from overlapping chunks)."""

    def rank(e: Event) -> tuple[bool, float]:
        located = e.source.char_end - e.source.char_start == len(e.source.quote)
        return (located, e.confidence)

    best, other = (a, b) if rank(a) >= rank(b) else (b, a)
    actors = list(best.actors) + [x for x in other.actors if x not in best.actors]
    return best.model_copy(
        update={
            "actors": actors,
            "confidence": max(a.confidence, b.confidence),
            "deadline_relevant": a.deadline_relevant or b.deadline_relevant,
        }
    )


def _ask(system: str, user: str) -> ExtractionResult:
    result = _chat_json(system, user, ExtractionResult)
    if not isinstance(result, ExtractionResult):
        result = ExtractionResult.model_validate(result, from_attributes=True)
    return result


def _summary(item: ExtractedEvent) -> str:
    return f"{_norm(item.date_text)}: {_norm(item.description)}"


def _extract_chunk(parsed: ParsedDocument, start: int, end: int, i: int, n: int) -> list[Event]:
    """First pass plus optional gleaning passes for one chunk."""
    user = prompts.extraction_user(
        parsed.doc.filename, parsed.text[start:end], i, n, head=_head(parsed.text, start)
    )
    items = list(_ask(prompts.EXTRACTION_SYSTEM, user).events)
    for _ in range(GLEAN_PASSES):
        try:
            more = _ask(
                prompts.EXTRACTION_SYSTEM,
                user + "\n\n" + prompts.glean_user([_summary(x) for x in items]),
            ).events
        except Exception as exc:  # gleaning is best effort; keep the first pass
            logger.warning("gleaning failed for %s chunk %d/%d: %s", parsed.doc.id, i, n, exc)
            break
        seen = {(_squash(x.quote), _squash(x.date_text)) for x in items}
        new = [x for x in more if (_squash(x.quote), _squash(x.date_text)) not in seen]
        logger.info("gleaning found %d new events in %s chunk %d/%d", len(new), parsed.doc.id, i, n)
        if not new:
            break
        items.extend(new)
    out: list[Event] = []
    for item in items:
        ev = _to_event(parsed, item, start, end)
        if ev is not None:
            out.append(ev)
    return out


def _map_chunks(
    parsed: ParsedDocument, spans: list[tuple[int, int]]
) -> list[list[Event] | Exception]:
    """Run ``_extract_chunk`` for every span, concurrently when allowed; results in span order.

    Each element is the chunk's event list or the exception its LLM call raised. The model
    requests share ``pipeline.llm``'s process-wide limiter, so this fan-out never exceeds
    ``settings.llm_concurrency`` requests in flight.
    """
    n = len(spans)

    def run(item: tuple[int, tuple[int, int]]) -> list[Event]:
        i, (start, end) = item
        return _extract_chunk(parsed, start, end, i, n)

    return map_ordered(run, list(enumerate(spans, start=1)))


def extract_events(parsed: ParsedDocument) -> list[Event]:
    """Chunk the document, ask the LLM for events per chunk, and return deduped Events.

    Chunks are sent to the model concurrently (bounded by ``settings.llm_concurrency``), and
    their results are combined in chunk order, so ids, cross-chunk dedupe and the returned
    order are the same as a sequential run. A chunk whose LLM call fails is logged and
    skipped; if every chunk fails the last error is re-raised so the caller can retry the
    document later.
    """
    spans = chunk_text(parsed.text, CHUNK_SIZE, CHUNK_OVERLAP)
    events: dict[str, Event] = {}
    failures = 0
    last_error: Exception | None = None
    for i, result in enumerate(_map_chunks(parsed, spans), start=1):
        if isinstance(result, Exception):  # one bad chunk must not sink the whole document
            failures += 1
            last_error = result
            logger.warning(
                "extraction failed for %s chunk %d/%d: %s", parsed.doc.id, i, len(spans), result
            )
            continue
        for ev in result:
            events[ev.id] = _merge_dup(events[ev.id], ev) if ev.id in events else ev
    if spans and failures == len(spans) and last_error is not None:
        raise last_error
    logger.info("extracted %d events from %s (%d chunks)", len(events), parsed.doc.id, len(spans))
    return sorted(events.values(), key=lambda e: (e.source.char_start, e.id))
