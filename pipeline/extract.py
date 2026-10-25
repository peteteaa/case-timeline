"""Turn a parsed document into proposed timeline events via the local LLM."""

from __future__ import annotations

import hashlib
import importlib
import logging
import re

from pydantic import BaseModel, Field

from pipeline import prompts
from pipeline.contracts import Event, EventStatus, ParsedDocument, SourceRef

logger = logging.getLogger(__name__)

CHUNK_SIZE = 6000
CHUNK_OVERLAP = 500

_WS = re.compile(r"\s+")


class ExtractedEvent(BaseModel):
    """One event as returned by the model (offsets are computed later, not by the LLM)."""

    quote: str = Field(description="Verbatim sentence or clause copied from the excerpt.")
    date_text: str = Field(description="The date exactly as written in the excerpt.")
    actors: list[str] = Field(default_factory=list, description="Named people/organizations.")
    description: str = Field(description="One neutral sentence describing the event.")
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
        confidence=min(1.0, max(0.0, item.confidence)),
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


def extract_events(parsed: ParsedDocument) -> list[Event]:
    """Chunk the document, ask the LLM for events per chunk, and return deduped Events.

    A chunk whose LLM call fails is logged and skipped; if every chunk fails the last
    error is re-raised so the caller can retry the document later.
    """
    spans = chunk_text(parsed.text, CHUNK_SIZE, CHUNK_OVERLAP)
    events: dict[str, Event] = {}
    failures = 0
    last_error: Exception | None = None
    for i, (start, end) in enumerate(spans, start=1):
        user = prompts.extraction_user(parsed.doc.filename, parsed.text[start:end], i, len(spans))
        try:
            result = _chat_json(prompts.EXTRACTION_SYSTEM, user, ExtractionResult)
        except Exception as exc:  # one bad chunk must not sink the whole document
            failures += 1
            last_error = exc
            logger.warning(
                "extraction failed for %s chunk %d/%d: %s", parsed.doc.id, i, len(spans), exc
            )
            continue
        if not isinstance(result, ExtractionResult):
            result = ExtractionResult.model_validate(result, from_attributes=True)
        for item in result.events:
            ev = _to_event(parsed, item, start, end)
            if ev is None:
                continue
            events[ev.id] = _merge_dup(events[ev.id], ev) if ev.id in events else ev
    if spans and failures == len(spans) and last_error is not None:
        raise last_error
    logger.info("extracted %d events from %s (%d chunks)", len(events), parsed.doc.id, len(spans))
    return sorted(events.values(), key=lambda e: (e.source.char_start, e.id))
