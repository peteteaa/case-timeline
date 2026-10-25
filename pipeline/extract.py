"""Draft facts from a parsed document with the local LLM (SPEC §6.1 step 1).

The document is split into overlapping chunks; the model returns facts with verbatim quotes per
chunk. Offsets are not the model's job: ``verify`` locates every quote afterwards.
"""

from __future__ import annotations

import hashlib
import logging
import re

from pydantic import BaseModel, Field

from pipeline import prompts
from pipeline.schema import DraftFact, ParsedDocument

logger = logging.getLogger(__name__)

CHUNK_SIZE = 6000
CHUNK_OVERLAP = 500
HEAD_CHARS = 600  # start of the document shown as context for later chunks
MAX_QUOTES = 4
MAX_TITLE_CHARS = 200
UNCOPIED_DATE_PENALTY = 0.8  # date_text not found in the document (the model computed it)

_WS = re.compile(r"\s+")
_NO_DATE = re.compile(
    r"^(?:\(?\s*(?:no|without|missing)\s+(?:date|time)(?:\s+(?:given|stated|provided))?\s*\)?"
    r"|n/?a|none|null|unknown|undated|not\s+(?:dated|stated|given|specified))\.?$",
    re.IGNORECASE,
)
_PRONOUNS = frozenset(
    "i me my he him his she her hers they them their we us our you your it its someone somebody "
    "nobody everyone anyone".split()
)
_ORG_HINT = re.compile(
    r"\b(?:inc|llc|llp|ltd|corp|corporation|co|company|department|dept|division|police|court|"
    r"agency|bureau|office|hr|freight|bank|university|county|city|state)\b\.?$",
    re.IGNORECASE,
)


class ExtractedFact(BaseModel):
    """One fact as returned by the model (offsets are computed later, not by the LLM)."""

    title: str = Field(
        description="Short neutral headline: who did what to whom (no date), 6 to 14 words."
    )
    details: str = Field(default="", description="One or two neutral sentences, or empty.")
    date_text: str = Field(
        default="",
        description="This fact's time reference copied verbatim from the excerpt (never "
        "computed), e.g. 'June 5, 2021' or 'three weeks later'; empty if none.",
    )
    people: list[str] = Field(default_factory=list, description="Person names only.")
    quotes: list[str] = Field(
        default_factory=list, description="Verbatim sentences or clauses copied from the excerpt."
    )
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    deadline_language: bool = False


class ExtractionResult(BaseModel):
    """Structured-output schema for one chunk."""

    facts: list[ExtractedFact] = Field(default_factory=list)


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


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def draft_key(doc_id: str, first_quote: str, title: str) -> str:
    """Deterministic 16-hex key from doc id, first quote and title (whitespace/case-insensitive)."""
    raw = "\x1f".join((doc_id, _norm(first_quote).casefold(), _norm(title).casefold()))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def clean_people(names: list[str]) -> list[str]:
    """Person names only: drop blanks, pronouns and organization-like names; dedupe."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = _norm(raw).strip(" ,;:")
        low = name.casefold()
        if len(name) < 2 or low in _PRONOUNS or _ORG_HINT.search(name):
            continue
        if not any(ch.isalpha() for ch in name) or low in seen:
            continue
        seen.add(low)
        out.append(name)
    return out


def _squash(s: str) -> str:
    """Lowercase, unify quotes/dashes and drop whitespace for lenient containment checks."""
    s = s.casefold().translate(str.maketrans("\u2018\u2019\u201c\u201d\u2013\u2014", "''\"\"--"))
    return _WS.sub("", s)


def head_context(text: str, chunk_start: int, limit: int = HEAD_CHARS) -> str:
    """The document's opening (title, date, parties) for chunks that do not include it."""
    if chunk_start <= 0:
        return ""
    end = min(limit, chunk_start, len(text))
    cut = text.rfind("\n", 0, end)
    return text[: cut if cut > limit // 2 else end]


def _to_draft(
    parsed: ParsedDocument, item: ExtractedFact, squashed_text: str | None = None
) -> DraftFact | None:
    title = _norm(item.title)[:MAX_TITLE_CHARS].strip()
    if not title:
        return None
    date_text = _norm(item.date_text)
    if _NO_DATE.match(date_text):  # "no date", "N/A", "unknown": the model's way of saying none
        date_text = ""
    confidence = min(1.0, max(0.0, item.confidence))
    squashed = squashed_text if squashed_text is not None else _squash(parsed.text)
    if date_text and _squash(date_text) not in squashed:
        # The model computed or reworded the date instead of copying it; keep the fact (the
        # date is often right) but don't let it look as solid as a copied one.
        logger.debug("date_text %r not found verbatim in %s", date_text, parsed.doc.doc_id)
        confidence *= UNCOPIED_DATE_PENALTY
    quotes: list[str] = []
    for q in item.quotes:
        q = q.strip()
        if q and q not in quotes:
            quotes.append(q)
    quotes = quotes[:MAX_QUOTES]
    return DraftFact(
        key=draft_key(parsed.doc.doc_id, quotes[0] if quotes else "", title),
        title=title,
        details=_norm(item.details) or None,
        date_text=date_text or None,
        people=clean_people(item.people),
        quotes=quotes,
        confidence=confidence,
        deadline_language=item.deadline_language,
    )


def _merge_dup(a: DraftFact, b: DraftFact) -> DraftFact:
    """Combine two sightings of the same fact (from overlapping chunks)."""
    best, other = (a, b) if a.confidence >= b.confidence else (b, a)
    return best.model_copy(
        update={
            "people": best.people + [p for p in other.people if p not in best.people],
            "quotes": (best.quotes + [q for q in other.quotes if q not in best.quotes])[
                :MAX_QUOTES
            ],
            "details": best.details or other.details,
            "date_text": best.date_text or other.date_text,
            "confidence": max(a.confidence, b.confidence),
            "deadline_language": a.deadline_language or b.deadline_language,
        }
    )


def _sighting(d: DraftFact) -> tuple[str, str]:
    """Overlap-duplicate signature: same first quote and same date as written."""
    first = d.quotes[0] if d.quotes else ""
    return _norm(first).casefold(), _norm(d.date_text or "").casefold()


def draft_facts(parsed: ParsedDocument) -> list[DraftFact]:
    """Chunk the document, ask the LLM for facts per chunk, return deduplicated drafts.

    Chunks are sent to the model concurrently (bounded by ``settings.llm_concurrency``; the
    requests share ``pipeline.llm``'s process-wide limiter) and their results are combined in
    chunk order, so keys, cross-chunk dedupe and the returned order match a sequential run.
    A chunk whose LLM call fails is logged and skipped; if every chunk fails the last error is
    re-raised so the document stays queued for the next scan.
    """
    from pipeline import llm

    spans = chunk_text(parsed.text, CHUNK_SIZE, CHUNK_OVERLAP)
    squashed = _squash(parsed.text)
    drafts: dict[str, DraftFact] = {}
    chunk_of: dict[str, int] = {}
    by_sighting: dict[tuple[str, str], str] = {}
    failures = 0
    last_error: Exception | None = None

    def ask(item: tuple[int, tuple[int, int]]) -> object:
        i, (start, end) = item
        user = prompts.extraction_user(
            parsed.doc.filename,
            parsed.text[start:end],
            i,
            len(spans),
            head=head_context(parsed.text, start),
        )
        return llm.chat_json(prompts.EXTRACTION_SYSTEM, user, ExtractionResult)

    results = llm.map_ordered(ask, list(enumerate(spans, start=1)))
    for i, result in enumerate(results, start=1):
        if isinstance(result, Exception):  # one bad chunk must not sink the whole document
            failures += 1
            last_error = result
            logger.warning(
                "extraction failed for %s chunk %d/%d: %s", parsed.doc.doc_id, i, len(spans), result
            )
            continue
        if not isinstance(result, ExtractionResult):
            result = ExtractionResult.model_validate(result, from_attributes=True)
        for item in result.facts:
            d = _to_draft(parsed, item, squashed)
            if d is None:
                continue
            sig = _sighting(d)
            key = d.key
            if key not in drafts and sig[0] and sig in by_sighting:
                other = by_sighting[sig]
                if chunk_of[other] != i:  # same quote and date seen in an overlapping chunk
                    key = other
            if key in drafts:
                drafts[key] = _merge_dup(drafts[key], d).model_copy(update={"key": key})
            else:
                drafts[key] = d
                chunk_of[key] = i
                if sig[0]:
                    by_sighting.setdefault(sig, key)
    if spans and failures == len(spans) and last_error is not None:
        raise last_error
    logger.info("drafted %d facts from %s (%d chunks)", len(drafts), parsed.doc.doc_id, len(spans))
    return list(drafts.values())
