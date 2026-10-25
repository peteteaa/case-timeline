"""Ground extracted events in the source text: locate each quote and record exact offsets."""

from __future__ import annotations

import logging
import re
import unicodedata

from rapidfuzz import fuzz

from pipeline.contracts import Event, EventStatus, Page, ParsedDocument, SourceRef

log = logging.getLogger(__name__)

FUZZY_THRESHOLD = 90.0
MIN_FUZZY_LEN = 12  # shorter quotes match too easily by chance
UNVERIFIED_PENALTY = 0.5

_CHAR_MAP = {
    **dict.fromkeys("‘’‚‛′", "'"),
    **dict.fromkeys("“”„‟″", '"'),
    **dict.fromkeys("‐‑‒–—―−", "-"),
    "­": "",  # soft hyphen
    "​": "",  # zero-width space
}
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\s*\.\s*\.|…)\s*")
_STRIP_CHARS = " \t\n\"'‘’“”"


def _normalize_with_map(s: str) -> tuple[str, list[int]]:
    """Casefold, straighten quotes/dashes and collapse whitespace.

    Returns the normalized string and, for each of its characters, the index of the
    original character it came from.
    """
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(s):
        ch = _CHAR_MAP.get(ch, ch)
        if not ch:
            continue
        if ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                idx.append(i)
            continue
        for c in unicodedata.normalize("NFKC", ch).lower():
            out.append(c)
            idx.append(i)
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def _normalize(s: str) -> str:
    return _normalize_with_map(unicodedata.normalize("NFC", s))[0]


def _span(idx: list[int], start: int, end: int) -> tuple[int, int]:
    """Map a [start, end) span in normalized text back to original offsets."""
    return idx[start], idx[end - 1] + 1


def _snap_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a fuzzy span so it does not cut words in half, then trim whitespace."""
    while start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum() and text[end - 1].isalnum():
        end += 1
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _locate_elided(nt: str, idx: list[int], quote: str) -> tuple[int, int] | None:
    """Handle quotes with "..." elisions: find each fragment in order."""
    frags = [_normalize(f) for f in _ELLIPSIS_RE.split(quote)]
    frags = [f for f in frags if f]
    if len(frags) < 2:
        return None
    first = nt.find(frags[0])
    if first < 0:
        return None
    pos = first + len(frags[0])
    for frag in frags[1:]:
        hit = nt.find(frag, pos)
        if hit < 0:
            return None
        pos = hit + len(frag)
    return _span(idx, first, pos)


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Find ``quote`` in ``text``; returns original (start, end) offsets or None.

    Tries an exact match, then a whitespace/quote/case-normalized match, then an
    elision-aware match, then rapidfuzz partial alignment (score >= 90).
    """
    quote = quote.strip(_STRIP_CHARS) or quote.strip()
    if not quote or not text:
        return None
    hit = text.find(quote)
    if hit >= 0:
        return hit, hit + len(quote)

    nt, idx = _normalize_with_map(text)
    nq = _normalize(quote)
    if not nq or not nt:
        return None
    hit = nt.find(nq)
    if hit >= 0:
        return _span(idx, hit, hit + len(nq))

    if span := _locate_elided(nt, idx, quote):
        return span

    if len(nq) < MIN_FUZZY_LEN or len(nq) > len(nt):
        return None
    al = fuzz.partial_ratio_alignment(nq, nt, score_cutoff=FUZZY_THRESHOLD)
    if al is None or al.dest_end <= al.dest_start:
        return None
    start, end = _snap_to_words(text, *_span(idx, al.dest_start, al.dest_end))
    if end <= start:
        return None
    log.debug("fuzzy quote match %.1f at %d-%d", al.score, start, end)
    return start, end


def page_for_offset(pages: list[Page], offset: int) -> int | None:
    """1-based page containing ``offset``; separator gaps count toward the next page."""
    ordered = sorted((p for p in pages if p.char_end > p.char_start), key=lambda p: p.char_start)
    for page in ordered:
        if offset < page.char_end:
            return page.number
    return ordered[-1].number if ordered else None


def verify_events(parsed: ParsedDocument, events: list[Event]) -> list[Event]:
    """Attach verbatim quotes, offsets and pages; mark unlocatable events UNVERIFIED."""
    text = parsed.text
    out: list[Event] = []
    for ev in events:
        span = locate_quote(text, ev.source.quote)
        if span is None:
            log.info("quote not found for event %s in %s", ev.id, parsed.doc.filename)
            source = SourceRef(
                doc_id=parsed.doc.id, page=None, char_start=0, char_end=0, quote=ev.source.quote
            )
            out.append(
                ev.model_copy(
                    update={
                        "source": source,
                        "status": EventStatus.UNVERIFIED,
                        "confidence": ev.confidence * UNVERIFIED_PENALTY,
                    }
                )
            )
            continue
        start, end = span
        source = SourceRef(
            doc_id=parsed.doc.id,
            page=page_for_offset(parsed.pages, start),
            char_start=start,
            char_end=end,
            quote=text[start:end],
        )
        out.append(ev.model_copy(update={"source": source}))
    return out
