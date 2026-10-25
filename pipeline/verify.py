"""Ground drafted facts in the source text (SPEC §6.1 step 2): no LLM here.

Every quote the model wrote is located in the parsed text (exact, then normalized, then
elision-aware, then fuzzy) and becomes a :class:`SpanIn` whose ``quote`` is exactly
``parsed.text[char_start:char_end]``, the check the store repeats before writing evidence.
Quotes that can't be located are dropped; drafts left without any span are discarded.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from rapidfuzz import fuzz

from pipeline.schema import DraftFact, Line, Page, ParsedDocument, SpanIn

log = logging.getLogger(__name__)

FUZZY_THRESHOLD = 90.0
MIN_FUZZY_LEN = 12  # shorter quotes match too easily by chance

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
    """1-based page containing ``offset``; separator gaps count toward the next page.

    Text before the first page (e.g. a caption ahead of a ``[Page 14]`` marker) has no page.
    """
    ordered = sorted((p for p in pages if p.char_end > p.char_start), key=lambda p: p.char_start)
    if not ordered or offset < ordered[0].char_start:
        return None
    for page in ordered:
        if offset < page.char_end:
            return page.number
    return ordered[-1].number


def lines_for_span(lines: list[Line], start: int, end: int) -> tuple[Line, Line] | None:
    """First and last transcript line overlapping ``[start, end)``, or None."""
    hits = [ln for ln in lines if ln.char_start < end and ln.char_end > start]
    if not hits:
        return None
    hits.sort(key=lambda ln: ln.char_start)
    return hits[0], hits[-1]


def make_span(parsed: ParsedDocument, start: int, end: int) -> SpanIn:
    """A span over ``parsed.text[start:end]`` with page and transcript lines filled."""
    page = page_for_offset(parsed.pages, start)
    line_start = line_end = None
    if found := lines_for_span(parsed.lines, start, end):
        first, last = found
        line_start, line_end = first.number, last.number
        page = page if page is not None else first.page
    return SpanIn(
        doc_id=parsed.doc.doc_id,
        page=page,
        char_start=start,
        char_end=end,
        line_start=line_start,
        line_end=line_end,
        quote=parsed.text[start:end],
    )


def verify_drafts(
    parsed: ParsedDocument, drafts: list[DraftFact]
) -> tuple[list[DraftFact], dict[str, int]]:
    """Locate every quote; returns ``(drafts with spans, counts)``.

    Counts: ``quotes_checked``, ``quotes_rejected`` (not found verbatim or near-verbatim) and
    ``drafts_discarded`` (no quote survived). Inputs are not mutated.
    """
    text = parsed.text
    counts = {"quotes_checked": 0, "quotes_rejected": 0, "drafts_discarded": 0}
    out: list[DraftFact] = []
    for draft in drafts:
        spans: list[SpanIn] = []
        seen: set[tuple[int, int]] = set()
        for quote in draft.quotes:
            if not quote.strip():
                continue
            counts["quotes_checked"] += 1
            found = locate_quote(text, quote)
            if found is not None:
                start, end = found
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                found = (start, end) if end > start else None
            if found is None:
                counts["quotes_rejected"] += 1
                log.info("quote not found in %s: %.80r", parsed.doc.filename, quote)
                continue
            if found in seen:
                continue
            seen.add(found)
            spans.append(make_span(parsed, *found))
        if not spans:
            counts["drafts_discarded"] += 1
            continue
        out.append(draft.model_copy(update={"spans": spans}))
    return out, counts
