"""Decide which drafts describe a fact already on the timeline (SPEC §6.1 step 4).

A draft merges into an existing fact only when all three hold:
- **dates compatible**: both dated with overlapping ranges, or both undated with the same (or no)
  date text;
- **people compatible**: the draft's witnesses and the fact's witnesses overlap, or either side
  names nobody;
- **similar**: rapidfuzz similarity of title + details (or a shared quote with a related
  title), falling back to embedding cosine for pairs that pass the first two gates.
Drafts that look like a fact but disagree on the date or the people are left alone: that is a
possible contradiction, which ``conflicts`` checks. Rejected facts are never merge targets.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date

from rapidfuzz import fuzz

from pipeline.schema import DraftFact, Fact

log = logging.getLogger(__name__)

FUZZ_THRESHOLD = 85.0
QUOTE_THRESHOLD = 90.0  # a quote that matches one already cited by the fact
QUOTE_TITLE_THRESHOLD = 50.0  # ...still needs a related title
COSINE_THRESHOLD = 0.9

_WS = re.compile(r"\s+")


def _norm_text(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def _preprocess(text: str) -> str:
    return _WS.sub(" ", re.sub(r"[^\w\s]", " ", text)).strip().casefold()


def similarity(a: str, b: str) -> float:
    """rapidfuzz token_set_ratio (0-100) on punctuation-free, case-folded text."""
    return fuzz.token_set_ratio(a, b, processor=_preprocess)


def fact_text(title: str, details: str | None) -> str:
    return f"{title}. {details}" if details else title


def fact_similarity(
    title_a: str, details_a: str | None, title_b: str, details_b: str | None
) -> float:
    """Best of title-only and title-plus-details similarity (details often differ in length)."""
    return max(
        similarity(title_a, title_b),
        similarity(fact_text(title_a, details_a), fact_text(title_b, details_b)),
    )


Interval = tuple[date, date]


def interval(start: date | None, end: date | None) -> Interval | None:
    if start is None:
        return None
    end = end or start
    return (min(start, end), max(start, end))


def gap_days(a: Interval | None, b: Interval | None) -> int | None:
    """Days between two date intervals (0 if they overlap); None if either is undated."""
    if a is None or b is None:
        return None
    if a[1] < b[0]:
        return (b[0] - a[1]).days
    if b[1] < a[0]:
        return (a[0] - b[1]).days
    return 0


def dates_compatible(draft: DraftFact, fact: Fact) -> bool:
    gap = gap_days(
        interval(draft.date_start, draft.date_end), interval(fact.date_start, fact.date_end)
    )
    if gap is not None:
        return gap == 0
    if draft.date_start is None and fact.date_start is None:
        ta, tb = _norm_text(draft.date_text or ""), _norm_text(fact.date_text or "")
        return not ta or not tb or ta == tb
    return False  # one dated, one not: not enough to call them the same event


def people_compatible(draft_witnesses: Sequence[str], fact: Fact) -> bool:
    """True unless both sides name people and the sets are disjoint."""
    a, b = set(draft_witnesses), set(fact.witness_ids)
    return not (a and b) or bool(a & b)


def _quote_hit(draft: DraftFact, fact: Fact) -> bool:
    """The draft cites a span the fact already cites, or a near-identical quote."""
    for span in draft.spans:
        for ev in fact.evidence:
            if ev.kind != "span" or ev.quote is None:
                continue
            if (
                ev.doc_id == span.doc_id
                and ev.char_start is not None
                and ev.char_end is not None
                and ev.char_start < span.char_end
                and span.char_start < ev.char_end
            ):
                return True
            if fuzz.ratio(_preprocess(span.quote), _preprocess(ev.quote)) >= QUOTE_THRESHOLD:
                return True
    return False


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v, strict=False))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv) if nu and nv else 0.0


def _embed(texts: list[str]) -> list[list[float]] | None:
    """Embed via the local model; None when the embedder is unavailable."""
    try:
        from pipeline import llm

        return llm.embed(texts)
    except Exception as exc:  # embedder optional: fall back to fuzzy matching only
        log.warning("embedding unavailable, using fuzzy matching only: %s", exc)
        return None


def plan(
    drafts: Sequence[DraftFact],
    existing: Sequence[Fact],
    *,
    witness_ids: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str | None]:
    """Map each draft key to the existing fact id it duplicates, or None for a new fact.

    ``witness_ids`` (draft key → matched witness ids) drives the people check; without it,
    people don't block a merge.
    """
    people = witness_ids or {}
    live = [f for f in existing if f.review != "rejected"]
    # (draft key) -> [(fact, fuzzy score, fuzzy hit)] for facts passing the date/people gates
    candidates: dict[str, list[tuple[Fact, float, bool]]] = {}
    for d in drafts:
        hits: list[tuple[Fact, float, bool]] = []
        for f in live:
            if not dates_compatible(d, f) or not people_compatible(people.get(d.key, ()), f):
                continue
            score = fact_similarity(d.title, d.details, f.title, f.details)
            hit = score >= FUZZ_THRESHOLD or (score >= QUOTE_TITLE_THRESHOLD and _quote_hit(d, f))
            hits.append((f, score, hit))
        candidates[d.key] = hits

    vectors = _embed_needed(drafts, candidates)
    out: dict[str, str | None] = {}
    for d in drafts:
        dtext = fact_text(d.title, d.details)
        best: tuple[float, float, str] | None = None
        for f, score, hit in candidates[d.key]:
            if not hit:
                cos = _cosine_of(vectors, f"d:{d.key}", f"f:{f.fact_id}")
                if cos is None or cos < COSINE_THRESHOLD:
                    continue
                score = max(score, cos * 100)
            # token_set_ratio ties at 100 for any subset; break ties on the plain ratio.
            tie = fuzz.ratio(_preprocess(dtext), _preprocess(fact_text(f.title, f.details)))
            if best is None or (score, tie) > best[:2]:
                best = (score, tie, f.fact_id)
        out[d.key] = best[2] if best else None
    n = sum(1 for v in out.values() if v)
    log.info("merge plan: %d new, %d attach to existing facts", len(out) - n, n)
    return out


def _embed_needed(
    drafts: Sequence[DraftFact], candidates: dict[str, list[tuple[Fact, float, bool]]]
) -> dict[str, list[float]]:
    """Embed only drafts (and facts) in candidate pairs that the fuzzy test didn't settle."""
    needed: dict[str, str] = {}
    for d in drafts:
        hits = candidates[d.key]
        if any(hit for _, _, hit in hits):
            continue
        for f, _, _ in hits:
            needed[f"d:{d.key}"] = fact_text(d.title, d.details)
            needed[f"f:{f.fact_id}"] = fact_text(f.title, f.details)
    if not needed:
        return {}
    ids = list(needed)
    vecs = _embed([needed[i] for i in ids])
    if not vecs or len(vecs) != len(ids):
        return {}
    return dict(zip(ids, vecs, strict=True))


def _cosine_of(vectors: dict[str, list[float]], a: str, b: str) -> float | None:
    va, vb = vectors.get(a), vectors.get(b)
    if va is None or vb is None:
        return None
    return _cosine(va, vb)
