"""Deduplicate newly extracted events against events already in the timeline."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from datetime import date

from rapidfuzz import fuzz

from pipeline.contracts import Event

log = logging.getLogger(__name__)

FUZZ_THRESHOLD = 85.0
COSINE_THRESHOLD = 0.9

_WS = re.compile(r"\s+")


def _norm_text(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def normalize_actor(name: str) -> str:
    """Case-fold and collapse whitespace so actor names compare loosely."""
    return _norm_text(name)


def actor_set(event: Event) -> set[str]:
    """Normalized, non-empty actor names of an event."""
    return {a for a in (normalize_actor(x) for x in event.actors) if a}


def _interval(event: Event) -> tuple[date, date] | None:
    if event.date_start is None:
        return None
    end = event.date_end or event.date_start
    return (min(event.date_start, end), max(event.date_start, end))


def date_gap_days(a: Event, b: Event) -> int | None:
    """Days between two resolved date intervals (0 if they overlap); None if either is unknown."""
    ia, ib = _interval(a), _interval(b)
    if ia is None or ib is None:
        return None
    if ia[1] < ib[0]:
        return (ib[0] - ia[1]).days
    if ib[1] < ia[0]:
        return (ia[0] - ib[1]).days
    return 0


def same_date(a: Event, b: Event) -> bool:
    """Resolved intervals overlap, or both unresolved with identical date text."""
    gap = date_gap_days(a, b)
    if gap is not None:
        return gap == 0
    if a.date_start is None and b.date_start is None:
        ta, tb = _norm_text(a.date_text), _norm_text(b.date_text)
        return bool(ta) and ta == tb
    return False


def actors_compatible(a: Event, b: Event) -> bool:
    """True unless both events name actors and the sets are disjoint."""
    sa, sb = actor_set(a), actor_set(b)
    if sa and sb:
        return bool(sa & sb)
    return True


def description_similarity(a: Event, b: Event) -> float:
    """rapidfuzz token_set_ratio (0-100) over the event descriptions."""
    return fuzz.token_set_ratio(a.description, b.description, processor=_preprocess)


def _preprocess(text: str) -> str:
    return _WS.sub(" ", re.sub(r"[^\w\s]", " ", text)).strip().casefold()


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


def duplicate_pairs(new: list[Event], existing: list[Event]) -> list[tuple[Event, str]]:
    """Return ``(new_event, canonical_event_id)`` for every new event that duplicates another.

    New events are compared with existing events and with earlier events of the same batch.
    Embeddings are computed only for pairs that pass the date/actor gate but not the fuzzy test.
    """
    pool: list[Event] = list(existing)
    existing_ids = {e.id for e in existing}
    # candidate matches per new event: (target_id, fuzzy_hit)
    candidates: dict[str, list[tuple[Event, bool]]] = {}
    for i, ev in enumerate(new):
        targets = pool + new[:i]
        hits: list[tuple[Event, bool]] = []
        for other in targets:
            if other.id == ev.id or not same_date(ev, other) or not actors_compatible(ev, other):
                continue
            hits.append((other, description_similarity(ev, other) >= FUZZ_THRESHOLD))
        candidates[ev.id] = hits

    vectors = _embed_needed(new, candidates)

    canonical: dict[str, str] = {}  # dup new-event id -> canonical id
    pairs: list[tuple[Event, str]] = []
    for ev in new:
        if ev.id in existing_ids:
            continue  # already stored; nothing to merge
        for other, fuzzy_hit in candidates[ev.id]:
            if not fuzzy_hit and not _cosine_hit(ev, other, vectors):
                continue
            target = canonical.get(other.id, other.id)
            canonical[ev.id] = target
            pairs.append((ev, target))
            break
    return pairs


def _embed_needed(
    new: list[Event], candidates: dict[str, list[tuple[Event, bool]]]
) -> dict[str, list[float]]:
    """Embed descriptions only for events in candidate pairs that need the semantic check."""
    needed: dict[str, str] = {}
    for ev in new:
        hits = candidates[ev.id]
        if any(fuzzy for _, fuzzy in hits):
            continue
        for other, _ in hits:
            needed[ev.id] = ev.description
            needed[other.id] = other.description
    if not needed:
        return {}
    ids = list(needed)
    vecs = _embed([needed[i] for i in ids])
    if not vecs or len(vecs) != len(ids):
        return {}
    return dict(zip(ids, vecs, strict=True))


def _cosine_hit(a: Event, b: Event, vectors: dict[str, list[float]]) -> bool:
    va, vb = vectors.get(a.id), vectors.get(b.id)
    return va is not None and vb is not None and _cosine(va, vb) >= COSINE_THRESHOLD


def merge_events_linked(
    new: list[Event], existing: list[Event]
) -> tuple[list[Event], list[tuple[Event, str]]]:
    """Like :func:`merge_events` but returns ``(to_insert, [(duplicate, canonical_id), ...])``."""
    pairs = duplicate_pairs(new, existing)
    dup_ids = {ev.id for ev, _ in pairs}
    existing_ids = {e.id for e in existing}
    to_insert = [e for e in new if e.id not in dup_ids and e.id not in existing_ids]
    log.info("merge: %d new, %d duplicates", len(to_insert), len(pairs))
    return to_insert, pairs


def merge_events(new: list[Event], existing: list[Event]) -> tuple[list[Event], list[Event]]:
    """Split ``new`` into ``(to_insert, duplicates)``.

    ``to_insert`` are genuinely new events. ``duplicates`` are new events that repeat an existing
    (or earlier in-batch) event; use :func:`merge_events_linked` / :func:`duplicate_pairs` for
    their canonical ids so the store can keep both sources via ``link_duplicate``.
    """
    to_insert, pairs = merge_events_linked(new, existing)
    return to_insert, [ev for ev, _ in pairs]
