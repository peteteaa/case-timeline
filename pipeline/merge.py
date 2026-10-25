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
# token_set_ratio scores 100 whenever one description's words are a subset of the other's
# ("Meeting held." vs "Meeting held with X about Y"), so a fuzzy hit also needs this much
# whole-string agreement (token_sort_ratio), which penalizes large length/content differences.
SORT_THRESHOLD = 55.0
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


# Titles and honorifics dropped before comparing names ("Dr. Lopez" ~ "Ana Lopez").
_HONORIFICS = frozenset(
    "mr mrs ms miss mx dr prof professor officer ofc sgt sergeant det detective cpl corporal "
    "lt lieutenant capt captain cmdr commander chief judge justice hon honorable atty attorney "
    "counsel esq sir madam dame rev father agent deputy sheriff trooper inspector investigator "
    "witness plaintiff defendant the".split()
)
# Words that mark an organization rather than a person.
_ORG_CUES = frozenset(
    "co company corp corporation inc incorporated llc llp lp ltd plc gmbh group holdings "
    "department dept division agency board bureau office administration authority commission "
    "council court police sheriff's university school hospital bank association union "
    "airlines airline services industries partners firm foundation trust committee state "
    "county city federal national".split()
)
_NAME_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?")


def name_tokens(name: str) -> tuple[str, ...]:
    """Lower-cased name tokens with honorifics, initials and possessives removed."""
    toks = []
    for raw in _NAME_TOKEN.findall(name.casefold()):
        tok = re.sub(r"['’]s$", "", raw)
        if tok in _HONORIFICS or (len(tok) == 1 and tok.isalpha()):
            continue
        toks.append(tok)
    return tuple(toks)


def is_organization(name: str) -> bool:
    """Heuristic: the name contains an organization cue word ("Co.", "Department", ...)."""
    return any(t in _ORG_CUES for t in _NAME_TOKEN.findall(name.casefold()))


def names_match(x: str, y: str) -> bool:
    """Loose person/org name match: equal, one name's tokens contained in the other's
    ("Lopez" ~ "Ana Lopez", "Mr. Okafor" ~ "Sam Okafor"), or a near-identical spelling."""
    tx, ty = name_tokens(x), name_tokens(y)
    if not tx or not ty:
        return bool(_norm_text(x)) and _norm_text(x) == _norm_text(y)
    sx, sy = set(tx), set(ty)
    if sx <= sy or sy <= sx:
        return True
    return fuzz.token_sort_ratio(" ".join(tx), " ".join(ty)) >= 92


def matched_actors(a: Event, b: Event) -> set[str]:
    """Actors of ``a`` (as written) that loosely match some actor of ``b``."""
    return {x for x in a.actors if x.strip() and any(names_match(x, y) for y in b.actors)}


def unmatched_people(a: Event, b: Event) -> list[str]:
    """People (not organizations) named in ``a`` that ``b`` does not name."""
    return [
        x
        for x in a.actors
        if x.strip()
        and name_tokens(x)
        and not is_organization(x)
        and not any(names_match(x, y) for y in b.actors)
    ]


def actors_overlap(a: Event, b: Event) -> bool:
    """True when the events share at least one (loosely matched) actor."""
    return bool(matched_actors(a, b))


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
    """True unless both events name actors and no name loosely matches across them."""
    if actor_set(a) and actor_set(b):
        return actors_overlap(a, b)
    return True


def description_similarity(a: Event, b: Event) -> float:
    """rapidfuzz token_set_ratio (0-100) over the event descriptions."""
    return fuzz.token_set_ratio(a.description, b.description, processor=_preprocess)


def fuzzy_duplicate(a: Event, b: Event) -> bool:
    """Descriptions near-identical: high token_set_ratio *and* reasonable token_sort_ratio."""
    if description_similarity(a, b) < FUZZ_THRESHOLD:
        return False
    sort = fuzz.token_sort_ratio(a.description, b.description, processor=_preprocess)
    return sort >= SORT_THRESHOLD


def people_agree(a: Event, b: Event) -> bool:
    """Neither event names a person the other leaves out (organizations are ignored).

    Events where one side names no actors at all are treated as agreeing.
    """
    if not actor_set(a) or not actor_set(b):
        return True
    return not unmatched_people(a, b) and not unmatched_people(b, a)


def may_merge(a: Event, b: Event) -> bool:
    """Date/actor gate for a duplicate link.

    Linked duplicates are hidden from the timeline and from contradiction checks, so a merge
    must never swallow a disagreement between two documents: across documents the resolved
    dates must overlap and the named people must be the same. Within one document (e.g. the
    same sentence seen in two overlapping chunks) sharing one actor is enough.
    """
    if not same_date(a, b):
        return False
    if a.source.doc_id != b.source.doc_id:
        return actors_compatible(a, b) and people_agree(a, b)
    return actors_compatible(a, b)


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
            if other.id == ev.id or not may_merge(ev, other):
                continue
            hits.append((other, fuzzy_duplicate(ev, other)))
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
