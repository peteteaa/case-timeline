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
# Two documents' accounts stamped with the same clock time (see ``clocks_agree``) are very
# likely the same occurrence, so for them a somewhat weaker (but still strong) description
# match is enough. Not within one document: a transcript lists several distinct utterances
# under one timestamp, and a document does not report one occurrence twice in other words.
TIMED_FUZZ_THRESHOLD = 75.0
TIMED_SORT_THRESHOLD = 50.0
TIMED_COSINE_THRESHOLD = 0.75
# How far apart two written clock times may be and still name the same moment. Times written
# to the second (recorder or log timestamps) put distinct occurrences only seconds apart, so
# they get a tight window; times written to the minute get a few minutes.
CLOCK_SLACK_TO_SECOND = 2  # seconds
CLOCK_SLACK_TO_MINUTE = 120  # seconds

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
    ("Lopez" ~ "Ana Lopez", "Mr. Adeyemi" ~ "Sam Adeyemi"), or a near-identical spelling."""
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


# --------------------------------------------------------------------------- clock times

# Time zone markers in a date as written ("0216 UTC", "3:40 PM CDT", "1816 central daylight
# time"); the spelled-out forms are normalized to their abbreviations.
_ZONE_RE = re.compile(
    r"\b(?:(?P<abbr>UTC|GMT|Zulu|Z|[ECMPA][SD]T)"
    r"|(?P<long>(?:eastern|central|mountain|pacific|alaska)\s+(?:daylight|standard)"
    r"(?:\s+saving)?\s+time))\b",
    re.IGNORECASE,
)
_UTC_NAMES = frozenset({"utc", "gmt", "z", "zulu"})
_CLOCK_RE = re.compile(
    # "09:42", "9:42:15", "3:40 p.m."
    r"\b(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)(?::(?P<s>[0-5]\d))?\s*(?P<ap>[ap]\.?\s?m\b\.?)?"
    # "0942:15" (HHMM:SS, as in recorder transcripts)
    r"|\b(?P<h2>[01]\d|2[0-3])(?P<m2>[0-5]\d):(?P<s2>[0-5]\d)\b"
    # "1405 local", "2316 UTC", "about 1816 central daylight time"
    r"|\b(?P<h3>[01]\d|2[0-3])(?P<m3>[0-5]\d)\b"
    r"(?=\s*(?:hours|hrs|local|lcl|UTC|GMT|Z\b|Zulu|[ECMPA][SD]T\b|"
    r"(?:eastern|central|mountain|pacific|alaska)\s+(?:daylight|standard)))"
    # "at 0940", "about 1816" (a four-digit time after a time preposition)
    r"|(?:(?<=\bat\s)|(?<=\babout\s)|(?<=\baround\s)|(?<=\bapproximately\s))"
    r"(?P<h4>[01]\d|2[0-3])(?P<m4>[0-5]\d)\b(?!\s*[,/-]?\s*\d)"
    # "2 p.m.", "11 AM" (an hour without minutes)
    r"|\b(?P<h5>1[0-2]|0?[1-9])\s*(?P<ap5>[ap]\.?\s?m\b\.?)",
    re.IGNORECASE,
)


def _zone(text: str) -> str:
    """Normalized time zone named in ``text`` ("utc", "cdt", ...), or "" if none."""
    m = _ZONE_RE.search(text)
    if m is None:
        return ""
    if m.group("abbr"):
        abbr = m.group("abbr").casefold()
        return "utc" if abbr in _UTC_NAMES else abbr
    words = m.group("long").casefold().split()
    return f"{words[0][0]}{words[1][0]}t"


def clock_of(event: Event) -> tuple[int, bool, str] | None:
    """``(seconds after midnight, written to the second?, zone)`` of the first clock time
    in the event's date as written, or None when it states no time of day."""
    text = event.date_text or ""
    m = _CLOCK_RE.search(text)
    if m is None:
        return None
    zone = _zone(text)
    if m.group("h2"):
        return int(m.group("h2")) * 3600 + int(m.group("m2")) * 60 + int(m.group("s2")), True, zone
    for h, mm in (("h3", "m3"), ("h4", "m4")):
        if m.group(h):
            return int(m.group(h)) * 3600 + int(m.group(mm)) * 60, False, zone
    if m.group("h5"):
        return _twelve_hour(int(m.group("h5")), m.group("ap5")) * 3600, False, zone
    hour = _twelve_hour(int(m.group("h")), m.group("ap"))
    seconds = m.group("s")
    return hour * 3600 + int(m.group("m")) * 60 + int(seconds or 0), seconds is not None, zone


def _twelve_hour(hour: int, marker: str | None) -> int:
    """24-hour clock hour for ``hour`` with an optional "a.m."/"p.m." marker."""
    ap = re.sub(r"[^ap]", "", (marker or "").casefold())
    if ap == "p" and hour < 12:
        return hour + 12
    if ap == "a" and hour == 12:
        return 0
    return hour


def _zones_comparable(za: str, zb: str) -> bool:
    """Same zone, or one unstated (local time) and the other a local zone (not UTC)."""
    if za == zb:
        return True
    return (not za and zb != "utc") or (not zb and za != "utc")


def clock_gap(a: Event, b: Event) -> tuple[int, int] | None:
    """``(seconds apart, tolerance)`` when both events state comparable clock times."""
    ca, cb = clock_of(a), clock_of(b)
    if ca is None or cb is None or not _zones_comparable(ca[2], cb[2]):
        return None
    tolerance = CLOCK_SLACK_TO_SECOND if ca[1] and cb[1] else CLOCK_SLACK_TO_MINUTE
    return abs(ca[0] - cb[0]), tolerance


def clocks_agree(a: Event, b: Event) -> bool | None:
    """True/False when both events state comparable clock times that do/do not name the
    same moment; None when at least one states no (comparable) time."""
    gap = clock_gap(a, b)
    if gap is None:
        return None
    return gap[0] <= gap[1]


def timed_pair(a: Event, b: Event) -> bool:
    """Accounts from two documents that state the same moment (lower similarity bars apply)."""
    return a.source.doc_id != b.source.doc_id and bool(clocks_agree(a, b))


def same_day(a: Event, b: Event) -> bool:
    """Both dates resolved to exactly the same day or span (no partial overlap)."""
    ia, ib = _interval(a), _interval(b)
    return ia is not None and ia == ib


# --------------------------------------------------------------------------- actors

# Generic role and object nouns: an actor named with one of these ("Oshkosh Tower", "SLC
# controller", "the pilot") is a role or a thing, not a named person, so two documents that
# call the same participant by different roles do not disagree about who was there.
_ROLE_CUES = frozenset(
    "tower controller controllers dispatcher dispatch operator operators pilot pilots copilot "
    "driver drivers crew passenger passengers witness witnesses occupant occupants staff team "
    "unit units center centre station desk aircraft airplane plane jet helicopter vehicle car "
    "truck vessel ship boat train recorder system camera alarm customer customers employee "
    "employees tenant landlord buyer seller caller victim suspect patient".split()
)


def is_named_person(name: str) -> bool:
    """Heuristic: ``name`` names a specific person ("Ana Lopez", "Mr. Keech"), not an
    organization, a role ("the pilot", "SLC controller"), a call sign or an acronym."""
    if not name.strip() or is_organization(name) or any(ch.isdigit() for ch in name):
        return False
    words = _NAME_TOKEN.findall(name)
    if not name_tokens(name):
        return False
    for word in words:
        low = word.casefold()
        if low in _ROLE_CUES:
            return False
        if low in _HONORIFICS or (len(word) == 1 and word.isalpha()):
            continue
        if not word[0].isupper() or (word.isupper() and len(words) == 1):
            return False  # lower-case descriptor ("controller of X") or a lone acronym
    return True


def named_people(event: Event) -> list[str]:
    """Actors of ``event`` that look like named people."""
    return [x for x in event.actors if is_named_person(x)]


def people_conflict(a: Event, b: Event) -> bool:
    """True when both events name people and one names a person the other does not.

    Only named people count: organizations, roles and objects are ignored, and an event that
    names nobody is silent rather than in disagreement.
    """
    pa, pb = named_people(a), named_people(b)
    if not pa or not pb:
        return False
    return any(not any(names_match(x, y) for y in b.actors) for x in pa) or any(
        not any(names_match(y, x) for x in a.actors) for y in pb
    )


def actors_compatible(a: Event, b: Event) -> bool:
    """True unless both events name actors and no name loosely matches across them."""
    if actor_set(a) and actor_set(b):
        return actors_overlap(a, b)
    return True


def description_similarity(a: Event, b: Event) -> float:
    """rapidfuzz token_set_ratio (0-100) over the event descriptions."""
    return fuzz.token_set_ratio(a.description, b.description, processor=_preprocess)


def fuzzy_duplicate(a: Event, b: Event, *, timed: bool = False) -> bool:
    """Descriptions near-identical: high token_set_ratio *and* reasonable token_sort_ratio.

    ``timed``: the two events state the same clock time, so slightly lower bars apply.
    """
    set_bar, sort_bar = (
        (TIMED_FUZZ_THRESHOLD, TIMED_SORT_THRESHOLD) if timed else (FUZZ_THRESHOLD, SORT_THRESHOLD)
    )
    if description_similarity(a, b) < set_bar:
        return False
    sort = fuzz.token_sort_ratio(a.description, b.description, processor=_preprocess)
    return sort >= sort_bar


def people_agree(a: Event, b: Event) -> bool:
    """Neither event names a person the other leaves out (organizations are ignored).

    Events where one side names no actors at all are treated as agreeing.
    """
    if not actor_set(a) or not actor_set(b):
        return True
    return not unmatched_people(a, b) and not unmatched_people(b, a)


def may_merge(a: Event, b: Event) -> bool:
    """Date/time/actor gate for a duplicate link.

    Linked duplicates are hidden from the timeline, so a merge must never swallow a
    disagreement between two documents (such pairs are left for the contradiction check).
    Clock times stated by both events must name the same moment (``clocks_agree``). Across
    documents the dates must resolve to exactly the same day (or span) and the named people
    must not conflict (roles, organizations and unnamed participants do not count). Within
    one document (e.g. the same sentence seen in two overlapping chunks) overlapping dates
    and sharing one actor are enough.
    """
    if clocks_agree(a, b) is False:
        return False
    if a.source.doc_id != b.source.doc_id:
        return same_day(a, b) and not people_conflict(a, b)
    return same_date(a, b) and actors_compatible(a, b)


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
            hits.append((other, fuzzy_duplicate(ev, other, timed=timed_pair(ev, other))))
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
    if va is None or vb is None:
        return False
    bar = TIMED_COSINE_THRESHOLD if timed_pair(a, b) else COSINE_THRESHOLD
    return _cosine(va, vb) >= bar


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
