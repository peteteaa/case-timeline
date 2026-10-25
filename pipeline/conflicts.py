"""Open contradictions between facts (SPEC §6.4): cheap candidate rules, then LLM confirmation.

A fact can cite several documents (merged evidence), so the rules compare documents, not just
facts: a pair is checked only when the facts' sources differ (a document is not checked against
itself). Candidate rules, over non-rejected facts, for pairs with at least one fact the job
touched; "linked" means the facts share a person (names compare loosely; a shared organization
alone doesn't link facts) and "topic" is the overlap of content words (names and dates removed):

- A plan or schedule ("set for", "rescheduled") is never compared with an occurrence, on any
  field: things get moved. Only plan-vs-plan and occurrence-vs-occurrence pairs qualify.
- Dates are compared through ``date_relation``. Precision is not disagreement: a period or
  range containing the other date, an approximate day ("early March", "around the 9th") within
  ``APPROX_SLACK_DAYS`` of it, a one-day gap between dates written in different time zones
  ("0216 UTC" vs local time), and clock times within ``CLOCK_SLACK_SECONDS`` are compatible.
  When a fact's quotes refer to calendar dates that don't include its own date, its date is
  most likely that of the message reporting the occurrence (a letter's date, an email's "Sent:"
  line), so the quoted dates are compared instead whenever that narrows the gap.
- **date**: linked facts (or, when one names nobody, near-identical ones) more than a day apart
  (a day apart when both dates are exact) that look like one occurrence: similar wording, a
  shared topic word within two weeks, or a strong topic overlap within two months. When either
  fact is about someone's knowledge ("first learned"), the candidate is "fact".
- **actor**: facts on the same single day that are similar but name disjoint people, or that
  share a person and a topic word but name different people. Never for someone's knowledge
  (when someone learned of a matter says nothing about who was present), and never for times
  given on that day that are clearly different moments (``distinct_moments``);
- **fact**: linked facts on the same single day where one negates ("never", "denied", ...) and
  the wording differs or the topic is shared; and linked facts where one is undated, or where
  one date is a period that merely contains the other, but the topic overlap is strong (a
  contained date is no evidence that two facts are one occurrence).

Each candidate is confirmed by the local LLM, which sees every quote of both facts with the
surrounding source text, plus plain-language notes computed from the dates (``date_notes``);
deterministic checks then veto impossible fields (a "date" conflict needs two resolved dates
that are not compatible; an "actor" conflict is never about a plan vs an occurrence or about
someone's knowledge, and between linked facts it needs the side that leaves a person out to
exclude people in its own quote or the text right after it). A disagreement already opened
(or dismissed) through another pair of facts about the same occurrences is not opened again.
A reviewer then confirms (facts become disputed) or dismisses each contradiction.
"""

from __future__ import annotations

import calendar
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pipeline import prompts
from pipeline.merge import (
    Interval,
    Person,
    fact_similarity,
    fact_text,
    gap_days,
    interval,
    overlap,
    people_of,
    proper_words,
    shares_person,
    topic_tokens,
    unmatched,
)
from pipeline.schema import ConflictField, Contradiction, Fact
from pipeline.store import Store

log = logging.getLogger(__name__)

SIMILAR_THRESHOLD = 70.0
STRONG_SIMILAR = 85.0  # date rule when one side names nobody
MAX_CANDIDATES = 50
MAX_QUOTES = 3  # evidence quotes per fact shown to the model
MAX_GAP_DAYS = 62  # facts further apart than this are rarely one misdated occurrence
NEAR_DAYS = 14  # within this many days, one shared topic word is enough for an LLM check
TOPIC_MIN = 0.3  # topic overlap for dated pairs further apart than NEAR_DAYS
TOPIC_MIN_UNDATED = 0.4  # stricter when one fact is undated (no date evidence)
SAME_OCCURRENCE_TOPIC = 0.5  # two facts on one day this alike are one occurrence (dedupe)
# An approximate or ranged date ("early June", "that week") has soft edges: a date this many
# days outside it is still a granularity difference, not a disagreement.
APPROX_SLACK_DAYS = 3
CLOCK_SLACK_SECONDS = 90  # two sources' clock times this close are one moment
CONTEXT_BEFORE = 300  # characters of source text shown around each quote
CONTEXT_AFTER = 400
LOCAL_AFTER = 200  # characters after a quote searched for an exclusion cue ("Only the two of us.")
VERDICT_MAX_TOKENS = 1200

NEGATION_CUES = re.compile(
    r"\b(not|never|no|none|nobody|nothing|neither|nor|without|denie[sd]|deny|denying|"
    r"refused?|refuses|absent|failed|cannot|can't|didn't|did\s+not|wasn't|weren't|isn't|"
    r"aren't|hasn't|haven't|hadn't|won't|wouldn't|couldn't|don't|doesn't)\b",
    re.IGNORECASE,
)
# A disagreement about *who* was there needs one source to exclude someone affirmatively;
# silence is not exclusion. One of these cues must appear in the quotes (or the text right
# after them) of the side that leaves a person out.
EXCLUSION_CUES = re.compile(
    r"\b(only|nobody|no\s+one|no-one|none|alone|nor|neither|never|not|no|absent|"
    r"wasn't|weren't|didn't|denie[sd]|deny|except|besides|other\s+than|else)\b",
    re.IGNORECASE,
)
# A plan or schedule ("set for", "due by"). A plan compared with what actually happened (or
# with a rescheduling) is not a date contradiction, so only plan-vs-plan or fact-vs-fact pairs
# qualify.
PLAN_CUES = re.compile(
    r"\b(schedul\w*|reschedul\w*|set\s+for|due\s+(?:by|on)|deadline|planned|plans?\s+to|"
    r"expected\s+to|will|postpon\w*|adjourn\w*|moved\s+(?:from|to))\b",
    re.IGNORECASE,
)
# Someone's knowledge or awareness ("first learned", "became aware"). When one side is about
# when someone knew of a matter, a timing disagreement is a FACT (notice) conflict about that
# knowledge, not two dates for one occurrence.
KNOWLEDGE_CUES = re.compile(
    r"\b(learn(?:ed|t|s|ing)?|became\s+aware|been\s+aware|was\s+aware|were\s+aware|unaware|"
    r"aware\s+of|knew|know(?:s|n)?\s+(?:of|about)|first\s+heard|found\s+out|discover(?:ed|s|y)|"
    r"realiz(?:ed|es)|realis(?:ed|es))\b",
    re.IGNORECASE,
)


def _all_required(schema: dict[str, Any]) -> None:
    """Mark every property required so constrained decoding always emits each field."""
    schema["required"] = list(schema.get("properties", {}))


class ConflictVerdict(BaseModel):
    """LLM confirmation of a candidate contradiction."""

    model_config = ConfigDict(json_schema_extra=_all_required)

    comparison: str = ""
    same_subject: bool = True
    reconciliation: str = ""
    is_contradiction: bool
    field: ConflictField
    other_fields: list[ConflictField] = Field(default_factory=list)
    explanation: str = ""

    def fields(self) -> list[ConflictField]:
        """Primary field first, then distinct additional date/actor fields.

        "fact" is the catch-all, so as a secondary field it almost always restates the primary
        disagreement; it is only kept as the primary field.
        """
        out: list[ConflictField] = [self.field]
        out += [f for f in self.other_fields if f not in out and f != "fact"]
        return out


@dataclass(frozen=True)
class Candidate:
    """A cheap-rule candidate pair awaiting LLM confirmation."""

    a: Fact
    b: Fact
    field: ConflictField
    score: float  # higher = more suspicious; used for capping


# --------------------------------------------------------------------------- fact features


def _evidence(f: Fact) -> list[Any]:
    return [e for e in f.evidence if e.review != "rejected"]


def _docs(f: Fact) -> frozenset[str]:
    return frozenset(e.doc_id for e in _evidence(f) if e.doc_id)


def _quotes(f: Fact) -> list[str]:
    return [e.quote for e in _evidence(f) if e.quote]


def _text(f: Fact) -> str:
    return " ".join([f.title, f.details or "", *_quotes(f)])


def _is_plan(f: Fact) -> bool:
    return bool(PLAN_CUES.search(_text(f)))


def _is_knowledge(f: Fact) -> bool:
    return bool(KNOWLEDGE_CUES.search(_text(f)))


def _has_negation(f: Fact) -> bool:
    return bool(NEGATION_CUES.search(_text(f)))


def _persons(f: Fact, witness_names: Mapping[str, Sequence[str]] | None) -> list[Person]:
    return people_of(f.witness_ids, witness_names)


def _people_only(ps: Sequence[Person]) -> list[Person]:
    return [p for p in ps if not p.is_org]


def linked(pa: Sequence[Person], pb: Sequence[Person]) -> bool:
    """The facts share someone; a shared person is required when both name people, so a
    common employer or agency alone does not link two facts."""
    qa, qb = _people_only(pa), _people_only(pb)
    if qa and qb:
        return shares_person(qa, qb)
    return shares_person(pa, pb)


def topic_overlap(
    a: Fact, b: Fact, witness_names: Mapping[str, Sequence[str]] | None = None
) -> tuple[float, int]:
    """Overlap coefficient (0..1) of the two facts' topic words, and the shared-word count.

    Names and date expressions are excluded so that the score reflects *what* happened
    (meeting, dismissal, injury...) rather than who or when.
    """
    texts = (_text(a), _text(b))
    names: list[str] = [*proper_words(*texts)]
    for w in {*a.witness_ids, *b.witness_ids}:
        names += list((witness_names or {}).get(w, ()))
    return overlap(topic_tokens(texts[0], names), topic_tokens(texts[1], names))


# --------------------------------------------------------------------------- date relation

_MONTH_NUM = {
    **{calendar.month_name[i].lower(): i for i in range(1, 13)},
    **{calendar.month_abbr[i].lower(): i for i in range(1, 13)},
    "sept": 9,
}
_MON = "|".join(sorted(_MONTH_NUM, key=len, reverse=True))
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_YEAR = r"(?:19|20)\d{2}"
_EXPLICIT_DATE = re.compile(
    rf"\b(?:(?P<m1>{_MON})\.?\s+(?P<d1>{_DAY})(?:,?\s+(?P<y1>{_YEAR}))?"  # June 5(, 2021)
    rf"|(?P<d2>{_DAY})\s+(?:of\s+)?(?P<m2>{_MON})\.?,?\s+(?P<y2>{_YEAR})"  # 5 June 2021
    rf"|(?P<y3>{_YEAR})-(?P<m3>\d{{2}})-(?P<d3>\d{{2}}))\b",  # 2021-06-05
    re.IGNORECASE,
)


def referenced_dates(f: Fact) -> list[date]:
    """Calendar dates written out in the fact's quotes ("June 5, 2021", "5 June 2021").

    A date without a year takes the year of the fact's own resolved date.
    """
    out: list[date] = []
    default_year = f.date_start.year if f.date_start else None
    for quote in _quotes(f):
        for m in _EXPLICIT_DATE.finditer(quote):
            try:
                if m.group("y3"):
                    d = date(int(m.group("y3")), int(m.group("m3")), int(m.group("d3")))
                else:
                    mon = _MONTH_NUM[(m.group("m1") or m.group("m2")).lower()]
                    day = int(re.sub(r"\D", "", m.group("d1") or m.group("d2")))
                    year = m.group("y1") or m.group("y2") or default_year
                    if year is None:
                        continue
                    d = date(int(year), mon, day)
            except (KeyError, ValueError):
                continue
            if d not in out:
                out.append(d)
    return out


def carrier_dates(f: Fact) -> list[date]:
    """Dates the fact's quotes refer to when its own date is not among them, else [].

    Then the fact's date is most likely the date of the message or document that reports the
    occurrence (an email's "Sent:" line, a letter's date), not the occurrence's own date: "we
    booked the site visit for June 9" in a message dated by its header. A fact citing several
    documents keeps its own date when any of its quotes states it.
    """
    own = interval(f.date_start, f.date_end)
    if own is None:
        return []
    refs = referenced_dates(f)
    if not refs or any(own[0] <= d <= own[1] for d in refs):
        return []
    return refs


# Time zone markers in a date as written ("0216 UTC", "3:40 PM CDT"). A UTC timestamp late in
# the evening falls on the next calendar day, so a one-day gap across zones is no conflict.
_ZONE = re.compile(r"\b(?:UTC|GMT|Z|Zulu|[ECMPA][SD]T|local(?:\s+time)?)\b", re.IGNORECASE)
# Clock times in a date as written: "09:42:15", "3:40 PM", "0942:15" (HHMM:SS).
_CLOCK = re.compile(
    r"\b(?:(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)(?::(?P<s>[0-5]\d))?\s*(?P<ap>[ap]\.?m\.?)?"
    r"|(?P<h2>[01]\d|2[0-3])(?P<m2>[0-5]\d):(?P<s2>[0-5]\d))(?!\w)",
    re.IGNORECASE,
)


def _zones(f: Fact) -> frozenset[str]:
    return frozenset(z.casefold() for z in _ZONE.findall(f.date_text or ""))


def clock_seconds(text: str | None) -> int | None:
    """Seconds after midnight of the first clock time in ``text``, or None."""
    m = _CLOCK.search(text or "")
    if m is None:
        return None
    if m.group("h2"):
        return int(m.group("h2")) * 3600 + int(m.group("m2")) * 60 + int(m.group("s2"))
    hour = int(m.group("h"))
    ap = (m.group("ap") or "").replace(".", "").lower()
    if ap == "pm" and hour < 12:
        hour += 12
    elif ap == "am" and hour == 12:
        hour = 0
    return hour * 3600 + int(m.group("m")) * 60 + int(m.group("s") or 0)


def clock_gap_seconds(a: Fact, b: Fact) -> int | None:
    """Seconds between the clock times written in both facts' dates (None unless both have
    one and their resolved dates, where both have one, overlap)."""
    ca, cb = clock_seconds(a.date_text), clock_seconds(b.date_text)
    if ca is None or cb is None:
        return None
    gap = gap_days(interval(a.date_start, a.date_end), interval(b.date_start, b.date_end))
    if gap is not None and gap > 0:
        return None
    return abs(ca - cb)


# Words that make the *day* itself soft ("early June", "a few days later", "sometime that
# week"), and hedges that do so unless they qualify a clock time ("around March 9" is soft,
# "approximately 3:40 PM" is a soft time on a definite day).
_PERIOD_HEDGE = re.compile(
    r"\b(?:early|mid|late|beginning|end|week|weekend|month|couple|few|several|sometime|"
    r"some\s+time|or\s+so)\b|\bmid-",
    re.IGNORECASE,
)
_POINT_HEDGE = re.compile(
    r"\b(?:around|about|approx\w*|circa|ca\.|roughly|nearly|on\s+or\s+about)\b", re.IGNORECASE
)


def _is_fuzzy(f: Fact) -> bool:
    """The resolved *day* has soft edges: a period ("March 2024", "early June", "that week")
    or a hedged day ("around March 9").

    A span stated to the day at both ends ("March 3-5, 2024", certainty "exact") has hard
    edges. "approximate" or "relative" alone is not enough either: the date resolver also
    uses them for a year inferred from context ("March 9"), a day a bare clock time inherits
    ("3:40 PM") or a computed offset ("three weeks later"). Those are single definite days,
    and a few days' gap to another date is a real disagreement.
    """
    if f.date_certainty == "exact":
        return False
    span = interval(f.date_start, f.date_end)
    if span is not None and span[0] != span[1]:
        return True
    if f.date_certainty not in ("approximate", "relative"):
        return False
    text = f.date_text or ""
    if _PERIOD_HEDGE.search(text):
        return True
    return bool(_POINT_HEDGE.search(text)) and clock_seconds(text) is None


@dataclass(frozen=True)
class DateRelation:
    """How the resolved dates of two facts relate."""

    gap: int | None  # days between them (0 if overlapping); None if either is undated
    same_day: bool  # both are the same single calendar day
    approximate: bool  # either date has soft edges (``_is_fuzzy``)
    carrier: bool = False  # a message date was replaced by the dates its quotes refer to
    zone_shift: bool = False  # the dates are written in different time zones
    clock_gap: int | None = None  # seconds between the clock times both dates give

    @property
    def same_moment(self) -> bool:
        """Both sources give clock times within ``CLOCK_SLACK_SECONDS``: one moment."""
        return self.clock_gap is not None and self.clock_gap <= CLOCK_SLACK_SECONDS

    @property
    def compatible(self) -> bool:
        """The dates cannot conflict: they overlap, an approximate period nearly touches the
        other date, a one-day gap comes from different time zones, or the clock times are one
        moment (precision, not disagreement)."""
        if self.same_moment:
            return True
        if self.gap is None:
            return False
        return (
            self.gap == 0
            or (self.approximate and self.gap <= APPROX_SLACK_DAYS)
            or (self.zone_shift and self.gap <= 1)
        )


def date_relation(a: Fact, b: Fact) -> DateRelation:
    """Compare two facts' dates, treating a message date as the dates its quotes refer to.

    The comparison only ever gets more lenient: the gap is the smaller of the plain gap and
    the gap after substituting quoted dates for a message date.
    """
    ia, ib = interval(a.date_start, a.date_end), interval(b.date_start, b.date_end)
    approximate = _is_fuzzy(a) or _is_fuzzy(b)
    clock_gap = clock_gap_seconds(a, b)
    if ia is None or ib is None:
        return DateRelation(None, False, approximate, clock_gap=clock_gap)
    gap = _span_gap(ia, ib)
    same_day = ia == ib and ia[0] == ia[1]
    carrier = False
    if gap > 0:
        spans_a = [(d, d) for d in carrier_dates(a)] or [ia]
        spans_b = [(d, d) for d in carrier_dates(b)] or [ib]
        quoted = min(_span_gap(x, y) for x in spans_a for y in spans_b)
        if quoted < gap:
            gap, carrier = quoted, True
    zone_shift = _zones(a) != _zones(b)
    return DateRelation(gap, same_day, approximate, carrier, zone_shift, clock_gap)


def _span_gap(x: Interval, y: Interval) -> int:
    return gap_days(x, y) or 0  # never None: both intervals are set


def _has_seconds(text: str | None) -> bool:
    m = _CLOCK.search(text or "")
    return m is not None and bool(m.group("s") or m.group("s2"))


def distinct_moments(a: Fact, b: Fact, rel: DateRelation | None = None) -> bool:
    """Both facts are timed on the same day, more than ``CLOCK_SLACK_SECONDS`` apart, and the
    times are precise enough to tell occurrences apart: both are given to the second (log or
    transcript timestamps), or both come from one document's own timeline (the facts share a
    document). A loose "about 2:30" vs "2:47" from two witnesses may still be one occurrence."""
    rel = rel or date_relation(a, b)
    if rel.clock_gap is None or rel.same_moment:
        return False
    if _docs(a) & _docs(b):
        return True
    return _has_seconds(a.date_text) and _has_seconds(b.date_text)


def date_notes(a: Fact, b: Fact) -> list[str]:
    """Plain-language facts about the two dates, given to the LLM with the candidate."""
    rel = date_relation(a, b)
    notes: list[str] = []
    for label, f in (("A", a), ("B", b)):
        refs = carrier_dates(f)
        if refs:
            notes.append(
                f"Fact {label}'s date ({f.date_text or _resolved(f)}) is not mentioned in its "
                f"quotes, which refer to {', '.join(d.isoformat() for d in refs)}: it is "
                "probably the date of the message or document, not of the occurrence it reports."
            )
    if rel.same_moment:
        notes.append(
            f"Both facts give clock times only {rel.clock_gap} seconds apart: for these "
            "sources that is the same moment, not a timing conflict."
        )
    elif rel.zone_shift and rel.gap == 1:
        notes.append(
            "The dates are written in different time zones (e.g. UTC vs local time); a "
            "one-day difference can be the same moment and is not a date conflict."
        )
    elif rel.gap is not None and rel.compatible and not rel.same_day:
        notes.append(
            "The two dates are compatible: one is an approximate period or range that contains "
            "or nearly touches the other date. This is a difference in precision, never a date "
            "conflict."
        )
    for label, f in (("A", a), ("B", b)):
        if _is_plan(f):
            notes.append(
                f"Fact {label} is about a plan, schedule or deadline; it is only in conflict "
                "with another statement about that same plan, not with what later happened."
            )
    return notes


# --------------------------------------------------------------------------- candidates


def classify(
    a: Fact, b: Fact, witness_names: Mapping[str, Sequence[str]] | None = None
) -> Candidate | None:
    """Apply the date / actor / fact candidate rules to one pair."""
    da, db = _docs(a), _docs(b)
    if da == db:
        return None  # one source is not checked against itself
    if _is_plan(a) != _is_plan(b):
        # A plan, schedule or deadline vs what actually happened: things get rescheduled, so
        # this is no conflict of any kind unless both facts make the same kind of claim.
        return None
    rel = date_relation(a, b)
    gap = rel.gap
    pa, pb = _persons(a, witness_names), _persons(b, witness_names)
    link = linked(pa, pb)
    sim = fact_similarity(a.title, a.details, b.title, b.details)
    topic, shared = topic_overlap(a, b, witness_names)

    if gap is None or (gap == 0 and not rel.same_day):
        # One side is undated, or one date is a period that merely contains the other: the
        # dates are no evidence that the facts are one occurrence, so only a strong
        # same-subject signal is worth an LLM call (e.g. "learned of it only after the
        # hearing" vs a dated written notice).
        if link and shared >= 2 and topic >= TOPIC_MIN_UNDATED and (a.date_start or b.date_start):
            return Candidate(a, b, "fact", 100.0 * topic)
        return None
    if gap > 0 and rel.compatible:
        return None  # precision only: an approximate edge, or a time-zone shift

    both_exact = a.date_certainty == "exact" and b.date_certainty == "exact"
    if gap > 1 or (gap == 1 and both_exact):
        # A one-day gap only counts when both dates are stated exactly ("March 11" vs "12").
        strong = (not pa or not pb) and sim >= STRONG_SIMILAR
        if not (link or strong):
            return None
        near = gap <= NEAR_DAYS and shared >= 1
        if sim >= SIMILAR_THRESHOLD or near or (gap <= MAX_GAP_DAYS and topic >= TOPIC_MIN):
            closeness = max(0.0, 1.0 - gap / MAX_GAP_DAYS)
            kind: ConflictField = "fact" if _is_knowledge(a) or _is_knowledge(b) else "date"
            return Candidate(a, b, kind, 100.0 * max(topic, sim / 100) + closeness)
        return None
    if gap != 0:
        return None
    if distinct_moments(a, b, rel):
        # Same day but clearly different moments (a log or transcript lists many timed
        # entries per day): the rules below assume one occurrence, so they do not apply.
        return None

    # The same single day on both sides.
    qa, qb = _people_only(pa), _people_only(pb)
    knowledge = _is_knowledge(a) or _is_knowledge(b)  # about who knew, not who was there
    if qa and qb and not link and sim >= SIMILAR_THRESHOLD and not knowledge:
        return Candidate(a, b, "actor", sim)
    if link and shared >= 1 and not knowledge and (unmatched(qa, qb) or unmatched(qb, qa)):
        # Same occurrence, different people named: an omission or a real disagreement about
        # who was there; the LLM decides using the surrounding source text.
        return Candidate(a, b, "actor", 100.0 * topic)
    negation = _has_negation(a) or _has_negation(b)
    if link and negation and (sim < SIMILAR_THRESHOLD or topic >= TOPIC_MIN):
        return Candidate(a, b, "fact", max(100.0 - sim, 100.0 * topic))
    return None


def candidate_pairs(
    facts: Sequence[Fact],
    focus_ids: set[str],
    *,
    max_candidates: int = MAX_CANDIDATES,
    skip_pairs: set[frozenset[str]] | None = None,
    witness_names: Mapping[str, Sequence[str]] | None = None,
) -> list[Candidate]:
    """Candidates involving at least one fact in ``focus_ids``, capped at ``max_candidates``."""
    skip = skip_pairs or set()
    live = [f for f in facts if f.review != "rejected"]
    found: list[Candidate] = []
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            if a.fact_id == b.fact_id:
                continue
            if a.fact_id not in focus_ids and b.fact_id not in focus_ids:
                continue
            if frozenset((a.fact_id, b.fact_id)) in skip:
                continue
            if cand := classify(a, b, witness_names):
                found.append(cand)
    found.sort(key=lambda c: (-c.score, c.a.fact_id, c.b.fact_id))
    if len(found) > max_candidates:
        dropped = found[max_candidates:]
        log.warning(
            "conflict candidates capped at %d; dropped %d: %s",
            max_candidates,
            len(dropped),
            [(c.a.fact_id, c.b.fact_id, c.field) for c in dropped],
        )
        found = found[:max_candidates]
    log.info("conflict candidates: %d", len(found))
    return found


# --------------------------------------------------------------------------- same disagreement


def same_occurrence(x: Fact, y: Fact) -> bool:
    """The same fact, or two facts on overlapping dates that read as one occurrence."""
    if x.fact_id == y.fact_id:
        return True
    gap = gap_days(interval(x.date_start, x.date_end), interval(y.date_start, y.date_end))
    if gap != 0:
        return False
    if fact_similarity(x.title, x.details, y.title, y.details) >= SIMILAR_THRESHOLD:
        return True
    names = proper_words(fact_text(x.title, x.details), fact_text(y.title, y.details))
    ta = topic_tokens(fact_text(x.title, x.details), names)
    tb = topic_tokens(fact_text(y.title, y.details), names)
    coeff, shared = overlap(ta, tb)
    return shared >= 1 and coeff >= SAME_OCCURRENCE_TOPIC


def same_disagreement(
    field: str, a: Fact, b: Fact, opened: Sequence[tuple[str, Fact, Fact]]
) -> bool:
    """``a`` vs ``b`` restates a disagreement already opened on ``field`` through other facts:
    each side is the same occurrence as one side of it (e.g. one account dated March 11 against
    a second March 12 account, after the first March 12 account was already flagged)."""
    for f, x, y in opened:
        if f != field or {a.fact_id, b.fact_id} == {x.fact_id, y.fact_id}:
            continue
        if (same_occurrence(a, x) and same_occurrence(b, y)) or (
            same_occurrence(a, y) and same_occurrence(b, x)
        ):
            return True
    return False


# --------------------------------------------------------------------------- source context

_WS = re.compile(r"\s+")
_BOUNDARY = re.compile(r"(?<=[.!?:;])\s+|\n")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _locate(text: str, quote: str, start: int | None, end: int | None) -> tuple[int, int] | None:
    span = (start, end) if start is not None and end is not None else None
    if span is not None and not 0 <= span[0] < span[1] <= len(text):
        span = None
    if span is not None and (not quote.strip() or _norm(text[span[0] : span[1]]) == _norm(quote)):
        return span
    if quote.strip():
        i = text.find(quote)
        if i >= 0:
            return i, i + len(quote)
    return span


def quote_context(
    text: str | None, quote: str, start: int | None = None, end: int | None = None
) -> str:
    """A few sentences of source text around a quote (whitespace collapsed)."""
    if not text:
        return ""
    span = _locate(text, quote, start, end)
    if span is None:
        return ""
    s, t = span
    lo, hi = max(0, s - CONTEXT_BEFORE), min(len(text), t + CONTEXT_AFTER)
    if lo > 0:  # start at a sentence/line boundary inside the window
        m = _BOUNDARY.search(text, lo, s)
        if m:
            lo = m.end()
    if hi < len(text):  # end at the last boundary inside the window
        last = None
        for m in _BOUNDARY.finditer(text, t, hi):
            last = m
        if last is not None:
            hi = last.start()
    return _WS.sub(" ", text[lo:hi]).strip()


def local_context(
    text: str | None,
    quote: str,
    start: int | None = None,
    end: int | None = None,
    after: int = LOCAL_AFTER,
) -> str:
    """The quote plus up to ``after`` characters of the text that follows it.

    Narrower than ``quote_context``: used to decide whether a source itself excludes people
    from an occurrence ("A. Only the two of us were there." right after the answer), so an
    exclusion stated about some other occurrence a few sentences earlier does not count.
    """
    span = _locate(text, quote, start, end) if text else None
    if not text or span is None:
        return _norm(quote)
    s, t = span
    hi = min(len(text), t + after)
    if hi < len(text):  # end at the last sentence/line boundary inside the window
        last = None
        for m in _BOUNDARY.finditer(text, t, hi):
            last = m
        if last is not None:
            hi = last.start()
    return _WS.sub(" ", text[s:hi]).strip()


class _Docs:
    """Cached document text and filenames from the store (best effort)."""

    def __init__(self, store: Store | None) -> None:
        self._store = store
        self._text: dict[str, str | None] = {}
        self._name: dict[str, str] = {}

    def text(self, doc_id: str) -> str | None:
        if doc_id not in self._text:
            parsed = None
            if self._store is not None:
                try:
                    parsed = self._store.get_parsed(doc_id)
                except Exception as exc:
                    log.debug("could not read text of %s: %s", doc_id, exc)
            self._text[doc_id] = parsed.text if parsed is not None else None
        return self._text[doc_id]

    def name(self, doc_id: str) -> str:
        if doc_id not in self._name:
            doc = None
            if self._store is not None:
                try:
                    doc = self._store.get_document(doc_id)
                except Exception:
                    doc = None
            self._name[doc_id] = f"{doc.filename} ({doc_id})" if doc is not None else doc_id
        return self._name[doc_id]


def _shown_evidence(f: Fact) -> list[Any]:
    """Up to MAX_QUOTES quoted evidence links, one per document first."""
    quoted = [e for e in _evidence(f) if e.quote]
    first: list[Any] = []
    seen: set[str | None] = set()
    for e in quoted:
        if e.doc_id not in seen:
            seen.add(e.doc_id)
            first.append(e)
    rest = [e for e in quoted if e not in first]
    return (first + rest)[:MAX_QUOTES]


def _evidence_block(f: Fact, docs: _Docs) -> tuple[str, str]:
    """The prompt's evidence list for one fact, and the raw context text (for cue checks)."""
    lines: list[str] = []
    contexts: list[str] = []
    for e in _shown_evidence(f):
        ctx = quote_context(docs.text(e.doc_id), e.quote, e.char_start, e.char_end)
        contexts.append(ctx)
        lines.append(f'  - document {docs.name(e.doc_id) if e.doc_id else "unknown"}: "{e.quote}"')
        lines.append(f"    surrounding text: <context>{ctx or '(not available)'}</context>")
    if not lines:
        names = ", ".join(docs.name(d) for d in sorted(_docs(f))) or "no document"
        lines.append(f"  - (no quote; cites {names})")
    return "\n".join(lines), " ".join(contexts)


def _local_text(f: Fact, docs: _Docs) -> str:
    """Each quote shown for the fact plus the text right after it (``local_context``)."""
    return " ".join(
        local_context(docs.text(e.doc_id), e.quote, e.char_start, e.char_end)
        for e in _shown_evidence(f)
    )


def _resolved(f: Fact) -> str:
    if f.date_start is None:
        return "undated"
    if f.date_end and f.date_end != f.date_start:
        return f"{f.date_start.isoformat()} to {f.date_end.isoformat()} ({f.date_certainty})"
    return f"{f.date_start.isoformat()} ({f.date_certainty})"


def _fields(f: Fact, suffix: str, names: Mapping[str, str], evidence: str) -> dict[str, str]:
    return {
        f"fact_{suffix}": f.fact_id,
        f"title_{suffix}": f.title,
        f"details_{suffix}": f.details or "none",
        f"date_{suffix}": f.date_text or "none",
        f"resolved_{suffix}": _resolved(f),
        f"people_{suffix}": ", ".join(names.get(w, w) for w in f.witness_ids) or "none",
        f"evidence_{suffix}": evidence,
    }


# --------------------------------------------------------------------------- confirmation


def confirm(
    cand: Candidate, names: Mapping[str, str], docs: _Docs | None = None
) -> tuple[ConflictVerdict, str]:
    """Ask the local LLM whether a candidate is a real contradiction.

    Returns the verdict and the surrounding source text both facts were shown with.
    """
    from pipeline import llm

    docs = docs or _Docs(None)
    ev_a, ctx_a = _evidence_block(cand.a, docs)
    ev_b, ctx_b = _evidence_block(cand.b, docs)
    user = prompts.CONFLICT_USER.format(
        field=cand.field, **_fields(cand.a, "a", names, ev_a), **_fields(cand.b, "b", names, ev_b)
    )
    notes = date_notes(cand.a, cand.b)
    if notes:
        user += "\n\nNotes from the date rules (computed, reliable):\n" + "\n".join(
            f"- {n}" for n in notes
        )
    result = llm.chat_json(
        prompts.CONFLICT_SYSTEM, user, ConflictVerdict, max_tokens=VERDICT_MAX_TOKENS
    )
    if not isinstance(result, ConflictVerdict):
        data = result.model_dump() if isinstance(result, BaseModel) else result
        result = ConflictVerdict.model_validate(data)
    return result, f"{ctx_a} {ctx_b}"


def final_fields(
    verdict: ConflictVerdict,
    a: Fact,
    b: Fact,
    context: str = "",
    witness_names: Mapping[str, Sequence[str]] | None = None,
    local: tuple[str, str] | None = None,
) -> list[ConflictField]:
    """Fields to open after deterministic sanity checks on the model's answer.

    - DATE is dropped when the dates are compatible (see ``date_relation``): overlapping, one
      approximate period containing or nearly touching the other date, a message date vs the
      date it reports, a one-day time-zone shift, or clock times within a minute and a half.
      Otherwise DATE needs two resolved dates; when either side has none the disagreement is
      reported as FACT, and also when either fact is about someone's knowledge ("first
      learned of it in mid-March").
    - ACTOR is dropped for a plan compared with an outcome and for knowledge facts (when
      someone learned of a matter says nothing about who was present). When the facts share
      a person, ACTOR also needs the side that lacks a person named by the other to exclude
      people affirmatively ("only", "no one else", "was not there") in its quotes or the text
      right after them (``local``: fact A's, fact B's; without it, either fact's quotes or
      ``context`` will do): a source that merely omits a person is not a conflict. (The same
      act attributed to entirely different people needs no cue.)
    """
    if not (verdict.is_contradiction and verdict.same_subject):
        return []
    rel = date_relation(a, b)
    out: list[ConflictField] = []
    for f in verdict.fields():
        if f == "date":
            if rel.compatible:
                continue
            if rel.gap is None or _is_knowledge(a) or _is_knowledge(b):
                f = "fact"
        if f == "actor" and not _actor_conflict_possible(a, b, context, witness_names, local):
            continue
        if f not in out:
            out.append(f)
    return out


def _actor_conflict_possible(
    a: Fact,
    b: Fact,
    context: str,
    witness_names: Mapping[str, Sequence[str]] | None,
    local: tuple[str, str] | None,
) -> bool:
    if _is_plan(a) != _is_plan(b) or _is_knowledge(a) or _is_knowledge(b):
        return False
    pa, pb = _persons(a, witness_names), _persons(b, witness_names)
    if not linked(pa, pb):
        return True
    if local is None:
        return bool(EXCLUSION_CUES.search(" ".join([*_quotes(a), *_quotes(b), context])))
    la, lb = local
    return bool(
        (unmatched(pb, pa) and EXCLUSION_CUES.search(la))
        or (unmatched(pa, pb) and EXCLUSION_CUES.search(lb))
    )


def detect(
    store: Store, case_id: str, fact_ids: Sequence[str], *, max_candidates: int = MAX_CANDIDATES
) -> list[Contradiction]:
    """Find, confirm and open contradictions touching ``fact_ids``; returns the new ones.

    Pairs that already have a contradiction (any status, any field) are not re-checked, so a
    dismissed conflict stays dismissed, and a pair that restates an existing contradiction's
    disagreement through other facts about the same occurrences is skipped. A failed
    confirmation call skips that candidate.
    """
    focus = set(fact_ids)
    if not focus:
        return []
    facts = store.list_facts(case_id)  # drafts and accepted; rejected facts are excluded
    by_id = {f.fact_id: f for f in facts}
    existing = store.list_contradictions(case_id)
    known = {frozenset(c.fact_ids) for c in existing}
    witnesses = store.list_witnesses(case_id)
    names = {w.witness_id: w.name for w in witnesses}
    witness_names = {w.witness_id: [w.name, *w.aliases] for w in witnesses}
    # Disagreements already on record (any status), as (field, fact, fact).
    opened_pairs: list[tuple[str, Fact, Fact]] = []
    for c in existing:
        fs = [by_id[i] for i in c.fact_ids if i in by_id]
        opened_pairs += [(c.field, x, y) for i, x in enumerate(fs) for y in fs[i + 1 :]]
    docs = _Docs(store)
    opened: list[Contradiction] = []
    cands = candidate_pairs(
        facts, focus, max_candidates=max_candidates, skip_pairs=known, witness_names=witness_names
    )
    for cand in cands:
        ids = [cand.a.fact_id, cand.b.fact_id]
        if same_disagreement(cand.field, cand.a, cand.b, opened_pairs):
            log.info("conflict %s (%s) restates an open disagreement; skipped", ids, cand.field)
            continue
        try:
            verdict, context = confirm(cand, names, docs)
        except Exception as exc:
            log.warning("conflict check failed for %s (%s): %s", ids, cand.field, exc)
            continue
        log.info(
            "conflict verdict %s suggested=%s score=%.1f same_subject=%s contradiction=%s "
            "fields=%s: %s",
            ids,
            cand.field,
            cand.score,
            verdict.same_subject,
            verdict.is_contradiction,
            verdict.fields(),
            (verdict.explanation or verdict.comparison).strip()[:300],
        )
        explanation = (verdict.explanation or verdict.comparison).strip()
        local = (_local_text(cand.a, docs), _local_text(cand.b, docs))
        for field in final_fields(verdict, cand.a, cand.b, context, witness_names, local):
            if same_disagreement(field, cand.a, cand.b, opened_pairs):
                log.info("conflict %s (%s) restates an open disagreement; skipped", ids, field)
                continue
            contradiction, created = store.add_contradiction(
                case_id, ids, field, explanation or f"Possible {field} conflict."
            )
            if created:
                opened.append(contradiction)
                opened_pairs.append((field, cand.a, cand.b))
    log.info("contradictions opened: %d (from %d candidates)", len(opened), len(cands))
    return opened
