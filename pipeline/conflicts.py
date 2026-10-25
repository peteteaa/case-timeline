"""Find contradictions between timeline events and derive content-free alerts."""

from __future__ import annotations

import calendar
import hashlib
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pipeline.contracts import (
    Alert,
    AlertType,
    Certainty,
    ConflictField,
    Contradiction,
    ContradictionStatus,
    Event,
    EventStatus,
)
from pipeline.llm import map_ordered
from pipeline.merge import (
    actor_set,
    actors_overlap,
    description_similarity,
    is_organization,
    name_tokens,
    names_match,
    unmatched_people,
)
from pipeline.prompts import CONFLICT_SYSTEM, CONFLICT_USER

log = logging.getLogger(__name__)

SIMILAR_THRESHOLD = 70.0  # legacy description-similarity rule (token_set_ratio)
MAX_CANDIDATES = 60
MAX_GAP_DAYS = 62  # events further apart than this are rarely one misdated occurrence
TOPIC_MIN = 0.3  # topic-word overlap for resolved-date pairs further apart than NEAR_DAYS
TOPIC_MIN_UNRESOLVED = 0.4  # stricter when one date is unresolved (no date evidence)
NEAR_DAYS = 14  # within this many days, one shared topic word is enough for an LLM check
# An approximate or ranged date ("early June", "that week") has soft edges: a date this many
# days outside it is still a granularity difference, not a disagreement.
APPROX_SLACK_DAYS = 3
CONTEXT_BEFORE = 300  # characters of source text shown around each quote
CONTEXT_AFTER = 400
LOCAL_AFTER = 200  # characters after a quote searched for an exclusion cue ("Only the two of us.")
VERDICT_MAX_TOKENS = 800

NEGATION_CUES = re.compile(
    r"\b(not|never|no|none|nobody|nothing|neither|nor|denie[sd]|deny|denying|"
    r"refused?|refuses|absent|failed|cannot|can't|didn't|did\s+not|wasn't|weren't|isn't|"
    r"aren't|hasn't|haven't|hadn't|won't|wouldn't|couldn't|don't|doesn't)\b",
    re.IGNORECASE,
)

# A disagreement about *who* was there needs one source to exclude someone affirmatively;
# silence is not exclusion. One of these cues must appear in either event's quote or context.
EXCLUSION_CUES = re.compile(
    r"\b(only|nobody|no\s+one|no-one|none|alone|nor|neither|never|not|no|absent|"
    r"wasn't|weren't|didn't|denie[sd]|deny|except|besides|other\s+than|else)\b",
    re.IGNORECASE,
)
# Words marking a plan or schedule ("set for", "due by"). A plan compared with what actually
# happened is not a date contradiction, so only plan-vs-plan or fact-vs-fact pairs qualify.
PLAN_CUES = re.compile(
    r"\b(schedul\w*|reschedul\w*|set\s+for|due\s+(?:by|on)|deadline|planned|plans?\s+to|"
    r"expected\s+to|will|postpon\w*|adjourn\w*|moved\s+(?:from|to))\b",
    re.IGNORECASE,
)
# Words marking someone's knowledge or awareness ("first learned", "became aware"). When one
# side is about when someone knew of a matter, a timing disagreement is a FACT (notice)
# conflict about that knowledge, not two dates for one occurrence.
KNOWLEDGE_CUES = re.compile(
    r"\b(learn(?:ed|t|s|ing)?|became\s+aware|been\s+aware|was\s+aware|were\s+aware|unaware|"
    r"aware\s+of|knew|know(?:s|n)?\s+(?:of|about)|first\s+heard|found\s+out|discover(?:ed|s|y)|"
    r"realiz(?:ed|es)|realis(?:ed|es))\b",
    re.IGNORECASE,
)


# Appended to the confirmation system prompt: the false-alarm patterns seen in live runs.
CONFIRM_RULES = """

Further rules (they override anything above):
- Precision is not disagreement. An approximate or ranged time ("early June", "mid-2021", \
"that week", "the spring") is compatible with any exact date inside it or just at its edge. \
Never call that a date conflict.
- Plan vs outcome. A statement that something is scheduled, planned, expected, due or set \
for a date is about the plan. It does not conflict with an account of what actually \
happened (rescheduling is normal). Such a pair has same_subject false unless both sources \
make the same kind of claim (two plans, or two accounts of what happened).
- Message date vs reported date. The date a letter, email, filing or statement was written or \
sent is not the date of an occurrence it reports. Compare the date the text gives for the \
occurrence itself.
- Stated, not inferred. Report a conflict only when what the two sources actually state \
cannot both be true. Do not build one from an inference (e.g. "this implies he already \
knew") or from a different occurrence mentioned nearby.
- When someone learned of something says nothing about who was present at an occurrence, \
so it is never an "actor" conflict."""


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
        out = [self.field]
        out += [f for f in self.other_fields if f not in out and f != ConflictField.FACT]
        return out


def _final_fields(
    verdict: ConflictVerdict,
    a: Event,
    b: Event,
    context: str = "",
    local: tuple[str, str] | None = None,
) -> list[ConflictField]:
    """Fields to report after deterministic sanity checks on the LLM's answer.

    - DATE needs two resolved dates; when either side has none the disagreement is reported as
      FACT (e.g. "learned of it after X" vs a dated notice), and also when either event is
      about someone's knowledge ("first learned of it in mid-June"). DATE is dropped when the
      dates are compatible (see ``date_relation``): overlapping, one approximate period
      containing or nearly touching the other date, or a message date vs the date it reports.
    - ACTOR is dropped for a plan compared with an outcome and for knowledge events (when
      someone learned of a matter says nothing about who was present). When the events share
      a person, ACTOR also needs the side that lacks a person named by the other to exclude
      people affirmatively ("only", "no one else", "was not there") in its quote or the
      sentences right around it (``local``, falling back to the quotes plus ``context``):
      a source that merely omits a person is not a conflict. (The same act attributed to
      entirely different people needs no cue.)
    """
    rel = date_relation(a, b)
    out: list[ConflictField] = []
    for f in verdict.fields():
        if f == ConflictField.DATE:
            if rel.compatible:
                continue
            if rel.gap is None or _is_knowledge(a) or _is_knowledge(b):
                f = ConflictField.FACT
        if f == ConflictField.ACTOR and not _actor_conflict_possible(a, b, context, local):
            continue
        if f not in out:
            out.append(f)
    return out


def _actor_conflict_possible(
    a: Event, b: Event, context: str, local: tuple[str, str] | None
) -> bool:
    if _is_plan(a) != _is_plan(b) or _is_knowledge(a) or _is_knowledge(b):
        return False
    if not linked(a, b):
        return True
    if local is None:
        return bool(EXCLUSION_CUES.search(" ".join([a.source.quote, b.source.quote, context])))
    la, lb = local
    return bool(
        (unmatched_people(b, a) and EXCLUSION_CUES.search(la))
        or (unmatched_people(a, b) and EXCLUSION_CUES.search(lb))
    )


@dataclass(frozen=True)
class Candidate:
    """A cheap-rule candidate pair awaiting LLM confirmation."""

    a: Event
    b: Event
    field: ConflictField
    score: float  # higher = more suspicious; used for capping


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def contradiction_id(event_ids: list[str], field: ConflictField | str) -> str:
    """Deterministic id from the sorted event ids and the conflict field."""
    key = "|".join(sorted(event_ids)) + "#" + str(field)
    return "ctr_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def _alert_id(case_id: str, kind: AlertType, key: str) -> str:
    raw = f"{case_id}|{kind}|{key}"
    return "alr_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _is_plan(event: Event) -> bool:
    return bool(PLAN_CUES.search(event.description) or PLAN_CUES.search(event.source.quote))


def _is_knowledge(event: Event) -> bool:
    return bool(KNOWLEDGE_CUES.search(f"{event.description} {event.source.quote}"))


def _has_negation(event: Event) -> bool:
    return bool(NEGATION_CUES.search(event.description) or NEGATION_CUES.search(event.source.quote))


# --------------------------------------------------------------------------- topic overlap

_STOPWORDS = frozenset(
    """a an the of to in on at for with by from and or but if was were is are be been being
    has had have having do does did that this these those his her hers their theirs its as
    into onto about after before during between over under than then there here also any all
    some each other another such same one two three first second he she they it i me my we
    our you your him them who whom whose which what when where while how why not no so very
    just only both either neither said says stated stating state told tells tell asked
    according approximately around about later earlier sometime time day days week weeks
    month months year years today yesterday morning afternoon evening night am pm
    mr mrs ms dr take took taken taking place held hold holds holding believe believed
    kept keep keeps went go goes going came come comes made make makes get got gave give
    given put way thing things matter re regarding concerning""".split()
)
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_WEEKDAYS = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thurs fri "
    "sat sun".split()
)
_DATE_NUMBERS = re.compile(
    rf"\b(?:(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})"
    rf"|\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}:\d{{2}}"
    rf"|(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W_]+")
_MONTH_SET = frozenset(_MONTHS.split("|"))


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def topic_tokens(event: Event, names: set[str] | None = None) -> set[str]:
    """Content words of the description and quote, minus dates, names and stopwords."""
    text = _DATE_NUMBERS.sub(" ", f"{event.description} {event.source.quote}".casefold())
    skip = names if names is not None else _actor_name_tokens(event)
    out: set[str] = set()
    for word in _WORD.findall(text):
        if word in _STOPWORDS or word in _MONTH_SET or word in _WEEKDAYS or word in skip:
            continue
        if len(word) < 2 and not word.isdigit():
            continue
        out.add(_stem(word))
    return out


def _actor_name_tokens(*events: Event) -> set[str]:
    return {t for e in events for name in e.actors for t in name_tokens(name)}


def topic_overlap(a: Event, b: Event) -> tuple[float, int]:
    """Overlap coefficient (0..1) of the two events' topic words, and the shared-word count.

    Actor names and date expressions are excluded so that the score reflects *what* happened
    (meeting, dismissal, injury...) rather than who or when.
    """
    names = _actor_name_tokens(a, b)
    ta, tb = topic_tokens(a, names), topic_tokens(b, names)
    if not ta or not tb:
        return 0.0, 0
    shared = len(ta & tb)
    return shared / min(len(ta), len(tb)), shared


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


def referenced_dates(e: Event) -> list[date]:
    """Calendar dates written out in the event's quote ("June 5, 2021", "5 June 2021").

    A date without a year takes the year of the event's own resolved date.
    """
    out: list[date] = []
    default_year = e.date_start.year if e.date_start else None
    for m in _EXPLICIT_DATE.finditer(e.source.quote):
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


def _interval(e: Event) -> tuple[date, date] | None:
    if e.date_start is None:
        return None
    end = e.date_end or e.date_start
    return min(e.date_start, end), max(e.date_start, end)


def _span_gap(x: tuple[date, date], y: tuple[date, date]) -> int:
    if x[1] < y[0]:
        return (y[0] - x[1]).days
    if y[1] < x[0]:
        return (x[0] - y[1]).days
    return 0


def carrier_dates(e: Event) -> list[date]:
    """Dates the event's quote refers to when its own date is not among them, else [].

    Then the event's date is most likely the date of the message or document that reports
    the occurrence (an email's "Sent:" line, a letter's date), not the occurrence's own date:
    "we booked the site visit for June 9" in a message dated by its header.
    """
    own = _interval(e)
    if own is None:
        return []
    refs = referenced_dates(e)
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
CLOCK_SLACK_SECONDS = 90  # two sources' clock times this close are one moment


def _zones(e: Event) -> frozenset[str]:
    return frozenset(z.casefold() for z in _ZONE.findall(e.date_text))


def clock_seconds(text: str) -> int | None:
    """Seconds after midnight of the first clock time in ``text``, or None."""
    m = _CLOCK.search(text)
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


def clock_gap_seconds(a: Event, b: Event) -> int | None:
    """Seconds between the clock times written in both events' dates (None unless both have
    one and their calendar dates, if resolved, are the same day)."""
    ca, cb = clock_seconds(a.date_text), clock_seconds(b.date_text)
    if ca is None or cb is None:
        return None
    ia, ib = _interval(a), _interval(b)
    if ia is not None and ib is not None and _span_gap(ia, ib) > 0:
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


def _is_fuzzy(e: Event) -> bool:
    """The resolved *day* has soft edges: a range or period ("early June", "that week",
    "between May and June") or a hedged day ("around March 9").

    APPROXIMATE alone is not enough: the date resolver also uses it for a year inferred from
    context ("March 9") and for the day a bare clock time inherits ("3:40 PM"). Those are
    single definite days, and a few days' gap to another date is a real disagreement.
    """
    span = _interval(e)
    if e.certainty == Certainty.RANGE or (span is not None and span[0] != span[1]):
        return True
    if e.certainty != Certainty.APPROXIMATE:
        return False
    text = e.date_text or ""
    if _PERIOD_HEDGE.search(text):
        return True
    return bool(_POINT_HEDGE.search(text)) and clock_seconds(text) is None


@dataclass(frozen=True)
class DateRelation:
    """How the resolved dates of two events relate."""

    gap: int | None  # days between them (0 if overlapping); None if either is unresolved
    same_day: bool  # both are the same single calendar day
    approximate: bool  # either date is approximate or a range
    carrier: bool = False  # a message date was replaced by the dates its quote refers to
    zone_shift: bool = False  # the dates are written in different time zones
    clock_gap: int | None = None  # seconds between the clock times both dates give

    @property
    def distinct_moments(self) -> bool:
        """The clock times are too far apart to be one occurrence (see ``distinct_moments``)."""
        return self.clock_gap is not None and self.clock_gap > CLOCK_SLACK_SECONDS

    @property
    def same_moment(self) -> bool:
        """Both sources give clock times within ``CLOCK_SLACK_SECONDS``: one moment."""
        return self.clock_gap is not None and self.clock_gap <= CLOCK_SLACK_SECONDS

    @property
    def compatible(self) -> bool:
        """The dates cannot conflict: they overlap, an approximate period nearly touches the
        other date, or a one-day gap comes from different time zones (precision, not
        disagreement)."""
        if self.same_moment:
            return True
        if self.gap is None:
            return False
        return (
            self.gap == 0
            or (self.approximate and self.gap <= APPROX_SLACK_DAYS)
            or (self.zone_shift and self.gap <= 1)
        )


def date_relation(a: Event, b: Event) -> DateRelation:
    """Compare two events' dates, treating a message date as the dates its quote refers to.

    The comparison only ever gets more lenient: the gap is the smaller of the plain gap and
    the gap after substituting quoted dates for a message date.
    """
    ia, ib = _interval(a), _interval(b)
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


def _has_seconds(text: str) -> bool:
    m = _CLOCK.search(text)
    return m is not None and bool(m.group("s") or m.group("s2"))


def distinct_moments(a: Event, b: Event, rel: DateRelation | None = None) -> bool:
    """Both events are timed on the same day, more than ``CLOCK_SLACK_SECONDS`` apart, and
    the times are precise enough to tell occurrences apart: both are given to the second
    (log or transcript timestamps), or both come from one source's own timeline. A loose
    "about 2:30" vs "2:47" from two witnesses may still be one occurrence."""
    rel = rel or date_relation(a, b)
    if not rel.distinct_moments:
        return False
    if a.source.doc_id == b.source.doc_id:
        return True
    return _has_seconds(a.date_text) and _has_seconds(b.date_text)


def date_notes(a: Event, b: Event) -> list[str]:
    """Plain-language facts about the two dates, given to the LLM with the candidate."""
    rel = date_relation(a, b)
    notes: list[str] = []
    for label, e in (("A", a), ("B", b)):
        refs = carrier_dates(e)
        if refs:
            notes.append(
                f"Event {label}'s date ({e.date_text}) is not mentioned in its quote, which "
                f"refers to {', '.join(d.isoformat() for d in refs)}: it is probably the date "
                "of the message or document, not of the occurrence it reports."
            )
    if rel.same_moment:
        notes.append(
            f"Both events give clock times only {rel.clock_gap} seconds apart: for these "
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
    for label, e in (("A", a), ("B", b)):
        if _is_plan(e):
            notes.append(
                f"Event {label} is about a plan, schedule or deadline; it is only in conflict "
                "with another statement about that same plan, not with what later happened."
            )
    return notes


# --------------------------------------------------------------------------- candidates


def _people(e: Event) -> list[str]:
    return [x for x in e.actors if x.strip() and name_tokens(x) and not is_organization(x)]


def linked(a: Event, b: Event) -> bool:
    """The events share an actor; a shared person is required when both name people, so a
    common employer or agency alone does not link two events."""
    pa, pb = _people(a), _people(b)
    if pa and pb:
        return any(names_match(x, y) for x in pa for y in pb)
    return actors_overlap(a, b)


def _classify(a: Event, b: Event) -> Candidate | None:
    """Apply the DATE / ACTOR / FACT candidate rules to one pair."""
    if _is_plan(a) != _is_plan(b):
        # A plan, schedule or deadline vs what actually happened: things get rescheduled, so
        # this is no conflict of any kind unless both sources make the same kind of claim.
        return None
    cross_doc = a.source.doc_id != b.source.doc_id
    rel = date_relation(a, b)
    gap = rel.gap
    sim = description_similarity(a, b)
    link = linked(a, b)
    topic, shared = topic_overlap(a, b) if cross_doc and link else (0.0, 0)

    if gap is None or (gap == 0 and not rel.same_day):
        # No usable date on one side, or one date is a period that merely contains the other:
        # the dates are no evidence that the events are one occurrence, so only a strong
        # same-subject signal is worth an LLM call (e.g. "learned of it only after the
        # hearing" vs a dated written notice).
        if cross_doc and link and topic >= TOPIC_MIN_UNRESOLVED:
            return Candidate(a, b, ConflictField.FACT, 100.0 * topic)
        return None
    if gap > 0 and rel.compatible:
        return None  # precision only: an approximate edge, or a time-zone shift

    both_exact = a.certainty == Certainty.EXACT and b.certainty == Certainty.EXACT
    if gap > 1 or (gap == 1 and both_exact):
        # A one-day gap only counts when both dates are stated exactly ("June 4" vs "June 5").
        if not cross_doc or not link:
            return None
        near = gap <= NEAR_DAYS and shared >= 1
        if sim >= SIMILAR_THRESHOLD or near or (gap <= MAX_GAP_DAYS and topic >= TOPIC_MIN):
            closeness = max(0.0, 1.0 - gap / MAX_GAP_DAYS)
            kind = (
                ConflictField.FACT if _is_knowledge(a) or _is_knowledge(b) else ConflictField.DATE
            )
            return Candidate(a, b, kind, 100.0 * max(topic, sim / 100) + closeness)
        return None
    if gap != 0:
        return None
    if distinct_moments(a, b, rel):
        # Same day but clearly different moments (a log or transcript lists many timed
        # entries per day): the rules below assume one occurrence, so they do not apply.
        return None

    # Same calendar day, stated precisely on both sides.
    sa, sb = actor_set(a), actor_set(b)
    knowledge = _is_knowledge(a) or _is_knowledge(b)  # about who knew, not who was there
    if cross_doc and sa and sb and not link and sim >= SIMILAR_THRESHOLD and not knowledge:
        return Candidate(a, b, ConflictField.ACTOR, sim)
    if cross_doc and link and shared >= 1 and not knowledge:
        differing = unmatched_people(a, b) + unmatched_people(b, a)
        if differing:
            # Same occurrence, different people named: an omission or a real disagreement
            # about who was there; the LLM decides using the surrounding source text.
            return Candidate(a, b, ConflictField.ACTOR, 100.0 * topic)
    if link and sim < SIMILAR_THRESHOLD and (_has_negation(a) or _has_negation(b)):
        return Candidate(a, b, ConflictField.FACT, 100.0 - sim)
    if cross_doc and link and topic >= TOPIC_MIN and (_has_negation(a) or _has_negation(b)):
        return Candidate(a, b, ConflictField.FACT, 100.0 * topic)
    return None


def candidate_pairs(
    events: list[Event],
    max_candidates: int = MAX_CANDIDATES,
    *,
    focus_ids: set[str] | None = None,
) -> list[Candidate]:
    """Generate candidate contradictions with cheap rules, capped at ``max_candidates``.

    With ``focus_ids``, only pairs involving at least one of those events are considered.
    """
    live = [e for e in events if e.status != EventStatus.REJECTED]
    found: list[Candidate] = []
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            if a.id == b.id:
                continue
            if focus_ids is not None and a.id not in focus_ids and b.id not in focus_ids:
                continue
            cand = _classify(a, b)
            if cand is not None:
                found.append(cand)
    found.sort(key=lambda c: (-c.score, c.a.id, c.b.id))
    if len(found) > max_candidates:
        dropped = found[max_candidates:]
        log.warning(
            "conflict candidates capped at %d; dropped %d: %s",
            max_candidates,
            len(dropped),
            [(c.a.id, c.b.id, str(c.field)) for c in dropped],
        )
        found = found[:max_candidates]
    log.info("conflict candidates: %d", len(found))
    return found


# --------------------------------------------------------------------------- source context

SourceText = Callable[[str], str | None]

_WS = re.compile(r"\s+")
_BOUNDARY = re.compile(r"(?<=[.!?:;])\s+|\n")


def _locate(text: str, e: Event) -> tuple[int, int] | None:
    s, t = e.source.char_start, e.source.char_end
    quote = e.source.quote
    if 0 <= s < t <= len(text):
        if _WS.sub(" ", text[s:t]).strip() == _WS.sub(" ", quote).strip() or not quote.strip():
            return s, t
    if quote.strip():
        i = text.find(quote)
        if i >= 0:
            return i, i + len(quote)
    if 0 <= s < t <= len(text):
        return s, t
    return None


def quote_context(text: str | None, e: Event) -> str:
    """Up to a few sentences of source text around the event's quote (whitespace collapsed)."""
    if not text:
        return ""
    span = _locate(text, e)
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


def local_context(text: str | None, e: Event, after: int = LOCAL_AFTER) -> str:
    """The quote plus up to ``after`` characters of the text that follows it.

    Narrower than ``quote_context``: used to decide whether a source itself excludes people
    from an occurrence ("A. Only the two of us were there." right after the answer), so an
    exclusion stated about some other occurrence a few sentences earlier does not count.
    """
    span = _locate(text, e) if text else None
    if not text or span is None:
        return _WS.sub(" ", e.source.quote).strip()
    s, t = span
    hi = min(len(text), t + after)
    if hi < len(text):  # end at the last sentence/line boundary inside the window
        last = None
        for m in _BOUNDARY.finditer(text, t, hi):
            last = m
        if last is not None:
            hi = last.start()
    return _WS.sub(" ", text[s:hi]).strip()


class _StoreDocs:
    """Best-effort read access to stored document text and filenames (cached per doc)."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._text: dict[str, str | None] = {}
        self._name: dict[str, str] = {}

    @classmethod
    def open_default(cls) -> _StoreDocs | None:
        try:
            from pipeline.config import get_settings
            from pipeline.store import Store

            db_path = Path(get_settings().db_path)
            if not db_path.exists():
                return None
            return cls(Store(db_path))
        except Exception as exc:
            log.debug("no stored document text for conflict context: %s", exc)
            return None

    def text(self, doc_id: str) -> str | None:
        if doc_id not in self._text:
            try:
                parsed = self._store.get_parsed_text(doc_id)
                self._text[doc_id] = parsed.text if parsed is not None else None
            except Exception as exc:
                log.debug("could not read text of %s: %s", doc_id, exc)
                self._text[doc_id] = None
        return self._text[doc_id]

    def name(self, doc_id: str) -> str:
        if doc_id not in self._name:
            try:
                doc = self._store.get_document(doc_id)
                self._name[doc_id] = doc.filename if doc is not None else doc_id
            except Exception:
                self._name[doc_id] = doc_id
        return self._name[doc_id]

    def linked_duplicates(self, case_id: str) -> list[Event]:
        """Events hidden from the timeline as duplicates of another event."""
        try:
            every = self._store.list_events(case_id, include_duplicates=True)
            shown = {e.id for e in self._store.list_events(case_id)}
        except Exception as exc:
            log.debug("could not list duplicate events: %s", exc)
            return []
        return [e for e in every if e.id not in shown]

    def close(self) -> None:
        try:
            self._store.close()
        except Exception:
            pass


def with_linked_duplicates(
    events: list[Event], duplicates: list[Event], focus_ids: set[str] | None
) -> tuple[list[Event], set[str] | None]:
    """Add linked duplicate events to the pool checked for contradictions.

    A duplicate is another document's own account of an occurrence; its quote and context can
    disagree with a third document even when the canonical event's do not. With ``focus_ids``,
    duplicates from the documents being scanned (those of the focus events) join the focus.
    """
    known = {e.id for e in events}
    extra = [d for d in duplicates if d.id not in known]
    if not extra:
        return events, focus_ids
    if focus_ids is not None:
        focus_docs = {e.source.doc_id for e in events if e.id in focus_ids}
        focus_ids = focus_ids | {d.id for d in extra if d.source.doc_id in focus_docs}
    return [*events, *extra], focus_ids


# --------------------------------------------------------------------------- confirmation


def _resolved(e: Event) -> str:
    if e.date_start is None:
        return "unresolved"
    if e.date_end and e.date_end != e.date_start:
        return f"{e.date_start.isoformat()} to {e.date_end.isoformat()}"
    return e.date_start.isoformat()


def _event_fields(e: Event, suffix: str, context: str, doc_name: str) -> dict[str, str]:
    """Per-event placeholders used by ``CONFLICT_USER`` (``doc_a``, ``quote_b``, ...)."""
    return {
        f"doc_{suffix}": doc_name,
        f"date_{suffix}": e.date_text,
        f"resolved_{suffix}": _resolved(e),
        f"actors_{suffix}": ", ".join(e.actors) or "none",
        f"description_{suffix}": e.description,
        f"quote_{suffix}": e.source.quote,
        f"context_{suffix}": context or "(not available)",
    }


def _context(e: Event, source_text: SourceText | None) -> str:
    if source_text is None:
        return ""
    try:
        return quote_context(source_text(e.source.doc_id), e)
    except Exception as exc:
        log.debug("context lookup failed for %s: %s", e.id, exc)
        return ""


def _local(e: Event, source_text: SourceText | None) -> str:
    if source_text is None:
        return local_context(None, e)
    try:
        return local_context(source_text(e.source.doc_id), e)
    except Exception as exc:
        log.debug("context lookup failed for %s: %s", e.id, exc)
        return local_context(None, e)


def _confirm(
    cand: Candidate,
    source_text: SourceText | None = None,
    doc_name: Callable[[str], str] | None = None,
    contexts: tuple[str, str] | None = None,
) -> ConflictVerdict | None:
    """Ask the local LLM whether a candidate is a real contradiction."""
    from pipeline import llm

    ctx_a, ctx_b = contexts or (_context(cand.a, source_text), _context(cand.b, source_text))

    def name(e: Event) -> str:
        return doc_name(e.source.doc_id) if doc_name is not None else e.source.doc_id

    values: dict[str, Any] = _SafeDict(
        field=str(cand.field),
        **_event_fields(cand.a, "a", ctx_a, name(cand.a)),
        **_event_fields(cand.b, "b", ctx_b, name(cand.b)),
    )
    user = CONFLICT_USER.format_map(values)
    notes = date_notes(cand.a, cand.b)
    if notes:
        user += "\n\nNotes from the date rules (computed, reliable):\n" + "\n".join(
            f"- {n}" for n in notes
        )
    result = llm.chat_json(
        CONFLICT_SYSTEM + CONFIRM_RULES, user, ConflictVerdict, max_tokens=VERDICT_MAX_TOKENS
    )
    if isinstance(result, ConflictVerdict):
        return result
    if isinstance(result, BaseModel):
        return ConflictVerdict.model_validate(result.model_dump())
    return ConflictVerdict.model_validate(result)


def find_contradictions(
    events: list[Event],
    case_id: str,
    *,
    max_candidates: int = MAX_CANDIDATES,
    skip_ids: set[str] | None = None,
    focus_ids: set[str] | None = None,
    source_text: SourceText | None = None,
    doc_name: Callable[[str], str] | None = None,
) -> list[Contradiction]:
    """Generate candidate pairs with cheap rules, then confirm each with the local LLM.

    ``skip_ids`` holds contradiction ids already stored; those candidates skip the LLM call.
    ``focus_ids`` restricts candidates to pairs involving at least one of those event ids.
    ``source_text(doc_id)`` returns a document's normalized text so the LLM sees the sentences
    around each quote and ``doc_name(doc_id)`` a display name; by default both are read from
    the configured store when one exists. In that case the store's linked duplicates (other
    documents' accounts of an occurrence, hidden from the timeline) are checked as well.
    A confirmed pair yields one contradiction per conflicting field.
    Candidates whose confirmation call fails are logged and skipped.
    """
    skip = skip_ids or set()
    out: dict[str, Contradiction] = {}
    docs = _StoreDocs.open_default() if source_text is None and events else None
    try:
        if docs is not None:
            source_text = docs.text
            doc_name = doc_name or docs.name
            events, focus_ids = with_linked_duplicates(
                events, docs.linked_duplicates(case_id), focus_ids
            )
        cands = candidate_pairs(events, max_candidates, focus_ids=focus_ids)
        _confirm_all(cands, case_id, skip, out, source_text, doc_name)
    finally:
        if docs is not None:
            docs.close()
    log.info("contradictions confirmed: %d (from %d candidates)", len(out), len(cands))
    return list(out.values())


def _redundant(cand: Candidate, f: ConflictField, covered: set[tuple[str, str, str]]) -> bool:
    return (cand.a.id, cand.b.source.doc_id, str(f)) in covered or (
        cand.b.id,
        cand.a.source.doc_id,
        str(f),
    ) in covered


def _confirm_all(
    cands: list[Candidate],
    case_id: str,
    skip: set[str],
    out: dict[str, Contradiction],
    source_text: SourceText | None,
    doc_name: Callable[[str], str] | None,
) -> None:
    """Confirm candidates with the LLM concurrently, then keep the verdicts in candidate order.

    The model calls run in parallel (bounded by ``settings.llm_concurrency`` via the shared
    limiter in ``pipeline.llm``); everything that depends on earlier results (the redundancy
    filter) runs afterwards, in candidate order, so the output equals a sequential run.
    """
    todo = [c for c in cands if contradiction_id([c.a.id, c.b.id], c.field) not in skip]
    # Source text is read here, on the calling thread (the store connection is not shared).
    contexts = [(_context(c.a, source_text), _context(c.b, source_text)) for c in todo]
    local = [(_local(c.a, source_text), _local(c.b, source_text)) for c in todo]
    names = {}
    if doc_name is not None:
        for c in todo:
            for e in (c.a, c.b):
                if e.source.doc_id not in names:
                    names[e.source.doc_id] = doc_name(e.source.doc_id)
    cached_name = names.__getitem__ if doc_name is not None else None

    def confirm(i: int) -> ConflictVerdict | None:
        return _confirm(todo[i], source_text, cached_name, contexts[i])

    verdicts = map_ordered(confirm, list(range(len(todo))))

    # (event id, other document id, field) already reported in this run. Candidates arrive best
    # first, so a weaker pair repeating the same event-vs-document disagreement is dropped.
    covered: set[tuple[str, str, str]] = set()
    for cand, ctx, loc, verdict in zip(todo, contexts, local, verdicts, strict=True):
        ids = sorted([cand.a.id, cand.b.id])
        if isinstance(verdict, Exception):
            log.warning("conflict check failed for %s (%s): %s", ids, cand.field, verdict)
            continue
        if verdict is None:
            continue
        confirmed = verdict.is_contradiction and verdict.same_subject
        log.info(
            "conflict verdict %s suggested=%s score=%.1f same_subject=%s contradiction=%s "
            "fields=%s: %s",
            ids,
            cand.field,
            cand.score,
            verdict.same_subject,
            verdict.is_contradiction,
            [str(f) for f in verdict.fields()],
            (verdict.explanation or verdict.comparison).strip()[:300],
        )
        if not confirmed:
            continue
        explanation = (verdict.explanation or verdict.comparison).strip()
        for f in _final_fields(verdict, cand.a, cand.b, " ".join(ctx), loc):
            cid = contradiction_id(ids, f)
            if cid in skip or cid in out:
                continue
            if _redundant(cand, f, covered):
                log.info("conflict %s (%s) already covered for these documents; skipped", ids, f)
                continue
            covered.add((cand.a.id, cand.b.source.doc_id, str(f)))
            covered.add((cand.b.id, cand.a.source.doc_id, str(f)))
            out[cid] = Contradiction(
                id=cid, case_id=case_id, event_ids=ids, field=f, explanation=explanation
            )


def alerts_for(
    contradictions: list[Contradiction],
    events: list[Event],
    case_id: str,
    base_url: str,
    *,
    now: datetime | None = None,
) -> list[Alert]:
    """Build messaging-safe alerts: types and ids only, never case content."""
    created = now or datetime.now(UTC)
    url = f"{base_url.rstrip('/')}/alerts"
    by_id = {e.id: e for e in events}
    alerts: list[Alert] = []

    def add(kind: AlertType, key: str, refs: list[str]) -> None:
        alerts.append(
            Alert(
                id=_alert_id(case_id, kind, key),
                case_id=case_id,
                type=kind,
                created_at=created,
                dashboard_url=url,
                ref_ids=refs,
            )
        )

    active = [c for c in contradictions if c.status != ContradictionStatus.DISMISSED]
    for c in active:
        add(AlertType.NEW_CONTRADICTION, c.id, [c.id])

    for e in events:
        if e.deadline_relevant and e.status != EventStatus.REJECTED:
            add(AlertType.DEADLINE_DATE, e.id, [e.id])

    involvement: dict[str, set[str]] = defaultdict(set)
    for c in active:
        actors: set[str] = set()
        for eid in c.event_ids:
            if eid in by_id:
                actors |= actor_set(by_id[eid])
        for actor in actors:
            involvement[actor].add(c.id)
    for actor in sorted(involvement):
        cids = involvement[actor]
        if len(cids) >= 2:
            add(AlertType.FLAGGED_WITNESS, actor, sorted(cids))
    return alerts
