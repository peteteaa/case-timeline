"""Resolve free-text dates on drafted facts into date ranges with an honest certainty (§9.10).

Precision is never invented: a month-only date becomes the whole month ("approximate"), a
season becomes the season's months, a span between two exact days ("March 3-5, 2024") is
"exact" with both ends set, and an unparseable date stays "unknown" with no dates. Relative
dates ("two days after the meeting") resolve against an anchor: another draft of the same
document (``anchor_key``), an existing fact (``anchor_fact_id``), the incident the document is
about, or the document's own date; unresolved ones stay undated with certainty "relative".

A bare time of day ("At 1816:19", "1729", "2130 eastern daylight time") is never a date on its
own: it takes the day stated in the same sentence, else the most recent earlier day in the
narrative, else the incident date. A time stated explicitly in UTC/GMT/Zulu is converted to the
case's time zone before its calendar day is taken.
"""

from __future__ import annotations

import calendar
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import dateparser
from dateutil.relativedelta import relativedelta
from rapidfuzz import fuzz

from pipeline.schema import Certainty, DraftFact, Fact, ParsedDocument

logger = logging.getLogger(__name__)

Span = tuple[date, date]

_MONTHS: dict[str, int] = {}
for _i in range(1, 13):
    _MONTHS[calendar.month_name[_i].lower()] = _i
    _MONTHS[calendar.month_abbr[_i].lower()] = _i
_MONTHS["sept"] = 9
_MONTH = r"(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"

_WEEKDAYS = {calendar.day_name[i].lower(): i for i in range(7)}
_WEEKDAY = r"(?:" + "|".join(_WEEKDAYS) + r")"

_NUMBERS: dict[str, tuple[int, int]] = {
    "a": (1, 1), "an": (1, 1), "one": (1, 1), "two": (2, 2), "three": (3, 3),
    "four": (4, 4), "five": (5, 5), "six": (6, 6), "seven": (7, 7), "eight": (8, 8),
    "nine": (9, 9), "ten": (10, 10), "eleven": (11, 11), "twelve": (12, 12),
    "thirteen": (13, 13), "fourteen": (14, 14), "fifteen": (15, 15), "sixteen": (16, 16),
    "seventeen": (17, 17), "eighteen": (18, 18), "nineteen": (19, 19), "twenty": (20, 20),
    "twenty-one": (21, 21), "thirty": (30, 30), "forty": (40, 40), "forty-five": (45, 45),
    "sixty": (60, 60), "ninety": (90, 90), "a couple of": (2, 2), "a couple": (2, 2),
    "couple of": (2, 2), "a few": (2, 4), "few": (2, 4), "several": (3, 7),
}  # fmt: skip
_NUM = r"(?:\d+|" + "|".join(re.escape(k) for k in sorted(_NUMBERS, key=len, reverse=True)) + r")"

_DATEPARSER_SETTINGS = {
    "PREFER_DAY_OF_MONTH": "first",
    "STRICT_PARSING": True,
    "REQUIRE_PARTS": ["day", "month", "year"],
    "DATE_ORDER": "MDY",
    "RETURN_AS_TIMEZONE_AWARE": False,
}

# Words that mean the text is relative; dateparser must not resolve these against "now".
_RELATIVE_WORDS = re.compile(
    r"\b(ago|yesterday|today|tomorrow|tonight|now|next|last|previous|following|later|earlier"
    r"|after|before|prior|this|that|same|days?|weeks?|months?|years?|hence|within)\b"
)

_APPROX_PREFIXES = (
    "on or about", "on or around", "in or about", "in or around", "at or about",
    "at or around", "sometime around", "sometime during", "sometime in", "sometime on",
    "some time in", "some time around", "sometime", "some time", "approximately",
    "approx.", "approx", "around", "about", "roughly", "circa", "ca.", "c.",
    "towards", "toward", "near",
)  # fmt: skip
_PLAIN_PREFIXES = (
    "effective as of", "no later than", "not later than", "on or before", "dated as of",
    "as of", "dated", "date:", "effective", "during", "on", "in", "by", "at", "of", "for",
)  # fmt: skip
_OPEN_BOUND = re.compile(
    r"^(?:shortly |soon |immediately |just |right |sometime |some time |not long |well )?"
    r"(before|after|prior to|since|until|till|following|preceding|up to|subsequent to)\s+(.+)$"
)
# A start point: "since 2006", "from 2006", "starting in 2006", "beginning in March 2007"
# ("beginning of March 2024" is the first part of March, not a start point).
_START_BOUND = re.compile(r"^(since|from|starting(?:\s+in)?|beginning\s+in)\s+(.+)$")


# ----------------------------------------------------------------------------- helpers


# Clock times, zones and day parts are dropped before a date is parsed: they say when in the
# day, not which day ("Mon, 25 Mar 2024 09:14:00 -0400", "at 1700 hours", "the morning of").
_TZ = r"(?:\s*(?:[+-]\d{2}:?\d{2}|z\b|(?:utc|gmt|[ecmp][sd]t)\b(?:\s*[+-]\d{1,2}(?::?\d{2})?)?))?"
_TIME_QUAL = (
    r"(?:(?:approximately|approx\.?|about|around|roughly|circa|ca\.|shortly before"
    r"|shortly after|just before|just after)\s+)?"
)
# Spelled-out zones ("central daylight time") and "local"/"lcl", as in incident reports.
_TZ_WORDS = (
    r"(?:(?:eastern|central|mountain|pacific|alaska|hawaii|atlantic)\s+(?:standard|daylight)"
    r"(?:\s+(?:saving\s+)?time)?|local(?:\s+time)?|lcl|[ecmp][sd]t|utc|gmt|zulu|z)\b"
)
_HHMM = r"(?:[01]\d|2[0-3])[0-5]\d"  # 24-hour clock without a colon ("1816")
_CLOCK = (
    rf"(?:{_HHMM}:[0-5]\d(?:\.\d+)?{_TZ}"  # "1810:33" (HHMM:SS, recorder transcripts)
    rf"|\d{{1,2}}:\d{{2}}(?::\d{{2}}(?:\.\d+)?)?(?:\s*[ap]\.?\s?m\b\.?)?{_TZ}"
    rf"|\d{{1,2}}\s*[ap]\.?\s?m\b\.?{_TZ}|[0-2]\d[0-5]\d\s*(?:hours|hrs\.?|h\b)(?:\s+local)?{_TZ}"
    rf"|{_HHMM}\s*{_TZ_WORDS}"  # "1829 eastern daylight time", "1342 cdt"
    rf"|noon|midnight)"
)
_TIME_OF_DAY = re.compile(
    rf"(?:,\s*|\s+|^)(?:at\s+|@\s*|by\s+)?{_TIME_QUAL}{_CLOCK}"
    rf"(?:\s+(?:hours|hrs\.?|local(?:\s+time)?))?"
)
# A bare 24-hour time is only a time (not a year) when introduced by "at" ("at 1729",
# "at about 0930"), or when qualified as approximate and too early to be a plausible year
# ("about 1815"). "About 2005" and "in 1999" stay years.
_BARE_HHMM = re.compile(
    rf"(?:,\s*|\s+|^)(?:(?P<at>at\s+(?:(?:approximately|approx\.?|about|around)\s+)?)"
    rf"|(?:approximately|approx\.?|about|around)\s+)(?P<hhmm>{_HHMM})\b"
    rf"(?!\s*(?:[-/.,]\s*)?\d|\s+(?:{_MONTH}))"
)
_DAY_PART = r"(?:early\s+|late\s+|mid-?)?(?:morning|afternoon|evening|night|day)"
_DAY_PART_PREFIX = re.compile(rf"^(?:in\s+|during\s+|on\s+)?the\s+{_DAY_PART}\s+of\s+")
_DAY_PART_SUFFIX = re.compile(rf",?\s+(?:in|during)\s+the\s+{_DAY_PART}$")
_LEADING_APPROX = re.compile(r"^(?:approximately|approx\.?|about|around|roughly|circa)\s+")


def _drop_bare_hhmm(m: re.Match[str]) -> str:
    return m.group(0) if not m.group("at") and int(m.group("hhmm")) >= 1900 else " "


def _strip_time(s: str) -> str:
    """Drop clock times, time zones and day parts from lowercased text.

    A time's own qualifier goes with it ("March 14, 2024 at approximately 5:40 PM" is still
    an exact date), but a qualifier that opens the whole phrase ("around 3pm on March 3,
    2024") is kept, so the date stays approximate. Text that is only a time becomes "".
    """
    s = re.sub(r"(\d)t(?=\d{1,2}:\d{2})", r"\1 ", s)  # ISO 8601 "2021-06-05T09:30"
    lead = _LEADING_APPROX.match(s)
    out = _BARE_HHMM.sub(_drop_bare_hhmm, _TIME_OF_DAY.sub(" ", s))
    if re.search(r"\d", out):  # never reduce "this morning" to "this"
        out = _DAY_PART_PREFIX.sub("", out.strip())
        out = _DAY_PART_SUFFIX.sub("", out)
    if out == s:
        return s
    out = re.sub(r"\b(?:between|from) (?:and|to|until|-) ", "", re.sub(r"\s+", " ", out))
    out = out.strip(" ,;:-")
    out = re.sub(r"(?:^|\s+)(?:on|at|of)$", "", out).strip()
    if out and lead and not _LEADING_APPROX.match(out):
        out = lead.group(0) + out
    return out


def _lower(text: str) -> str:
    """Lowercased, normalized text with its times of day still in it."""
    s = text.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s*[–—]\s*", " - ", s)
    s = re.sub(r"\(\s*\d+\s*\)", " ", s)  # "thirty (30) days"
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"(\d{1,2}/\d{1,2}/\d{4})(?=\d{1,2}:\d{2})", r"\1 ", s)  # OCR "07/02/202118:17"
    s = re.sub(r"(?<=[a-z])/(?=\d)", " / ", s)  # one time in two zones: "2316 utc/1816 cdt"
    return re.sub(r"\s+", " ", s).strip()


def _clean(text: str) -> str:
    s = _strip_time(_lower(text))  # "09:14:00 -0400", "at 4:02 pm", "at 1729": a day, not a year
    return s.strip(" .,;:\"'/")


_TIME_JOINERS = re.compile(r"\b(?:or|to|and|until|till|through|between|from)\b")
_APPROX_TIME = re.compile(
    r"\b(?:approximately|approx\.?|about|around|roughly|circa|ca\.)\s+(?:at\s+)?\d"
)


def is_time_only(text: str | None) -> bool:
    """`text` states only a time of day, no date ("at 1816:19", "6:15 pm", "about 2:30 or
    3:00", "2316:19 utc/1816:19 cdt")."""
    low = _lower(text or "")
    if not low:
        return False
    rest = _strip_time(low)
    if rest and rest != low:  # "2:30 or 3:00", "between 3 pm and 5 pm"
        rest = _TIME_JOINERS.sub(" ", rest)
    return not rest.strip(" ,;:-/.")


def _has_time(text: str) -> bool:
    low = _lower(text)
    return bool(low) and _strip_time(low) != low


# ------------------------------------------------------------------------- time zones

# A clock time stated in UTC: "2316 utc", "2316:57 utc", "23:16z", "0016 z", "11:16 pm gmt",
# "t00:16:00z", "00:16+00:00".
_UTC_CLOCK = re.compile(
    r"(?<![\d:.])(?P<h>[01]?\d|2[0-3]):?(?P<m>[0-5]\d)(?::[0-5]\d(?:\.\d+)?)?"
    r"(?:\s*(?P<ap>[ap])\.?\s?m\b\.?)?"
    r"\s*(?:(?:utc|gmt|zulu|z)\b(?!\s*[+-]\s*\d)|[+-]00:?00\b)"
)
# A local zone: abbreviations, spelled-out zones, "local", or a non-zero numeric offset.
_LOCAL_ZONE = re.compile(
    r"\b(?:[ecmp][sd]t|ak[sd]t|h[sd]t|local|lcl"
    r"|(?:eastern|central|mountain|pacific|alaska|hawaii|atlantic)\s+(?:standard|daylight))\b"
    r"|\d\s*[+-](?!00:?00\b)\d{2}:?\d{2}\b"
)


def case_zone(tz: str | tzinfo | None) -> tzinfo | None:
    """The case time zone from an IANA name ("America/Chicago"); None when unset or unknown."""
    if tz is None or isinstance(tz, tzinfo):
        return tz
    name = tz.strip()
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("unknown case time zone %r; UTC times are not converted", name)
        return None


def _utc_to_case_day(low: str, day: date, tz: tzinfo | None) -> date:
    """`day` as a calendar day in the case zone `tz`, when `low` (lowercased, times kept)
    gives the time of day explicitly in UTC/GMT/Zulu ("July 28, 2010, 0016 UTC" is July 27
    in Chicago). A local zone ("1816 cdt", "local", "-0500") keeps its own date, and so does
    text that also names one; a time with no zone is never shifted.
    """
    if tz is None:
        return day
    m = _UTC_CLOCK.search(low)
    if m is None or _LOCAL_ZONE.search(low):
        return day
    hour, minute = int(m.group("h")), int(m.group("m"))
    if m.group("ap"):
        if hour > 12:
            return day
        hour = hour % 12 + (12 if m.group("ap") == "p" else 0)
    when = datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)
    return when.astimezone(tz).date()


def _last_day(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def _month_span(y: int, m: int) -> Span:
    return date(y, m, 1), date(y, m, _last_day(y, m))


def _year_span(y: int) -> Span:
    return date(y, 1, 1), date(y, 12, 31)


def _month_num(token: str) -> int:
    return _MONTHS[token.rstrip(".")]


def _has_year(s: str) -> bool:
    return bool(re.search(r"\b\d{4}\b", s) or re.search(r"\d{1,2}([/-])\d{1,2}\1\d{2}\b", s))


def _nearest_year(month: int, day: int, ref: date) -> date | None:
    """The month/day occurrence closest to ref (year inferred, so callers mark APPROXIMATE)."""
    options = []
    for y in (ref.year - 1, ref.year, ref.year + 1):
        try:
            options.append(date(y, month, day))
        except ValueError:
            continue
    return min(options, key=lambda d: abs((d - ref).days)) if options else None


# ----------------------------------------------------------------------- absolute dates


def parse_absolute(
    text: str, ref: date | None = None, tz: str | tzinfo | None = None
) -> tuple[Span, Certainty] | None:
    """Parse a date that needs no anchor event. `ref` only helps infer a missing year.

    With a case time zone `tz`, a single day stated with an explicit UTC time ("July 28,
    2010, 0016 UTC") becomes that instant's calendar day in `tz`.
    """
    found = _parse_absolute(text, ref)
    zone = case_zone(tz)
    if found is not None and zone is not None and found[0][0] == found[0][1]:
        day = _utc_to_case_day(_lower(text), found[0][0], zone)
        found = ((day, day), found[1])
    return found


def _parse_absolute(text: str, ref: date | None) -> tuple[Span, Certainty] | None:
    s = _clean(text)
    if not s:
        return None
    try:
        found = _parse_range(s, ref) or _parse_point(s, ref)
    except (ValueError, OverflowError):  # e.g. year 0000, Feb 30
        return None
    if found is None:
        # "March 3, 2024, after the meeting", "March 3, 2024 between 2 and 3": keep the
        # leading part up to the first year, else the longest leading comma-part that parses.
        heads = []
        if (m := re.match(r"^(.*?\b\d{4})\b(?!\s*[-/]\s*\d)", s)) and m.group(1) != s:
            heads.append(m.group(1))
        parts = s.split(",")
        heads += [",".join(parts[:k]).strip() for k in range(len(parts) - 1, 0, -1)]
        for head in heads:
            if head and re.search(r"\d|" + _MONTH, head):
                found = _parse_absolute(head, ref)
                if found is not None:
                    break
    return found


def _parse_point(s: str, ref: date | None) -> tuple[Span, Certainty] | None:
    for prefix in _APPROX_PREFIXES:
        if s.startswith(prefix + " "):
            inner = parse_absolute(s[len(prefix) + 1 :], ref)
            if inner is None:
                return None
            return inner[0], "approximate"

    m = _START_BOUND.match(s)
    if m:
        inner = parse_absolute(m.group(2), ref)
        if inner is None:
            return None
        (lo, hi), cert = inner
        if lo != hi:  # "since 2006" began at some point in 2006, not on January 1
            return (lo, hi), "approximate"
        if m.group(1) != "since":  # "from March 3, 2024": that day
            return inner

    m = _OPEN_BOUND.match(s)
    if m:
        inner = parse_absolute(m.group(2), ref)
        if inner is None:
            return None
        (lo, hi), _ = inner
        bound = lo if m.group(1) in ("before", "prior to", "preceding", "since") else hi
        return (bound, bound), "approximate"

    for prefix in _PLAIN_PREFIXES:
        if s.startswith(prefix + " "):
            return parse_absolute(s[len(prefix) + 1 :], ref)

    return _parse_partial(s, ref) or _parse_full(s, ref) or _parse_month_day(s, ref)


def _parse_partial(s: str, ref: date | None) -> tuple[Span, Certainty] | None:
    """Month-only, year-only, seasons, quarters, early/mid/late, week of."""
    approx: Certainty = "approximate"

    m = re.match(
        r"^(?:the\s+)?(?:(?:early|mid|late)[\s-]+)?(spring|summer|fall|autumn|winter)"
        r"(?:\s+of)?,?\s+(\d{4})(?:\s*[-/]\s*(\d{2,4}))?$",
        s,
    )
    if m:
        season, y = m.group(1), int(m.group(2))
        if season == "spring":
            return (date(y, 3, 1), date(y, 5, 31)), approx
        if season == "summer":
            return (date(y, 6, 1), date(y, 8, 31)), approx
        if season in ("fall", "autumn"):
            return (date(y, 9, 1), date(y, 11, 30)), approx
        if m.group(3):  # "winter 2023-24": Dec 2023 - Feb 2024
            return (date(y, 12, 1), date(y + 1, 2, _last_day(y + 1, 2))), approx
        return (date(y - 1, 12, 1), date(y, 2, _last_day(y, 2))), approx

    part = r"(?P<part>early|mid|middle|late|beginning|start|end|close)"
    joiner = r"(?:\s*-\s*|\s+of\s+(?:the\s+month\s+of\s+)?|\s+)"
    m = re.match(rf"^(?:the\s+)?{part}{joiner}(?P<mon>{_MONTH})(?:\s+of)?,?\s+(?P<y>\d{{4}})$", s)
    if m:
        y, mo = int(m.group("y")), _month_num(m.group("mon"))
        p = m.group("part")
        if p in ("early", "beginning", "start"):
            return (date(y, mo, 1), date(y, mo, 10)), approx
        if p in ("mid", "middle"):
            return (date(y, mo, 11), date(y, mo, 20)), approx
        return (date(y, mo, 21), date(y, mo, _last_day(y, mo))), approx
    m = re.match(rf"^(?:the\s+)?{part}{joiner}(?:the\s+year\s+)?(?P<y>\d{{4}})$", s)
    if m:
        y, p = int(m.group("y")), m.group("part")
        if p in ("early", "beginning", "start"):
            return (date(y, 1, 1), date(y, 4, 30)), approx
        if p in ("mid", "middle"):
            return (date(y, 5, 1), date(y, 8, 31)), approx
        return (date(y, 9, 1), date(y, 12, 31)), approx

    m = re.match(
        r"^(?:q([1-4])|(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter)"
        r"(?:\s+of)?,?\s+(?:fy\s*)?(\d{4})$",
        s,
    )
    if m:
        order = {"first": 1, "second": 2, "third": 3, "fourth": 4}
        q = int(m.group(1)) if m.group(1) else order.get(m.group(2), 0) or int(m.group(2)[0])
        y = int(m.group(3))
        first = 3 * (q - 1) + 1
        return (date(y, first, 1), date(y, first + 2, _last_day(y, first + 2))), approx

    m = re.match(r"^(?:the\s+)?(week|weekend)\s+of\s+(.+)$", s)
    if m:
        inner = parse_absolute(m.group(2), ref)
        if inner is None:
            return None
        (lo, _), _ = inner
        return (lo, lo + timedelta(days=6 if m.group(1) == "week" else 2)), approx

    m = re.match(rf"^(?:the\s+month\s+of\s+)?({_MONTH})(?:\s+of)?,?\s+(\d{{4}})$", s)
    if m:
        return _month_span(int(m.group(2)), _month_num(m.group(1))), approx
    m = re.match(r"^(\d{1,2})/(\d{4})$", s) or re.match(r"^(\d{4})-(\d{1,2})$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        y, mo = (b, a) if a <= 12 and b > 31 else (a, b)
        if 1 <= mo <= 12:
            return _month_span(y, mo), approx
    m = re.match(r"^(?:the\s+year\s+)?(\d{4})$", s)
    if m:
        return _year_span(int(m.group(1))), approx
    m = re.match(r"^(?:the\s+)?(?:(early|mid|late)[\s-]+)?(\d{3})0'?s$", s)
    if m:
        start = int(m.group(2)) * 10
        lo, hi = {"early": (0, 3), "mid": (3, 6), "late": (6, 9)}.get(m.group(1) or "", (0, 9))
        return (date(start + lo, 1, 1), date(start + hi, 12, 31)), approx
    return None


def _parse_full(s: str, ref: date | None) -> tuple[Span, Certainty] | None:
    """A full day/month/year date (time of day is dropped)."""
    s = re.sub(
        rf"^(?:this\s+|the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+day\s+of\s+({_MONTH}),?\s+(\d{{4}})",
        r"\2 \1, \3",
        s,
    )
    if _RELATIVE_WORDS.search(s) or not _has_year(s):
        return None
    settings: dict[str, object] = dict(_DATEPARSER_SETTINGS)
    if ref is not None:
        settings["RELATIVE_BASE"] = datetime(ref.year, ref.month, ref.day)
    parsed = dateparser.parse(s, languages=["en"], settings=settings)
    if parsed is None:
        return None
    d = parsed.date()
    return (d, d), "exact"


def _parse_month_day(s: str, ref: date | None) -> tuple[Span, Certainty] | None:
    """'March 3' / '3 March' with no year: infer the year from context, mark APPROXIMATE."""
    if ref is None or _has_year(s):
        return None
    m = re.match(
        rf"^(?:{_WEEKDAY},?\s+)?(?P<mon>{_MONTH})\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?$", s
    ) or re.match(
        rf"^(?:{_WEEKDAY},?\s+)?(?:the\s+)?(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mon>{_MONTH})$",
        s,
    )
    if not m:
        # "mid-March", "March": the month (or part of it) in the nearest year.
        part = r"(?:early|mid|middle|late|beginning|start|end|close)"
        mm = re.match(
            rf"^(?:the\s+)?(?:{part}(?:\s*-\s*|\s+of\s+|\s+))?(?:the\s+month\s+of\s+)?"
            rf"(?P<mon>{_MONTH})$",
            s,
        )
        if not mm:
            return None
        near = _nearest_year(_month_num(mm.group("mon")), 15, ref)
        return _parse_partial(f"{s} {near.year}", ref) if near else None
    d = _nearest_year(_month_num(m.group("mon")), int(m.group("d")), ref)
    return ((d, d), "approximate") if d else None


def _parse_range(s: str, ref: date | None) -> tuple[Span, Certainty] | None:
    """'between X and Y', 'from X to Y', 'X - Y', 'March 3-5, 2024', '2023-24'."""
    m = re.match(r"^(\d{4})\s*-\s*(\d{2}|\d{4})$", s)
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2)) if len(m.group(2)) == 4 else (y1 // 100) * 100 + int(m.group(2))
        if y2 > y1:
            return (date(y1, 1, 1), date(y2, 12, 31)), "approximate"
        return None

    pairs: list[tuple[str, str]] = []
    m = re.match(r"^(?:between|from)\s+(.+?)\s+(?:and|to|through|thru|until|till|-)\s+(.+)$", s)
    if m:
        pairs.append((m.group(1), m.group(2)))
    else:
        m = re.match(r"^(.+?)\s+(?:to|through|thru|until|till)\s+(.+)$", s)
        if m:
            pairs.append((m.group(1), m.group(2)))
        # Alternatives span both: "May or June of 2007", "March 3 or 4, 2024".
        m = re.match(rf"^((?:{_MONTH}|\d)[^,]*?)\s+or\s+(.+)$", s)
        if m:
            pairs.append((m.group(1), m.group(2)))
        numeric = re.search(r"\d{1,4}-\d{1,2}-\d{1,4}", s)
        part_prefix = re.match(r"^(?:the\s+)?(?:early|mid|middle|late)\s*-", s)
        if not numeric and not part_prefix:
            for hy in re.finditer(r"\s*-\s*", s):
                pairs.append((s[: hy.start()], s[hy.end() :]))
    for left, right in pairs:
        result = _range_sides(left.strip(" ,"), right.strip(" ,"), ref)
        if result:
            return result
    return None


def _range_sides(left: str, right: str, ref: date | None) -> tuple[Span, Certainty] | None:
    if not left or not right:
        return None
    day = r"(\d{1,2})(?:st|nd|rd|th)?"
    # "March 3 - 5, 2024": borrow the month (and year) for the bare right-hand day.
    mr = re.match(rf"^{day}(?:,?\s*(\d{{4}}))?$", right)
    ml = re.match(rf"^({_MONTH})\s+{day}(?:,?\s*(\d{{4}}))?$", left)
    if mr and ml:
        year = mr.group(2) or ml.group(3) or ""
        left = f"{ml.group(1)} {ml.group(2)} {year}".strip()
        right = f"{ml.group(1)} {mr.group(1)} {year}".strip()
    # "3 - 5 March 2024": borrow month/year from the right.
    ml2 = re.match(rf"^{day}$", left)
    mr2 = re.match(rf"^{day}\s+(?:of\s+)?(.+)$", right)
    if ml2 and mr2:
        left = f"{ml2.group(1)} {mr2.group(2)}"
    borrowed = False
    year_m = re.search(r"\b(\d{4})\b", right)
    if not _has_year(left) and year_m:
        left = f"{left} {year_m.group(1)}"
        borrowed = True
    lp = parse_absolute(left, ref)
    rp = parse_absolute(right, ref)
    if lp is None or rp is None:
        return None
    if min(lp[0][0].year, rp[0][0].year) < 1000:  # "-0400" and the like are not years
        return None
    (lo, _), _ = lp
    (_, hi), _ = rp
    if lo > hi and borrowed:
        lo = lo - relativedelta(years=1)
    if lo > hi:
        lo, hi = rp[0][0], lp[0][1]
    # Both ends stated to the day: an exact span. Otherwise the ends are themselves fuzzy.
    cert: Certainty = "exact" if lp[1] == "exact" and rp[1] == "exact" else "approximate"
    return (lo, hi), cert


# ----------------------------------------------------------------------- relative dates


@dataclass(frozen=True)
class RelativeSpec:
    """A relative date: `apply(anchor_span, anchor_is_event)` gives the resolved span."""

    apply: Callable[[Span, bool], Span]
    deictic: bool = False  # relative to the document's own date ("yesterday", "3 days ago")
    phrase: str | None = None  # explicit anchor ("the meeting", "March 3, 2024")
    clock: bool = False  # a bare time of day: the day of the narrative around it
    day: bool = False  # "that day", "the next morning": refers back to a single day


def _shift(d: date, n: int, unit: str, business: bool = False) -> date:
    if unit == "day" and business:
        step = 1 if n >= 0 else -1
        out, left = d, abs(n)
        while left:
            out += timedelta(days=step)
            if out.weekday() < 5:
                left -= 1
        return out
    if unit == "day":
        return d + timedelta(days=n)
    if unit == "week":
        return d + timedelta(weeks=n)
    if unit == "month":
        return d + relativedelta(months=n)
    return d + relativedelta(years=n)


def _weekday_after(wd: int, inclusive: bool) -> Callable[[date], date]:
    def f(d: date) -> date:
        delta = (wd - d.weekday()) % 7
        return d + timedelta(days=delta or (0 if inclusive else 7))

    return f


def _weekday_before(wd: int, inclusive: bool) -> Callable[[date], date]:
    def f(d: date) -> date:
        delta = (d.weekday() - wd) % 7
        return d - timedelta(days=delta or (0 if inclusive else 7))

    return f


def _ordered(a: date, b: date) -> Span:
    return (a, b) if a <= b else (b, a)


def _period(span: Span, unit: str, n: int) -> Span:
    """The calendar week/weekend/month/year n periods from the anchor span."""
    lo, hi = span
    if unit in ("week", "weekend"):
        mon_lo = lo - timedelta(days=lo.weekday()) + timedelta(weeks=n)
        mon_hi = hi - timedelta(days=hi.weekday()) + timedelta(weeks=n)
        if unit == "weekend":
            return mon_lo + timedelta(days=5), mon_hi + timedelta(days=6)
        return mon_lo, mon_hi + timedelta(days=6)
    if unit == "month":
        a, b = lo + relativedelta(months=n), hi + relativedelta(months=n)
        return date(a.year, a.month, 1), date(b.year, b.month, _last_day(b.year, b.month))
    if unit == "year":
        return date(lo.year + n, 1, 1), date(hi.year + n, 12, 31)
    return lo + timedelta(days=n), hi + timedelta(days=n)


_SUB_DAY = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}


def _sub_day_days(n: int, unit: str, up: bool = False) -> int:
    """Whole days spanned by n seconds/minutes/hours; `up` rounds a partial day up, since a
    few hours after an event may already be the next day. Seconds and minutes never are."""
    secs = n * _SUB_DAY[unit]
    days, rest = divmod(secs, 86400)
    return days + (1 if up and rest and _SUB_DAY[unit] == 3600 else 0)


_DIR_BACK = {"before", "earlier", "prior to", "ago", "beforehand", "preceding", "previously"}
_DAY_UNITS = {"day", "morning", "afternoon", "evening", "night"}
_PRONOUN_ANCHOR = re.compile(r"^(?:that|this|then|it|which|those|these|there|so)$")


def _anchor_phrase(rest: str | None) -> str | None:
    """The anchor a relative date names; None for a pronoun ("after that", "before this"),
    which refers back to the previous fact rather than naming one."""
    if rest is None or _PRONOUN_ANCHOR.match(rest.strip()):
        return None
    return rest


def parse_relative(text: str) -> RelativeSpec | None:
    """Recognize a relative date expression; None if the text is not relative."""
    if is_time_only(text):
        return RelativeSpec(lambda a, _e: a, clock=True)  # "at about 6:15 pm": same day
    s = _clean(text)
    if not s:
        return None
    s = re.sub(r"^(?:on|in)\s+", "", s)

    m = re.match(
        r"^(?:(the day before yesterday|the day after tomorrow)"
        r"|(today|tonight|now|yesterday|tomorrow)"
        r"|this (morning|afternoon|evening))(?:\s+(?:morning|afternoon|evening|night))?$",
        s,
    )
    if m:
        word = m.group(1) or m.group(2) or "today"
        n = {"yesterday": -1, "tomorrow": 1, "the day before yesterday": -2}.get(word, 0)
        if word == "the day after tomorrow":
            n = 2
        return RelativeSpec(lambda a, _e, n=n: _period(a, "day", n), deictic=True)

    m = re.match(
        rf"^(?P<within>within\s+|no later than\s+|not later than\s+)?"
        rf"(?P<qual>(?:about|approximately|approx\.?|roughly|around|nearly|almost|some|over"
        rf"|more than|less than|just over|just under)\s+)?(?P<n>{_NUM})\s+"
        rf"(?P<biz>business\s+|calendar\s+|working\s+|court\s+)?"
        rf"(?P<unit>day|week|month|year|second|sec|minute|min|hour|hr)s?"
        rf"(?:\s+(?P<dir>after|later|before|earlier|prior to|following|from|of|ago|thereafter"
        rf"|afterwards?|beforehand|hence|previously))?(?:\s+(?P<rest>.+))?$",
        s,
    )
    if m and (m.group("dir") or m.group("within")):
        n_word = m.group("n")
        lo_n, hi_n = (int(n_word), int(n_word)) if n_word.isdigit() else _NUMBERS[n_word]
        unit = m.group("unit")
        business = bool(m.group("biz") and m.group("biz").strip() in ("business", "working"))
        direction = m.group("dir") or "after"
        sign = -1 if direction in _DIR_BACK else 1
        within = bool(m.group("within"))
        fuzzy = bool(m.group("qual"))
        rest = _anchor_phrase(m.group("rest"))
        if direction in ("later", "earlier", "ago", "thereafter", "afterwards", "afterward",
                         "beforehand", "hence", "previously"):  # fmt: skip
            rest = None

        def apply(a: Span, _e: bool) -> Span:
            if unit in _SUB_DAY:  # "15 seconds after the crash", "two hours later"
                lo_d, hi_d = _sub_day_days(lo_n, unit), _sub_day_days(hi_n, unit, up=True)
                if within or sign > 0:
                    return a[0] + timedelta(days=0 if within else lo_d), a[1] + timedelta(days=hi_d)
                return a[0] - timedelta(days=hi_d), a[1] - timedelta(days=0 if within else lo_d)
            if within:  # a window: from the anchor up to n units away
                if sign > 0:
                    return a[0], _shift(a[1], hi_n, unit, business)
                return _shift(a[0], -hi_n, unit, business), a[1]
            lo, hi = _ordered(
                _shift(a[0], sign * lo_n, unit, business), _shift(a[1], sign * hi_n, unit, business)
            )
            if fuzzy:  # "about two weeks later": widen by ~20% of the offset
                offset = (hi - a[1]).days if sign > 0 else (a[0] - lo).days
                slack = timedelta(days=max(1, round(0.2 * abs(offset))))
                lo, hi = lo - slack, hi + slack
            return lo, hi

        return RelativeSpec(apply, deictic=direction in ("ago", "hence"), phrase=rest)

    m = re.match(
        r"^(?P<the>the\s+)?(?P<which>next|following|previous|preceding|prior|last|this|that"
        r"|same|very same|that same|subsequent)\s+(?P<unit>day|date|morning|afternoon|evening"
        r"|night|week|weekend|month|year)$",
        s,
    )
    if m:
        which, unit = m.group("which"), m.group("unit")
        unit = "day" if unit in _DAY_UNITS or unit == "date" else unit
        n = {"next": 1, "following": 1, "subsequent": 1}.get(which, 0)
        if which in ("previous", "preceding", "prior", "last"):
            n = -1
        # "on this date" in a narrative is the date just mentioned, not the document's own.
        deictic = not m.group("the") and which in ("last", "this") and m.group("unit") != "date"
        return RelativeSpec(
            lambda a, _e, n=n, u=unit: _period(a, u, n), deictic=deictic, day=unit == "day"
        )

    m = re.match(
        r"^(?:later|earlier) (?:that|the same|that same) (?:day|morning|afternoon|evening|night)$",
        s,
    )
    if m:
        return RelativeSpec(lambda a, _e: a, day=True)

    # "earlier in the week", "later that month", "that same year": the anchor's period.
    m = re.match(
        r"^(?:(?P<le>later|earlier)\s+(?P<w>that|the same|that same|in the|in that|this)"
        r"|(?:that|the same|that same))\s+(?P<unit>week|weekend|month|year)$",
        s,
    )
    if m:
        unit = m.group("unit")
        return RelativeSpec(lambda a, _e, u=unit: _period(a, u, 0), deictic=m.group("w") == "this")

    m = re.match(
        rf"^(?:the\s+)?(?:(?P<which>next|following|last|previous|preceding|this|that|coming)\s+)?"
        rf"(?P<wd>{_WEEKDAY})(?:\s+(?:morning|afternoon|evening|night))?$",
        s,
    )
    if m:
        which, wd = m.group("which"), _WEEKDAYS[m.group("wd")]
        deictic = which in ("last", "this") and not s.startswith("the ")

        def apply_wd(a: Span, from_event: bool, which=which, wd=wd) -> Span:
            if which in ("next", "following", "coming"):
                f = _weekday_after(wd, inclusive=False)
            elif which in ("last", "previous", "preceding"):
                f = _weekday_before(wd, inclusive=False)
            elif from_event:  # narrative "on Monday" after an event: on/after the anchor
                f = _weekday_after(wd, inclusive=True)
            else:  # "on Monday" in a dated letter: on/before the document date
                f = _weekday_before(wd, inclusive=True)
            return f(a[0]), f(a[1])

        return RelativeSpec(apply_wd, deictic=deictic)

    m = re.match(
        r"^(?:the\s+)?(?P<unit>day|morning|evening|night|week|weekend|month|year)\s+"
        r"(?P<dir>before|after|following|prior to|preceding|of)(?:\s+(?P<rest>.+))?$",
        s,
    )
    if m:
        unit = "day" if m.group("unit") in _DAY_UNITS else m.group("unit")
        d = m.group("dir")
        n = 0 if d == "of" else (-1 if d in ("before", "prior to", "preceding") else 1)
        return RelativeSpec(
            lambda a, _e, n=n, u=unit: _period(a, u, n), phrase=_anchor_phrase(m.group("rest"))
        )

    m = re.match(
        r"^(?P<q>shortly|soon|immediately|right|just|directly|not long|sometime|some time)?\s*"
        r"(?P<dir>after|before|following|prior to|preceding|since|until|upon|at the time of"
        r"|on the day of|on the date of|the day of|the date of|during|afterwards?|thereafter"
        r"|subsequently|later)(?:\s+(?P<rest>.+))?$",
        s,
    )
    if m:
        q, d = m.group("q") or "", m.group("dir")
        back = d in ("before", "prior to", "preceding", "until")
        same = d in ("upon", "at the time of", "on the day of", "on the date of", "the day of",
                     "the date of", "during")  # fmt: skip
        width = 1 if q in ("immediately", "right", "just", "directly") else 0
        width = 7 if q in ("shortly", "soon", "not long") else width

        def apply_b(a: Span, _e: bool, back=back, same=same, width=width) -> Span:
            if same:
                return a
            if back:
                return a[0] - timedelta(days=width), a[0]
            return a[1], a[1] + timedelta(days=width)

        return RelativeSpec(apply_b, phrase=_anchor_phrase(m.group("rest")))
    return None


# --------------------------------------------------------------------- anchor selection

_ARTICLES = re.compile(r"^(?:the|a|an|his|her|their|its|our|my|your|said|that|this)\s+")
_ANCHOR_MIN_SCORE = 85


@dataclass
class _Anchor:
    """A dated candidate anchor: a draft of this document or an existing fact."""

    span: Span
    certainty: str
    text: str  # lowercased title, details and quotes, for phrase matching
    pos: int | None = None  # text position (drafts only)
    key: str | None = None  # draft key
    fact_id: str | None = None
    date_text: str = ""
    quote: str = ""  # the draft's first located quote
    end: int | None = None  # end of that quote in the text

    @property
    def single_day(self) -> bool:
        """Pinned to one day: not a span, and not an open bound ("since 2006")."""
        return self.span[0] == self.span[1] and not _open_ended(self.date_text)

    @property
    def timed(self) -> bool:
        """Stated with a time of day (in its date or its quote)."""
        return _has_time(self.date_text) or _has_time(self.quote)

    @property
    def dateline(self) -> bool:
        """The quote is just its date (a letter's or section's own date line)."""
        rest = _clean(self.quote).replace(_clean(self.date_text), "")
        return bool(self.quote) and not re.sub(r"[\W_]+", "", rest)


def _draft_text(d: DraftFact) -> str:
    quotes = [s.quote for s in d.spans] or d.quotes
    return " ".join([d.title, d.details or "", *quotes]).lower()


def _fact_text(f: Fact) -> str:
    quotes = [e.quote for e in f.evidence if e.quote]
    return " ".join([f.title, f.details or "", *quotes]).lower()


def _match_anchor(phrase: str, pos: int, candidates: list[_Anchor]) -> _Anchor | None:
    """Best candidate whose text mentions the anchor phrase ("the meeting").

    Drafts of the same document win over existing facts; then match score, then a stated
    (not relative) date, then a position before the relative date, then proximity.
    """
    key = _ARTICLES.sub("", _clean(phrase)).strip()
    if len(key) < 3:
        return None
    best: tuple[tuple, _Anchor] | None = None
    for c in candidates:
        score = fuzz.partial_ratio(key, c.text)
        if score < _ANCHOR_MIN_SCORE:
            continue
        in_doc = c.pos is not None
        cpos = c.pos if c.pos is not None else 0
        rank = (
            in_doc,
            score,
            c.certainty != "relative",
            in_doc and cpos < pos,
            -abs(cpos - pos) if in_doc else 0,
        )
        if best is None or rank > best[0]:
            best = (rank, c)
    return best[1] if best else None


def _prior(pos: int, candidates: list[_Anchor]) -> _Anchor | None:
    prior = [c for c in candidates if c.pos is not None and c.pos < pos]
    return max(prior, key=lambda c: c.pos or 0) if prior else None


def _pos(d: DraftFact, i: int) -> int:
    return min(s.char_start for s in d.spans) if d.spans else i


def _first_span(d: DraftFact):  # noqa: ANN202 - SpanIn | None
    return min(d.spans, key=lambda s: s.char_start) if d.spans else None


_PERIOD_WORD = r"(?:day|date|morning|afternoon|evening|night|week|weekend|month|year)s?\b"
_BOUND_TEXT = re.compile(
    r"^(?:sometime |some time |well )?"
    r"(?:before|after|prior to|since|until|till|following|preceding|up to|subsequent to|by"
    r"|no later than|not later than|on or before|on or after|no earlier than|not earlier than"
    r"|from|starting(?: in)?|beginning in)"
    rf"(?:\s+(?!{_PERIOD_WORD})\S|$)"
)


def is_open_bound(date_text: str | None) -> bool:
    """The date only bounds the event ("before March 3", "since 2006", "by May 6").

    Its resolved day is a limit, not when the event happened, so it is never an anchor for a
    relative date and never the context year for a later month/day without a year.
    """
    return bool(date_text) and bool(_BOUND_TEXT.match(_clean(date_text or "")))


def _open_ended(date_text: str | None) -> bool:
    """An open bound or start point ("since 2006", "before March 3, 2024"), not a period."""
    s = _clean(date_text or "")
    return bool(s) and bool(_OPEN_BOUND.match(s) or _START_BOUND.match(s))


def _draft_anchor(d: DraftFact, pos: int) -> _Anchor:
    assert d.date_start is not None
    first = _first_span(d)
    return _Anchor(
        span=(d.date_start, d.date_end or d.date_start),
        certainty=d.date_certainty,
        text=_draft_text(d),
        pos=pos,
        key=d.key,
        date_text=d.date_text or "",
        quote=first.quote if first else "",
        end=first.char_end if first else None,
    )


# ------------------------------------------------------------------ document context

# The occurrence a document is about ("at the time of the accident", "after the crash").
_INCIDENT_REF = re.compile(
    r"^(?:the|this|that|said)\s+(?:[a-z-]+\s+)?"
    r"(?:accident|incident|crash|collision|impact|mishap|occurrence|wreck)$"
)
_INCIDENT_WORD = re.compile(r"\b(?:accident|incident|crash|collision|mishap|wreck)\b", re.I)
# A full explicit date written in the document text (the day is stated).
_TEXT_DATE = re.compile(
    rf"(?:\b{_WEEKDAY},?\s+)?(?:\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{{4}}"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH},?\s+\d{{4}}"
    r"|\b\d{4}-\d{2}-\d{2}|\b\d{1,2}/\d{1,2}/(?:\d{4}|\d{2}))\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*(?=\s)|\n\s*\n")
_CONTEXT_CHARS = 60  # how far around a date to look for a time of day in the same sentence
_HEADING_LINES = 25  # how far down a document's own date (its heading) is looked for


@dataclass(frozen=True)
class _Mention:
    """An explicit full date in the document text."""

    pos: int
    end: int
    day: date  # in the case zone when a UTC time is stated with it
    timed: bool  # a time of day is stated with it ("June 5, 2021, about 0940 cdt")
    dateline: bool  # the date stands alone on its line (a letter or section date)
    stated: date  # the day as written


@dataclass
class _DocContext:
    doc_date: date | None
    incident: date | None = None
    incident_key: str | None = None  # the draft stating the incident date, if any
    text: str = ""
    mentions: list[_Mention] = field(default_factory=list)
    tz: tzinfo | None = None
    heading_end: int = 0  # end of the line stating the document's own date near the top
    waiting: set[int] = field(default_factory=set)  # relative drafts not yet resolved


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """The sentence(s) around text[start:end]; without a visible boundary, just that span.
    The start of the text is a boundary."""
    lo = 0 if start <= 400 else start
    for m in _SENTENCE_END.finditer(text, max(0, start - 400), start):
        lo = m.end()
    m = _SENTENCE_END.search(text, max(end - 1, start), end + 400)
    return min(lo, start), (m.start() if m else end)


def _text_mentions(text: str, tz: tzinfo | None = None) -> list[_Mention]:
    out: list[_Mention] = []
    for m in _TEXT_DATE.finditer(text):
        found = _parse_absolute(m.group(0), None)
        if found is None or found[0][0] != found[0][1]:
            continue
        lo, hi = _sentence_bounds(text, m.start(), m.end())
        around = _lower(
            text[max(lo, m.start() - _CONTEXT_CHARS) : m.start()]
            + " , "
            + text[m.end() : min(hi, m.end() + _CONTEXT_CHARS)]
        )
        line_lo = text.rfind("\n", 0, m.start()) + 1
        line_hi = text.find("\n", m.end())
        line = text[line_lo : len(text) if line_hi < 0 else line_hi]
        dateline = not re.sub(r"[\W_]+", "", line.replace(m.group(0), ""))
        stated = found[0][0]
        day = _utc_to_case_day(around, stated, tz)  # "July 28, 2010, 0016 UTC"
        timed = _strip_time(around) != around
        out.append(_Mention(m.start(), m.end(), day, timed, dateline, stated))
    return out


def _heading_end(text: str, doc_date: date | None, mentions: list[_Mention]) -> int:
    """End of the line near the top that states the document's own date, else 0.

    Everything up to there is the document's heading (an interview's caption, a letter's
    date line), not the account it goes on to give.
    """
    if doc_date is None or not text:
        return 0
    limit = 0
    for _ in range(_HEADING_LINES):
        nl = text.find("\n", limit)
        if nl < 0:
            limit = len(text)
            break
        limit = nl + 1
    for m in mentions:
        if m.pos >= limit:
            break
        if doc_date in (m.stated, m.day):
            nl = text.find("\n", m.end)
            return len(text) if nl < 0 else nl
    return 0


def _in_heading(pos: int | None, ctx: _DocContext) -> bool:
    return pos is not None and pos < ctx.heading_end


def _own_date_heading(a: _Anchor, ctx: _DocContext) -> bool:
    """A draft in the document's heading dated with the document's own date (the interview
    being held, the letter being written) in a document whose account is of an incident on
    another day: it never anchors that account's times or "that day"."""
    return (
        ctx.incident is not None
        and ctx.doc_date is not None
        and ctx.doc_date != ctx.incident
        and a.span[0] == ctx.doc_date
        and _in_heading(a.pos, ctx)
    )


def _covering(pos: int, anchors: list[_Anchor]) -> str | None:
    """The draft whose quote contains text position `pos` (the draft stating that date)."""
    for a in anchors:
        if a.pos is not None and a.end is not None and a.pos <= pos < a.end:
            return a.key
    return None


# --------------------------------------------------------------------- resolution


def resolve_drafts(
    drafts: list[DraftFact],
    doc_date: date | None,
    existing_facts: list[Fact] | None = None,
    *,
    parsed: ParsedDocument | None = None,
    incident_date: date | None = None,
    case_incident_date: date | None = None,
    tz: str | tzinfo | None = None,
) -> list[DraftFact]:
    """Fill date_start/date_end/date_certainty (and anchors for relative dates).

    ``drafts`` all come from one document and are walked in text order (first span).
    Absolute dates are resolved first; a month/day without a year takes the year nearest the
    most recent prior dated draft (else the incident date, else ``doc_date``). Relative ones
    then anchor to the draft or existing fact they name, else the most recent prior dated
    draft, else the incident date, else ``doc_date`` ("yesterday" and "3 days ago" prefer
    ``doc_date``). Open-ended bounds ("before X", "since 2006", "by May 6") are never anchors
    or year context. Drafts that already carry a date_start are left untouched. Input order
    is preserved.

    ``parsed`` (the drafts' document) supplies the text, so that a bare time of day
    ("0942:15", "at 1405") takes the day stated in its sentence ("exact"), else the most
    recent earlier day in the narrative ("approximate"), and the incident date
    (``parse.incident_date``) that "at the time of the accident" and otherwise unanchored
    accounts refer to; ``incident_date`` overrides the detected one.

    ``case_incident_date`` is the incident date of the whole case (the consensus of its
    documents). A document about an incident that never states its date itself (an interview
    held weeks later) resolves its account ("that day", bare times) against it, while drafts
    about the document itself (the interview being held, in its heading) keep the document's
    own date.

    ``tz`` is the case time zone (IANA name): a day stated with an explicit UTC/GMT/Zulu time
    is converted to it ("July 28, 2010, 0016 UTC" is July 27 in America/Chicago). Local zones
    keep their own date, and times without a zone are never shifted.
    """
    if not drafts:
        return []
    zone = case_zone(tz)
    ctx = _DocContext(doc_date=doc_date, tz=zone)
    if parsed is not None:
        if incident_date is None:
            from pipeline import parse  # local import: parse pulls in optional deps lazily

            incident_date = parse.incident_date(parsed)
        ctx.text = parsed.text
        ctx.mentions = _text_mentions(parsed.text, zone)
        ctx.heading_end = _heading_end(parsed.text, doc_date, ctx.mentions)
    ctx.incident = incident_date
    if ctx.incident is None and case_incident_date is not None:
        hay = ctx.text or " ".join(_draft_text(d) for d in drafts)
        if _INCIDENT_WORD.search(hay):
            ctx.incident = case_incident_date

    out: list[DraftFact] = list(drafts)
    order = sorted(range(len(out)), key=lambda i: (_pos(out[i], i), i))
    pos_of = {i: _pos(out[i], i) for i in order}
    facts = [
        _Anchor(
            span=(f.date_start, f.date_end or f.date_start),
            certainty=f.date_certainty,
            text=_fact_text(f),
            fact_id=f.fact_id,
            date_text=f.date_text or "",
        )
        for f in (existing_facts or [])
        if f.date_start is not None and f.review != "rejected" and not is_open_bound(f.date_text)
    ]
    pending: list[tuple[int, RelativeSpec]] = []
    last_dated: date | None = None

    for i in order:
        d = out[i]
        if d.date_start is not None:
            if not is_open_bound(d.date_text):
                last_dated = d.date_start
            continue
        text = (d.date_text or "").strip()
        undated = {"date_start": None, "date_end": None, "date_certainty": "unknown"}
        if not text:
            out[i] = d.model_copy(update=undated)
            continue
        absolute = (
            None
            if is_time_only(text)
            else parse_absolute(text, last_dated or ctx.incident or doc_date, zone)
        )
        if absolute is not None:
            (lo, hi), cert = absolute
            out[i] = d.model_copy(update={"date_start": lo, "date_end": hi, "date_certainty": cert})
            if not is_open_bound(text):
                last_dated = lo  # year context for a later "March 3" / "mid-March"
            continue
        spec = parse_relative(text)
        if spec is not None:
            pending.append((i, spec))
        else:
            out[i] = d.model_copy(update=undated)

    if ctx.incident is not None:
        stating = [
            j
            for j in order
            if out[j].date_start == ctx.incident
            and (out[j].date_end in (None, ctx.incident))
            and out[j].date_certainty in ("exact", "approximate")
            and not _open_ended(out[j].date_text)
        ]
        ctx.incident_key = out[stating[0]].key if stating else None

    bound = {i: is_open_bound(out[i].date_text) for i in order}
    ctx.waiting = {i for i, _spec in pending}
    for final in (False, True):
        still: list[tuple[int, RelativeSpec]] = []
        for i, spec in pending:
            candidates = [
                _draft_anchor(out[j], pos_of[j])
                for j in order
                if j != i and out[j].date_start is not None and not bound[j]
            ]
            resolved = _resolve_relative(
                out[i], i, pos_of, order, spec, candidates, facts, ctx, final
            )
            if resolved is None:
                still.append((i, spec))
            else:
                out[i] = resolved
                ctx.waiting.discard(i)
        pending = still
    for i, _spec in pending:
        out[i] = out[i].model_copy(
            update={"date_start": None, "date_end": None, "date_certainty": "relative"}
        )
    return out


def _waits_for_prior(
    i: int, pool: list[_Anchor], order: list[int], pos_of: dict[int, int], ctx: _DocContext
) -> bool:
    """The nearest earlier draft that could anchor draft `i` (any of `pool`, or an earlier
    relative draft still being resolved) is unresolved: wait for it, so "after that" follows
    the draft just before it rather than an older one."""
    anchors = {a.pos for a in pool}
    for j in reversed([j for j in order if pos_of[j] < pos_of[i]]):
        if j in ctx.waiting:
            return True
        if pos_of[j] in anchors:
            return False
    return False


def _resolve_clock(
    d: DraftFact,
    i: int,
    pos_of: dict[int, int],
    order: list[int],
    drafts: list[_Anchor],
    ctx: _DocContext,
    final: bool,
) -> tuple[Span, Certainty, str | None] | None:
    """The day of a bare time of day ("0942:15", "At 1405", "2130 eastern daylight time").

    In order: a date stated in the same sentence ("exact"); the most recent earlier day that
    was itself stated with a time of day (a draft, or an explicit date in the text); the most
    recent earlier single day; the incident date; the document date. Inherited days are
    "approximate". A time never continues a month- or year-wide span ("since 2006").
    """
    first = _first_span(d)
    start = first.char_start if first else None
    approx_time = bool(_APPROX_TIME.search(_lower(d.date_text or "")))

    if first is not None and ctx.mentions:
        lo, hi = _sentence_bounds(ctx.text, first.char_start, first.char_end)
        same = [m for m in ctx.mentions if lo <= m.pos < hi]
        if same:
            m = min(same, key=lambda m: abs(m.pos - first.char_start))
            day = m.day
            if m.day == m.stated and not _LOCAL_ZONE.search(_lower(ctx.text[lo:hi])):
                day = _utc_to_case_day(_lower(d.date_text or ""), m.day, ctx.tz)  # "0016 UTC"
            cert: Certainty = "approximate" if approx_time else "exact"
            return (day, day), cert, _covering(m.pos, drafts)

    # Skip the document's own date lines ("March 3, 2024" heading a letter) once the document
    # names the incident: a witness's account happened then, not when it was written.
    skip_datelines = ctx.incident is not None
    narrative = not _in_heading(start, ctx)
    pos = pos_of[i]
    earlier = [
        a
        for a in drafts
        if a.pos is not None
        and a.pos < pos
        and a.single_day
        and not (skip_datelines and a.dateline)
        and not (narrative and _own_date_heading(a, ctx))
    ]
    mentions = [
        m
        for m in ctx.mentions
        if start is not None
        and m.pos < start
        and not (skip_datelines and (m.dateline or m.day == ctx.doc_date != ctx.incident))
    ]
    if not final and _waits_for_prior(i, earlier, order, pos_of, ctx):
        return None
    timed = [a for a in earlier if a.timed]
    for pool, m_pool in ((timed, [m for m in mentions if m.timed]), (earlier, mentions)):
        best = max(pool, key=lambda a: a.pos or 0) if pool else None
        best_m = max(m_pool, key=lambda m: m.pos) if m_pool else None
        if best_m is not None and (best is None or best_m.pos > (best.pos or 0)):
            return (best_m.day, best_m.day), "approximate", _covering(best_m.pos, drafts)
        if best is not None:
            return best.span, "approximate", best.key
    if not final:
        return None
    if ctx.incident is not None:
        return (ctx.incident, ctx.incident), "approximate", ctx.incident_key
    if ctx.doc_date is not None:
        return (ctx.doc_date, ctx.doc_date), "approximate", None
    return None


def _resolve_relative(
    d: DraftFact,
    i: int,
    pos_of: dict[int, int],
    order: list[int],
    spec: RelativeSpec,
    drafts: list[_Anchor],
    facts: list[_Anchor],
    ctx: _DocContext,
    final: bool,
) -> DraftFact | None:
    pos = pos_of[i]
    doc_date = ctx.doc_date
    if spec.clock:
        clock = _resolve_clock(d, i, pos_of, order, drafts, ctx, final)
        if clock is None:
            return None
        (lo, hi), cert, key = clock
        return d.model_copy(
            update={
                "date_start": lo,
                "date_end": hi,
                "date_certainty": cert,
                "anchor_key": key,
                "anchor_fact_id": None,
            }
        )

    first = _first_span(d)
    narrative = not _in_heading(first.char_start if first else None, ctx)
    # What an unnamed anchor ("that day", "after that") can refer back to: not the document's
    # own heading date when the account is of an incident on another day.
    context = [a for a in drafts if not (narrative and _own_date_heading(a, ctx))]
    anchor_span: Span | None = None
    anchor: _Anchor | None = None
    incident_anchor = False

    if spec.phrase and ctx.incident is not None and _INCIDENT_REF.match(_clean(spec.phrase)):
        # "at the time of the accident", "after the crash": the occurrence the document is about.
        anchor_span, incident_anchor = (ctx.incident, ctx.incident), True
    elif spec.phrase:
        prior = _prior(pos, context)
        ref = (prior.span[0] if prior else None) or ctx.incident or doc_date
        absolute = parse_absolute(spec.phrase, ref, ctx.tz)
        if absolute is not None:
            anchor_span = absolute[0]
        else:
            anchor = _match_anchor(spec.phrase, pos, drafts + facts)
            if anchor is None and final:
                anchor = _prior(pos, context)
            if anchor is None and final and ctx.incident is not None:
                anchor_span, incident_anchor = (ctx.incident, ctx.incident), True
    elif spec.deictic and doc_date is not None:
        anchor_span = (doc_date, doc_date)
    else:
        # "after that" never refers back to background ("since 2006"); "that day" never to a
        # month or a year either.
        context = [a for a in context if not _open_ended(a.date_text)]
        if spec.day:
            context = [a for a in context if a.single_day]
        if not final and _waits_for_prior(i, context, order, pos_of, ctx):
            return None
        anchor = _prior(pos, context)
        if anchor is None and final and ctx.incident is not None:
            # An undated account in a document about an incident: not the authoring date.
            anchor_span, incident_anchor = (ctx.incident, ctx.incident), True
        elif anchor is None and doc_date is not None and final:
            anchor_span = (doc_date, doc_date)

    if anchor is not None:
        anchor_span = anchor.span
    if anchor_span is None:
        return None
    from_event = anchor is not None or (incident_anchor and ctx.incident_key is not None)
    try:
        lo, hi = spec.apply(anchor_span, from_event)
    except (ValueError, OverflowError) as exc:
        logger.debug("could not apply relative date %r: %s", d.date_text, exc)
        return None
    return d.model_copy(
        update={
            "date_start": lo,
            "date_end": hi,
            "date_certainty": "relative",
            "anchor_key": anchor.key if anchor else (ctx.incident_key if incident_anchor else None),
            "anchor_fact_id": anchor.fact_id if anchor else None,
        }
    )
