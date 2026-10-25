"""Resolve free-text dates on events into date ranges with an honest certainty level.

Precision is never invented: a month-only date becomes the whole month (APPROXIMATE), a
season becomes the season's months, a year-only date the whole year, and an unparseable date
stays UNKNOWN with no dates. A time stated explicitly in UTC ("2316 UTC", "0016Z") is
converted to the case's time zone (``tz``) before its calendar day is taken.
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

from pipeline.contracts import Certainty, Event, ParsedDocument

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


def _open_ended(cleaned: str) -> bool:
    """`cleaned` is an open bound ("since 2006", "before March 3, 2024"), not a period."""
    return bool(_OPEN_BOUND.match(cleaned) or _START_BOUND.match(cleaned))


# ----------------------------------------------------------------------------- helpers


def _clean(text: str) -> str:
    s = text.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s*[–—]\s*", " - ", s)
    s = re.sub(r"\(\s*\d+\s*\)", " ", s)  # "thirty (30) days"
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"(\d{1,2}/\d{1,2}/\d{4})(?=\d{1,2}:\d{2})", r"\1 ", s)  # OCR "07/02/202118:17"
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,;:\"'")


_TZ = r"(?:\s*(?:[+-]\d{2}:?\d{2}|z\b|(?:utc|gmt|[ecmp][sd]t)\b(?:\s*[+-]\d{1,2}(?::?\d{2})?)?))?"
_TIME_QUAL = (
    r"(?P<q>(?:approximately|approx\.?|about|around|roughly|circa|ca\.|shortly before"
    r"|shortly after|just before|just after)\s+)?"
)
# Spelled-out zones ("central daylight time") and "local"/"lcl", as in incident reports.
_TZ_WORDS = (
    r"(?:(?:eastern|central|mountain|pacific|alaska|hawaii|atlantic)\s+(?:standard|daylight)"
    r"(?:\s+(?:saving\s+)?time)?|local(?:\s+time)?|lcl|[ecmp][sd]t|utc|gmt|zulu|z)\b"
)
_HHMM = r"(?:[01]\d|2[0-3])[0-5]\d"  # 24-hour clock without a colon ("1405")
_CLOCK = (
    rf"(?:{_HHMM}:[0-5]\d(?:\.\d+)?{_TZ}"  # "0942:15" (HHMM:SS, recorder transcripts)
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
# ("about 1405"). "About 2005" stays a year.
_BARE_HHMM = re.compile(
    rf"(?:,\s*|\s+|^)(?:(?P<at>at\s+(?P<aq>(?:approximately|approx\.?|about|around)\s+)?)"
    rf"|(?P<q>(?:approximately|approx\.?|about|around)\s+))(?P<hhmm>{_HHMM})\b"
    rf"(?!\s*(?:[-/.,]\s*)?\d|\s+(?:{_MONTH}))"
)


def _sub_times(s: str) -> tuple[str, bool]:
    """Replace every time of day in `s` with a space; also say if one was approximate."""
    approx = False

    def drop(m: re.Match[str]) -> str:
        nonlocal approx
        approx = approx or bool(m.group("q"))
        return " "

    def drop_bare(m: re.Match[str]) -> str:
        nonlocal approx
        if not m.group("at") and int(m.group("hhmm")) >= 1900:
            return m.group(0)
        approx = approx or bool(m.group("q") or m.group("aq"))
        return " "

    out = _TIME_OF_DAY.sub(drop, s)
    return _BARE_HHMM.sub(drop_bare, out), approx


_TIME_JOINERS = re.compile(r"\b(?:or|to|and|until|till|through|between|from)\b")


def _split_zones(s: str) -> str:
    """Separate one time given in two zones ("2316 utc/1816 cdt") so both read as times."""
    return re.sub(r"(?<=[a-z])/(?=\d)", " / ", s)


def _is_time_only(s: str) -> bool:
    """`s` (already cleaned) states only a time of day, no date ("at 1405:12", "6:40 pm",
    "about 2:30 or 3:00", "2316:19 utc/1816:19 cdt")."""
    s = _split_zones(s)
    rest, _ = _sub_times(s)
    if rest.strip(" ,;:-/") and rest != s:  # "2:30 or 3:00", "between 3 pm and 5 pm"
        rest = _TIME_JOINERS.sub(" ", rest)
    return bool(s) and not rest.strip(" ,;:-/") and s.strip() != ""


def _has_time(s: str) -> bool:
    return _sub_times(s)[0] != s


_DAY_PART = r"(?:early\s+|late\s+|mid-?)?(?:morning|afternoon|evening|night|day)"
_DAY_PART_PREFIX = re.compile(rf"^(?:in\s+|during\s+|on\s+)?the\s+{_DAY_PART}\s+of\s+")
_DAY_PART_SUFFIX = re.compile(rf",?\s+(?:in|during)\s+the\s+{_DAY_PART}$")


def _strip_time(s: str) -> tuple[str, bool]:
    """Drop clock times, time zones and day parts ("at approximately 5:40 pm", "-0400").

    Returns the remaining text and whether a removed time was qualified as approximate
    ("around 3pm"), which keeps the date APPROXIMATE rather than EXACT.
    """
    s = re.sub(r"(\d)t(?=\d{1,2}:\d{2})", r"\1 ", s)  # ISO 8601 "2021-06-05T09:30"
    out, approx = _sub_times(_split_zones(s))
    if re.search(r"\d", out):  # never reduce "this morning" to "this"
        out = _DAY_PART_PREFIX.sub("", out.strip())
        out = _DAY_PART_SUFFIX.sub("", out)
    out = re.sub(r"\s+", " ", out).strip(" ,;:-/")
    out = re.sub(r"\s+(?:on|at|of)$", "", out)
    return (out, approx) if out else (s, False)


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


def _utc_to_case_day(text: str, day: date, tz: tzinfo | None) -> date:
    """`day` as a calendar day in the case zone `tz`, when `text` (cleaned) gives the time
    of day explicitly in UTC/GMT/Zulu ("July 28, 2010, 0016 UTC" is July 27 in Chicago).

    A local zone ("1816 cdt", "central daylight time", "local", "-0500") keeps its own date,
    and so does text that also names one; a time with no zone is never shifted.
    """
    if tz is None:
        return day
    m = _UTC_CLOCK.search(text)
    if m is None or _LOCAL_ZONE.search(text):
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
    cleaned = _clean(text)
    if _is_time_only(cleaned):  # "at 1729" is a time of day, never the year 1729
        return None
    s, approx_time = _strip_time(cleaned)
    if not s:
        return None
    try:
        found = _parse_range(s, ref) or _parse_point(s, ref)
    except (ValueError, OverflowError):  # e.g. year 0000, Feb 30
        return None
    if found is None and "," in s:
        # "March 3, 2024, after the meeting": keep the longest leading part that parses.
        parts = s.split(",")
        for k in range(len(parts) - 1, 0, -1):
            head = ",".join(parts[:k]).strip()
            if head and re.search(r"\d|" + _MONTH, head):
                found = parse_absolute(head, ref)
                if found is not None:
                    break
    if found is not None and approx_time and found[1] == Certainty.EXACT:
        found = (found[0], Certainty.APPROXIMATE)
    zone = case_zone(tz)
    if found is not None and zone is not None and found[0][0] == found[0][1]:
        day = _utc_to_case_day(cleaned, found[0][0], zone)
        found = ((day, day), found[1])
    return found


def _parse_point(s: str, ref: date | None) -> tuple[Span, Certainty] | None:
    for prefix in _APPROX_PREFIXES:
        if s.startswith(prefix + " "):
            inner = parse_absolute(s[len(prefix) + 1 :], ref)
            if inner is None:
                return None
            span, cert = inner
            return span, Certainty.RANGE if cert == Certainty.RANGE else Certainty.APPROXIMATE

    m = _START_BOUND.match(s)
    if m:
        inner = parse_absolute(m.group(2), ref)
        if inner is None:
            return None
        (lo, hi), cert = inner
        if lo != hi:  # "since 2006" began at some point in 2006, not on January 1
            return (lo, hi), Certainty.APPROXIMATE if cert == Certainty.EXACT else cert
        if m.group(1) != "since":  # "from March 3, 2024": that day
            return inner

    m = _OPEN_BOUND.match(s)
    if m:
        inner = parse_absolute(m.group(2), ref)
        if inner is None:
            return None
        (lo, hi), _ = inner
        bound = lo if m.group(1) in ("before", "prior to", "preceding", "since") else hi
        return (bound, bound), Certainty.APPROXIMATE

    for prefix in _PLAIN_PREFIXES:
        if s.startswith(prefix + " "):
            return parse_absolute(s[len(prefix) + 1 :], ref)

    return _parse_partial(s, ref) or _parse_full(s, ref) or _parse_month_day(s, ref)


def _parse_partial(s: str, ref: date | None) -> tuple[Span, Certainty] | None:
    """Month-only, year-only, seasons, quarters, early/mid/late, week of."""
    approx = Certainty.APPROXIMATE

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
    return _partial_without_year(s, ref)


def _partial_without_year(s: str, ref: date | None) -> tuple[Span, Certainty] | None:
    """'mid-June' / 'early June' / 'June' with no year: carry the year over from `ref`.

    The year is the one that puts the span nearest to `ref` (the previous dated event in
    the document, else the document date); the result is always APPROXIMATE.
    """
    if ref is None or _has_year(s) or re.search(r"\d", s):
        return None
    part = r"(?:(?:early|mid|middle|late|beginning|start|end|close)(?:\s*-\s*|\s+of\s+|\s+))"
    m = re.match(rf"^(?:the\s+)?(?:month\s+of\s+)?{part}?(?:the\s+month\s+of\s+)?({_MONTH})$", s)
    if not m:
        return None
    best: tuple[int, tuple[Span, Certainty]] | None = None
    for y in (ref.year - 1, ref.year, ref.year + 1):
        found = _parse_partial(f"{s} {y}", None)
        if found is None:
            continue
        lo, hi = found[0]
        dist = 0 if lo <= ref <= hi else min(abs((lo - ref).days), abs((hi - ref).days))
        if best is None or dist < best[0]:
            best = (dist, (found[0], Certainty.APPROXIMATE))
    return best[1] if best else None


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
    return (d, d), Certainty.EXACT


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
        return None
    d = _nearest_year(_month_num(m.group("mon")), int(m.group("d")), ref)
    return ((d, d), Certainty.APPROXIMATE) if d else None


def _parse_range(s: str, ref: date | None) -> tuple[Span, Certainty] | None:
    """'between X and Y', 'from X to Y', 'X - Y', 'March 3-5, 2024', '2023-24'."""
    m = re.match(r"^(\d{4})\s*-\s*(\d{2}|\d{4})$", s)
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2)) if len(m.group(2)) == 4 else (y1 // 100) * 100 + int(m.group(2))
        if y2 > y1:
            return (date(y1, 1, 1), date(y2, 12, 31)), Certainty.RANGE
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
    (lo, _), _ = lp
    (_, hi), _ = rp
    if lo > hi and borrowed:
        lo = lo - relativedelta(years=1)
    if lo > hi:
        lo, hi = rp[0][0], lp[0][1]
    return (lo, hi), Certainty.RANGE


# ----------------------------------------------------------------------- relative dates


@dataclass(frozen=True)
class RelativeSpec:
    """A relative date: `apply(anchor_span, anchor_is_event)` gives the resolved span."""

    apply: Callable[[Span, bool], Span]
    deictic: bool = False  # relative to the document's own date ("yesterday", "3 days ago")
    phrase: str | None = None  # explicit anchor ("the meeting", "March 3, 2024")
    clock: bool = False  # a bare time of day: continues the last event that had a time
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
    which refers back to the previous event rather than naming one."""
    if rest is None or _PRONOUN_ANCHOR.match(rest.strip()):
        return None
    return rest


def parse_relative(text: str) -> RelativeSpec | None:
    """Recognize a relative date expression; None if the text is not relative."""
    s = _clean(text)
    if _is_time_only(s):
        return RelativeSpec(lambda a, _e: a, clock=True)  # "at about 3:40 pm": same day
    s = _strip_time(s)[0]
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

    m = re.match(
        r"^(?:(?:later|earlier)\s+(?:that|the same|that same|in the|in that|this)"
        r"|(?:that|the same|that same))\s+(?P<unit>week|weekend|month|year)$",
        s,
    )
    if m:
        period_unit = m.group("unit")
        return RelativeSpec(lambda a, _e: _period(a, period_unit, 0))

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


def _match_anchor(phrase: str, pos: int, candidates: list[tuple[int, Event]]) -> Event | None:
    """Best event whose description/quote mentions the anchor phrase ("the meeting")."""
    key = _ARTICLES.sub("", _clean(phrase)).strip()
    if len(key) < 3:
        return None
    best: tuple[tuple[float, bool, bool, int], Event] | None = None
    for cpos, ev in candidates:
        hay = f"{ev.description} {ev.source.quote}".lower()
        score = fuzz.partial_ratio(key, hay)
        if score < 85:
            continue
        rank = (score, ev.certainty != Certainty.RELATIVE, cpos < pos, -abs(cpos - pos))
        if best is None or rank > best[0]:
            best = (rank, ev)
    return best[1] if best else None


def _waits_for_prior(
    i: int, pool: list[int], order: list[int], pos_of: dict[int, int], ctx: _DocContext
) -> bool:
    """The nearest earlier event that could anchor event `i` (any of `pool`, or an earlier
    relative event still being resolved) is still unresolved: wait for it, so "after that"
    follows the event just before it rather than an older one."""
    before = [j for j in order if pos_of[j] < pos_of[i]]
    anchors = set(pool)
    for j in reversed(before):
        if j in ctx.waiting:
            return True
        if j in anchors:
            return False
    return False


def _prior(pos: int, candidates: list[tuple[int, Event]]) -> Event | None:
    prior = [(cpos, ev) for cpos, ev in candidates if cpos < pos]
    return max(prior, key=lambda x: x[0])[1] if prior else None


def _single_day(ev: Event) -> bool:
    """The event is pinned to one day: not a span, and not an open bound ("since 2006")."""
    if ev.date_start is None or (ev.date_end is not None and ev.date_end != ev.date_start):
        return False
    return not _open_ended(_clean(ev.date_text))


def _span_of(ev: Event) -> Span:
    start = ev.date_start or date.min
    return start, ev.date_end or start


# ------------------------------------------------------------------ document context

# The occurrence a document is about ("at the time of the accident", "after the crash").
_INCIDENT_REF = re.compile(
    r"^(?:the|this|that|said)\s+(?:[a-z-]+\s+)?"
    r"(?:accident|incident|crash|collision|impact|mishap|occurrence|wreck)$"
)
# A full explicit date written in the document text (the day is stated).
_TEXT_DATE = re.compile(
    rf"(?:\b{_WEEKDAY},?\s+)?(?:\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{{4}}"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH},?\s+\d{{4}}"
    r"|\b\d{4}-\d{2}-\d{2}|\b\d{1,2}/\d{1,2}/(?:\d{4}|\d{2}))\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*(?=\s)|\n\s*\n")
_CONTEXT_CHARS = 60  # how far around a date to look for a time of day in the same sentence


@dataclass(frozen=True)
class _Mention:
    """An explicit full date in the document text."""

    pos: int
    end: int
    day: date
    timed: bool  # a time of day is stated with it ("June 5, 2021, about 0940 cdt")
    dateline: bool  # the date stands alone on its line (a letter or section date)
    stated: date | None = None  # the day as written, before a UTC time moved it to the case zone


@dataclass
class _DocContext:
    doc_date: date | None
    incident: date | None = None
    incident_event_id: str | None = None
    text: str = ""
    mentions: list[_Mention] = field(default_factory=list)
    tz: tzinfo | None = None
    heading_end: int = 0  # end of the line stating the document's own date near the top
    waiting: set[int] = field(default_factory=set)  # relative events not yet resolved


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """The sentence(s) around text[start:end]; without a visible boundary, just that span."""
    lo = start
    for m in _SENTENCE_END.finditer(text, max(0, start - 400), start):
        lo = m.end()
    m = _SENTENCE_END.search(text, max(end - 1, start), end + 400)
    return min(lo, start), (m.start() if m else end)


def _text_mentions(text: str, tz: tzinfo | None = None) -> list[_Mention]:
    out: list[_Mention] = []
    for m in _TEXT_DATE.finditer(text):
        found = parse_absolute(m.group(0))
        if found is None or found[0][0] != found[0][1]:
            continue
        lo, hi = _sentence_bounds(text, m.start(), m.end())
        around = _clean(
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
        out.append(_Mention(m.start(), m.end(), day, _has_time(around), dateline, stated))
    return out


_HEADING_LINES = 25  # how far down a document's own date (its heading) is looked for


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
        if m.stated == doc_date or m.day == doc_date:
            nl = text.find("\n", m.end)
            return len(text) if nl < 0 else nl
    return 0


def _in_heading(ev: Event, ctx: _DocContext) -> bool:
    return _known_pos(ev) and ev.source.char_start < ctx.heading_end


def _own_date_heading(ev: Event, ctx: _DocContext) -> bool:
    """An event in the document's heading dated with the document's own date (the interview
    being held, the letter being written) in a document whose account is of an incident on
    another day: it never anchors that account's times or "that day"."""
    return (
        ctx.incident is not None
        and ctx.doc_date is not None
        and ctx.doc_date != ctx.incident
        and ev.date_start == ctx.doc_date
        and _in_heading(ev, ctx)
    )


_INCIDENT_WORD = re.compile(r"\b(?:accident|incident|crash|collision|mishap|wreck)\b", re.I)


def _about_an_incident(text: str, events: list[Event]) -> bool:
    """The document refers to an incident at all (so the case's incident date can apply)."""
    hay = text or " ".join(f"{e.description} {e.source.quote}" for e in events)
    return bool(_INCIDENT_WORD.search(hay))


_APPROX_BEFORE = re.compile(r"\b(?:about|around|approximately|approx|circa|roughly)\W*$", re.I)


def _refine_from_quote(
    ev: Event, found: tuple[Span, Certainty], tz: tzinfo | None
) -> tuple[Span, Certainty]:
    """A year- or month-wide date ("2010", "in July 2010") whose own quote states one full
    date inside that period ("On July 27, 2010, ...") takes that day: a coarse date_text never
    overrides the better date the source gives. Open bounds ("since 2006") are kept."""
    (lo, hi), cert = found
    if lo == hi or cert != Certainty.APPROXIMATE or _open_ended(_clean(ev.date_text)):
        return found
    if _TEXT_DATE.search(ev.date_text):  # "the week of March 3, 2024" names its own day
        return found
    quote = ev.source.quote
    inside = [m for m in _text_mentions(quote, tz) if lo <= m.day <= hi]
    if len({m.day for m in inside}) != 1:
        return found
    m = inside[0]
    approx = _APPROX_BEFORE.search(quote[max(0, m.pos - 25) : m.pos])
    return (m.day, m.day), Certainty.APPROXIMATE if approx else Certainty.EXACT


def _known_pos(ev: Event) -> bool:
    """The quote was located in the text (unverified events carry no real offsets)."""
    return ev.source.char_end > ev.source.char_start


def _covering_event(pos: int, out: list[Event], order: list[int]) -> str | None:
    """The event whose quote contains text position `pos` (the event stating that date)."""
    for j in order:
        src = out[j].source
        if _known_pos(out[j]) and src.char_start <= pos < src.char_end and out[j].date_start:
            return out[j].id
    return None


def _is_dateline_event(ev: Event) -> bool:
    """An event whose quote is just a date (a letter's or section's own date line)."""
    return not re.sub(r"[\W_]+", "", _clean(ev.source.quote).replace(_clean(ev.date_text), ""))


# --------------------------------------------------------------------- resolution


def resolve_dates(
    events: list[Event],
    doc_date: date | None,
    *,
    parsed: ParsedDocument | None = None,
    incident_date: date | None = None,
    case_incident_date: date | None = None,
    tz: str | tzinfo | None = None,
) -> list[Event]:
    """Fill date_start/date_end/certainty (and anchor_event_id for anchored dates).

    Events are grouped per source document and walked in text order. Absolute dates are
    resolved first; relative ones then anchor to the event they name, else the most recent
    prior dated event in the same document, else the incident date, else `doc_date`
    ("yesterday", "3 days ago" prefer `doc_date`). Events that already carry a date_start
    are left untouched.

    `parsed` (the source document of the events) supplies the document text, so that a bare
    time of day ("0942:15", "at 1405") takes the day of the most recent explicit date
    before it in the narrative, and the incident date (``parse.incident_date``) that
    "at the time of the accident" and otherwise unanchored times refer to. `incident_date`
    overrides the detected one. Days inherited this way are APPROXIMATE; a day stated in the
    same sentence as the time is EXACT.

    `case_incident_date` is the incident date of the whole case (the consensus of its
    documents). A document about an incident that never states its date itself (an interview
    held weeks later) resolves its account ("that day", "after the show", bare times)
    against it, while events about the document itself (the interview being held, at the
    top of the document) keep the document's own date.

    `tz` is the case time zone (IANA name): a day stated with an explicit UTC/GMT/Zulu time
    is converted to it ("July 28, 2010, 0016 UTC" is July 27 in America/Chicago). Local
    zones keep their own date, and times without a zone are never shifted.
    """
    if not events:
        return []
    zone = case_zone(tz)
    if parsed is not None and incident_date is None:
        from pipeline import parse  # local import: parse pulls in heavy optional deps lazily

        incident_date = parse.incident_date(parsed)
    out: list[Event] = list(events)
    by_doc: dict[str, list[int]] = {}
    for i, ev in enumerate(out):
        by_doc.setdefault(ev.source.doc_id, []).append(i)

    for doc_id, idxs in by_doc.items():
        ctx = _DocContext(doc_date=doc_date, tz=zone)
        if parsed is None or parsed.doc.id == doc_id:
            ctx.incident = incident_date
        if parsed is not None and parsed.doc.id == doc_id:
            ctx.text = parsed.text
            ctx.mentions = _text_mentions(parsed.text, zone)
            ctx.heading_end = _heading_end(parsed.text, doc_date, ctx.mentions)
        if (
            ctx.incident is None
            and case_incident_date is not None
            and _about_an_incident(ctx.text, [out[i] for i in idxs])
        ):
            ctx.incident = case_incident_date
        order = sorted(idxs, key=lambda i: (out[i].source.char_start, i))
        pos_of = {i: p for p, i in enumerate(order)}
        pending: list[tuple[int, RelativeSpec]] = []
        last_dated: date | None = None

        for i in order:
            ev = out[i]
            if ev.date_start is not None:
                last_dated = ev.date_start
                continue
            ref = last_dated or ctx.incident or doc_date
            absolute = parse_absolute(ev.date_text, ref, zone)
            if absolute is not None:
                (lo, hi), cert = _refine_from_quote(ev, absolute, zone)
                out[i] = ev.model_copy(update={"date_start": lo, "date_end": hi, "certainty": cert})
                if not _open_ended(_clean(ev.date_text)):  # "since 2006" is background
                    last_dated = lo
                continue
            spec = parse_relative(ev.date_text)
            if spec is not None:
                pending.append((i, spec))
            else:
                out[i] = ev.model_copy(
                    update={"date_start": None, "date_end": None, "certainty": Certainty.UNKNOWN}
                )

        if ctx.incident is not None:
            stating = [
                j
                for j in order
                if out[j].date_start == ctx.incident
                and _single_day(out[j])
                and out[j].certainty in (Certainty.EXACT, Certainty.APPROXIMATE)
            ]
            ctx.incident_event_id = out[stating[0]].id if stating else None

        ctx.waiting = {i for i, _spec in pending}
        for final in (False, True):
            still: list[tuple[int, RelativeSpec]] = []
            for i, spec in pending:
                resolved = _resolve_relative(out, i, spec, order, pos_of, ctx, final)
                if resolved is None:
                    still.append((i, spec))
                else:
                    out[i] = resolved
                    ctx.waiting.discard(i)
            pending = still
        for i, _spec in pending:
            out[i] = out[i].model_copy(
                update={"date_start": None, "date_end": None, "certainty": Certainty.RELATIVE}
            )
    return out


def _resolve_clock(
    out: list[Event],
    i: int,
    order: list[int],
    pos_of: dict[int, int],
    ctx: _DocContext,
    final: bool,
) -> tuple[Span, Certainty, str | None] | None:
    """The day of a bare time of day ("0942:15", "At 1405", "2130 eastern daylight time").

    In order: a date stated in the same sentence (EXACT); the most recent earlier day that
    was itself stated with a time of day (an event, or an explicit date in the text); the
    most recent earlier single day; the incident date; the document date. Inherited days
    are APPROXIMATE. A time never continues a month- or year-wide span ("since 2006").
    """
    ev = out[i]
    known = _known_pos(ev)
    start = ev.source.char_start
    approx_time = _sub_times(_clean(ev.date_text))[1]

    if known and ctx.mentions:
        lo, hi = _sentence_bounds(ctx.text, start, ev.source.char_end)
        same = [m for m in ctx.mentions if lo <= m.pos < hi]
        if same:
            m = min(same, key=lambda m: abs(m.pos - start))
            cert = Certainty.APPROXIMATE if approx_time else Certainty.EXACT
            day = m.day
            if m.day == m.stated and not _LOCAL_ZONE.search(_clean(ctx.text[lo:hi])):
                day = _utc_to_case_day(_clean(ev.date_text), m.day, ctx.tz)  # "at 0016 UTC"
            return (day, day), cert, _covering_event(m.pos, out, order)

    # Skip the document's own date lines ("March 3, 2024" heading a letter) once the
    # document names the incident: a witness's account happened then, not when written.
    skip_datelines = ctx.incident is not None
    narrative = not _in_heading(ev, ctx)

    def offset(j: int) -> int:
        return out[j].source.char_start

    events = [
        j
        for j in order
        if j != i
        and out[j].date_start is not None
        and pos_of[j] < pos_of[i]
        and _single_day(out[j])
        and not (skip_datelines and _is_dateline_event(out[j]))
        and not (narrative and _own_date_heading(out[j], ctx))
    ]
    timed_ev = [
        j
        for j in events
        if _has_time(_clean(out[j].date_text)) or _has_time(_clean(out[j].source.quote))
    ]
    mentions = [
        m
        for m in ctx.mentions
        if known
        and m.pos < start
        and not (skip_datelines and (m.dateline or m.day == ctx.doc_date != ctx.incident))
    ]

    if not final and _waits_for_prior(i, events, order, pos_of, ctx):
        return None
    for ev_pool, m_pool in ((timed_ev, [m for m in mentions if m.timed]), (events, mentions)):
        best_ev = max(ev_pool, key=lambda j: (offset(j), pos_of[j])) if ev_pool else None
        best_m = max(m_pool, key=lambda m: m.pos) if m_pool else None
        if best_m is not None and (best_ev is None or best_m.pos > offset(best_ev)):
            anchor = _covering_event(best_m.pos, out, order)
            return (best_m.day, best_m.day), Certainty.APPROXIMATE, anchor
        if best_ev is not None:
            return _span_of(out[best_ev]), Certainty.APPROXIMATE, out[best_ev].id
    if not final:
        return None
    if ctx.incident is not None:
        return (ctx.incident, ctx.incident), Certainty.APPROXIMATE, ctx.incident_event_id
    if ctx.doc_date is not None:
        return (ctx.doc_date, ctx.doc_date), Certainty.APPROXIMATE, None
    return None


def _resolve_relative(
    out: list[Event],
    i: int,
    spec: RelativeSpec,
    order: list[int],
    pos_of: dict[int, int],
    ctx: _DocContext,
    final: bool,
) -> Event | None:
    ev = out[i]
    pos = pos_of[i]
    doc_date = ctx.doc_date
    if spec.clock:
        clock = _resolve_clock(out, i, order, pos_of, ctx, final)
        if clock is None:
            return None
        (lo, hi), cert, anchor = clock
        return ev.model_copy(
            update={"date_start": lo, "date_end": hi, "certainty": cert, "anchor_event_id": anchor}
        )

    candidates = [(pos_of[j], out[j]) for j in order if j != i and out[j].date_start is not None]
    # What an unnamed anchor ("that day", "after that") can refer back to: not the
    # document's own heading date when the account is of an incident on another day.
    narrative = not _in_heading(ev, ctx)
    context = [(c, e) for c, e in candidates if not (narrative and _own_date_heading(e, ctx))]
    anchor_span: Span | None = None
    anchor_id: str | None = None
    from_event = False

    if spec.phrase and ctx.incident is not None and _INCIDENT_REF.match(_clean(spec.phrase)):
        # "at the time of the accident", "after the crash": the occurrence the document is about.
        anchor_span, anchor_id = (ctx.incident, ctx.incident), ctx.incident_event_id
        from_event = anchor_id is not None
    elif spec.phrase:
        prior = _prior(pos, context)
        ref = (prior.date_start if prior else None) or ctx.incident or doc_date
        absolute = parse_absolute(spec.phrase, ref, ctx.tz)
        if absolute is not None:
            anchor_span = absolute[0]
        else:
            match = _match_anchor(spec.phrase, pos, candidates)
            if match is None and final:
                match = _prior(pos, context)
            if match is not None:
                anchor_span, anchor_id, from_event = _span_of(match), match.id, True
            elif final and ctx.incident is not None:
                # "at the time of the event" with nothing before it: the incident's account.
                anchor_span, anchor_id = (ctx.incident, ctx.incident), ctx.incident_event_id
                from_event = anchor_id is not None
    elif spec.deictic and doc_date is not None:
        anchor_span = (doc_date, doc_date)
    else:
        # "after that" never refers back to background ("since 2006"); "that day" never
        # to a month or year either.
        context = [(c, e) for c, e in context if not _open_ended(_clean(e.date_text))]
        if spec.day:
            context = [(c, e) for c, e in context if _single_day(e)]
        ids = {e.id for _c, e in context}
        if not final and _waits_for_prior(
            i, [j for j in order if out[j].id in ids], order, pos_of, ctx
        ):
            return None
        prior = _prior(pos, context)
        if prior is not None:
            anchor_span, anchor_id, from_event = _span_of(prior), prior.id, True
        elif final and ctx.incident is not None:
            # An undated account in a document about an incident: not the authoring date.
            anchor_span, anchor_id = (ctx.incident, ctx.incident), ctx.incident_event_id
            from_event = anchor_id is not None
        elif doc_date is not None and final:
            anchor_span = (doc_date, doc_date)

    if anchor_span is None:
        return None
    try:
        lo, hi = spec.apply(anchor_span, from_event)
    except (ValueError, OverflowError) as exc:
        logger.debug("could not apply relative date %r: %s", ev.date_text, exc)
        return None
    return ev.model_copy(
        update={
            "date_start": lo,
            "date_end": hi,
            "certainty": Certainty.RELATIVE,
            "anchor_event_id": anchor_id,
        }
    )
