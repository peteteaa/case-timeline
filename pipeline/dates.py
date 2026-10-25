"""Resolve free-text dates on events into date ranges with an honest certainty level.

Precision is never invented: a month-only date becomes the whole month (APPROXIMATE), a
season becomes the season's months, and an unparseable date stays UNKNOWN with no dates.
"""

from __future__ import annotations

import calendar
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import dateparser
from dateutil.relativedelta import relativedelta
from rapidfuzz import fuzz

from pipeline.contracts import Certainty, Event

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


# ----------------------------------------------------------------------------- helpers


def _clean(text: str) -> str:
    s = text.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s*[–—]\s*", " - ", s)
    s = re.sub(r"\(\s*\d+\s*\)", " ", s)  # "thirty (30) days"
    s = re.sub(r"\([^)]*\)", " ", s)
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
# ("about 1815"). "About 2005" stays a year.
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


def _is_time_only(s: str) -> bool:
    """`s` (already cleaned) states only a time of day, no date ("at 1816:19", "6:15 pm")."""
    return bool(s) and not _sub_times(s)[0].strip(" ,;:-") and s.strip() != ""


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
    out, approx = _sub_times(s)
    if re.search(r"\d", out):  # never reduce "this morning" to "this"
        out = _DAY_PART_PREFIX.sub("", out.strip())
        out = _DAY_PART_SUFFIX.sub("", out)
    out = re.sub(r"\s+", " ", out).strip(" ,;:-")
    out = re.sub(r"\s+(?:on|at|of)$", "", out)
    return (out, approx) if out else (s, False)


def _last_day(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def _month_span(y: int, m: int) -> Span:
    return date(y, m, 1), date(y, m, _last_day(y, m))


def _year_span(y: int) -> Span:
    return date(y, 1, 1), date(y, 12, 31)


def _month_num(token: str) -> int:
    return _MONTHS[token.rstrip(".")]


def _has_year(s: str) -> bool:
    return bool(re.search(r"\b\d{4}\b", s) or re.search(r"\d{1,2}/\d{1,2}/\d{2}\b", s))


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


def parse_absolute(text: str, ref: date | None = None) -> tuple[Span, Certainty] | None:
    """Parse a date that needs no anchor event. `ref` only helps infer a missing year."""
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
    return found


def _parse_point(s: str, ref: date | None) -> tuple[Span, Certainty] | None:
    for prefix in _APPROX_PREFIXES:
        if s.startswith(prefix + " "):
            inner = parse_absolute(s[len(prefix) + 1 :], ref)
            if inner is None:
                return None
            span, cert = inner
            return span, Certainty.RANGE if cert == Certainty.RANGE else Certainty.APPROXIMATE

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


_DIR_BACK = {"before", "earlier", "prior to", "ago", "beforehand", "preceding", "previously"}
_DAY_UNITS = {"day", "morning", "afternoon", "evening", "night"}


def parse_relative(text: str) -> RelativeSpec | None:
    """Recognize a relative date expression; None if the text is not relative."""
    s = _clean(text)
    if _is_time_only(s):
        return RelativeSpec(lambda a, _e: a, clock=True)  # "at about 6:15 pm": same day
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
        rf"(?P<biz>business\s+|calendar\s+|working\s+|court\s+)?(?P<unit>day|week|month|year)s?"
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
        rest = m.group("rest")
        if direction in ("later", "earlier", "ago", "thereafter", "afterwards", "afterward",
                         "beforehand", "hence", "previously"):  # fmt: skip
            rest = None

        def apply(a: Span, _e: bool) -> Span:
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
        deictic = not m.group("the") and which in ("last", "this")
        return RelativeSpec(lambda a, _e, n=n, u=unit: _period(a, u, n), deictic=deictic)

    m = re.match(
        r"^(?:later|earlier) (?:that|the same|that same) (?:day|morning|afternoon|evening|night)$",
        s,
    )
    if m:
        return RelativeSpec(lambda a, _e: a)

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
        return RelativeSpec(lambda a, _e, n=n, u=unit: _period(a, u, n), phrase=m.group("rest"))

    m = re.match(
        r"^(?P<q>shortly|soon|immediately|right|just|directly|not long|sometime|some time)?\s*"
        r"(?P<dir>after|before|following|prior to|preceding|since|until|upon|at the time of"
        r"|on the day of|on the date of|the day of|the date of|during)(?:\s+(?P<rest>.+))?$",
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

        return RelativeSpec(apply_b, phrase=m.group("rest"))
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


def _prior(pos: int, candidates: list[tuple[int, Event]]) -> Event | None:
    prior = [(cpos, ev) for cpos, ev in candidates if cpos < pos]
    return max(prior, key=lambda x: x[0])[1] if prior else None


def _single_day(ev: Event) -> bool:
    """The event is pinned to one day: not a span, and not an open bound ("since 2006")."""
    if ev.date_start is None or (ev.date_end is not None and ev.date_end != ev.date_start):
        return False
    return not _OPEN_BOUND.match(_clean(ev.date_text))


def _span_of(ev: Event) -> Span:
    start = ev.date_start or date.min
    return start, ev.date_end or start


def resolve_dates(events: list[Event], doc_date: date | None) -> list[Event]:
    """Fill date_start/date_end/certainty (and anchor_event_id for relative dates).

    Events are grouped per source document and walked in text order. Absolute dates are
    resolved first; relative ones then anchor to the event they name, else the most recent
    prior dated event in the same document, else `doc_date` ("yesterday", "3 days ago"
    prefer `doc_date`). Events that already carry a date_start are left untouched.
    """
    if not events:
        return []
    out: list[Event] = list(events)
    by_doc: dict[str, list[int]] = {}
    for i, ev in enumerate(out):
        by_doc.setdefault(ev.source.doc_id, []).append(i)

    for idxs in by_doc.values():
        order = sorted(idxs, key=lambda i: (out[i].source.char_start, i))
        pos_of = {i: p for p, i in enumerate(order)}
        pending: list[tuple[int, RelativeSpec]] = []
        last_dated: date | None = None

        for i in order:
            ev = out[i]
            if ev.date_start is not None:
                last_dated = ev.date_start
                continue
            ref = last_dated or doc_date
            absolute = parse_absolute(ev.date_text, ref)
            if absolute is not None:
                (lo, hi), cert = absolute
                out[i] = ev.model_copy(update={"date_start": lo, "date_end": hi, "certainty": cert})
                last_dated = lo
                continue
            spec = parse_relative(ev.date_text)
            if spec is not None:
                pending.append((i, spec))
            else:
                out[i] = ev.model_copy(
                    update={"date_start": None, "date_end": None, "certainty": Certainty.UNKNOWN}
                )

        for final in (False, True):
            still: list[tuple[int, RelativeSpec]] = []
            for i, spec in pending:
                resolved = _resolve_relative(out, i, spec, order, pos_of, doc_date, final)
                if resolved is None:
                    still.append((i, spec))
                else:
                    out[i] = resolved
            pending = still
        for i, _spec in pending:
            out[i] = out[i].model_copy(
                update={"date_start": None, "date_end": None, "certainty": Certainty.RELATIVE}
            )
    return out


def _resolve_relative(
    out: list[Event],
    i: int,
    spec: RelativeSpec,
    order: list[int],
    pos_of: dict[int, int],
    doc_date: date | None,
    final: bool,
) -> Event | None:
    ev = out[i]
    pos = pos_of[i]
    candidates = [(pos_of[j], out[j]) for j in order if j != i and out[j].date_start is not None]
    anchor_span: Span | None = None
    anchor_id: str | None = None
    from_event = False

    if spec.phrase:
        prior = _prior(pos, candidates)
        ref = (prior.date_start if prior else None) or doc_date
        absolute = parse_absolute(spec.phrase, ref)
        if absolute is not None:
            anchor_span = absolute[0]
        else:
            match = _match_anchor(spec.phrase, pos, candidates)
            if match is None and final:
                match = _prior(pos, candidates)
            if match is not None:
                anchor_span, anchor_id, from_event = _span_of(match), match.id, True
    elif spec.deictic and doc_date is not None:
        anchor_span = (doc_date, doc_date)
    else:
        prior = _prior(pos, candidates)
        if spec.clock:
            # Narrative times ("At 6:15 pm they left", "1810:33") continue the most recent
            # event that was itself stated with a time of day (in its date text or quote), not
            # a date quoted in between ("the group met on <next day>"). A time of day only
            # continues a single day, never a month- or year-wide span ("since 2006").
            day = [(c, e) for c, e in candidates if _single_day(e)]
            timed = [
                (c, e)
                for c, e in day
                if _has_time(_clean(e.date_text)) or _has_time(_clean(e.source.quote))
            ]
            prior = _prior(pos, timed) or _prior(pos, day)
        if prior is not None:
            anchor_span, anchor_id, from_event = _span_of(prior), prior.id, True
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
