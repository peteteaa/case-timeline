"""Resolve free-text dates on drafted facts into date ranges with an honest certainty (§9.10).

Precision is never invented: a month-only date becomes the whole month ("approximate"), a
season becomes the season's months, a span between two exact days ("March 3-5, 2024") is
"exact" with both ends set, and an unparseable date stays "unknown" with no dates. Relative
dates ("two days after the meeting") resolve against an anchor: another draft of the same
document (``anchor_key``), an existing fact (``anchor_fact_id``) or the document's own date;
unresolved ones stay undated with certainty "relative".
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

from pipeline.schema import Certainty, DraftFact, Fact

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


_TIME_OF_DAY = re.compile(
    r"(?:\s+(?:at|@))?(?:\s+(?:approximately|approx\.?|about|around))?\s*\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\b\.?)?"
    r"(?:\s*[+-]\d{4}\b|\s+(?:utc|gmt|[ecmp][sd]t)\b)?"
)


def _clean(text: str) -> str:
    s = text.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s*[–—]\s*", " - ", s)
    s = re.sub(r"\(\s*\d+\s*\)", " ", s)  # "thirty (30) days"
    s = re.sub(r"\([^)]*\)", " ", s)
    s = _TIME_OF_DAY.sub(" ", s)  # "09:14:00 -0400", "at 4:02 pm": a date, not a range
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,;:\"'")


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
    s = _clean(text)
    if not s:
        return None
    try:
        return _parse_range(s, ref) or _parse_point(s, ref)
    except (ValueError, OverflowError):  # e.g. year 0000, Feb 30
        return None


def _parse_point(s: str, ref: date | None) -> tuple[Span, Certainty] | None:
    for prefix in _APPROX_PREFIXES:
        if s.startswith(prefix + " "):
            inner = parse_absolute(s[len(prefix) + 1 :], ref)
            if inner is None:
                return None
            return inner[0], "approximate"

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
        r"|same|very same|subsequent)\s+(?P<unit>day|date|morning|afternoon|evening|night|week"
        r"|weekend|month|year)$",
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
        r"^(?:later|earlier) (?:that|the same) (?:day|morning|afternoon|evening|night)$", s
    )
    if m:
        return RelativeSpec(lambda a, _e: a)

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


def _draft_anchor(d: DraftFact, pos: int) -> _Anchor:
    assert d.date_start is not None
    return _Anchor(
        span=(d.date_start, d.date_end or d.date_start),
        certainty=d.date_certainty,
        text=_draft_text(d),
        pos=pos,
        key=d.key,
    )


def resolve_drafts(
    drafts: list[DraftFact], doc_date: date | None, existing_facts: list[Fact] | None = None
) -> list[DraftFact]:
    """Fill date_start/date_end/date_certainty (and anchors for relative dates).

    ``drafts`` all come from one document and are walked in text order (first span).
    Absolute dates are resolved first; relative ones then anchor to the draft or existing
    fact they name, else the most recent prior dated draft, else ``doc_date`` ("yesterday"
    and "3 days ago" prefer ``doc_date``). Drafts that already carry a date_start are left
    untouched. Input order is preserved.
    """
    if not drafts:
        return []
    out: list[DraftFact] = list(drafts)
    order = sorted(range(len(out)), key=lambda i: (_pos(out[i], i), i))
    pos_of = {i: _pos(out[i], i) for i in order}
    facts = [
        _Anchor(
            span=(f.date_start, f.date_end or f.date_start),
            certainty=f.date_certainty,
            text=_fact_text(f),
            fact_id=f.fact_id,
        )
        for f in (existing_facts or [])
        if f.date_start is not None and f.review != "rejected"
    ]
    pending: list[tuple[int, RelativeSpec]] = []
    last_dated: date | None = None

    for i in order:
        d = out[i]
        if d.date_start is not None:
            last_dated = d.date_start
            continue
        text = (d.date_text or "").strip()
        undated = {"date_start": None, "date_end": None, "date_certainty": "unknown"}
        if not text:
            out[i] = d.model_copy(update=undated)
            continue
        absolute = parse_absolute(text, last_dated or doc_date)
        if absolute is not None:
            (lo, hi), cert = absolute
            out[i] = d.model_copy(update={"date_start": lo, "date_end": hi, "date_certainty": cert})
            last_dated = lo
            continue
        spec = parse_relative(text)
        if spec is not None:
            pending.append((i, spec))
        else:
            out[i] = d.model_copy(update=undated)

    for final in (False, True):
        still: list[tuple[int, RelativeSpec]] = []
        for i, spec in pending:
            candidates = [
                _draft_anchor(out[j], pos_of[j])
                for j in order
                if j != i and out[j].date_start is not None
            ]
            resolved = _resolve_relative(
                out[i], pos_of[i], spec, candidates, facts, doc_date, final
            )
            if resolved is None:
                still.append((i, spec))
            else:
                out[i] = resolved
        pending = still
    for i, _spec in pending:
        out[i] = out[i].model_copy(
            update={"date_start": None, "date_end": None, "date_certainty": "relative"}
        )
    return out


def _resolve_relative(
    d: DraftFact,
    pos: int,
    spec: RelativeSpec,
    drafts: list[_Anchor],
    facts: list[_Anchor],
    doc_date: date | None,
    final: bool,
) -> DraftFact | None:
    anchor_span: Span | None = None
    anchor: _Anchor | None = None

    if spec.phrase:
        prior = _prior(pos, drafts)
        ref = (prior.span[0] if prior else None) or doc_date
        absolute = parse_absolute(spec.phrase, ref)
        if absolute is not None:
            anchor_span = absolute[0]
        else:
            anchor = _match_anchor(spec.phrase, pos, drafts + facts)
            if anchor is None and final:
                anchor = _prior(pos, drafts)
    elif spec.deictic and doc_date is not None:
        anchor_span = (doc_date, doc_date)
    else:
        anchor = _prior(pos, drafts)
        if anchor is None and doc_date is not None and final:
            anchor_span = (doc_date, doc_date)

    if anchor is not None:
        anchor_span = anchor.span
    if anchor_span is None:
        return None
    try:
        lo, hi = spec.apply(anchor_span, anchor is not None)
    except (ValueError, OverflowError) as exc:
        logger.debug("could not apply relative date %r: %s", d.date_text, exc)
        return None
    return d.model_copy(
        update={
            "date_start": lo,
            "date_end": hi,
            "date_certainty": "relative",
            "anchor_key": anchor.key if anchor else None,
            "anchor_fact_id": anchor.fact_id if anchor else None,
        }
    )
