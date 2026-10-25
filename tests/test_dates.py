from datetime import date

import pytest

from pipeline.contracts import Certainty, Event, SourceRef
from pipeline.dates import parse_absolute, parse_relative, resolve_dates

DOC_DATE = date(2024, 6, 15)  # a Saturday
EXACT, APPROXIMATE, RANGE = Certainty.EXACT, Certainty.APPROXIMATE, Certainty.RANGE


def ev(
    eid: str, date_text: str, pos: int, description: str = "something happened", doc: str = "d1"
) -> Event:
    return Event(
        id=eid,
        case_id="demo",
        date_text=date_text,
        description=description,
        source=SourceRef(doc_id=doc, char_start=pos, char_end=pos + 10, quote=description),
        confidence=0.9,
    )


def one(date_text: str, doc_date: date | None = DOC_DATE) -> Event:
    return resolve_dates([ev("e1", date_text, 0)], doc_date)[0]


# ------------------------------------------------------------------------ absolute dates


@pytest.mark.parametrize(
    ("text", "start", "end", "certainty"),
    [
        ("March 3, 2024", date(2024, 3, 3), date(2024, 3, 3), EXACT),
        ("3 March 2024", date(2024, 3, 3), date(2024, 3, 3), EXACT),
        ("2024-03-03", date(2024, 3, 3), date(2024, 3, 3), EXACT),
        ("03/04/2024", date(2024, 3, 4), date(2024, 3, 4), EXACT),
        ("Mar. 3, 2024", date(2024, 3, 3), date(2024, 3, 3), EXACT),
        ("on March 3rd, 2024", date(2024, 3, 3), date(2024, 3, 3), EXACT),
        ("this 3rd day of March, 2024", date(2024, 3, 3), date(2024, 3, 3), EXACT),
        ("Monday, March 4, 2024 at 10:00 a.m.", date(2024, 3, 4), date(2024, 3, 4), EXACT),
        ("no later than April 1, 2024", date(2024, 4, 1), date(2024, 4, 1), EXACT),
        ("on or about March 3, 2024", date(2024, 3, 3), date(2024, 3, 3), APPROXIMATE),
        ("around 3pm on March 3, 2024", date(2024, 3, 3), date(2024, 3, 3), APPROXIMATE),
        ("March 2024", date(2024, 3, 1), date(2024, 3, 31), APPROXIMATE),
        ("in February 2024", date(2024, 2, 1), date(2024, 2, 29), APPROXIMATE),
        ("03/2024", date(2024, 3, 1), date(2024, 3, 31), APPROXIMATE),
        ("early March 2024", date(2024, 3, 1), date(2024, 3, 10), APPROXIMATE),
        ("mid-March 2024", date(2024, 3, 11), date(2024, 3, 20), APPROXIMATE),
        ("late February 2024", date(2024, 2, 21), date(2024, 2, 29), APPROXIMATE),
        ("the end of March 2024", date(2024, 3, 21), date(2024, 3, 31), APPROXIMATE),
        ("late 2023", date(2023, 9, 1), date(2023, 12, 31), APPROXIMATE),
        ("in or around late 2023", date(2023, 9, 1), date(2023, 12, 31), APPROXIMATE),
        ("spring 2024", date(2024, 3, 1), date(2024, 5, 31), APPROXIMATE),
        ("the summer of 2022", date(2022, 6, 1), date(2022, 8, 31), APPROXIMATE),
        ("winter 2023-24", date(2023, 12, 1), date(2024, 2, 29), APPROXIMATE),
        ("Q2 2024", date(2024, 4, 1), date(2024, 6, 30), APPROXIMATE),
        ("third quarter of 2023", date(2023, 7, 1), date(2023, 9, 30), APPROXIMATE),
        ("2024", date(2024, 1, 1), date(2024, 12, 31), APPROXIMATE),
        ("sometime in 2023", date(2023, 1, 1), date(2023, 12, 31), APPROXIMATE),
        ("the 1990s", date(1990, 1, 1), date(1999, 12, 31), APPROXIMATE),
        ("the week of March 4, 2024", date(2024, 3, 4), date(2024, 3, 10), APPROXIMATE),
        ("after March 3, 2024", date(2024, 3, 3), date(2024, 3, 3), APPROXIMATE),
        ("between May and June 2024", date(2024, 5, 1), date(2024, 6, 30), RANGE),
        ("between May 1, 2024 and May 9, 2024", date(2024, 5, 1), date(2024, 5, 9), RANGE),
        ("from March 3 to March 5, 2024", date(2024, 3, 3), date(2024, 3, 5), RANGE),
        ("March 3-5, 2024", date(2024, 3, 3), date(2024, 3, 5), RANGE),
        ("3-5 March 2024", date(2024, 3, 3), date(2024, 3, 5), RANGE),
        ("March 3 – April 2, 2024", date(2024, 3, 3), date(2024, 4, 2), RANGE),
        ("December 2023 - January 2024", date(2023, 12, 1), date(2024, 1, 31), RANGE),
        ("between December and January 2024", date(2023, 12, 1), date(2024, 1, 31), RANGE),
        ("2023-2024", date(2023, 1, 1), date(2024, 12, 31), RANGE),
        (
            "between early May and late June 2024",
            date(2024, 5, 1),
            date(2024, 6, 30),
            RANGE,
        ),
    ],
)
def test_absolute(text: str, start: date, end: date, certainty: Certainty) -> None:
    out = one(text)
    assert (out.date_start, out.date_end, out.certainty) == (start, end, certainty)
    assert out.anchor_event_id is None


@pytest.mark.parametrize(
    "text", ["", "unknown", "n/a", "the meeting", "at some point", "13/45/2024"]
)
def test_unparseable_is_unknown(text: str) -> None:
    out = one(text)
    assert out.certainty == Certainty.UNKNOWN
    assert out.date_start is None and out.date_end is None


def test_month_only_never_becomes_exact() -> None:
    out = one("June 2024")
    assert out.certainty != Certainty.EXACT
    assert out.date_start != out.date_end


def test_year_is_never_taken_from_today() -> None:
    # No year and no context: we must not silently use the current year.
    out = one("March 3", doc_date=None)
    assert out.date_start is None
    assert out.certainty == Certainty.UNKNOWN


def test_missing_year_inferred_from_doc_date_is_approximate() -> None:
    out = one("March 3", doc_date=date(2024, 6, 15))
    assert out.date_start == date(2024, 3, 3)
    assert out.certainty == Certainty.APPROXIMATE


def test_missing_year_picks_nearest_occurrence() -> None:
    out = one("December 20", doc_date=date(2024, 1, 5))
    assert out.date_start == date(2023, 12, 20)


def test_parse_helpers() -> None:
    assert parse_absolute("two days after the meeting") is None
    assert parse_relative("two days after the meeting") is not None
    assert parse_relative("March 3, 2024") is None


# ------------------------------------------------------------------------ relative dates


def test_relative_resolves_against_named_event() -> None:
    meeting = ev("m", "March 3, 2024", 0, "Smith and Jones held a meeting at Acme.")
    letter = ev("l", "March 10, 2024", 50, "Jones sent a demand letter.")
    rel = ev("r", "two days after the meeting", 100, "Smith called the bank.")
    out = {e.id: e for e in resolve_dates([meeting, letter, rel], DOC_DATE)}
    assert out["r"].date_start == date(2024, 3, 5) == out["r"].date_end
    assert out["r"].certainty == Certainty.RELATIVE
    assert out["r"].anchor_event_id == "m"


def test_relative_anaphoric_uses_most_recent_prior_event() -> None:
    a = ev("a", "March 1, 2024", 0)
    b = ev("b", "March 6, 2024", 50)  # Wednesday
    c = ev("c", "the following week", 100)
    out = {e.id: e for e in resolve_dates([a, b, c], DOC_DATE)}
    assert (out["c"].date_start, out["c"].date_end) == (date(2024, 3, 11), date(2024, 3, 17))
    assert out["c"].anchor_event_id == "b"
    assert out["c"].certainty == Certainty.RELATIVE


def test_next_weekday_after_anchor() -> None:
    a = ev("a", "March 8, 2024", 0)  # Friday
    b = ev("b", "next Monday", 50)
    out = resolve_dates([a, b], DOC_DATE)[1]
    assert out.date_start == date(2024, 3, 11)
    assert out.anchor_event_id == "a"


def test_previous_weekday() -> None:
    a = ev("a", "March 8, 2024", 0)  # Friday
    b = ev("b", "the previous Friday", 50)
    assert resolve_dates([a, b], DOC_DATE)[1].date_start == date(2024, 3, 1)


def test_relative_uses_text_order_not_input_order() -> None:
    b = ev("b", "the next day", 100)
    a = ev("a", "April 2, 2024", 0)
    out = resolve_dates([b, a], DOC_DATE)
    assert [e.id for e in out] == ["b", "a"]  # input order preserved
    assert out[0].date_start == date(2024, 4, 3)
    assert out[0].anchor_event_id == "a"


def test_relative_chains_through_relative_events() -> None:
    a = ev("a", "April 2, 2024", 0)
    b = ev("b", "the next day", 50)
    c = ev("c", "three days later", 100)
    out = resolve_dates([a, b, c], DOC_DATE)
    assert out[1].date_start == date(2024, 4, 3)
    assert out[2].date_start == date(2024, 4, 6)
    assert out[2].anchor_event_id == "b"


def test_relative_to_later_named_event() -> None:
    rel = ev("r", "two days before the hearing", 0, "Counsel filed a brief.")
    hearing = ev("h", "May 10, 2024", 100, "The court held a hearing on the motion.")
    out = resolve_dates([rel, hearing], DOC_DATE)[0]
    assert out.date_start == date(2024, 5, 8)
    assert out.anchor_event_id == "h"


def test_relative_falls_back_to_doc_date() -> None:
    out = one("the next day", doc_date=date(2024, 6, 15))
    assert out.date_start == date(2024, 6, 16)
    assert out.anchor_event_id is None
    assert out.certainty == Certainty.RELATIVE


def test_deictic_prefers_doc_date_over_prior_event() -> None:
    a = ev("a", "January 2, 2024", 0)
    b = ev("b", "yesterday", 50)
    c = ev("c", "three days ago", 60)
    out = resolve_dates([a, b, c], date(2024, 6, 15))
    assert out[1].date_start == date(2024, 6, 14)
    assert out[1].anchor_event_id is None
    assert out[2].date_start == date(2024, 6, 12)


def test_relative_without_any_anchor_stays_unresolved() -> None:
    out = one("two days later", doc_date=None)
    assert out.certainty == Certainty.RELATIVE
    assert out.date_start is None and out.date_end is None


def test_relative_to_explicit_date() -> None:
    out = one("two days after March 3, 2024")
    assert out.date_start == date(2024, 3, 5)
    assert out.certainty == Certainty.RELATIVE
    assert out.anchor_event_id is None


def test_relative_keeps_anchor_imprecision() -> None:
    a = ev("a", "early March 2024", 0)
    b = ev("b", "the following month", 50)
    c = ev("c", "two days later", 60)
    out = resolve_dates([a, b, c], DOC_DATE)
    assert (out[1].date_start, out[1].date_end) == (date(2024, 4, 1), date(2024, 4, 30))
    assert (out[2].date_start, out[2].date_end) == (date(2024, 4, 3), date(2024, 5, 2))


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("later that day", date(2024, 3, 6), date(2024, 3, 6)),
        ("the same day", date(2024, 3, 6), date(2024, 3, 6)),
        ("the next morning", date(2024, 3, 7), date(2024, 3, 7)),
        ("the day before", date(2024, 3, 5), date(2024, 3, 5)),
        ("a week later", date(2024, 3, 13), date(2024, 3, 13)),
        ("two months later", date(2024, 5, 6), date(2024, 5, 6)),
        ("a few days later", date(2024, 3, 8), date(2024, 3, 10)),
        ("about two weeks later", date(2024, 3, 17), date(2024, 3, 23)),
        ("two business days after", date(2024, 3, 8), date(2024, 3, 8)),
        ("within thirty (30) days", date(2024, 3, 6), date(2024, 4, 5)),
        ("shortly after", date(2024, 3, 6), date(2024, 3, 13)),
        ("the following weekend", date(2024, 3, 16), date(2024, 3, 17)),
    ],
)
def test_relative_offsets(text: str, start: date, end: date) -> None:
    anchor = ev("a", "March 6, 2024", 0)  # Wednesday
    out = resolve_dates([anchor, ev("b", text, 50)], DOC_DATE)[1]
    assert (out.date_start, out.date_end) == (start, end)
    assert out.certainty == Certainty.RELATIVE
    assert out.anchor_event_id == "a"


def test_within_days_of_named_event() -> None:
    service = ev("s", "January 10, 2024", 0, "Service of the complaint on Acme was completed.")
    other = ev("o", "January 20, 2024", 50, "Smith emailed Jones.")
    due = ev("d", "within 21 days of service", 100, "Acme's answer is due.")
    out = resolve_dates([service, other, due], DOC_DATE)[2]
    assert (out.date_start, out.date_end) == (date(2024, 1, 10), date(2024, 1, 31))
    assert out.anchor_event_id == "s"


def test_documents_do_not_share_anchors() -> None:
    a = ev("a", "March 1, 2024", 0, doc="d1")
    b = ev("b", "the next day", 50, doc="d2")
    out = resolve_dates([a, b], None)[1]
    assert out.date_start is None
    assert out.certainty == Certainty.RELATIVE


def test_already_resolved_event_is_untouched() -> None:
    e = ev("a", "garbled", 0).model_copy(
        update={"date_start": date(2020, 1, 1), "date_end": date(2020, 1, 1)}
    )
    e = e.model_copy(update={"certainty": Certainty.EXACT})
    assert resolve_dates([e], DOC_DATE)[0] == e


def test_empty() -> None:
    assert resolve_dates([], DOC_DATE) == []


@pytest.mark.parametrize("text", ["0000", "February 30, 2024", "Q2 0000", "the 0000s"])
def test_invalid_calendar_values_are_unknown(text: str) -> None:
    assert one(text).certainty == Certainty.UNKNOWN


# ------------------------------------------------------------ times of day, year carry-over


@pytest.mark.parametrize(
    ("text", "start", "certainty"),
    [
        ("March 14, 2024 at 5:40 PM", date(2024, 3, 14), EXACT),
        ("March 14, 2024 at approximately 5:40 PM", date(2024, 3, 14), APPROXIMATE),
        ("Wednesday, March 20, 2024 4:02 PM", date(2024, 3, 20), EXACT),
        ("Mon, 25 Mar 2024 09:14:00 -0400", date(2024, 3, 25), EXACT),
        ("2024-03-14T17:40:00", date(2024, 3, 14), EXACT),
        ("10/14/2010 13:42 CDT", date(2010, 10, 14), EXACT),
        ("May 6, 2024 at 1700 hours", date(2024, 5, 6), EXACT),
        ("at noon on June 1, 2024", date(2024, 6, 1), EXACT),
        ("the morning of March 3, 2024", date(2024, 3, 3), EXACT),
        ("March 3, 2024, after the meeting", date(2024, 3, 3), EXACT),
    ],
)
def test_time_of_day_is_dropped(text: str, start: date, certainty: Certainty) -> None:
    result = parse_absolute(text, None)
    assert result is not None
    (lo, hi), cert = result
    assert (lo, hi, cert) == (start, start, certainty)


def test_timezone_offset_is_not_a_range() -> None:
    e = one("Mon, 25 Mar 2024 09:14:00 -0400")
    assert (e.date_start, e.date_end, e.certainty) == (date(2024, 3, 25), date(2024, 3, 25), EXACT)


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("mid-March", date(2024, 3, 11), date(2024, 3, 20)),
        ("early March", date(2024, 3, 1), date(2024, 3, 10)),
        ("Sometime in mid-March, after that meeting", date(2024, 3, 11), date(2024, 3, 20)),
        ("March", date(2024, 3, 1), date(2024, 3, 31)),
        ("late December", date(2023, 12, 21), date(2023, 12, 31)),
    ],
)
def test_month_without_year_carries_year_over(text: str, start: date, end: date) -> None:
    assert parse_absolute(text, date(2024, 3, 12)) == ((start, end), APPROXIMATE)


def test_month_without_year_and_no_reference_is_unknown() -> None:
    assert parse_absolute("mid-March", None) is None
    assert one("mid-March", doc_date=None).certainty == Certainty.UNKNOWN


def test_month_without_year_uses_prior_dated_event() -> None:
    events = [ev("a", "March 12, 2024", 0), ev("b", "sometime in mid-March", 50)]
    out = {e.id: e for e in resolve_dates(events, date(2024, 9, 10))}
    assert (out["b"].date_start, out["b"].date_end) == (date(2024, 3, 11), date(2024, 3, 20))
    assert out["b"].certainty == APPROXIMATE


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("earlier in the week", date(2024, 3, 11), date(2024, 3, 17)),
        ("later that month", date(2024, 3, 1), date(2024, 3, 31)),
        ("that same day", date(2024, 3, 12), date(2024, 3, 12)),
        ("Two days later at 3 PM", date(2024, 3, 14), date(2024, 3, 14)),
        ("approximately 6:15 PM", date(2024, 3, 12), date(2024, 3, 12)),
    ],
)
def test_more_relative_phrases(text: str, start: date, end: date) -> None:
    spec = parse_relative(text)
    assert spec is not None
    anchor = date(2024, 3, 12)
    assert spec.apply((anchor, anchor), True) == (start, end)


def test_bare_time_continues_last_timed_event_not_quoted_date() -> None:
    events = [
        ev("a", "March 14, 2024 at approximately 5:40 PM", 0, "officer responded"),
        ev("b", "March 12, 2024", 100, "she said a meeting happened earlier"),
        ev("c", "approximately 6:15 PM", 200, "both parties left"),
    ]
    out = {e.id: e for e in resolve_dates(events, None)}
    assert out["c"].date_start == date(2024, 3, 14)
    assert out["c"].anchor_event_id == "a"
    assert out["c"].certainty == Certainty.APPROXIMATE  # the day is inherited, not stated


def test_earlier_that_day_anchors_to_prior_timed_event() -> None:
    events = [
        ev("a", "March 14, 2024 at 5:40 PM", 0, "officer responded"),
        ev("b", "earlier that day", 80, "she was sent home"),
    ]
    out = {e.id: e for e in resolve_dates(events, date(2024, 3, 18))}
    assert out["b"].date_start == date(2024, 3, 14)
    assert out["b"].anchor_event_id == "a"


# --------------------------------------------------------------- 24-hour clock times


@pytest.mark.parametrize(
    "text",
    [
        "1810:33",
        "At 1816:19",
        "about 1816:45",
        "1829 Eastern daylight time",
        "1342 cdt",
        "at 1729",
        "at about 0930",
        "about 1815",
        "at 1200 hours",
    ],
)
def test_24_hour_times_are_times_not_years(text: str) -> None:
    assert parse_absolute(text) is None
    spec = parse_relative(text)
    assert spec is not None and spec.clock


@pytest.mark.parametrize("text", ["about 2005", "in 1999", "2010"])
def test_plausible_years_stay_years(text: str) -> None:
    found = parse_absolute(text)
    assert found is not None and found[0][0].month == 1 and found[0][1].month == 12


def test_date_with_24_hour_time_and_zone_name() -> None:
    found = parse_absolute("June 5, 2021, about 1816 central daylight time")
    assert found == ((date(2021, 6, 5), date(2021, 6, 5)), APPROXIMATE)
    assert parse_absolute("On June 5, 2021 at 1729") == ((date(2021, 6, 5),) * 2, EXACT)


def test_time_only_events_take_the_day_of_the_last_dated_event() -> None:
    events = [
        ev("a", "June 5, 2021, at 1729", 0, "The aircraft departed."),
        ev("b", "1810:33", 50, "The pilot reported the field in sight."),
        ev("c", "At 1816:19", 90, "The controller cleared the aircraft to land."),
    ]
    out = {e.id: e for e in resolve_dates(events, date(2021, 9, 1))}
    for eid in ("b", "c"):
        assert out[eid].date_start == date(2021, 6, 5)
        assert out[eid].certainty == Certainty.APPROXIMATE
        assert out[eid].anchor_event_id in ("a", "b")


def test_time_only_events_skip_untimed_later_dates_and_wide_spans() -> None:
    events = [
        ev("acc", "June 5, 2021", 0, "On June 5, 2021, about 1816 cdt, the truck overturned."),
        ev("bg", "since 2006", 40, "The driver had held a commercial license since 2006."),
        ev("meet", "June 6, 2021", 80, "The review team met the next day."),
        ev("t1", "1810:33", 120, "The driver radioed dispatch."),
        ev("t2", "1811:02", 160, "Dispatch replied."),
    ]
    out = {e.id: e for e in resolve_dates(events, date(2022, 2, 1))}
    assert out["t1"].date_start == date(2021, 6, 5) and out["t1"].anchor_event_id == "acc"
    assert out["t2"].date_start == date(2021, 6, 5)


def test_time_only_event_never_anchors_to_a_year_span() -> None:
    events = [
        ev("bg", "since 2006", 0, "He has flown this aircraft type since 2006."),
        ev("t", "8:00", 40, "He flew a morning mission."),
    ]
    out = {e.id: e for e in resolve_dates(events, date(2021, 8, 30))}
    assert out["t"].date_start == date(2021, 8, 30)  # the document date, not 2006-01-01


# ------------------------------------------------- narrative context: text + incident date


def _doc(text: str, doc_id: str = "d1"):
    from pipeline.contracts import Document, ParsedDocument

    doc = Document(id=doc_id, case_id="demo", path="/x", filename="r.pdf", mime="x", sha256="x")
    return ParsedDocument(doc=doc, text=text)


def at(eid: str, date_text: str, text: str, quote: str, unverified: bool = False) -> Event:
    """An event whose quote is located in `text` (offsets 0/0 when unverified)."""
    start = 0 if unverified else text.index(quote)
    end = 0 if unverified else start + len(quote)
    return Event(
        id=eid,
        case_id="demo",
        date_text=date_text,
        description=quote,
        source=SourceRef(doc_id="d1", char_start=start, char_end=end, quote=quote),
        confidence=0.9,
    )


REPORT = (
    "Regional Safety Office\n"
    "September 9, 2022\n"
    "FACTUAL REPORT\n"
    "B. VEHICLE ACCIDENT\n"
    "Location: Springfield Date: June 5, 2021 Time: 0940 central daylight time\n\n"
    "On June 5, 2021, about 0940 central daylight time, a delivery truck overturned. "
    "The review team met at the depot on June 7, 2021. "
    "The team collected logs.\n\n"
    "History of the trip\n\n"
    "The truck left the yard at 0805 local time. "
    "At 0931:12, the driver radioed dispatch. "
    "Dispatch replied. "
    "At the time of the accident, the road was wet.\n"
)


def test_time_only_takes_last_timed_narrative_date_from_text() -> None:
    events = [
        at("meet", "June 7, 2021", REPORT, "The review team met at the depot on June 7, 2021."),
        at("t1", "0805 local time", REPORT, "The truck left the yard at 0805 local time."),
        at("t2", "At 0931:12", REPORT, "At 0931:12, the driver radioed dispatch."),
    ]
    out = {e.id: e for e in resolve_dates(events, date(2022, 9, 9), parsed=_doc(REPORT))}
    # the timed "On June 5, 2021, about 0940" beats the later, untimed review-team date
    assert out["t1"].date_start == out["t1"].date_end == date(2021, 6, 5)
    assert out["t1"].certainty == APPROXIMATE
    assert out["t2"].date_start == date(2021, 6, 5)
    assert out["t2"].certainty == APPROXIMATE
    assert out["t2"].anchor_event_id == "t1"  # continues the previous timed event


def test_time_only_with_the_day_in_the_same_sentence_is_exact() -> None:
    text = "Notes.\n\nThe truck left at 0805 on June 5, 2021, heading north. It rained.\n"
    e = at("t", "at 0805", text, "The truck left at 0805")
    out = resolve_dates([e], date(2022, 1, 1), parsed=_doc(text))[0]
    assert out.date_start == date(2021, 6, 5)
    assert out.certainty == EXACT


def test_time_only_falls_back_to_incident_date_not_doc_date() -> None:
    text = "Statement taken March 1, 2022\nDate of Incident: 06/05/2021\n\nAt 0931 I saw it.\n"
    e = at("t", "At 0931", text, "At 0931 I saw it.")
    out = resolve_dates([e], date(2022, 3, 1), parsed=_doc(text))[0]
    assert out.date_start == date(2021, 6, 5)
    assert out.certainty == APPROXIMATE


def test_time_only_without_text_uses_explicit_incident_date() -> None:
    out = resolve_dates([ev("t", "at 1405", 0)], date(2022, 3, 1), incident_date=date(2021, 6, 5))
    assert out[0].date_start == date(2021, 6, 5)
    assert out[0].certainty == APPROXIMATE


def test_time_only_without_incident_still_uses_doc_date() -> None:
    out = resolve_dates([ev("t", "at 1405", 0)], date(2022, 3, 1))[0]
    assert out.date_start == date(2022, 3, 1)
    assert out.certainty == APPROXIMATE


def test_at_the_time_of_the_accident_anchors_to_incident_date() -> None:
    events = [
        at("hdr", "June 5, 2021", REPORT, "Date: June 5, 2021 Time: 0940"),
        at("meet", "June 7, 2021", REPORT, "The review team met at the depot on June 7, 2021."),
        at("wet", "At the time of the accident", REPORT, "the road was wet."),
        at("u", "at the time of the accident", REPORT, "nowhere", unverified=True),
    ]
    out = {e.id: e for e in resolve_dates(events, date(2022, 9, 9), parsed=_doc(REPORT))}
    for eid in ("wet", "u"):
        assert out[eid].date_start == date(2021, 6, 5)
        assert out[eid].anchor_event_id == "hdr"


def test_undated_account_takes_incident_date_not_authoring_date() -> None:
    text = (
        "September 2, 2021\nDear Ms. Ortiz:\nThis concerns the collision involving my truck "
        "on June 5, 2021.\n\nThe next day I called the insurer.\n"
    )
    e = at("call", "The next day", text, "The next day I called the insurer.", unverified=True)
    out = resolve_dates([e], date(2021, 9, 2), parsed=_doc(text))[0]
    assert out.date_start == date(2021, 6, 6)  # the day after the incident, not Sept 3
    assert out.certainty == Certainty.RELATIVE


def test_letter_date_line_is_not_a_narrative_anchor_when_incident_known() -> None:
    text = (
        "Statement\nIncident occurred Saturday, June 5, 2021 near the depot.\n\n"
        "September 4, 2021\n\nI was at the gate. At 0931 the truck tipped over.\n"
    )
    events = [
        at("letter", "September 4, 2021", text, "September 4, 2021"),
        at("t", "At 0931", text, "At 0931 the truck tipped over."),
    ]
    out = {e.id: e for e in resolve_dates(events, date(2021, 9, 4), parsed=_doc(text))}
    assert out["t"].date_start == date(2021, 6, 5)


def test_on_this_date_is_anaphoric_not_the_document_date() -> None:
    events = [ev("a", "June 5, 2021", 0), ev("b", "On this date", 50)]
    out = resolve_dates(events, date(2021, 9, 2))[1]
    assert out.date_start == date(2021, 6, 5)
    assert out.anchor_event_id == "a"


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("about 15 seconds after the crash", date(2021, 6, 5), date(2021, 6, 5)),
        ("ten minutes later", date(2021, 6, 5), date(2021, 6, 5)),
        ("two hours later", date(2021, 6, 5), date(2021, 6, 6)),  # may cross midnight
        ("30 hours later", date(2021, 6, 6), date(2021, 6, 7)),
    ],
)
def test_sub_day_offsets(text: str, start: date, end: date) -> None:
    out = resolve_dates([ev("a", "June 5, 2021", 0, "the crash"), ev("b", text, 50)], None)[1]
    assert (out.date_start, out.date_end) == (start, end)


def test_parsed_context_applies_only_to_its_own_document() -> None:
    other = ev("x", "at 1405", 0, doc="d2")
    out = resolve_dates([other], date(2022, 3, 1), parsed=_doc(REPORT))[0]
    assert out.date_start == date(2022, 3, 1)  # d2 has no text/incident context


def test_that_day_never_refers_back_to_an_open_bound_or_year() -> None:
    events = [
        ev("a", "June 5, 2021", 0, "the trip"),
        ev("bg", "since 2006", 40, "he has driven trucks since 2006"),
        ev("c", "that day", 80, "he drove two routes that day"),
    ]
    out = {e.id: e for e in resolve_dates(events, date(2021, 9, 1))}
    assert out["c"].date_start == date(2021, 6, 5)
    assert out["c"].anchor_event_id == "a"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("07-28-21", date(2021, 7, 28)),
        ("12-31-99", date(1999, 12, 31)),
        ("07/02/202118:17", date(2021, 7, 2)),  # OCR glued the time to the year
    ],
)
def test_numeric_dates_with_short_years_and_glued_times(text: str, expected: date) -> None:
    assert parse_absolute(text) == ((expected, expected), EXACT)


@pytest.mark.parametrize("text", ["about 2:30 or 3:00", "between 3 pm and 5 pm"])
def test_time_alternatives_are_times(text: str) -> None:
    assert parse_absolute(text) is None
    spec = parse_relative(text)
    assert spec is not None and spec.clock


def test_shortly_afterward_follows_prior_event() -> None:
    out = resolve_dates([ev("a", "June 5, 2021", 0), ev("b", "Shortly afterward", 50)], None)[1]
    assert out.date_start == date(2021, 6, 5) and out.anchor_event_id == "a"
    assert out.certainty == Certainty.RELATIVE


# ------------------------------------------------------------- time zones (case_tz)

CHICAGO = "America/Chicago"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("July 28, 2010, 0016 UTC", date(2010, 7, 27)),  # 19:16 CDT the day before
        ("July 28, 2010, 00:16Z", date(2010, 7, 27)),
        ("July 28, 2010 0016 Z", date(2010, 7, 27)),
        ("2010-07-28T00:16:57Z", date(2010, 7, 27)),
        ("July 28, 2010, 12:16 am GMT", date(2010, 7, 27)),
        ("07/28/2010 0016 zulu", date(2010, 7, 27)),
        ("July 27, 2010, 2316:57 UTC", date(2010, 7, 27)),  # 18:16 CDT, same day
        ("July 28, 2010, 0016 CDT", date(2010, 7, 28)),  # a local zone keeps its date
        ("July 28, 2010, 0016 central daylight time", date(2010, 7, 28)),
        ("July 28, 2010, 00:16-05:00", date(2010, 7, 28)),
        ("July 28, 2010, 0016 UTC/1916 CDT", date(2010, 7, 28)),  # local time given too
        ("July 28, 2010, at 00:16", date(2010, 7, 28)),  # no zone: never shifted
    ],
)
def test_utc_times_take_the_case_zone_day(text: str, expected: date) -> None:
    (lo, hi), cert = parse_absolute(text, tz=CHICAGO)
    assert lo == hi == expected
    assert cert == EXACT


def test_utc_times_are_not_converted_without_a_case_zone() -> None:
    assert parse_absolute("July 28, 2010, 0016 UTC")[0] == (date(2010, 7, 28),) * 2
    assert parse_absolute("July 28, 2010, 0016 UTC", tz="")[0] == (date(2010, 7, 28),) * 2


def test_resolve_dates_converts_utc_with_the_case_zone() -> None:
    out = resolve_dates([ev("a", "July 28, 2010, 0016 UTC", 0)], None, tz=CHICAGO)[0]
    assert out.date_start == out.date_end == date(2010, 7, 27)
    out = resolve_dates([ev("a", "July 28, 2010, 0016 UTC", 0)], None)[0]
    assert out.date_start == date(2010, 7, 28)


def test_utc_time_with_its_date_in_the_same_sentence_takes_the_case_zone_day() -> None:
    text = "Summary.\n\nOn July 28, 2010, the crew called the tower at 0016 UTC. It rained.\n"
    e = at("t", "at 0016 UTC", text, "the crew called the tower at 0016 UTC")
    out = resolve_dates([e], None, parsed=_doc(text), tz=CHICAGO)[0]
    assert out.date_start == date(2010, 7, 27)
    out = resolve_dates([e], None, parsed=_doc(text))[0]
    assert out.date_start == date(2010, 7, 28)


def test_narrative_date_stated_with_a_utc_time_is_converted() -> None:
    text = (
        "Notes.\n\nThe flight ended July 28, 2010, 0016 UTC, at the field.\n\n"
        "At 0020 it was towed.\n"
    )
    e = at("t", "At 0020", text, "At 0020 it was towed.")
    out = resolve_dates([e], None, parsed=_doc(text), tz=CHICAGO)[0]
    assert out.date_start == date(2010, 7, 27)  # inherits the converted narrative day


def test_one_time_in_two_zones_is_a_time_of_day() -> None:
    text = "2316:19 UTC/1816:19 CDT"
    assert parse_absolute(text) is None
    spec = parse_relative(text)
    assert spec is not None and spec.clock


# ------------------------------------------------- case incident date (case_incident_date)

INTERVIEW = (
    "Investigation of: TRUCK ACCIDENT\n"
    "Interview of: J. Doe\n"
    "Monday, August 30, 2021\n"
    "The above matter convened.\n\n"
    "Q. Tell me about that day.\n"
    "A. I had an 8:00 delivery. And then I did a pickup after that. "
    "The depot opened at about 2:30 or 3:00. After the depot opened I loaded up. "
    "I was wearing glasses that day. "
    "I had been driving trucks since 2006.\n"
    "Q. When did you last see a doctor?\n"
    "A. Yesterday.\n"
)


def _interview_events() -> list[Event]:
    return [
        at("held", "Monday, August 30, 2021", INTERVIEW, "Monday, August 30, 2021"),
        at("t8", "8:00", INTERVIEW, "I had an 8:00 delivery."),
        at("after", "after that", INTERVIEW, "And then I did a pickup after that."),
        at("open", "about 2:30 or 3:00", INTERVIEW, "The depot opened at about 2:30 or 3:00."),
        at("load", "After the depot opened", INTERVIEW, "After the depot opened I loaded up."),
        at("glasses", "that day", INTERVIEW, "I was wearing glasses that day."),
        at("since", "since 2006", INTERVIEW, "I had been driving trucks since 2006."),
        at("doc", "Yesterday", INTERVIEW, "A. Yesterday."),
        at("event", "at the time of the event", INTERVIEW, "Tell me about that day."),
    ]


def test_document_without_incident_date_uses_the_case_incident_date() -> None:
    incident = date(2021, 7, 27)
    out = {
        e.id: e
        for e in resolve_dates(
            _interview_events(),
            date(2021, 8, 30),
            parsed=_doc(INTERVIEW),
            case_incident_date=incident,
        )
    }
    assert out["held"].date_start == date(2021, 8, 30)  # the interview itself
    for eid in ("t8", "after", "open", "load", "glasses", "event"):
        assert out[eid].date_start == out[eid].date_end == incident, eid
    assert out["after"].anchor_event_id == "t8"  # the event just before, not "since 2006"
    assert out["since"].date_start == date(2006, 1, 1)
    assert out["since"].date_end == date(2006, 12, 31)
    assert out["doc"].date_start == date(2021, 8, 29)  # "yesterday" is the document's own


def test_without_case_incident_date_the_interview_date_is_kept() -> None:
    out = {
        e.id: e
        for e in resolve_dates(_interview_events(), date(2021, 8, 30), parsed=_doc(INTERVIEW))
    }
    assert out["t8"].date_start == date(2021, 8, 30)


def test_own_incident_date_beats_the_case_incident_date() -> None:
    text = "Statement\nDate of Incident: 06/05/2021\n\nAt 0931 I saw the crash.\n"
    e = at("t", "At 0931", text, "At 0931 I saw the crash.")
    out = resolve_dates(
        [e], date(2021, 9, 1), parsed=_doc(text), case_incident_date=date(2021, 7, 27)
    )[0]
    assert out.date_start == date(2021, 6, 5)


def test_case_incident_date_needs_a_document_about_an_incident() -> None:
    text = "Invoice\nSeptember 1, 2021\n\nDelivered at 0931 to the dock.\n"
    e = at("t", "at 0931", text, "Delivered at 0931 to the dock.")
    out = resolve_dates(
        [e], date(2021, 9, 1), parsed=_doc(text), case_incident_date=date(2021, 7, 27)
    )[0]
    assert out.date_start == date(2021, 9, 1)


def test_time_stated_with_the_document_date_keeps_it_with_a_case_incident_date() -> None:
    head = "Interview about the accident\nRecording started at 10:05 on August 30, 2021.\n\n"
    e = at("rec", "at 10:05", head, "Recording started at 10:05")
    out = resolve_dates(
        [e], date(2021, 8, 30), parsed=_doc(head), case_incident_date=date(2021, 7, 27)
    )[0]
    assert out.date_start == date(2021, 8, 30)  # the day stated with it wins


# ------------------------------------------------------------------- year-only dates


@pytest.mark.parametrize(
    "text", ["2006", "in 2006", "In 2006", "since 2006", "from 2006", "starting in 2006"]
)
def test_year_only_is_an_approximate_whole_year(text: str) -> None:
    e = one(text)
    assert (e.date_start, e.date_end) == (date(2006, 1, 1), date(2006, 12, 31))
    assert e.certainty == APPROXIMATE


def test_since_a_season_or_month_is_that_period() -> None:
    assert parse_absolute("since the fall of 2006") == (
        (date(2006, 9, 1), date(2006, 11, 30)),
        APPROXIMATE,
    )
    assert parse_absolute("since March 2007")[0] == (date(2007, 3, 1), date(2007, 3, 31))
    # a full day keeps its open-bound meaning
    assert parse_absolute("since March 3, 2024")[0] == (date(2024, 3, 3), date(2024, 3, 3))


def test_alternatives_span_both() -> None:
    assert parse_absolute("May or June of 2007") == (
        (date(2007, 5, 1), date(2007, 6, 30)),
        RANGE,
    )
    assert parse_absolute("on or about March 3, 2024") == (
        (date(2024, 3, 3), date(2024, 3, 3)),
        APPROXIMATE,
    )


def test_year_only_never_overrides_a_full_date_in_its_quote() -> None:
    e = ev("a", "2010", 0, "On July 27, 2010, the truck overturned near the depot.")
    out = resolve_dates([e], date(2010, 9, 1))[0]
    assert out.date_start == out.date_end == date(2010, 7, 27)
    assert out.certainty == EXACT
    e = ev("b", "in 2010", 0, "In 2010 he moved; on about March 3, 2010 he left.")
    out = resolve_dates([e], None)[0]
    assert out.date_start == date(2010, 3, 3) and out.certainty == APPROXIMATE
    e = ev("c", "since 2006", 0, "Since 2006 he has driven; on June 5, 2006 he started.")
    out = resolve_dates([e], None)[0]
    assert (out.date_start, out.date_end) == (date(2006, 1, 1), date(2006, 12, 31))


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("the week of March 3, 2024", date(2024, 3, 3), date(2024, 3, 9)),
        ("beginning of March 2024", date(2024, 3, 1), date(2024, 3, 10)),
        ("beginning in March 2024", date(2024, 3, 1), date(2024, 3, 31)),
    ],
)
def test_periods_that_name_their_own_day_are_kept(text: str, start: date, end: date) -> None:
    out = resolve_dates([ev("a", text, 0, f"It happened {text}.")], None)[0]
    assert (out.date_start, out.date_end) == (start, end)


def test_background_year_does_not_set_the_year_of_later_dates() -> None:
    events = [
        ev("bg", "since 2006", 0, "he has driven since 2006"),
        ev("d", "June 5", 40, "the truck tipped over"),
    ]
    out = resolve_dates(events, None, incident_date=date(2021, 6, 9))[1]
    assert out.date_start == date(2021, 6, 5)  # near the incident, not 2006


def test_after_that_follows_the_previous_event_not_background() -> None:
    events = [
        ev("a", "June 5, 2021", 0, "the trip"),
        ev("bg", "since 2006", 40, "he has driven trucks since 2006"),
        ev("c", "after that", 80, "he went home after that"),
    ]
    out = resolve_dates(events, None)[2]
    assert out.anchor_event_id == "a"
    assert out.date_start == date(2021, 6, 5)
