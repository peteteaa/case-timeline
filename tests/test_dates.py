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
    assert out["c"].certainty == Certainty.RELATIVE


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
        assert out[eid].certainty == Certainty.RELATIVE


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
