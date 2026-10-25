from datetime import UTC, date, datetime

import pytest

from pipeline.dates import is_open_bound, parse_absolute, parse_relative, resolve_drafts
from pipeline.schema import Document, DraftFact, ParsedDocument, SpanIn
from tests.pipeline_helpers import make_fact

DOC_DATE = date(2024, 6, 15)  # a Saturday
EXACT, APPROXIMATE, RELATIVE, UNKNOWN = "exact", "approximate", "relative", "unknown"


def ev(key: str, date_text: str, pos: int, title: str = "something happened") -> DraftFact:
    return DraftFact(
        key=key,
        title=title,
        date_text=date_text,
        quotes=[title],
        spans=[SpanIn(doc_id="D-0001", char_start=pos, char_end=pos + 10, quote=title)],
        confidence=0.9,
    )


def one(date_text: str, doc_date: date | None = DOC_DATE) -> DraftFact:
    return resolve_drafts([ev("e1", date_text, 0)], doc_date, [])[0]


def resolve(drafts: list[DraftFact], doc_date: date | None = DOC_DATE, facts=None):
    return resolve_drafts(drafts, doc_date, facts or [])


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
        ("between May and June 2024", date(2024, 5, 1), date(2024, 6, 30), APPROXIMATE),
        ("between May 1, 2024 and May 9, 2024", date(2024, 5, 1), date(2024, 5, 9), EXACT),
        ("from March 3 to March 5, 2024", date(2024, 3, 3), date(2024, 3, 5), EXACT),
        ("March 3-5, 2024", date(2024, 3, 3), date(2024, 3, 5), EXACT),
        ("3-5 March 2024", date(2024, 3, 3), date(2024, 3, 5), EXACT),
        ("March 3 – April 2, 2024", date(2024, 3, 3), date(2024, 4, 2), EXACT),
        ("December 2023 - January 2024", date(2023, 12, 1), date(2024, 1, 31), APPROXIMATE),
        ("between December and January 2024", date(2023, 12, 1), date(2024, 1, 31), APPROXIMATE),
        ("2023-2024", date(2023, 1, 1), date(2024, 12, 31), APPROXIMATE),
        (
            "between early May and late June 2024",
            date(2024, 5, 1),
            date(2024, 6, 30),
            APPROXIMATE,
        ),
    ],
)
def test_absolute(text: str, start: date, end: date, certainty: str) -> None:
    out = one(text)
    assert (out.date_start, out.date_end, out.date_certainty) == (start, end, certainty)
    assert out.anchor_key is None and out.anchor_fact_id is None


@pytest.mark.parametrize(
    "text", ["", "unknown", "n/a", "the meeting", "at some point", "13/45/2024"]
)
def test_unparseable_is_unknown(text: str) -> None:
    out = one(text)
    assert out.date_certainty == UNKNOWN
    assert out.date_start is None and out.date_end is None


@pytest.mark.parametrize(
    "text",
    [
        "Mon, 25 Mar 2024 09:14:00 -0400",
        "March 25, 2024 at 9:14 AM",
        "2024-03-25 09:14 EST",
        "March 25, 2024 at approximately 5:40 PM",
    ],
)
def test_times_and_offsets_are_not_ranges(text: str) -> None:
    out = one(text)
    assert (out.date_start, out.date_end, out.date_certainty) == (
        date(2024, 3, 25),
        date(2024, 3, 25),
        EXACT,
    )


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("mid-March", date(2024, 3, 11), date(2024, 3, 20)),
        ("March", date(2024, 3, 1), date(2024, 3, 31)),
        ("late December", date(2023, 12, 21), date(2023, 12, 31)),
    ],
)
def test_month_without_year_uses_nearest_year(text: str, start: date, end: date) -> None:
    out = one(text, doc_date=date(2024, 4, 2))
    assert (out.date_start, out.date_end, out.date_certainty) == (start, end, APPROXIMATE)
    assert one(text, doc_date=None).date_start is None


def test_time_of_day_takes_the_day_of_the_timed_fact_before_it() -> None:
    # "approximately 6:15 PM" names no day: it continues the narrative's last timed day.
    a = ev("a", "March 14, 2024 at approximately 5:40 PM", 0)
    b = ev("b", "approximately 6:15 PM", 50)
    out = resolve([a, b], DOC_DATE)[1]
    assert out.date_start == out.date_end == date(2024, 3, 14)
    assert out.date_certainty == APPROXIMATE and out.anchor_key == "a"


def test_month_only_never_becomes_exact() -> None:
    out = one("June 2024")
    assert out.date_certainty != EXACT
    assert out.date_start != out.date_end


def test_year_is_never_taken_from_today() -> None:
    # No year and no context: we must not silently use the current year.
    out = one("March 3", doc_date=None)
    assert out.date_start is None
    assert out.date_certainty == UNKNOWN


def test_missing_year_inferred_from_doc_date_is_approximate() -> None:
    out = one("March 3", doc_date=date(2024, 6, 15))
    assert out.date_start == date(2024, 3, 3)
    assert out.date_certainty == APPROXIMATE


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
    out = {e.key: e for e in resolve([meeting, letter, rel], DOC_DATE)}
    assert out["r"].date_start == date(2024, 3, 5) == out["r"].date_end
    assert out["r"].date_certainty == RELATIVE
    assert out["r"].anchor_key == "m"


def test_relative_anaphoric_uses_most_recent_prior_event() -> None:
    a = ev("a", "March 1, 2024", 0)
    b = ev("b", "March 6, 2024", 50)  # Wednesday
    c = ev("c", "the following week", 100)
    out = {e.key: e for e in resolve([a, b, c], DOC_DATE)}
    assert (out["c"].date_start, out["c"].date_end) == (date(2024, 3, 11), date(2024, 3, 17))
    assert out["c"].anchor_key == "b"
    assert out["c"].date_certainty == RELATIVE


def test_next_weekday_after_anchor() -> None:
    a = ev("a", "March 8, 2024", 0)  # Friday
    b = ev("b", "next Monday", 50)
    out = resolve([a, b], DOC_DATE)[1]
    assert out.date_start == date(2024, 3, 11)
    assert out.anchor_key == "a"


def test_previous_weekday() -> None:
    a = ev("a", "March 8, 2024", 0)  # Friday
    b = ev("b", "the previous Friday", 50)
    assert resolve([a, b], DOC_DATE)[1].date_start == date(2024, 3, 1)


def test_relative_uses_text_order_not_input_order() -> None:
    b = ev("b", "the next day", 100)
    a = ev("a", "April 2, 2024", 0)
    out = resolve([b, a], DOC_DATE)
    assert [e.key for e in out] == ["b", "a"]  # input order preserved
    assert out[0].date_start == date(2024, 4, 3)
    assert out[0].anchor_key == "a"


def test_relative_chains_through_relative_events() -> None:
    a = ev("a", "April 2, 2024", 0)
    b = ev("b", "the next day", 50)
    c = ev("c", "three days later", 100)
    out = resolve([a, b, c], DOC_DATE)
    assert out[1].date_start == date(2024, 4, 3)
    assert out[2].date_start == date(2024, 4, 6)
    assert out[2].anchor_key == "b"


def test_relative_to_later_named_event() -> None:
    rel = ev("r", "two days before the hearing", 0, "Counsel filed a brief.")
    hearing = ev("h", "May 10, 2024", 100, "The court held a hearing on the motion.")
    out = resolve([rel, hearing], DOC_DATE)[0]
    assert out.date_start == date(2024, 5, 8)
    assert out.anchor_key == "h"


def test_relative_falls_back_to_doc_date() -> None:
    out = one("the next day", doc_date=date(2024, 6, 15))
    assert out.date_start == date(2024, 6, 16)
    assert out.anchor_key is None
    assert out.date_certainty == RELATIVE


def test_deictic_prefers_doc_date_over_prior_event() -> None:
    a = ev("a", "January 2, 2024", 0)
    b = ev("b", "yesterday", 50)
    c = ev("c", "three days ago", 60)
    out = resolve([a, b, c], date(2024, 6, 15))
    assert out[1].date_start == date(2024, 6, 14)
    assert out[1].anchor_key is None
    assert out[2].date_start == date(2024, 6, 12)


def test_relative_without_any_anchor_stays_unresolved() -> None:
    out = one("two days later", doc_date=None)
    assert out.date_certainty == RELATIVE
    assert out.date_start is None and out.date_end is None


def test_relative_to_explicit_date() -> None:
    out = one("two days after March 3, 2024")
    assert out.date_start == date(2024, 3, 5)
    assert out.date_certainty == RELATIVE
    assert out.anchor_key is None


def test_relative_keeps_anchor_imprecision() -> None:
    a = ev("a", "early March 2024", 0)
    b = ev("b", "the following month", 50)
    c = ev("c", "two days later", 60)
    out = resolve([a, b, c], DOC_DATE)
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
    out = resolve([anchor, ev("b", text, 50)], DOC_DATE)[1]
    assert (out.date_start, out.date_end) == (start, end)
    assert out.date_certainty == RELATIVE
    assert out.anchor_key == "a"


def test_within_days_of_named_event() -> None:
    service = ev("s", "January 10, 2024", 0, "Service of the complaint on Acme was completed.")
    other = ev("o", "January 20, 2024", 50, "Smith emailed Jones.")
    due = ev("d", "within 21 days of service", 100, "Acme's answer is due.")
    out = resolve([service, other, due], DOC_DATE)[2]
    assert (out.date_start, out.date_end) == (date(2024, 1, 10), date(2024, 1, 31))
    assert out.anchor_key == "s"


def test_relative_resolves_against_existing_fact() -> None:
    hearing = make_fact("F-0007", "Court holds hearing on the motion", date(2024, 5, 10))
    rel = ev("r", "two days after the hearing", 0, "Counsel files a reply.")
    out = resolve([rel], None, [hearing])[0]
    assert out.date_start == date(2024, 5, 12)
    assert out.anchor_fact_id == "F-0007" and out.anchor_key is None
    assert out.date_certainty == RELATIVE


def test_draft_anchor_in_same_document_beats_existing_fact() -> None:
    other_doc = make_fact("F-0001", "Meeting in Ellison's office", date(2024, 3, 12))
    meeting = ev("m", "March 11, 2024", 0, "The meeting in Ellison's office took place.")
    rel = ev("r", "Two days after the meeting", 80, "Rivera was sent home without pay.")
    out = resolve([meeting, rel], date(2024, 4, 2), [other_doc])[1]
    assert out.date_start == date(2024, 3, 13)
    assert out.anchor_key == "m" and out.anchor_fact_id is None


def test_rejected_and_undated_facts_are_not_anchors() -> None:
    rejected = make_fact("F-0001", "Court holds hearing", date(2024, 5, 10), review="rejected")
    undated = make_fact("F-0002", "Court holds hearing", None)
    rel = ev("r", "two days after the hearing", 0, "Counsel files a reply.")
    out = resolve([rel], None, [rejected, undated])[0]
    assert out.date_start is None and out.date_certainty == RELATIVE


def test_undated_draft_stays_undated() -> None:
    d = ev("a", "", 0).model_copy(update={"date_text": None})
    out = resolve([d])[0]
    assert (out.date_start, out.date_end, out.date_certainty) == (None, None, UNKNOWN)


def test_already_resolved_draft_is_untouched() -> None:
    e = ev("a", "garbled", 0).model_copy(
        update={
            "date_start": date(2020, 1, 1),
            "date_end": date(2020, 1, 1),
            "date_certainty": EXACT,
        }
    )
    assert resolve([e], DOC_DATE)[0] == e


def test_drafts_without_spans_use_input_order() -> None:
    a = DraftFact(key="a", title="x", date_text="April 2, 2024")
    b = DraftFact(key="b", title="y", date_text="the next day")
    out = resolve([a, b], None)
    assert out[1].date_start == date(2024, 4, 3) and out[1].anchor_key == "a"


def test_empty() -> None:
    assert resolve([], DOC_DATE) == []


@pytest.mark.parametrize("text", ["0000", "February 30, 2024", "Q2 0000", "the 0000s"])
def test_invalid_calendar_values_are_unknown(text: str) -> None:
    assert one(text).date_certainty == UNKNOWN


# ------------------------------------------------------------ times of day, year carry-over


@pytest.mark.parametrize(
    ("text", "start", "certainty"),
    [
        ("March 14, 2024 at 5:40 PM", date(2024, 3, 14), EXACT),
        # The qualifier belongs to the time; the day itself is stated exactly.
        ("March 14, 2024 at approximately 5:40 PM", date(2024, 3, 14), EXACT),
        ("Wednesday, March 20, 2024 4:02 PM", date(2024, 3, 20), EXACT),
        ("Mon, 25 Mar 2024 09:14:00 -0400", date(2024, 3, 25), EXACT),
        ("2024-03-14T17:40:00", date(2024, 3, 14), EXACT),
        ("10/14/2010 13:42 CDT", date(2010, 10, 14), EXACT),
        ("May 6, 2024 at 1700 hours", date(2024, 5, 6), EXACT),
        ("at noon on June 1, 2024", date(2024, 6, 1), EXACT),
        ("the morning of March 3, 2024", date(2024, 3, 3), EXACT),
        ("March 3, 2024, in the afternoon", date(2024, 3, 3), EXACT),
        ("March 3, 2024, after the meeting", date(2024, 3, 3), EXACT),
        ("June 5, 2021, about 1816 central daylight time", date(2021, 6, 5), EXACT),
        ("On June 5, 2021 at 1729", date(2021, 6, 5), EXACT),
        ("around 3pm on March 3, 2024", date(2024, 3, 3), APPROXIMATE),
    ],
)
def test_time_of_day_is_dropped(text: str, start: date, certainty: str) -> None:
    assert parse_absolute(text, None) == ((start, start), certainty)
    out = one(text)
    assert (out.date_start, out.date_end, out.date_certainty) == (start, start, certainty)


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
    assert one("mid-March", doc_date=None).date_certainty == UNKNOWN


def test_month_without_year_uses_prior_dated_draft() -> None:
    drafts = [ev("a", "March 12, 2024", 0), ev("b", "sometime in mid-March", 50)]
    out = resolve(drafts, date(2025, 9, 10))[1]  # the doc date alone would say 2025
    assert (out.date_start, out.date_end) == (date(2024, 3, 11), date(2024, 3, 20))
    assert out.date_certainty == APPROXIMATE


def test_month_day_without_year_takes_year_from_prior_draft() -> None:
    drafts = [ev("a", "April 2, 2023", 0), ev("b", "April 9", 50)]
    out = resolve(drafts, date(2024, 6, 1))[1]
    assert out.date_start == date(2023, 4, 9) and out.date_certainty == APPROXIMATE


def test_open_bound_is_not_year_context() -> None:
    # "since 2006" is background; "March 3" belongs to the document's own time.
    drafts = [ev("a", "since 2006", 0), ev("b", "March 3", 50)]
    out = resolve(drafts, date(2024, 6, 1))
    assert out[0].date_start == date(2006, 1, 1)
    assert out[1].date_start == date(2024, 3, 3)


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("earlier in the week", date(2024, 3, 11), date(2024, 3, 17)),
        ("later that month", date(2024, 3, 1), date(2024, 3, 31)),
        ("that same year", date(2024, 1, 1), date(2024, 12, 31)),
        ("that same day", date(2024, 3, 12), date(2024, 3, 12)),
        ("later that same evening", date(2024, 3, 12), date(2024, 3, 12)),
        ("Two days later at 3 PM", date(2024, 3, 14), date(2024, 3, 14)),
    ],
)
def test_more_relative_phrases(text: str, start: date, end: date) -> None:
    spec = parse_relative(text)
    assert spec is not None and not spec.deictic
    anchor = date(2024, 3, 12)
    assert spec.apply((anchor, anchor), True) == (start, end)


def test_earlier_this_week_is_relative_to_the_document_date() -> None:
    a = ev("a", "January 2, 2024", 0)
    b = ev("b", "earlier this week", 50)
    out = resolve([a, b], DOC_DATE)[1]  # Saturday, June 15, 2024
    assert (out.date_start, out.date_end) == (date(2024, 6, 10), date(2024, 6, 16))
    assert out.anchor_key is None


def test_earlier_in_the_week_anchors_to_prior_draft() -> None:
    a = ev("a", "March 15, 2024", 0, "Officer Ortiz interviewed Becker by phone.")
    b = ev("b", "earlier in the week", 80, "Becker heard raised voices.")
    out = resolve([a, b], date(2024, 3, 18))[1]
    assert (out.date_start, out.date_end) == (date(2024, 3, 11), date(2024, 3, 17))
    assert out.anchor_key == "a" and out.date_certainty == RELATIVE


def test_earlier_that_day_anchors_to_prior_draft() -> None:
    a = ev("a", "March 14, 2024 at 5:40 PM", 0, "officer responded")
    b = ev("b", "earlier that day", 80, "she was sent home")
    out = resolve([a, b], date(2024, 3, 18))[1]
    assert out.date_start == date(2024, 3, 14) and out.anchor_key == "a"


# --------------------------------------------------------------- open-ended bounds


@pytest.mark.parametrize(
    "bound", ["before March 10, 2024", "after March 10, 2024", "by March 10, 2024", "since 2006"]
)
def test_open_bound_is_never_an_anchor(bound: str) -> None:
    a = ev("a", "March 4, 2024", 0)
    b = ev("b", bound, 50)
    c = ev("c", "two days later", 100)
    out = resolve([a, b, c], DOC_DATE)
    assert out[2].anchor_key == "a"
    assert out[2].date_start == date(2024, 3, 6)


def test_open_bound_is_never_a_named_anchor() -> None:
    bound = ev("b", "before March 10, 2024", 0, "The inspection happened before the audit.")
    audit = ev("a", "March 20, 2024", 50, "The audit of the warehouse was completed.")
    rel = ev("r", "two days after the inspection", 100, "The report was filed.")
    out = resolve([bound, audit, rel], DOC_DATE)[2]
    assert out.anchor_key != "b"


def test_open_bound_fact_is_never_an_anchor() -> None:
    fact = make_fact("F-0001", "Court holds hearing", date(2024, 5, 10)).model_copy(
        update={"date_text": "after May 10, 2024"}
    )
    rel = ev("r", "two days after the hearing", 0, "Counsel files a reply.")
    out = resolve([rel], None, [fact])[0]
    assert out.date_start is None and out.anchor_fact_id is None


@pytest.mark.parametrize(
    ("text", "bound"),
    [
        ("before March 3, 2024", True),
        ("shortly after March 3, 2024", False),  # a near window, not open-ended
        ("since 2006", True),
        ("no later than April 1, 2024", True),
        ("the following week", False),
        ("following the meeting", True),
        ("March 3, 2024", False),
        (None, False),
    ],
)
def test_is_open_bound(text: str | None, bound: bool) -> None:
    assert is_open_bound(text) is bound


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
        "approximately 6:15 PM",
    ],
)
def test_time_only_is_never_a_year(text: str) -> None:
    assert parse_absolute(text) is None
    spec = parse_relative(text)
    assert spec is not None and spec.clock
    # With nothing else to go on, a bare time falls back to the document's own day.
    out = one(text)
    assert out.date_start == out.date_end == DOC_DATE and out.date_certainty == APPROXIMATE
    assert one(text, doc_date=None).date_start is None


@pytest.mark.parametrize("text", ["about 2005", "in 1999", "2010"])
def test_plausible_years_stay_years(text: str) -> None:
    found = parse_absolute(text)
    assert found is not None and found[0][0].month == 1 and found[0][1].month == 12


@pytest.mark.parametrize(
    "text",
    [
        "March 3, 2024 between 2 and 3 pm",
        "between 9 am and 5 pm on March 3, 2024",
        "from 9:00 to 11:30 on March 3, 2024",
    ],
)
def test_time_ranges_within_a_day(text: str) -> None:
    assert parse_absolute(text, None) == ((date(2024, 3, 3), date(2024, 3, 3)), EXACT)


# ------------------------------------------------- times of day, incident date, time zone

ACCIDENT = date(2010, 7, 27)


def doc(text: str, doc_id: str = "D-0001") -> ParsedDocument:
    return ParsedDocument(
        doc=Document(
            doc_id=doc_id,
            case_id="ntsb",
            filename="report.pdf",
            sha256="0" * 64,
            location="evidence",
            kind="other",
            pages=1,
            parse_status="parsed",
            added_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        text=text,
    )


def located(key: str, date_text: str, text: str, quote: str, title: str = "x") -> DraftFact:
    start = text.index(quote)
    return DraftFact(
        key=key,
        title=title,
        date_text=date_text,
        quotes=[quote],
        spans=[SpanIn(doc_id="D-0001", char_start=start, char_end=start + len(quote), quote=quote)],
    )


def test_time_takes_the_day_stated_in_its_sentence() -> None:
    text = "On July 27, 2010, at 1729, N6JR departed Willow Run Airport for Oshkosh."
    d = located("a", "at 1729", text, "at 1729, N6JR departed Willow Run Airport")
    out = resolve_drafts([d], date(2010, 9, 1), [], parsed=doc(text))[0]
    assert out.date_start == out.date_end == ACCIDENT and out.date_certainty == EXACT


def test_times_continue_the_narratives_last_explicit_date() -> None:
    text = (
        "HISTORY OF FLIGHT\n"
        "On July 27, 2010, about 1817 central daylight time, a Premier jet crashed at Oshkosh.\n"
        "At 1810:33 the pilot called the tower.\n"
        "At 1816:19 the pilot asked whether 6JR was going to be okay.\n"
    )
    a = located("a", "1810:33", text, "the pilot called the tower")
    b = located("b", "1816:19", text, "the pilot asked whether 6JR")
    out = resolve_drafts([a, b], date(2010, 9, 1), [], parsed=doc(text))
    assert [o.date_start for o in out] == [ACCIDENT, ACCIDENT]
    assert all(o.date_certainty == APPROXIMATE for o in out)


def test_a_time_never_continues_a_year_wide_span() -> None:
    a = ev("a", "since 2006", 0)
    b = ev("b", "at 1729", 50)
    out = resolve([a, b], DOC_DATE)
    assert out[1].date_start == DOC_DATE  # not January 1, 2006


def test_at_the_time_of_the_accident_is_the_incident_date() -> None:
    d = ev("a", "at the time of the accident", 40)
    out = resolve_drafts([d], date(2010, 8, 17), [], incident_date=ACCIDENT)[0]
    assert out.date_start == out.date_end == ACCIDENT and out.date_certainty == RELATIVE


def test_incident_date_is_read_from_the_document_header() -> None:
    text = "Date of Accident: 07/27/2010\nLocation: Oshkosh, WI\n\nAt 1816:57 the airplane crashed."
    d = located("a", "1816:57", text, "the airplane crashed")
    out = resolve_drafts([d], date(2010, 8, 2), [], parsed=doc(text))[0]
    assert out.date_start == ACCIDENT


INTERVIEW = (  # laid out like the NTSB docket's interview transcript
    "NATIONAL TRANSPORTATION SAFETY BOARD\n"
    "Investigation of: AIRPLANE ACCIDENT, OSHKOSH, WISCONSIN\n"
    "Interview of: JACK ROUSH\n"
    "Interview conducted by telephone\n"
    "Monday,\n"
    "August 30, 2010\n\n"
    "Q. Tell us about the accident flight.\n"
    "A. That day I cancelled the IFR flight plan to Milwaukee and continued to Oshkosh.\n"
    "A. The stick shaker and the pusher went off as the right wing stalled.\n"
)


def test_an_interview_dates_its_account_by_the_case_incident() -> None:
    held = located("held", "August 30, 2010", INTERVIEW, "Interview conducted by telephone")
    cancel = located("cancel", "that day", INTERVIEW, "That day I cancelled the IFR flight plan")
    stall = located("stall", "that day", INTERVIEW, "the right wing stalled")
    drafts = [held, cancel, stall]
    out = resolve_drafts(
        drafts, date(2010, 8, 30), [], parsed=doc(INTERVIEW), case_incident_date=ACCIDENT
    )
    assert out[0].date_start == date(2010, 8, 30)  # the interview itself keeps its own date
    assert out[1].date_start == out[2].date_start == ACCIDENT
    # Without the case's incident date, "that day" can only mean the interview's day.
    plain = resolve_drafts(drafts, date(2010, 8, 30), [], parsed=doc(INTERVIEW))
    assert plain[1].date_start == date(2010, 8, 30)


def test_case_incident_date_needs_a_document_about_an_incident() -> None:
    text = "Memo dated March 3, 2024.\nThat day the board met to approve the budget.\n"
    d = located("a", "that day", text, "the board met to approve the budget")
    out = resolve_drafts([d], date(2024, 3, 3), [], parsed=doc(text), case_incident_date=ACCIDENT)
    assert out[0].date_start == date(2024, 3, 3)


@pytest.mark.parametrize(
    ("text", "tz", "day"),
    [
        ("July 28, 2010, 0016 UTC", "America/Chicago", date(2010, 7, 27)),
        ("July 28, 2010 at 00:16Z", "America/Chicago", date(2010, 7, 27)),
        ("July 28, 2010, 0016 UTC", None, date(2010, 7, 28)),  # no case zone: as written
        ("July 28, 2010, 0016 cdt", "America/Chicago", date(2010, 7, 28)),  # already local
        ("July 28, 2010, 1816", "America/Chicago", date(2010, 7, 28)),  # no zone: never moved
        ("July 27, 2010, 2316 UTC/1816 CDT", "America/Chicago", date(2010, 7, 27)),
    ],
)
def test_utc_times_move_to_the_case_time_zone(text: str, tz: str | None, day: date) -> None:
    found = parse_absolute(text, tz=tz)
    assert found is not None and found[0] == (day, day)


def test_unknown_case_time_zone_is_ignored() -> None:
    found = parse_absolute("July 28, 2010, 0016 UTC", tz="Mars/Olympus")
    assert found is not None and found[0][0] == date(2010, 7, 28)


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("since 2006", date(2006, 1, 1), date(2006, 12, 31)),
        ("starting in March 2007", date(2007, 3, 1), date(2007, 3, 31)),
        ("from 2006", date(2006, 1, 1), date(2006, 12, 31)),
    ],
)
def test_a_start_point_in_a_year_or_month_is_the_whole_period(
    text: str, start: date, end: date
) -> None:
    found = parse_absolute(text)
    assert found == ((start, end), APPROXIMATE)
    assert is_open_bound(text)


def test_year_only_is_the_whole_year_never_january_first() -> None:
    out = one("2010")
    assert (out.date_start, out.date_end) == (date(2010, 1, 1), date(2010, 12, 31))
    assert out.date_certainty == APPROXIMATE


def test_alternatives_span_both_dates() -> None:
    assert parse_absolute("May or June of 2007") == (
        (date(2007, 5, 1), date(2007, 6, 30)),
        APPROXIMATE,
    )


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("two hours later", date(2024, 3, 3), date(2024, 3, 4)),  # may cross midnight
        ("15 seconds later", date(2024, 3, 3), date(2024, 3, 3)),
        ("30 minutes before", date(2024, 3, 3), date(2024, 3, 3)),
    ],
)
def test_sub_day_offsets_stay_on_or_near_the_anchor_day(text: str, start: date, end: date) -> None:
    a = ev("a", "March 3, 2024", 0)
    b = ev("b", text, 50)
    out = resolve([a, b])[1]
    assert (out.date_start, out.date_end) == (start, end) and out.anchor_key == "a"


def test_after_that_refers_back_to_the_previous_fact() -> None:
    a = ev("a", "March 3, 2024", 0, title="the board met")
    b = ev("b", "March 10, 2024", 50, title="that was approved")
    c = ev("c", "shortly after that", 100)
    out = resolve([a, b, c])[2]
    assert out.anchor_key == "b" and out.date_start == date(2024, 3, 10)


def test_on_this_date_is_the_date_just_mentioned_not_the_documents() -> None:
    a = ev("a", "March 3, 2024", 0)
    b = ev("b", "on this date", 50)
    out = resolve([a, b])[1]
    assert out.date_start == date(2024, 3, 3)


def test_ocr_date_run_into_a_time_still_parses() -> None:
    found = parse_absolute("07/02/202118:17")
    assert found is not None and found[0] == (date(2021, 7, 2), date(2021, 7, 2))
