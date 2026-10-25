from datetime import date

import pytest

from pipeline.dates import parse_absolute, parse_relative, resolve_drafts
from pipeline.schema import DraftFact, SpanIn
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


def test_time_of_day_only_stays_undated() -> None:
    # "approximately 6:15 PM" names no day; guessing one from a nearby fact is too risky.
    a = ev("a", "March 14, 2024 at approximately 5:40 PM", 0)
    b = ev("b", "approximately 6:15 PM", 50)
    out = resolve([a, b], DOC_DATE)[1]
    assert out.date_start is None and out.date_certainty == UNKNOWN


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
