"""Tests for pipeline.verify, including that every gold quote grounds in the fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.contracts import Document, Event, EventStatus, Page, ParsedDocument, SourceRef
from pipeline.parse import discover, parse
from pipeline.verify import locate_quote, page_for_offset, verify_events

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "case_demo"
GOLD = ROOT / "eval" / "gold"

TEXT = (
    "Ms. Rivera stated that on\nMarch 12, 2024, during a meeting in Mr. Ellison’s office,\n"
    "Mr. Ellison  threatened to fire her if she “kept writing up the forklifts.”"
)


@pytest.fixture(scope="module")
def parsed_fixtures() -> dict[str, ParsedDocument]:
    mp = pytest.MonkeyPatch()
    mp.setenv("CT_DISABLE_DOCLING", "1")
    try:
        return {d.filename: parse(d) for d in discover(FIXTURES, "demo", inbox_dirs=[])}
    finally:
        mp.undo()


def _event(quote: str, *, confidence: float = 0.8, doc_id: str = "x") -> Event:
    return Event(
        id="e1",
        case_id="demo",
        date_text="March 12, 2024",
        description="meeting",
        source=SourceRef(doc_id=doc_id, char_start=0, char_end=0, quote=quote),
        confidence=confidence,
    )


def _parsed(text: str, pages: list[Page] | None = None) -> ParsedDocument:
    doc = Document(
        id="d1", case_id="demo", path="/x", filename="x.txt", mime="text/plain", sha256="0" * 64
    )
    return ParsedDocument(doc=doc, text=text, pages=pages or [])


def test_exact_match() -> None:
    q = "during a meeting in Mr. Ellison’s office"
    s, e = locate_quote(TEXT, q)
    assert TEXT[s:e] == q


def test_normalized_match_maps_to_original_offsets() -> None:
    q = "stated that on March 12, 2024, during a meeting in mr. ellison's office"
    s, e = locate_quote(TEXT, q)
    assert TEXT[s:e] == "stated that on\nMarch 12, 2024, during a meeting in Mr. Ellison’s office"


def test_normalized_curly_quotes_and_double_spaces() -> None:
    q = 'Mr. Ellison threatened to fire her if she "kept writing up the forklifts."'
    s, e = locate_quote(TEXT, q)
    assert TEXT[s:e].startswith("Mr. Ellison  threatened")
    assert TEXT[s:e].endswith("forklifts.”") or TEXT[s:e].endswith("forklifts.")


def test_surrounding_quote_marks_are_ignored() -> None:
    s, e = locate_quote(TEXT, "“March 12, 2024”")
    assert TEXT[s:e] == "March 12, 2024"


def test_elided_quote() -> None:
    q = "Ms. Rivera stated that ... threatened to fire her"
    s, e = locate_quote(TEXT, q)
    assert TEXT[s:e].startswith("Ms. Rivera stated that")
    assert TEXT[s:e].endswith("threatened to fire her")


def test_fuzzy_match_snaps_to_words() -> None:
    q = "during a meting in Mr. Elison's ofice"
    span = locate_quote(TEXT, q)
    assert span is not None
    s, e = span
    assert TEXT[s:e] == "during a meeting in Mr. Ellison’s office"


def test_not_found() -> None:
    assert locate_quote(TEXT, "the forklift was repaired on April 9, 2024") is None
    assert locate_quote(TEXT, "") is None
    assert locate_quote("", "anything") is None


def test_short_quote_needs_exact_or_normalized() -> None:
    assert locate_quote(TEXT, "Marhc 12") is None
    assert locate_quote(TEXT, "march 12") is not None


def test_page_for_offset() -> None:
    pages = [Page(number=1, char_start=0, char_end=10), Page(number=2, char_start=12, char_end=20)]
    assert page_for_offset(pages, 0) == 1
    assert page_for_offset(pages, 9) == 1
    assert page_for_offset(pages, 10) == 2  # separator gap belongs to the next page
    assert page_for_offset(pages, 15) == 2
    assert page_for_offset(pages, 20) == 2
    assert page_for_offset([], 3) is None


def test_verify_events_sets_offsets_and_page() -> None:
    text = "Page one text.\n\nOn page two the MEETING happened."
    pages = [
        Page(number=1, char_start=0, char_end=14),
        Page(number=2, char_start=16, char_end=len(text)),
    ]
    parsed = _parsed(text, pages)
    ev = _event("the meeting happened", confidence=0.9).model_copy(
        update={"status": EventStatus.ACCEPTED}
    )
    (out,) = verify_events(parsed, [ev])
    assert out.source.quote == "the MEETING happened"
    assert text[out.source.char_start : out.source.char_end] == out.source.quote
    assert out.source.page == 2
    assert out.source.doc_id == "d1"
    assert out.status == EventStatus.ACCEPTED
    assert out.confidence == 0.9
    assert ev.source.char_end == 0  # input not mutated


def test_verify_events_unverified() -> None:
    parsed = _parsed("Nothing relevant here at all.")
    (out,) = verify_events(parsed, [_event("a quote the model invented", confidence=0.8)])
    assert out.status == EventStatus.UNVERIFIED
    assert out.confidence == pytest.approx(0.4)
    assert out.source.doc_id == "d1"
    assert out.source.page is None
    assert (out.source.char_start, out.source.char_end) == (0, 0)
    assert out.source.quote == "a quote the model invented"


def test_gold_quotes_ground_in_fixtures(parsed_fixtures: dict[str, ParsedDocument]) -> None:
    gold = json.loads((GOLD / "demo.json").read_text(encoding="utf-8"))
    assert 25 <= len(gold) <= 35
    assert len({g["event_key"] for g in gold}) >= 15
    assert any(g["deadline_relevant"] for g in gold)
    certainties = {g["certainty"] for g in gold}
    assert {"exact", "approximate", "relative"} <= certainties
    for g in gold:
        parsed = parsed_fixtures[g["doc"]]
        events = [_event(g["quote"], doc_id=parsed.doc.id)]
        (out,) = verify_events(parsed, events)
        assert out.status != EventStatus.UNVERIFIED, g["quote"]
        if g["doc"].endswith(".pdf"):
            assert out.source.page in (1, 2)


def test_pdf_quotes_land_on_right_page(parsed_fixtures: dict[str, ParsedDocument]) -> None:
    parsed = parsed_fixtures["incident_report.pdf"]
    evs = [
        _event("only he and Ms. Rivera were present in the office"),  # wraps a line in the PDF
        _event("conducted a follow-up telephone interview with Tom Becker"),
    ]
    out = verify_events(parsed, evs)
    assert [e.source.page for e in out] == [1, 2]


def test_contradiction_quotes_ground_in_fixtures(
    parsed_fixtures: dict[str, ParsedDocument],
) -> None:
    items = json.loads((GOLD / "demo_contradictions.json").read_text(encoding="utf-8"))
    assert sum(1 for c in items if c["planted"]) >= 2
    assert {"date", "actor"} <= {c["field"] for c in items}
    for c in items:
        assert len(c["sides"]) >= 2
        for side in c["sides"]:
            text = parsed_fixtures[side["doc"]].text
            assert locate_quote(text, side["quote"]) is not None, side
