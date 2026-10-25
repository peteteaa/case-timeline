"""Tests for pipeline.verify, including that every gold quote grounds in the fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline.parse import parse_file
from pipeline.schema import DraftFact, Line, Page, ParsedDocument
from pipeline.verify import (
    lines_for_span,
    locate_quote,
    page_for_offset,
    verify_drafts,
)
from tests.pipeline_helpers import make_doc, make_parsed

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
        out = {}
        for i, path in enumerate(sorted(FIXTURES.iterdir()), start=1):
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            doc = make_doc(f"D-{i:04d}", path.name).model_copy(update={"sha256": sha})
            out[path.name] = parse_file(path, doc)
        return out
    finally:
        mp.undo()


def _gold(*names: str) -> list[dict]:
    """The eval's gold file (its name has changed over time); skip if it isn't there."""
    for name in names:
        if (GOLD / name).is_file():
            return json.loads((GOLD / name).read_text(encoding="utf-8"))
    pytest.skip(f"no gold file among {names}")


def _draft(*quotes: str, key: str = "k1") -> DraftFact:
    return DraftFact(key=key, title="meeting", quotes=list(quotes), confidence=0.8)


# --------------------------------------------------------------------------- locate_quote


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
    s, e = locate_quote(TEXT, "Ms. Rivera stated that ... threatened to fire her")
    assert TEXT[s:e].startswith("Ms. Rivera stated that")
    assert TEXT[s:e].endswith("threatened to fire her")


def test_fuzzy_match_snaps_to_words() -> None:
    span = locate_quote(TEXT, "during a meting in Mr. Elison's ofice")
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
    marker_pages = [Page(number=14, char_start=50, char_end=90)]
    assert page_for_offset(marker_pages, 10) is None  # caption before the first page marker
    assert page_for_offset(marker_pages, 60) == 14


def test_lines_for_span() -> None:
    lines = [Line(page=3, number=n, char_start=10 * n, char_end=10 * n + 8) for n in range(1, 5)]
    first, last = lines_for_span(lines, 15, 35)
    assert (first.number, last.number) == (1, 3)
    assert lines_for_span(lines, 0, 9) is None


# --------------------------------------------------------------------------- verify_drafts


def test_verify_drafts_sets_spans_and_page() -> None:
    text = "Page one text.\n\nOn page two the MEETING happened."
    pages = [
        Page(number=1, char_start=0, char_end=14),
        Page(number=2, char_start=16, char_end=len(text)),
    ]
    parsed = make_parsed(text, pages)
    draft = _draft("the meeting happened", "the meeting happened", "Page one text.")
    (out,), counts = verify_drafts(parsed, [draft])
    assert [s.quote for s in out.spans] == ["the MEETING happened", "Page one text."]
    for s in out.spans:
        assert text[s.char_start : s.char_end] == s.quote
        assert s.doc_id == "D-0001"
    assert [s.page for s in out.spans] == [2, 1]
    assert counts == {"quotes_checked": 3, "quotes_rejected": 0, "drafts_discarded": 0}
    assert draft.spans == []  # input not mutated
    assert out.confidence == 0.8 and out.quotes == draft.quotes


def test_verify_drops_bad_quotes_and_discards_empty_drafts() -> None:
    parsed = make_parsed("Smith signed the lease on March 3, 2024. Nothing else happened.")
    keep = _draft("Smith signed the lease", "a quote the model invented", key="a")
    gone = _draft("another invented sentence about rent", key="b")
    no_quotes = _draft(key="c")
    out, counts = verify_drafts(parsed, [keep, gone, no_quotes])
    assert [d.key for d in out] == ["a"]
    assert [s.quote for s in out[0].spans] == ["Smith signed the lease"]
    assert counts == {"quotes_checked": 3, "quotes_rejected": 2, "drafts_discarded": 2}


def test_verify_fills_transcript_lines(parsed_fixtures: dict[str, ParsedDocument]) -> None:
    parsed = parsed_fixtures["deposition_excerpt_ellison.txt"]
    draft = _draft(
        "A. Priya Nand from HR sat in. Nobody else.",
        "Q. When was Ms. Rivera's employment terminated?\nA. April 5, 2024.",
        "DEPOSITION OF MARK ELLISON",
    )
    (out,), _ = verify_drafts(parsed, [draft])
    refs = [(s.page, s.line_start, s.line_end) for s in out.spans]
    assert refs == [(14, 4, 4), (15, 5, 6), (None, None, None)]


def test_gold_quotes_ground_in_fixtures(parsed_fixtures: dict[str, ParsedDocument]) -> None:
    gold = _gold("demo_timeline.json", "demo.json")
    assert gold
    for g in gold:
        parsed = parsed_fixtures[g["doc"]]
        (out,), counts = verify_drafts(parsed, [_draft(g["quote"])])
        assert counts["quotes_rejected"] == 0, g["quote"]
        (span,) = out.spans
        assert parsed.text[span.char_start : span.char_end] == span.quote
        if g["doc"].endswith(".pdf"):
            assert span.page in (1, 2)
        if g["doc"].startswith("deposition") and span.char_start > parsed.pages[0].char_start:
            assert span.page in (14, 15) and span.line_start is not None


def test_pdf_quotes_land_on_right_page(parsed_fixtures: dict[str, ParsedDocument]) -> None:
    parsed = parsed_fixtures["incident_report.pdf"]
    drafts = [
        _draft("only he and Ms. Rivera were present in the office", key="a"),  # wraps a line
        _draft("conducted a follow-up telephone interview with Tom Becker", key="b"),
    ]
    out, _ = verify_drafts(parsed, drafts)
    assert [d.spans[0].page for d in out] == [1, 2]


def test_contradiction_quotes_ground_in_fixtures(
    parsed_fixtures: dict[str, ParsedDocument],
) -> None:
    items = _gold("demo_disputes.json", "demo_contradictions.json")
    for c in items:
        for side in c["sides"]:
            text = parsed_fixtures[side["doc"]].text
            assert locate_quote(text, side["quote"]) is not None, side
