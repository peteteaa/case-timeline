"""Tests for pipeline.parse (no LLM, no network)."""

from __future__ import annotations

import hashlib
from datetime import date
from email.message import EmailMessage
from pathlib import Path

import pytest

from pipeline import parse as parse_mod
from pipeline.parse import (
    FoundFile,
    detect_kind,
    discover,
    doc_date,
    normalize_text,
    parse_file,
)
from pipeline.schema import ParsedDocument
from tests.pipeline_helpers import make_doc

FIXTURES = Path(__file__).parent / "fixtures" / "case_demo"
FIXTURE_NAMES = {
    "deposition_excerpt_ellison.txt",
    "email_thread.eml",
    "hr_memo.md",
    "incident_report.pdf",
    "witness_statement_becker.docx",
}


@pytest.fixture(autouse=True)
def _no_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep PDF output deterministic even if Docling gets installed later."""
    monkeypatch.setenv("CT_DISABLE_DOCLING", "1")


def _parse(path: Path) -> ParsedDocument:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    doc = make_doc("D-0001", path.name).model_copy(update={"sha256": sha})
    return parse_file(path, doc)


def _fixture(name: str) -> ParsedDocument:
    return _parse(FIXTURES / name)


def _snapshot(folder: Path) -> dict[str, tuple[str, int]]:
    return {
        p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(folder.iterdir())
    }


def test_normalize_text() -> None:
    raw = "a\r\nb\rc\x00d é"
    assert normalize_text(raw) == "a\nb\ncd é"


# --------------------------------------------------------------------------- discover


def test_discover_fixtures() -> None:
    found = discover(FIXTURES, None)
    assert {f.rel_path for f in found} == FIXTURE_NAMES
    for f in found:
        assert isinstance(f, FoundFile)
        assert f.location == "evidence"
        assert f.sha256 == hashlib.sha256(f.path.read_bytes()).hexdigest()
        assert f.path == FIXTURES / f.rel_path


def test_discover_skips_hidden_unsupported_and_duplicates(tmp_path: Path) -> None:
    ev = tmp_path / "evidence"
    (ev / "sub").mkdir(parents=True)
    (ev / ".git").mkdir()
    (ev / "a.txt").write_text("alpha")
    (ev / "sub" / "b.MD").write_text("beta")
    (ev / ".hidden.txt").write_text("hidden")
    (ev / ".git" / "c.txt").write_text("in a hidden folder")
    (ev / "~$lock.docx").write_bytes(b"lock")
    (ev / "image.png").write_bytes(b"png")
    inbox = tmp_path / "inbox"
    (inbox / "nested").mkdir(parents=True)
    (inbox / "dup.txt").write_text("alpha")  # same bytes as a.txt
    (inbox / "nested" / "new.eml").write_text("Subject: hi\n\nbody")
    found = discover(ev, inbox)
    assert [(f.rel_path, f.location) for f in found] == [
        ("a.txt", "evidence"),
        ("sub/b.MD", "evidence"),
        ("nested/new.eml", "inbox"),
    ]


def test_discover_missing_folders(tmp_path: Path) -> None:
    assert discover(tmp_path / "nope", tmp_path / "missing") == []


# --------------------------------------------------------------------------- formats


def test_parse_pdf_pages() -> None:
    parsed = _fixture("incident_report.pdf")
    assert parsed.doc.pages == 2
    assert [p.number for p in parsed.pages] == [1, 2]
    p1, p2 = parsed.pages
    assert p1.char_start == 0 and p1.char_end <= p2.char_start
    assert p2.char_end == len(parsed.text)
    assert parsed.text[p1.char_start : p1.char_end].startswith("LAKEVIEW POLICE DEPARTMENT\n")
    page2 = parsed.text[p2.char_start : p2.char_end]
    assert page2.startswith("LAKEVIEW POLICE DEPARTMENT - Report No. LPD-24-03318 (continued)")
    assert "approved by Sgt. Karen Whitfield on March 18, 2024." in page2
    assert "March 15, 2024" not in parsed.text[: p1.char_end]
    assert parsed.lines == []  # not a transcript


def test_parse_docx_includes_tables() -> None:
    parsed = _fixture("witness_statement_becker.docx")
    assert parsed.pages == []
    assert parsed.text.startswith("WITNESS STATEMENT OF TOM BECKER\n")
    assert "took place on March 11, 2024." in parsed.text
    assert "“done here.”" in parsed.text
    assert "Date signed | April 2, 2024" in parsed.text


def test_parse_eml_thread() -> None:
    text = _fixture("email_thread.eml").text
    assert "\r" not in text
    assert text.startswith("Date: Mon, 25 Mar 2024 09:14:00 -0400\nFrom: Priya Nand")
    assert "Subject: RE: Formal complaint - safety retaliation" in text
    assert "moved from March 27, 2024 to March 28, 2024" in text
    assert "On March 4, 2024 I reported in writing to Mark Ellison" in text


def test_parse_txt_md() -> None:
    txt = _fixture("deposition_excerpt_ellison.txt")
    md = _fixture("hr_memo.md")
    assert "A. Priya Nand from HR sat in. Nobody else." in txt.text
    assert "The appeal must be received by May 6, 2024." in md.text
    assert "—" in md.text
    assert md.pages == [] and md.lines == []


def test_parse_normalizes_before_offsets(tmp_path: Path) -> None:
    f = tmp_path / "crlf.txt"
    f.write_bytes("﻿Line one\r\nCafé\x00 on March 3, 2024\r\n".encode())
    parsed = _parse(f)
    assert parsed.text == "Line one\nCafé on March 3, 2024\n"
    assert parsed.text.index("March") == 17


def test_parse_cp1252_fallback(tmp_path: Path) -> None:
    (tmp_path / "old.txt").write_bytes("Smith’s note".encode("cp1252"))
    assert _parse(tmp_path / "old.txt").text == "Smith’s note"


def test_parse_eml_html_only_with_attachment(tmp_path: Path) -> None:
    msg = EmailMessage()
    msg["From"] = "a@example.test"
    msg["To"] = "b@example.test"
    msg["Date"] = "Tue, 02 Apr 2024 10:00:00 +0000"
    msg["Subject"] = "Notes"
    msg.set_content("<p>Hello <b>there</b></p><script>x()</script><p>Second</p>", subtype="html")
    msg.add_attachment(b"%PDF-1.4", maintype="application", subtype="pdf", filename="x.pdf")
    (tmp_path / "m.eml").write_bytes(bytes(msg))
    parsed = _parse(tmp_path / "m.eml")
    assert "Hello there" in parsed.text
    assert "Second" in parsed.text
    assert "x()" not in parsed.text
    assert "[Attachment: x.pdf]" in parsed.text
    assert doc_date(parsed) == date(2024, 4, 2)


def test_parse_survives_awkward_real_world_files(tmp_path: Path) -> None:
    import pymupdf

    files = {
        "empty.docx": b"",
        "empty.pdf": b"",
        "utf16.txt": "On March 4, 2024 Zoë left.".encode("utf-16"),
        "cp1252.txt": "José “quoted”".encode("cp1252"),
        "bogus.eml": b"Subject: hi\nContent-Type: text/plain; charset=bogus-charset\n\nWe met.\n",
    }
    blank = pymupdf.open()
    blank.new_page()
    files["scanned.pdf"] = blank.tobytes()
    texts = {}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
        texts[name] = _parse(tmp_path / name).text
    assert texts["empty.docx"] == texts["empty.pdf"] == texts["scanned.pdf"] == ""
    assert texts["utf16.txt"] == "On March 4, 2024 Zoë left."
    assert texts["cp1252.txt"] == "José “quoted”"
    assert "We met." in texts["bogus.eml"]


def test_parse_unsupported_type_raises(tmp_path: Path) -> None:
    (tmp_path / "x.rtf").write_text("hello")
    with pytest.raises(ValueError):
        _parse(tmp_path / "x.rtf")


def test_parse_never_writes_evidence() -> None:
    before = _snapshot(FIXTURES)
    for name in FIXTURE_NAMES:
        _fixture(name)
    assert _snapshot(FIXTURES) == before


def test_parse_warns_on_changed_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    f = tmp_path / "a.txt"
    f.write_text("first")
    doc = make_doc("D-0001", "a.txt").model_copy(
        update={"sha256": hashlib.sha256(b"first").hexdigest()}
    )
    f.write_text("second")
    assert parse_file(f, doc).text == "second"
    assert "sha256 mismatch" in caplog.text


def test_docling_fallback_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CT_DISABLE_DOCLING", raising=False)
    monkeypatch.setattr(parse_mod, "_parse_pdf_docling", lambda path: None)
    assert len(_fixture("incident_report.pdf").pages) == 2


# --------------------------------------------------------------------------- transcripts


def test_page_markers_and_counted_lines_in_deposition() -> None:
    parsed = _fixture("deposition_excerpt_ellison.txt")
    assert [p.number for p in parsed.pages] == [14, 15]
    assert parsed.doc.pages == 2
    p14, p15 = parsed.pages
    assert parsed.text[p14.char_start :].startswith("[Page 14]")
    assert parsed.text[: p14.char_start].startswith("DEPOSITION OF MARK ELLISON")
    # No printed line numbers: non-empty lines are counted 1..n within each page.
    by_ref = {(ln.page, ln.number): parsed.text[ln.char_start : ln.char_end] for ln in parsed.lines}
    assert by_ref[(14, 1)] == "Q. When did your meeting with Ms. Rivera take place?"
    assert by_ref[(14, 4)] == "A. Priya Nand from HR sat in. Nobody else."
    assert by_ref[(15, 6)] == "A. April 5, 2024."
    assert max(n for p, n in by_ref if p == 14) == 6
    assert all("[Page" not in t for t in by_ref.values())


def test_page_markers_without_transcript_get_no_lines(tmp_path: Path) -> None:
    f = tmp_path / "scan_notes.txt"
    f.write_text("[Page 1]\nThe lease was signed.\n\n[Page 2]\nRent was paid.\n")
    parsed = _parse(f)
    assert [p.number for p in parsed.pages] == [1, 2]
    assert parsed.lines == []


def test_form_feed_pages(tmp_path: Path) -> None:
    f = tmp_path / "pages.txt"
    f.write_text("first page text\n\fsecond page text\n\fthird")
    parsed = _parse(f)
    assert [p.number for p in parsed.pages] == [1, 2, 3]
    assert [parsed.text[p.char_start : p.char_end] for p in parsed.pages] == [
        "first page text",
        "second page text",
        "third",
    ]


def _numbered_pdf(path: Path, inline: bool) -> None:
    import pymupdf

    pdf = pymupdf.open()
    for page_no in (14, 15):
        page = pdf.new_page()
        page.insert_text((300, 40), str(page_no))  # printed page number
        for i in range(1, 9):
            body = f"Q. Question {page_no}-{i} about the forklift?"
            if inline:
                page.insert_text((40, 60 + i * 20), f"{i:>2}   {body}")
            else:
                page.insert_text((40, 60 + i * 20), f"{i:>2}")
                page.insert_text((80, 60 + i * 20), body)
    path.write_bytes(pdf.tobytes())


@pytest.mark.parametrize("inline", [True, False])
def test_printed_line_numbers_in_pdf_transcript(tmp_path: Path, inline: bool) -> None:
    f = tmp_path / "depo.pdf"
    _numbered_pdf(f, inline)
    parsed = _parse(f)
    assert len(parsed.pages) == 2
    lines = {(ln.page, ln.number): parsed.text[ln.char_start : ln.char_end] for ln in parsed.lines}
    assert len(lines) == 16
    assert "Question 14-3 about" in lines[(1, 3)]
    assert "Question 15-8 about" in lines[(2, 8)]


def test_printed_line_numbers_in_txt(tmp_path: Path) -> None:
    body = "\n".join(f"{i:>4}  {'Q' if i % 2 else 'A'}. line {i}" for i in range(1, 26))
    f = tmp_path / "hearing.txt"
    f.write_text(f"[Page 3]\n{body}\n[Page 4]\n{body}\n")
    parsed = _parse(f)
    refs = [(ln.page, ln.number) for ln in parsed.lines]
    assert refs[:2] == [(3, 1), (3, 2)] and refs[-1] == (4, 25)
    assert len(refs) == 50
    (ln,) = [x for x in parsed.lines if (x.page, x.number) == (4, 7)]
    assert parsed.text[ln.char_start : ln.char_end].endswith("Q. line 7")


# --------------------------------------------------------------------------- kind and date


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("deposition_excerpt_ellison.txt", "transcript"),
        ("email_thread.eml", "email"),
        ("hr_memo.md", "other"),
        ("incident_report.pdf", "other"),
        ("witness_statement_becker.docx", "other"),
    ],
)
def test_detect_kind_fixtures(name: str, kind: str) -> None:
    assert detect_kind(name, _fixture(name).text) == kind


@pytest.mark.parametrize(
    ("filename", "text", "kind"),
    [
        ("first_amended_complaint.pdf", "", "pleading"),
        ("Answer and Counterclaims.docx", "", "pleading"),
        ("petition.pdf", "", "pleading"),
        ("def_motion_to_dismiss.pdf", "", "motion"),
        ("msj.pdf", "", "motion"),
        ("order_granting_msj.pdf", "", "order"),
        ("Exhibit_12.pdf", "", "exhibit"),
        ("ex3.pdf", "", "exhibit"),
        ("hearing_transcript.txt", "", "transcript"),
        ("scan001.pdf", "SUPERIOR COURT\nCOMPLAINT FOR DAMAGES\nPlaintiff alleges", "pleading"),
        ("scan002.pdf", "NOTICE OF MOTION AND MOTION TO COMPEL", "motion"),
        ("scan003.pdf", "Having considered the papers, IT IS HEREBY ORDERED that", "order"),
        ("scan004.pdf", "EXHIBIT B\n\nInvoice", "exhibit"),
        ("scan005.txt", "From: a@x.test\nTo: b@x.test\nSubject: hi\n\nBody", "email"),
        ("scan006.docx", "April 1, 2024\n\nDear Mr. Smith,\n\nWe write", "letter"),
        ("notes.txt", "Re: complaint investigation\nThe meeting happened.", "other"),
        ("notes2.txt", "Q. Where were you?\nA. Home.", "other"),
    ],
)
def test_detect_kind_heuristics(filename: str, text: str, kind: str) -> None:
    assert detect_kind(filename, text) == kind


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("incident_report.pdf", date(2024, 3, 18)),
        ("witness_statement_becker.docx", date(2024, 4, 2)),
        ("email_thread.eml", date(2024, 3, 25)),
        ("deposition_excerpt_ellison.txt", date(2024, 9, 10)),
        ("hr_memo.md", date(2024, 4, 8)),
    ],
)
def test_doc_date_fixtures(name: str, expected: date) -> None:
    assert doc_date(_fixture(name)) == expected


def test_doc_date_none(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("No dates here.\nJust words.")
    assert doc_date(_parse(tmp_path / "x.txt")) is None
