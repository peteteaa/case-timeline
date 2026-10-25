"""Tests for pipeline.parse (no LLM, no network)."""

from __future__ import annotations

import hashlib
from datetime import date
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import parse as parse_mod
from pipeline.contracts import Document
from pipeline.parse import discover, doc_date, normalize_text, parse

FIXTURES = Path(__file__).parent / "fixtures" / "case_demo"
EXPECTED = {
    "deposition_excerpt_ellison.txt": "text/plain",
    "email_thread.eml": "message/rfc822",
    "hr_memo.md": "text/markdown",
    "incident_report.pdf": "application/pdf",
    "witness_statement_becker.docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
}


@pytest.fixture(autouse=True)
def _no_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep PDF output deterministic even if Docling gets installed later."""
    monkeypatch.setenv("CT_DISABLE_DOCLING", "1")


def _docs() -> dict[str, Document]:
    return {d.filename: d for d in discover(FIXTURES, "demo", inbox_dirs=[])}


def _snapshot(folder: Path) -> dict[str, tuple[str, int]]:
    return {
        p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(folder.iterdir())
    }


def test_normalize_text() -> None:
    raw = "a\r\nb\rc\x00d é"
    assert normalize_text(raw) == "a\nb\ncd é"


def test_discover_fixtures() -> None:
    docs = _docs()
    assert {n: d.mime for n, d in docs.items()} == EXPECTED
    for d in docs.values():
        data = Path(d.path).read_bytes()
        assert d.sha256 == hashlib.sha256(data).hexdigest()
        assert d.id == d.sha256[:16]
        assert d.case_id == "demo"
        assert Path(d.path).is_absolute()


def test_discover_skips_hidden_unsupported_and_duplicates(tmp_path: Path) -> None:
    ev = tmp_path / "evidence"
    (ev / "sub").mkdir(parents=True)
    (ev / "a.txt").write_text("alpha")
    (ev / "sub" / "b.MD").write_text("beta")
    (ev / ".hidden.txt").write_text("hidden")
    (ev / "~$lock.docx").write_bytes(b"lock")
    (ev / "image.png").write_bytes(b"png")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "dup.txt").write_text("alpha")  # same bytes as a.txt
    (inbox / "new.eml").write_text("Subject: hi\n\nbody")
    docs = discover(ev, "c1", inbox_dirs=[inbox, tmp_path / "missing"])
    assert [d.filename for d in docs] == ["a.txt", "b.MD", "new.eml"]
    assert docs[1].mime == "text/markdown"


def test_discover_default_inbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pipeline.config

    inbox = tmp_path / "inbox"
    (inbox / "demo").mkdir(parents=True)
    (inbox / "other").mkdir()
    (inbox / "loose.txt").write_text("loose")
    (inbox / "demo" / "mine.txt").write_text("mine")
    (inbox / "other" / "theirs.txt").write_text("theirs")
    monkeypatch.setattr(pipeline.config, "get_settings", lambda: SimpleNamespace(inbox=inbox))
    ev = tmp_path / "ev"
    ev.mkdir()
    names = {d.filename for d in discover(ev, "demo")}
    assert names == {"loose.txt", "mine.txt"}


def test_parse_pdf_pages() -> None:
    parsed = parse(_docs()["incident_report.pdf"])
    assert parsed.doc.num_pages == 2
    assert [p.number for p in parsed.pages] == [1, 2]
    p1, p2 = parsed.pages
    assert p1.char_start == 0 and p1.char_end <= p2.char_start
    assert p2.char_end == len(parsed.text)
    assert parsed.text[p1.char_start : p1.char_end].startswith("LAKEVIEW POLICE DEPARTMENT\n")
    page2 = parsed.text[p2.char_start : p2.char_end]
    assert page2.startswith("LAKEVIEW POLICE DEPARTMENT - Report No. LPD-24-03318 (continued)")
    assert "approved by Sgt. Karen Whitfield on March 18, 2024." in page2
    assert "March 15, 2024" not in parsed.text[: p1.char_end]


def test_parse_docx_includes_tables() -> None:
    parsed = parse(_docs()["witness_statement_becker.docx"])
    assert parsed.pages == []
    assert parsed.text.startswith("WITNESS STATEMENT OF TOM BECKER\n")
    assert "took place on March 11, 2024." in parsed.text
    assert "“done here.”" in parsed.text
    assert "Date signed | April 2, 2024" in parsed.text


def test_parse_eml_thread() -> None:
    parsed = parse(_docs()["email_thread.eml"])
    text = parsed.text
    assert "\r" not in text
    assert text.startswith("Date: Mon, 25 Mar 2024 09:14:00 -0400\nFrom: Priya Nand")
    assert "Subject: RE: Formal complaint - safety retaliation" in text
    assert "moved from March 27, 2024 to March 28, 2024" in text
    assert "On March 4, 2024 I reported in writing to Mark Ellison" in text


def test_parse_txt_md() -> None:
    docs = _docs()
    txt = parse(docs["deposition_excerpt_ellison.txt"])
    md = parse(docs["hr_memo.md"])
    assert "A. Priya Nand from HR sat in. Nobody else." in txt.text
    assert "The appeal must be received by May 6, 2024." in md.text
    assert "—" in md.text


def test_parse_normalizes_before_offsets(tmp_path: Path) -> None:
    f = tmp_path / "crlf.txt"
    f.write_bytes("﻿Line one\r\nCafé\x00 on March 3, 2024\r\n".encode())
    (doc,) = discover(tmp_path, "c", inbox_dirs=[])
    parsed = parse(doc)
    assert parsed.text == "Line one\nCafé on March 3, 2024\n"
    assert parsed.text.index("March") == 17


def test_parse_cp1252_fallback(tmp_path: Path) -> None:
    (tmp_path / "old.txt").write_bytes("Smith’s note".encode("cp1252"))
    (doc,) = discover(tmp_path, "c", inbox_dirs=[])
    assert parse(doc).text == "Smith’s note"


def test_parse_eml_html_only_with_attachment(tmp_path: Path) -> None:
    msg = EmailMessage()
    msg["From"] = "a@example.test"
    msg["To"] = "b@example.test"
    msg["Date"] = "Tue, 02 Apr 2024 10:00:00 +0000"
    msg["Subject"] = "Notes"
    msg.set_content("<p>Hello <b>there</b></p><script>x()</script><p>Second</p>", subtype="html")
    msg.add_attachment(b"%PDF-1.4", maintype="application", subtype="pdf", filename="x.pdf")
    (tmp_path / "m.eml").write_bytes(bytes(msg))
    (doc,) = discover(tmp_path, "c", inbox_dirs=[])
    parsed = parse(doc)
    assert "Hello there" in parsed.text
    assert "Second" in parsed.text
    assert "x()" not in parsed.text
    assert "[Attachment: x.pdf]" in parsed.text
    assert doc_date(parsed) == date(2024, 4, 2)


def test_parse_never_writes_evidence() -> None:
    before = _snapshot(FIXTURES)
    for d in _docs().values():
        parse(d)
    assert _snapshot(FIXTURES) == before


def test_parse_warns_on_changed_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    f = tmp_path / "a.txt"
    f.write_text("first")
    (doc,) = discover(tmp_path, "c", inbox_dirs=[])
    f.write_text("second")
    assert parse(doc).text == "second"
    assert "sha256 mismatch" in caplog.text


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
    assert doc_date(parse(_docs()[name])) == expected


def test_doc_date_none(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("No dates here.\nJust words.")
    (doc,) = discover(tmp_path, "c", inbox_dirs=[])
    assert doc_date(parse(doc)) is None


def test_docling_fallback_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CT_DISABLE_DOCLING", raising=False)
    monkeypatch.setattr(parse_mod, "_parse_pdf_docling", lambda path: None)
    parsed = parse(_docs()["incident_report.pdf"])
    assert len(parsed.pages) == 2


# --------------------------------------------------------------------------- real-world PDFs
# Synthetic PDFs built on the fly: layouts seen in real case files (court-reporter
# transcripts, forms, running headers, scans), with no case-specific content.


def _pdf(tmp_path: Path, pages: list[list[tuple[float, float, str]]], name: str = "x.pdf") -> Path:
    """Write a PDF whose pages hold (x, y, text) snippets; returns its path."""
    import pymupdf

    pdf = pymupdf.open()
    for snippets in pages:
        page = pdf.new_page(width=612, height=792)
        for x, y, text in snippets:
            page.insert_text((x, y), text, fontsize=11)
    path = tmp_path / name
    pdf.save(path)
    pdf.close()
    return path


def _image_pdf(tmp_path: Path, n_pages: int, text_pages: dict[int, str] | None = None) -> Path:
    """Pages that carry only an image (a "scan"); ``text_pages`` adds real text to some."""
    import pymupdf

    pdf = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), 0)
    pix.clear_with(200)
    for i in range(n_pages):
        page = pdf.new_page(width=612, height=792)
        page.insert_image(pymupdf.Rect(50, 50, 550, 700), pixmap=pix)
        page.insert_text((300, 760), "(R)", fontsize=8)  # stamp, as on redacted scans
        for j, line in enumerate((text_pages or {}).get(i, "").split("\n")):
            if line:
                page.insert_text((72, 100 + 14 * j), line, fontsize=11)
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    pdf.close()
    return path


def _parse_path(path: Path) -> object:
    (doc,) = [d for d in discover(path.parent, "c", inbox_dirs=[]) if d.filename == path.name]
    return parse(doc)


def _transcript_page(page_no: int, lines: list[str]) -> list[tuple[float, float, str]]:
    """Court-reporter layout: margin line numbers, a Q/A column, then the text column."""
    out: list[tuple[float, float, str]] = [(540, 40, str(page_no))]
    for i, line in enumerate(lines, start=1):
        y = 80 + 24 * i
        out.append((80, y, str(i)))
        if line[:2] in {"Q.", "A."}:
            out.append((120, y, line[:2]))
            out.append((160, y, line[3:]))
        else:
            out.append((160, y, line))
    out.append((200, 740, "Example Court Reporting, Inc."))
    return out


def test_transcript_rows_rejoined_and_line_numbers_stripped(tmp_path: Path) -> None:
    body = [
        "Q. Where were you on the morning in question?",
        "A. I was at the warehouse, near the north loading",
        "dock, from about eight o'clock.",
        "Q. Who else was there?",
        "A. Two drivers and the shift supervisor.",
        "Q. Did you speak with the supervisor?",
        "A. Only briefly, before the delivery arrived.",
        "Q. What did the supervisor say?",
        "A. That the truck was running late.",
        "Q. Anything else?",
    ]
    pages = [_transcript_page(n, body) for n in (5, 6, 7)]
    parsed = _parse_path(_pdf(tmp_path, pages))
    text = parsed.text
    assert "Q. Where were you on the morning in question?" in text
    assert "A. I was at the warehouse, near the north loading\ndock, from about" in text
    assert not any(ln.strip().isdigit() for ln in text.split("\n"))  # margin/page numbers
    assert text.count("Example Court Reporting, Inc.") == 1  # running footer kept once
    for page in parsed.pages:  # offsets still index the cleaned text
        assert text[page.char_start : page.char_end].strip()
    assert parsed.pages[-1].char_end == len(text)


def test_form_labels_joined_with_values(tmp_path: Path) -> None:
    parsed = _parse_path(
        _pdf(tmp_path, [[(72, 100, "Date of Report:"), (250, 100, "June 3, 2021"), (72, 130, "x")]])
    )
    assert "Date of Report: June 3, 2021" in parsed.text
    assert doc_date(parsed) == date(2021, 6, 3)


def test_two_column_prose_not_merged(tmp_path: Path) -> None:
    import pymupdf

    left = "The left column discusses the first topic in some detail here. " * 6
    right = "The right column covers an unrelated second subject entirely. " * 6
    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(40, 60, 296, 400), left, fontsize=10)
    page.insert_textbox(pymupdf.Rect(316, 60, 572, 400), right, fontsize=10)
    path = tmp_path / "cols.pdf"
    pdf.save(path)
    pdf.close()
    text = _parse_path(path).text
    assert "first topic in some detail here." in text.replace("\n", " ")
    first_right = text.index("The right column")
    assert "left column" not in text[first_right:]  # columns read one after the other


def test_running_header_footer_and_page_numbers(tmp_path: Path) -> None:
    body = [
        "Officers arrived at the gate.",
        "Records were requested from the clinic.",
        "A neighbour gave a short statement.",
        "No further action was taken that week.",
    ]
    pages = []
    for n in range(1, 5):
        pages.append(
            [
                (72, 40, "COUNTY OFFICE OF EXAMPLE - CONFIDENTIAL"),
                (72, 56, f"Report No. 21-{n}004"),
                (72, 120, body[n - 1]),
                (72, 740, f"Page {n} of 4"),
            ]
        )
    parsed = _parse_path(_pdf(tmp_path, pages))
    text = parsed.text
    assert text.count("COUNTY OFFICE OF EXAMPLE - CONFIDENTIAL") == 1
    assert text.startswith("COUNTY OFFICE OF EXAMPLE - CONFIDENTIAL\nReport No. 21-1004\n")
    assert "Report No. 21-3004" not in text
    assert "Page 2 of 4" not in text and "Page 1 of 4" not in text
    for n in range(1, 5):
        page = parsed.pages[n - 1]
        assert body[n - 1] in text[page.char_start : page.char_end]


def test_torn_running_footer_fragments_removed(tmp_path: Path) -> None:
    pages = []
    for n in range(1, 9):
        footer = "Acme Transcription Services, Inc." if n % 2 else "Inc."
        body = ["Testimony resumed after a short recess.", "The witness was shown a map."]
        pages.append([(72, 100, body[n % 2]), (72, 120, "x" * n), (200, 740, footer)])
    text = _parse_path(_pdf(tmp_path, pages)).text
    assert text.count("Acme Transcription Services, Inc.") == 1
    assert "\nInc." not in text


def test_short_documents_keep_repeated_lines(tmp_path: Path) -> None:
    pages = [[(72, 100, "Signed: J. Doe")], [(72, 100, "Signed: J. Doe")]]
    assert _parse_path(_pdf(tmp_path, pages)).text.count("Signed: J. Doe") == 2


def test_scanned_pdf_without_ocr_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parse_mod, "ocr_engine", lambda: None)
    with pytest.raises(parse_mod.NeedsOCRError, match="needs OCR, no text layer: 3 of 3"):
        _parse_path(_image_pdf(tmp_path, 3))


def test_scanned_pdf_is_ocrd_when_engine_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_ocr(png: bytes, engine: str) -> str:
        calls.append(engine)
        assert png.startswith(b"\x89PNG")
        return f"Statement page {len(calls)}: the meeting happened on May 2, 2022."

    monkeypatch.setattr(parse_mod, "ocr_engine", lambda: "tesseract")
    monkeypatch.setattr(parse_mod, "_ocr_png", fake_ocr)
    parsed = _parse_path(_image_pdf(tmp_path, 2))
    assert calls == ["tesseract", "tesseract"]
    assert "Statement page 2: the meeting happened on May 2, 2022." in parsed.text
    assert len(parsed.pages) == 2


def test_partly_scanned_pdf_keeps_text_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(parse_mod, "ocr_engine", lambda: None)
    text_page = "This page has a real text layer with plenty of words in it."
    parsed = _parse_path(_image_pdf(tmp_path, 3, {1: text_page}))
    assert text_page in parsed.text
    assert "2 of 3 pages have no text layer" in caplog.text


def test_tesseract_engine_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_run(cmd: list[str], **kw: object) -> SimpleNamespace:
        seen["cmd"], seen["input"] = cmd, kw["input"]
        return SimpleNamespace(stdout=b"OCR text\n")

    monkeypatch.setattr(parse_mod.subprocess, "run", fake_run)
    assert parse_mod._ocr_png(b"PNG", "tesseract") == "OCR text\n"
    assert seen["cmd"][:3] == ["tesseract", "stdin", "stdout"] and seen["input"] == b"PNG"


def test_ocr_boxes_grouped_into_rows() -> None:
    items = [
        (300.0, 101.0, 10.0, "world"),
        (10.0, 100.0, 10.0, "Hello"),
        (10.0, 130.0, 10.0, "Next"),
    ]
    assert parse_mod._boxes_to_text(items) == "Hello world\nNext"


def test_redaction_marks_normalized(tmp_path: Path) -> None:
    (tmp_path / "r.txt").write_text("Phone: •••••  Address: ■■■ ■■ Main\nBullet • stays")
    (doc,) = discover(tmp_path, "c", inbox_dirs=[])
    text = parse(doc).text
    assert text == "Phone: [REDACTED]  Address: [REDACTED] Main\nBullet • stays"


def test_numbered_lists_are_not_transcripts() -> None:
    text = "1 Introduction\n2 Background\nSome prose here.\n3 Findings"
    assert parse_mod._strip_line_numbers(text) == text


def test_docling_pages_get_the_same_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CT_DISABLE_DOCLING", raising=False)
    bodies = ["Arrival noted.", "Second visit.\n\nNotes taken.", "Phone call.", "Exit."]
    pages = [f"Header Of The Example Firm\n\n{b}\n\n{n}" for n, b in enumerate(bodies, 1)]
    monkeypatch.setattr(parse_mod, "_parse_pdf_docling", lambda path: list(pages))
    parsed = parse(_docs()["incident_report.pdf"])
    assert parsed.text.count("Header Of The Example Firm") == 1
    assert "Exit." in parsed.text and "\n4" not in parsed.text
    assert len(parsed.pages) == 4


def test_vertical_margin_text_does_not_break_row_merge(tmp_path: Path) -> None:
    import pymupdf

    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    page.insert_text((72, 100), "Label:", fontsize=11)
    page.insert_text((200, 100), "value one", fontsize=11)
    page.insert_text((590, 600), "MARGIN STAMP 0001", fontsize=8, rotate=90)
    path = tmp_path / "stamp.pdf"
    pdf.save(path)
    pdf.close()
    text = _parse_path(path).text
    assert text.startswith("Label: value one")
    assert "MARGIN STAMP 0001" in text


# --------------------------------------------------------------------------- incident date


def _parsed_text(text: str, filename: str = "x.pdf"):
    from pipeline.contracts import ParsedDocument

    doc = Document(id="d1", case_id="c", path="/x", filename=filename, mime="x", sha256="x")
    return ParsedDocument(doc=doc, text=text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # explicit form labels
        ("CLAIM FORM\nDate of Loss: 03/14/2024\nDate of Report: 03/20/2024", date(2024, 3, 14)),
        ("Incident Date: June 5, 2021\nPrepared: June 30, 2021", date(2021, 6, 5)),
        (
            "4. LOCATION OF ACCIDENT 5. DATE/TIMEOF ACCIDENT/INCIDENT (UTC) 6. UNIT\n"
            "Springfield depot June 6, 2021,0410 UTC",
            date(2021, 6, 6),
        ),
        ("Occurred On: 6/5/21 Report No. 7", date(2021, 6, 5)),
        # a where/when block under the report's own date
        (
            "Regional Safety Office\nSeptember 9, 2022\nFACTUAL REPORT\n"
            "Location: Springfield Date: June 5, 2021 Time: 0940 local\n",
            date(2021, 6, 5),
        ),
        ("March 1, 2022\nB. VEHICLE ACCIDENT\nDate: June 5, 2021\nUnit: 12", date(2021, 6, 5)),
        # an opening sentence
        (
            "September 2, 2021\nDear Ms. Ortiz:\nThis concerns the collision involving my\n"
            "truck on Saturday, June 5, 2021. I was driving north.",
            date(2021, 6, 5),
        ),
        ("Statement\nIncident occurred Saturday, June 5, 2021 near the depot.", date(2021, 6, 5)),
    ],
)
def test_incident_date_found(text: str, expected: date) -> None:
    assert parse_mod.incident_date(_parsed_text(text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "MEMORANDUM\nDate: April 8, 2024\nRe: attendance\nOn March 12, 2024 we met.",
        "INCIDENT REPORT\nDate: March 18, 2024\nThe officer responded.",  # the report's date
        "Date of Report: March 18, 2024\nReporting Officer: J. Ortiz",
        "Inc Occurred End:\nReported Date/Time: 07/27/2021 18:17",
        "No dates at all here.",
    ],
)
def test_incident_date_not_found(text: str) -> None:
    assert parse_mod.incident_date(_parsed_text(text)) is None


def test_doc_date_skips_the_incident_date_field() -> None:
    text = (
        "Regional Safety Office\nSeptember 9, 2022\nFACTUAL REPORT\nB. VEHICLE ACCIDENT\n"
        "Location: Springfield\nDate: June 5, 2021\nTime: 0940 local\n"
    )
    parsed = _parsed_text(text)
    assert doc_date(parsed) == date(2022, 9, 9)
    assert parse_mod.incident_date(parsed) == date(2021, 6, 5)


def test_doc_date_accepts_dates_without_space_after_comma() -> None:
    assert doc_date(_parsed_text("Letter\nJune 5,2021\nDear Sir")) == date(2021, 6, 5)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("incident_report.pdf", date(2024, 3, 14)),  # "Incident Type: ... On March 14, 2024"
        ("witness_statement_becker.docx", None),
        ("email_thread.eml", None),
        ("deposition_excerpt_ellison.txt", None),
        ("hr_memo.md", None),
    ],
)
def test_incident_date_fixtures(name: str, expected: date | None) -> None:
    assert parse_mod.incident_date(parse(_docs()[name])) == expected
