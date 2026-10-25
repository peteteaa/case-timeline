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
