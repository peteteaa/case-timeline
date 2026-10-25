"""Discover case files and turn them into normalized text with page (and transcript line) offsets.

Evidence files are only ever opened for reading. All offsets index into the normalized text
(NFC, ``\\n`` line endings, no NUL characters), which is what the store keeps and what evidence
spans cite.

Pages:
- PDFs: one page per PDF page, joined with a blank line (``PAGE_SEPARATOR``).
- Text files: form feeds split pages (numbered 1..n), or ``[Page N]`` marker lines start page N
  (text before the first marker belongs to no page).

Transcript lines (page:line citations):
- Printed line numbers ("  12  Q. Where were you?") are kept as printed, per page.
- A transcript with ``[Page N]`` markers but no printed numbers (e.g. a deposition excerpt typed
  into a .txt) gets its non-empty lines numbered 1..n within each page, skipping the marker
  line itself. That is our convention, not the court reporter's, and only applies to documents
  that look like transcripts (Q./A. lines, or "deposition"/"transcript" in the name).
"""

from __future__ import annotations

import codecs
import email
import email.policy
import hashlib
import io
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from email.message import EmailMessage, Message
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal

from pipeline.schema import DocKind, Document, Line, Page, ParsedDocument

log = logging.getLogger(__name__)

MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".eml": "message/rfc822",
}
SUPPORTED_EXTENSIONS = frozenset(MIME_TYPES)
PAGE_SEPARATOR = "\n\n"
_DOC_DATE_SCAN_LINES = 25
_HEADER_FIELDS = ("Date", "From", "To", "Cc", "Subject")

# --------------------------------------------------------------------------- normalization


def normalize_text(text: str) -> str:
    """NFC-normalize, unify line endings to ``\\n`` and drop NUL characters."""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- discovery


@dataclass(frozen=True)
class FoundFile:
    """A supported file found by :func:`discover`."""

    path: Path
    rel_path: str  # POSIX path relative to its root folder
    location: Literal["evidence", "inbox"]
    sha256: str


def _candidate_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*"):
        rel_parts = p.relative_to(root).parts
        if any(part.startswith((".", "~$")) for part in rel_parts):
            continue  # hidden files and folders, Office lock files
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file():
            out.append(p)
    return sorted(out)


def discover(evidence_dir: Path, inbox_dir: Path | None) -> list[FoundFile]:
    """Hash every supported file in the evidence folder and the inbox (both recursive).

    Identical content is listed once, preferring the evidence folder.
    """
    evidence_dir = Path(evidence_dir)
    if not evidence_dir.is_dir():
        log.warning("evidence dir %s does not exist", evidence_dir)
    roots: list[tuple[Path, Literal["evidence", "inbox"]]] = [(evidence_dir, "evidence")]
    if inbox_dir is not None:
        roots.append((Path(inbox_dir), "inbox"))
    found: list[FoundFile] = []
    seen: set[str] = set()
    for root, location in roots:
        for path in _candidate_files(root):
            try:
                sha = hash_file(path)
            except OSError:
                log.warning("cannot read %s; skipping", path, exc_info=True)
                continue
            if sha in seen:
                continue
            seen.add(sha)
            rel = path.relative_to(root).as_posix()
            found.append(FoundFile(path=path, rel_path=rel, location=location, sha256=sha))
    return found


# --------------------------------------------------------------------------- format readers


def _join_pages(page_texts: list[str]) -> tuple[str, list[Page]]:
    """Normalize each page, join with PAGE_SEPARATOR and record offsets."""
    parts: list[str] = []
    pages: list[Page] = []
    pos = 0
    for i, raw in enumerate(page_texts, start=1):
        txt = normalize_text(raw).strip("\n")
        if parts:
            parts.append(PAGE_SEPARATOR)
            pos += len(PAGE_SEPARATOR)
        pages.append(Page(number=i, char_start=pos, char_end=pos + len(txt)))
        parts.append(txt)
        pos += len(txt)
    return "".join(parts), pages


def _docling_enabled() -> bool:
    return os.environ.get("CT_DISABLE_DOCLING", "").lower() not in {"1", "true", "yes"}


def _parse_pdf_docling(path: Path) -> list[str] | None:
    """Per-page markdown via Docling, or None if Docling is unavailable or fails."""
    if not _docling_enabled():
        return None
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        result = DocumentConverter().convert(str(path))
        ddoc = result.document
        n = len(ddoc.pages)
        pages = [ddoc.export_to_markdown(page_no=i) for i in range(1, n + 1)]
    except Exception:
        log.warning("docling failed on %s; falling back to PyMuPDF", path.name, exc_info=True)
        return None
    return pages if any(p.strip() for p in pages) else None


def _parse_pdf_pymupdf(data: bytes) -> list[str]:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        return [page.get_text("text") for page in pdf]


def _parse_docx(data: bytes) -> str:
    import docx
    from docx.table import Table

    document = docx.Document(io.BytesIO(data))
    blocks: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            for row in item.rows:
                cells = [c.text.strip() for c in row.cells]
                blocks.append(" | ".join(cells))
        else:
            blocks.append(item.text)
    return "\n".join(blocks)


class _HTMLText(HTMLParser):
    _BLOCK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "blockquote"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _html_to_text(html: str) -> str:
    p = _HTMLText()
    p.feed(html)
    return re.sub(r"\n{3,}", "\n\n", "".join(p.parts)).strip()


def _email_text(msg: Message) -> str:
    """Headers plus body text; nested message/rfc822 parts are rendered recursively."""
    lines = [f"{h}: {msg[h]}" for h in _HEADER_FIELDS if msg[h]]
    body_parts: list[str] = []
    attachments: list[str] = []
    html_fallback: list[str] = []
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        ctype = part.get_content_type()
        if part.is_multipart():
            if ctype == "message/rfc822":
                for inner in part.get_payload():
                    body_parts.append("\n" + _email_text(inner))
            continue
        filename = part.get_filename()
        if filename or part.get_content_disposition() == "attachment":
            attachments.append(filename or "(unnamed)")
            continue
        if ctype in {"text/plain", "text/html"}:
            try:
                content = part.get_content() if isinstance(part, EmailMessage) else None
            except (LookupError, ValueError):
                content = None
            if content is None:
                payload = part.get_payload(decode=True) or b""
                try:
                    content = payload.decode(part.get_content_charset() or "utf-8", "replace")
                except LookupError:  # bogus/unknown charset label: best-effort decode
                    content = _decode_text(payload)
            (body_parts if ctype == "text/plain" else html_fallback).append(content)
    if not body_parts:
        body_parts = [_html_to_text(h) for h in html_fallback]
    text = "\n".join(lines) + "\n\n" + "\n".join(b.strip("\n") for b in body_parts)
    if attachments:
        text += "\n\n" + "\n".join(f"[Attachment: {a}]" for a in attachments)
    return text


def _parse_eml(data: bytes) -> str:
    msg = email.message_from_bytes(data, policy=email.policy.default)
    return _email_text(msg)


def _decode_text(data: bytes) -> str:
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return data.decode("utf-32", "replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", "replace")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


# --------------------------------------------------------------------------- pages and lines

_PAGE_MARKER_RE = re.compile(r"^[ \t]*\[\s*page\s+(\d{1,5})\s*\][ \t]*$", re.IGNORECASE | re.M)
_NUMBERED_LINE_RE = re.compile(r"^[ \t]*(\d{1,2})(?:[ \t]+(\S.*))?$")
_QA_LINE_RE = re.compile(r"^[ \t]*(?:\d{1,2}[ \t]+)?(?:Q|A)[.:][ \t]", re.M)
_MIN_NUMBERED_LINES = 5  # per page, before we trust printed line numbers


def _line_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """``(start, end)`` of each line (without its newline) in ``text[start:end]``."""
    spans = []
    pos = start
    while pos < end:
        nl = text.find("\n", pos, end)
        stop = end if nl < 0 else nl
        spans.append((pos, stop))
        pos = stop + 1
    return spans


def _printed_lines(text: str, page: Page) -> list[Line]:
    """Lines with printed line numbers on one page ("  12  Q. ...", or "12" then the line)."""
    candidates: list[tuple[int, int, int]] = []  # (number, char_start, char_end)
    pending: int | None = None
    for s, e in _line_spans(text, page.char_start, page.char_end):
        raw = text[s:e]
        if not raw.strip():
            continue
        m = _NUMBERED_LINE_RE.match(raw)
        if m and m.group(2) is None:  # a bare number: the content follows on the next line
            pending = int(m.group(1))
            continue
        if m:
            candidates.append((int(m.group(1)), s, e))
        elif pending is not None:
            candidates.append((pending, s, e))
        pending = None
    kept: list[tuple[int, int, int]] = []
    expected = 1
    for number, s, e in candidates:
        skipped_ok = bool(kept) and 0 < number - expected <= 2  # a missing line or two
        if number == expected or (not kept and number == 2) or skipped_ok:
            kept.append((number, s, e))
            expected = number + 1
    if len(kept) < _MIN_NUMBERED_LINES:
        return []
    return [Line(page=page.number, number=n, char_start=s, char_end=e) for n, s, e in kept]


def _counted_lines(text: str, page: Page) -> list[Line]:
    """Non-empty lines of a page numbered 1..n (no printed numbers; marker lines skipped)."""
    out: list[Line] = []
    for s, e in _line_spans(text, page.char_start, page.char_end):
        raw = text[s:e]
        if not raw.strip() or _PAGE_MARKER_RE.fullmatch(raw):
            continue
        out.append(Line(page=page.number, number=len(out) + 1, char_start=s, char_end=e))
    return out


def looks_like_transcript(filename: str, text: str) -> bool:
    name = filename.lower()
    if any(w in name for w in ("depo", "transcript", "hearing", "trial")):
        return True
    return len(_QA_LINE_RE.findall(text[:20000])) >= 3


def _marker_pages(text: str) -> list[Page]:
    """Pages from ``[Page N]`` marker lines; each page runs to the next marker (or the end)."""
    marks = list(_PAGE_MARKER_RE.finditer(text))
    pages: list[Page] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        stop = end
        while stop > m.start() and text[stop - 1] in "\n \t":
            stop -= 1
        pages.append(Page(number=int(m.group(1)), char_start=m.start(), char_end=stop))
    return pages


def transcript_lines(text: str, pages: list[Page], *, count_if_unnumbered: bool) -> list[Line]:
    """Printed line numbers per page; else (optionally) non-empty lines counted per page."""
    printed = [ln for p in pages for ln in _printed_lines(text, p)]
    if printed:
        return printed
    if count_if_unnumbered:
        return [ln for p in pages for ln in _counted_lines(text, p)]
    return []


# --------------------------------------------------------------------------- parse


def parse_file(path: Path, doc: Document) -> ParsedDocument:
    """Extract normalized text, page offsets and transcript lines from one file (read-only)."""
    path = Path(path)
    data = path.read_bytes()  # read-only access to evidence
    if _hash_bytes(data) != doc.sha256:
        log.warning("%s changed since discovery (sha256 mismatch)", doc.filename)
    ext = path.suffix.lower()
    pages: list[Page] = []
    lines: list[Line] = []
    if not data:  # zero-byte file: nothing to parse (and not a corrupt PDF/DOCX)
        text = ""
    elif ext == ".pdf":
        raw_pages = _parse_pdf_pymupdf(data)
        text, pages = _join_pages(raw_pages)
        lines = transcript_lines(text, pages, count_if_unnumbered=False)
        if not lines:  # Docling's markdown loses printed line numbers; keep PyMuPDF for those
            docling_pages = _parse_pdf_docling(path)
            if docling_pages:
                text, pages = _join_pages(docling_pages)
    elif ext == ".docx":
        text = normalize_text(_parse_docx(data))
    elif ext == ".eml":
        text = normalize_text(_parse_eml(data))
    elif ext in {".txt", ".md"}:
        decoded = _decode_text(data)
        if "\f" in decoded.strip("\f\n\r \t"):
            text, pages = _join_pages(decoded.split("\f"))
        else:
            text = normalize_text(decoded)
            pages = _marker_pages(text)
        if pages:
            lines = transcript_lines(
                text, pages, count_if_unnumbered=looks_like_transcript(doc.filename, text)
            )
    else:
        raise ValueError(f"unsupported file type: {doc.filename}")
    if not text.strip():
        log.warning("%s produced no text (scanned PDF without OCR?)", doc.filename)
    if pages:
        doc = doc.model_copy(update={"pages": len(pages)})
    return ParsedDocument(doc=doc, text=text, pages=pages, lines=lines)


# --------------------------------------------------------------------------- document kind

_KIND_TEXT_RULES: tuple[tuple[str, DocKind], ...] = (
    (r"\b(?:deposition|transcript) of\b|\breporter'?s transcript\b", "transcript"),
    (r"^\s*(?:(?:verified|amended|first amended|second amended)\s+)?(?:complaint|petition)\b"
     r"|^\s*(?:defendants?'?s?\s+)?answer\b|^\s*counterclaims?\b", "pleading"),
    (r"^\s*(?:notice of\s+)?motion\b|\bmotion (?:to|for|in limine)\b", "motion"),
    (r"^\s*(?:\[proposed\]\s*)?order\b|\bit is (?:hereby )?ordered\b|\bso ordered\b", "order"),
    (r"^\s*exhibit\s+[a-z0-9-]+\s*$", "exhibit"),
)  # fmt: skip
_EMAIL_HEADER_RE = re.compile(r"^(from|to|subject|sent|date|cc):[ \t]*\S", re.I | re.M)
_SALUTATION_RE = re.compile(r"^[ \t]*dear\s+\S", re.I | re.M)


def _token_kind(t: str) -> DocKind | None:
    if "depo" in t or "transcript" in t:
        return "transcript"
    if any(w in t for w in ("complaint", "counterclaim", "petition", "pleading")) or t == "answer":
        return "pleading"
    if "motion" in t or t in ("mtn", "msj", "mtd"):
        return "motion"
    if t in ("order", "ruling", "judgment", "decree"):
        return "order"
    if "exhibit" in t or t in ("ex", "exh") or re.fullmatch(r"exh?\d+", t):
        return "exhibit"
    if "email" in t or t == "mail":
        return "email"
    if "letter" in t or t in ("ltr", "correspondence"):
        return "letter"
    return None


def _kind_from_name(filename: str) -> DocKind | None:
    """The kind named first in the filename ("order_granting_msj" is an order)."""
    for t in re.split(r"[^a-z0-9]+", Path(filename).stem.lower()):
        if t and (kind := _token_kind(t)):
            return kind
    return None


def detect_kind(filename: str, text: str) -> DocKind:
    """Guess the document kind from the filename, then the first page (SPEC §6.5)."""
    if Path(filename).suffix.lower() == ".eml":
        return "email"
    if kind := _kind_from_name(filename):
        return kind
    head = text[:3000]
    first_lines = "\n".join(head.split("\n")[:15]).lower()
    for pattern, kind in _KIND_TEXT_RULES:
        if re.search(pattern, first_lines, re.M):
            return kind
    if len({m.lower() for m in _EMAIL_HEADER_RE.findall(head[:1500])}) >= 3:
        return "email"
    if _SALUTATION_RE.search(head):
        return "letter"
    return "other"


# --------------------------------------------------------------------------- document date

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_DATE_RE = re.compile(
    rf"\b(?:{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS},?\s+\d{{4}}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{4})\b",
    re.IGNORECASE,
)
_DATE_LINE_RE = re.compile(r"^\W*(?:date|dated|sent)\b\s*[:\-]?\s*(.+)$", re.IGNORECASE)


def _parse_date_str(s: str) -> date | None:
    import dateparser

    dt = dateparser.parse(
        s,
        languages=["en"],
        settings={"PREFER_DAY_OF_MONTH": "first", "STRICT_PARSING": True},
    )
    return dt.date() if dt else None


def _email_header_date(value: str) -> date | None:
    try:
        return parsedate_to_datetime(value.strip()).date()
    except (TypeError, ValueError, IndexError):
        return _parse_date_str(value)


def doc_date(parsed: ParsedDocument) -> date | None:
    """Best-guess authoring date: EML ``Date`` header, else a date-like line near the top."""
    lines = parsed.text.split("\n")[:_DOC_DATE_SCAN_LINES]
    if parsed.doc.filename.lower().endswith(".eml"):
        for line in lines:
            if line.startswith("Date:"):
                if found := _email_header_date(line[5:]):
                    return found
                break
    # Prefer explicit "Date: ..." lines, then any line with a full calendar date.
    for line in lines:
        m = _DATE_LINE_RE.match(line)
        if m and (dm := _DATE_RE.search(m.group(1))):
            if found := _parse_date_str(dm.group(0)):
                return found
    for line in lines:
        if dm := _DATE_RE.search(line):
            if found := _parse_date_str(dm.group(0)):
                return found
    return None
