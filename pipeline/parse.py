"""Discover case files and turn them into normalized text with page (and transcript line) offsets.

Evidence files are only ever opened for reading. All offsets index into the normalized text
(NFC, ``\\n`` line endings, no NUL characters), which is what the store keeps and what evidence
spans cite.

Pages:
- PDFs: one page per PDF page, joined with a blank line (``PAGE_SEPARATOR``). PyMuPDF text is
  rebuilt row by row (so transcript line numbers, Q./A. columns and form labels stay on one
  line with their text); repeated running headers/footers and page numbers are removed (the
  first copy of a running line is kept) and redaction marks become ``[REDACTED]``, all before
  offsets are computed. Scanned pages are OCR'd if an engine is installed; a PDF whose pages
  are all images with no readable text raises :class:`NeedsOCRError`.
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
import shutil
import subprocess
import threading
import unicodedata
from dataclasses import dataclass
from datetime import date
from email.message import EmailMessage, Message
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

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


class NeedsOCRError(RuntimeError):
    """A PDF has no usable text layer and no OCR engine could read it.

    Raised from :func:`parse_file` so the scan records it as the document's parse error
    instead of silently producing a document with no text (and no facts).
    """


_docling_lock = threading.Lock()
_docling_converter: Any = None


def _parse_pdf_docling(path: Path) -> list[str] | None:
    """Per-page markdown via Docling (which OCRs scanned pages if it can), else None."""
    global _docling_converter
    if not _docling_enabled():
        return None
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with _docling_lock:  # the converter loads its models once; reuse it
            if _docling_converter is None:
                _docling_converter = DocumentConverter()
            result = _docling_converter.convert(str(path))
        ddoc = result.document
        n = len(ddoc.pages)
        pages = [ddoc.export_to_markdown(page_no=i) for i in range(1, n + 1)]
    except Exception:
        log.warning("docling failed on %s; falling back to PyMuPDF", path.name, exc_info=True)
        return None
    return pages if any(p.strip() for p in pages) else None


# ---- PyMuPDF: rebuild visual rows so labels, values and transcript columns stay together

_ROW_OVERLAP = 0.5  # min vertical overlap (fraction of the smaller line) to share a row
_COLUMN_MIN_WIDTH = 0.3  # a prose column spans at least this fraction of the page width
_COLUMN_MIN_LINES = 3
_COLUMN_MIN_CHARS = 25  # average characters per line for a block to count as prose


def _page_lines(page: Any) -> list[tuple[tuple[float, ...], str, tuple[float, ...]]]:
    """(bbox, text, direction) for each text line on the page."""
    import pymupdf

    out = []
    raw = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if text:
                out.append((tuple(line["bbox"]), text, tuple(line.get("dir", (1, 0)))))
    return out


def _is_multi_column(page: Any) -> bool:
    """True if the page has side-by-side prose columns (rows must not be merged across)."""
    width = page.rect.width or 1.0
    prose = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        lines = [ln for ln in str(text).split("\n") if ln.strip()]
        if (
            len(lines) >= _COLUMN_MIN_LINES
            and (x1 - x0) >= _COLUMN_MIN_WIDTH * width
            and sum(len(ln) for ln in lines) / len(lines) >= _COLUMN_MIN_CHARS
        ):
            prose.append((x0, y0, x1, y1))
    for i, a in enumerate(prose):
        for b in prose[i + 1 :]:
            side_by_side = a[2] <= b[0] or b[2] <= a[0]
            v_overlap = min(a[3], b[3]) - max(a[1], b[1])
            if side_by_side and v_overlap > 0.5 * min(a[3] - a[1], b[3] - b[1]):
                return True
    return False


def _page_text_rows(page: Any) -> str:
    """Page text with lines that sit on the same visual row joined left to right.

    PyMuPDF's default order emits each block separately, which tears apart transcript
    line numbers / Q. / A. columns, form labels and their values, and fully justified
    lines. Multi-column prose and rotated pages keep PyMuPDF's own reading order.
    """
    if page.rotation or _is_multi_column(page):
        return page.get_text("text")
    lines = _page_lines(page)
    # Vertical margin text (stamps, Bates labels) is kept, but after the page body.
    sideways = [t for _, t, d in lines if abs(d[0]) < 0.9]
    lines = [ln for ln in lines if abs(ln[2][0]) >= 0.9]
    lines.sort(key=lambda ln: ((ln[0][1] + ln[0][3]) / 2, ln[0][0]))
    rows: list[tuple[float, float, list[tuple[float, str]]]] = []
    for (x0, y0, _x1, y1), text, _ in lines:
        h = max(y1 - y0, 0.1)
        for i in range(len(rows) - 1, max(len(rows) - 4, -1), -1):
            ry0, ry1, items = rows[i]
            overlap = min(y1, ry1) - max(y0, ry0)
            if overlap >= _ROW_OVERLAP * min(h, max(ry1 - ry0, 0.1)):
                items.append((x0, text))
                break
        else:
            rows.append((y0, y1, [(x0, text)]))
    rows.sort(key=lambda r: r[0])
    body = [" ".join(t for _, t in sorted(items)) for _, _, items in rows]
    return "\n".join(body + sideways)


def _parse_pdf_pymupdf(data: bytes) -> list[str]:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        return [_page_text_rows(page) for page in pdf]


# ---- OCR for pages without a text layer

_MIN_PAGE_CHARS = 20  # fewer alphanumerics than this (after redaction marks): no text layer
_OCR_DPI = 300
_OCR_TIMEOUT = 180  # seconds per page


def _meaningful_chars(text: str) -> int:
    return sum(ch.isalnum() for ch in _REDACTION_RE.sub("", text).replace(REDACTED, ""))


def _has_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def ocr_engine() -> str | None:
    """Name of the first available OCR engine: tesseract binary, rapidocr or easyocr."""
    if shutil.which("tesseract"):
        return "tesseract"
    for mod in ("rapidocr_onnxruntime", "rapidocr"):
        if _has_module(mod):
            return mod
    if _has_module("easyocr"):
        return "easyocr"
    return None


_ocr_models: dict[str, Any] = {}


def _boxes_to_text(items: list[tuple[float, float, float, str]]) -> str:
    """Group OCR boxes (x0, y_center, height, text) into rows, top to bottom, left to right."""
    rows: list[tuple[float, float, list[tuple[float, str]]]] = []
    for x0, yc, h, text in sorted(items, key=lambda it: (it[1], it[0])):
        if rows and abs(yc - rows[-1][0]) <= 0.5 * max(h, rows[-1][1]):
            rows[-1][2].append((x0, text))
        else:
            rows.append((yc, h, [(x0, text)]))
    return "\n".join(" ".join(t for _, t in sorted(r[2])) for r in rows)


def _ocr_png(png: bytes, engine: str) -> str:
    if engine == "tesseract":
        proc = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", "6"],
            input=png,
            capture_output=True,
            timeout=_OCR_TIMEOUT,
            check=True,
        )
        return proc.stdout.decode("utf-8", "replace")
    if engine in {"rapidocr_onnxruntime", "rapidocr"}:
        if engine not in _ocr_models:
            mod = __import__(engine, fromlist=["RapidOCR"])
            _ocr_models[engine] = mod.RapidOCR()
        out = _ocr_models[engine](png)
        if isinstance(out, tuple):  # rapidocr_onnxruntime: ([[box, text, score], ...], elapse)
            results = [(r[0], r[1]) for r in (out[0] or [])]
        else:  # rapidocr >= 2: object with .boxes / .txts
            boxes = out.boxes if out.boxes is not None else []
            results = list(zip(boxes, out.txts or (), strict=False))
    elif engine == "easyocr":
        if engine not in _ocr_models:
            import easyocr  # type: ignore[import-not-found]

            _ocr_models[engine] = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = [(r[0], r[1]) for r in _ocr_models[engine].readtext(png)]
    else:
        raise ValueError(f"unknown OCR engine {engine}")
    items = []
    for box, text in results:
        ys = [float(pt[1]) for pt in box]
        xs = [float(pt[0]) for pt in box]
        items.append((min(xs), (min(ys) + max(ys)) / 2, max(ys) - min(ys), str(text)))
    return _boxes_to_text(items)


def _ocr_missing_pages(data: bytes, page_texts: list[str], name: str) -> list[str]:
    """OCR scanned pages that have (almost) no text; raise NeedsOCRError if none is readable.

    Only pages that carry an image can be scans: blank or nearly empty vector pages are left
    alone (a PDF of blank pages parses to empty text, with a warning).
    """
    import pymupdf

    missing = [i for i, t in enumerate(page_texts) if _meaningful_chars(t) < _MIN_PAGE_CHARS]
    if not missing:
        return page_texts
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        missing = [i for i in missing if i < len(pdf) and pdf[i].get_images(full=True)]
        if not missing:
            return page_texts
        engine = ocr_engine()
        out = list(page_texts)
        if engine:
            for i in missing:
                try:
                    pix = pdf[i].get_pixmap(dpi=_OCR_DPI)
                    out[i] = _ocr_png(pix.tobytes("png"), engine)
                except Exception:
                    log.warning("OCR (%s) failed on %s page %d", engine, name, i + 1, exc_info=True)
            log.info("%s: OCR'd %d page(s) with %s", name, len(missing), engine)
            still = [i for i in missing if _meaningful_chars(out[i]) < _MIN_PAGE_CHARS]
        else:
            still = missing
    total = len(page_texts)
    if total and sum(_meaningful_chars(t) for t in out) < _MIN_PAGE_CHARS:
        raise NeedsOCRError(
            f"needs OCR, no text layer: {len(missing)} of {total} pages are scanned images "
            "with no extractable text"
            + (
                f" and OCR ({engine}) returned nothing"
                if engine
                else "; no OCR engine available (install the tesseract-ocr binary, the "
                "rapidocr-onnxruntime or easyocr Python package, or Docling)"
            )
        )
    if still:
        log.warning(
            "%s: %d of %d pages have no text layer and were not OCR'd (pages %s)",
            name,
            len(still),
            total,
            ", ".join(str(i + 1) for i in still[:20]),
        )
    return out


# ---- clean-up of page furniture, applied before offsets are computed

REDACTED = "[REDACTED]"
# Runs of bullets / blocks that redaction tools leave in the text layer ("•••••", "■■■ ■■").
# The middle dot is left out: dot leaders in tables of contents use it.
_REDACTION_RE = re.compile(r"[•●▪■□█▇▆▓▒░◼◾]{2,}(?:[ \t]*[•●▪■□█▇▆▓▒░◼◾]+)*")
_PAGE_NUM_RE = re.compile(
    r"^(?:(?:page|pg\.?|p\.)\s*)?[-–—(\[]?\s*\d{1,3}\s*[-–—)\]]?"
    r"(?:\s*(?:of|/)\s*\d{1,4})?$",
    re.IGNORECASE,
)
_EDGE_LINES = 4  # lines at the top / bottom of a page that can be running furniture
_FUZZY_KEY_LEN = 12
_FUZZY_KEY_RATIO = 88


def _clean_redactions(text: str) -> str:
    return _REDACTION_RE.sub(REDACTED, text)


def _furniture_key(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _edge_indices(lines: list[str], keep: set[int]) -> list[int]:
    """Top and bottom non-blank lines, skipping printed transcript lines (never furniture)."""
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    edges = set(nonblank[:_EDGE_LINES] + nonblank[-_EDGE_LINES:])
    return sorted(edges - keep)


def _same_furniture(a: str, b: str) -> bool:
    """Equal keys, or long keys that differ only by OCR noise ("Wl10" vs "Wl1 0")."""
    if a == b:
        return True
    from rapidfuzz import fuzz

    return min(len(a), len(b)) >= _FUZZY_KEY_LEN and fuzz.ratio(a, b) >= _FUZZY_KEY_RATIO


def _fragment_of(key: str, running: set[str]) -> bool:
    """A torn-off start or end of a running line ("Inc." from "Acme Reporting, Inc.")."""
    if sum(ch.isalpha() for ch in key) < 3:
        return False
    from rapidfuzz import fuzz

    n = len(key)
    return any(
        n < len(k) and max(fuzz.ratio(key, k[:n]), fuzz.ratio(key, k[-n:])) >= 90 for k in running
    )


def _edge_positions(lines: list[str], idx: list[int]) -> dict[str, set[tuple[str, int]]]:
    """Key -> positions ("t", n-th line from the top) / ("b", n-th from the bottom)."""
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    rank = {i: r for r, i in enumerate(nonblank)}
    out: dict[str, set[tuple[str, int]]] = {}
    for i in idx:
        r = rank[i]
        pos = out.setdefault(_furniture_key(lines[i]), set())
        if r < _EDGE_LINES:
            pos.add(("t", r))
        if len(nonblank) - 1 - r < _EDGE_LINES:
            pos.add(("b", len(nonblank) - 1 - r))
    return out


def _support(similar: set[str], per_page: list[dict[str, set[tuple[str, int]]]]) -> int:
    """Pages holding one of ``similar`` at a consistent edge position (+/- one line)."""
    best = 0
    for side in ("t", "b"):
        for anchor in range(_EDGE_LINES):
            ok = {(side, o) for o in (anchor - 1, anchor, anchor + 1)}
            count = sum(
                1
                for page in per_page
                if any(not ok.isdisjoint(page[k]) for k in similar.intersection(page))
            )
            best = max(best, count)
    return best


def _running_keys(pages: list[list[str]], edges: list[list[int]]) -> set[str]:
    """Keys found at the same top/bottom position on at least half the pages (min. 3)."""
    n = len(pages)
    if n < 3:
        return set()
    from rapidfuzz import fuzz, process

    per_page = [_edge_positions(lines, idx) for lines, idx in zip(pages, edges, strict=True)]
    keys = sorted({k for page_keys in per_page for k in page_keys if len(k) >= 3})
    need = max(3, -(-n // 2))
    running = set()
    for key in keys:
        similar = {key}
        if len(key) >= _FUZZY_KEY_LEN:
            hits = process.extract(
                key, keys, scorer=fuzz.ratio, score_cutoff=_FUZZY_KEY_RATIO, limit=None
            )
            similar |= {k for k, _, _ in hits if len(k) >= _FUZZY_KEY_LEN}
        support = _support(similar, per_page)
        if support >= need:
            running.add(key)
        elif support >= max(3, -(-n // 4)) and len(key) >= _FUZZY_KEY_LEN:
            # OCR'd text layers often tear a footer apart ("Inc." left of "Acme, Inc.").
            torn = sum(
                1
                for lines, idx, pk in zip(pages, edges, per_page, strict=True)
                if similar.isdisjoint(pk)
                and idx
                and any(_fragment_of(_furniture_key(lines[i]), {key}) for i in (idx[0], idx[-1]))
            )
            if support + torn >= need:
                running.add(key)
    return running


def _strip_furniture_once(pages: list[list[str]]) -> list[list[str]]:
    keep = [_numbered_row_indices(lines) for lines in pages]
    edges = [_edge_indices(lines, k) for lines, k in zip(pages, keep, strict=True)]
    running = _running_keys(pages, edges)
    seen: set[str] = set()
    out = []
    for lines, idx in zip(pages, edges, strict=True):
        drop = set()
        for i in idx:
            line = lines[i].strip()
            key = _furniture_key(line)
            outermost = i in (idx[0], idx[-1])
            if len(pages) >= 2 and outermost and _PAGE_NUM_RE.match(line):
                drop.add(i)
            elif outermost and running and _fragment_of(key, running):
                drop.add(i)
            elif (match := next((k for k in running if _same_furniture(key, k)), None)) is not None:
                if match in seen:  # keep the first occurrence for context
                    drop.add(i)
                seen.add(match)
        out.append([ln for i, ln in enumerate(lines) if i not in drop])
    return out


def _strip_page_furniture(page_texts: list[str]) -> list[str]:
    """Remove page numbers and running headers/footers repeated across pages.

    A running line is one whose digit-insensitive form (OCR noise tolerated) sits in the top
    or bottom lines of at least half the pages (min. 3); its first occurrence is kept for
    context. Two passes, so a header block taller than the edge window is removed too.
    Lines with no letters or digits at all (rules, stray marks) are dropped. Printed
    transcript lines ("12 Q. Where were you?") are never treated as furniture, so page:line
    citations survive.
    """
    pages = [
        [ln for ln in t.split("\n") if not ln.strip() or any(c.isalnum() for c in ln)]
        for t in page_texts
    ]
    for _ in range(2):
        pages = _strip_furniture_once(pages)
    return ["\n".join(lines) for lines in pages]


def _clean_pdf_pages(page_texts: list[str]) -> list[str]:
    return _strip_page_furniture([_clean_redactions(normalize_text(t)) for t in page_texts])


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


def _printed_numbering(rows: list[str]) -> list[tuple[int, int, int | None]]:
    """``(number, content_row, number_row)`` for rows carrying printed transcript line numbers.

    A number either starts its row ("  12  Q. ...") or sits alone on the row before the
    content ("12" then the line; ``number_row`` is then that bare row). The numbers must run
    1, 2, 3 ... (a missing line or two is tolerated) for at least ``_MIN_NUMBERED_LINES``
    rows, else nothing is returned.
    """
    candidates: list[tuple[int, int, int | None]] = []
    pending: tuple[int, int] | None = None  # (number, row) of a bare number
    for i, raw in enumerate(rows):
        if not raw.strip():
            continue
        m = _NUMBERED_LINE_RE.match(raw)
        if m and m.group(2) is None:  # a bare number: the content follows on the next line
            pending = (int(m.group(1)), i)
            continue
        if m:
            candidates.append((int(m.group(1)), i, None))
        elif pending is not None:
            candidates.append((pending[0], i, pending[1]))
        pending = None
    kept: list[tuple[int, int, int | None]] = []
    expected = 1
    for number, row, num_row in candidates:
        skipped_ok = bool(kept) and 0 < number - expected <= 2  # a missing line or two
        if number == expected or (not kept and number == 2) or skipped_ok:
            kept.append((number, row, num_row))
            expected = number + 1
    return kept if len(kept) >= _MIN_NUMBERED_LINES else []


def _numbered_row_indices(rows: list[str]) -> set[int]:
    """Rows that belong to printed transcript lines (content rows and bare number rows)."""
    out: set[int] = set()
    for _n, row, num_row in _printed_numbering(rows):
        out.add(row)
        if num_row is not None:
            out.add(num_row)
    return out


def _printed_lines(text: str, page: Page) -> list[Line]:
    """Lines with printed line numbers on one page ("  12  Q. ...", or "12" then the line)."""
    spans = _line_spans(text, page.char_start, page.char_end)
    kept = _printed_numbering([text[s:e] for s, e in spans])
    return [
        Line(page=page.number, number=n, char_start=spans[row][0], char_end=spans[row][1])
        for n, row, _num_row in kept
    ]


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


def _pdf_text(page_texts: list[str], *, numbered: bool) -> tuple[str, list[Page], list[Line]]:
    text, pages = _join_pages(_clean_pdf_pages(page_texts))
    lines = transcript_lines(text, pages, count_if_unnumbered=False) if numbered else []
    return text, pages, lines


def _parse_pdf(path: Path, data: bytes, name: str) -> tuple[str, list[Page], list[Line]]:
    """PyMuPDF rows first; Docling unless the PDF has printed transcript line numbers (its
    markdown loses them); OCR for scanned pages; then page furniture is stripped."""
    raw = _parse_pdf_pymupdf(data)
    text, pages, lines = _pdf_text(raw, numbered=True)
    from_docling = False
    if not lines:
        docling_pages = _parse_pdf_docling(path)
        if docling_pages:
            raw, from_docling = docling_pages, True
            text, pages, lines = _pdf_text(raw, numbered=False)
    ocrd = _ocr_missing_pages(data, raw, name)  # raises NeedsOCRError if nothing is readable
    if ocrd is not raw:
        text, pages, lines = _pdf_text(ocrd, numbered=not from_docling)
    return text, pages, lines


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
        text, pages, lines = _parse_pdf(path, data, doc.filename)
    elif ext == ".docx":
        text = _clean_redactions(normalize_text(_parse_docx(data)))
    elif ext == ".eml":
        text = _clean_redactions(normalize_text(_parse_eml(data)))
    elif ext in {".txt", ".md"}:
        decoded = _clean_redactions(_decode_text(data))
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
        log.warning("%s produced no text", doc.filename)
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
