"""Discover evidence files and turn them into normalized text with page offsets.

Evidence files are only ever opened for reading. All offsets index into the
normalized text (NFC, ``\\n`` line endings, no NUL characters).
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
from datetime import date
from email.message import EmailMessage, Message
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pipeline.contracts import Document, Page, ParsedDocument

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


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- discovery


def _default_inbox(case_id: str) -> list[tuple[Path, bool]]:
    """Settings inbox as (root, recursive): loose ``inbox/`` files plus all of ``inbox/<case>``."""
    try:
        from pipeline.config import get_settings, valid_case_id

        valid_case_id(case_id)
        inbox = Path(get_settings().inbox)
    except Exception:  # settings unavailable or misconfigured: no inbox
        log.debug("no inbox from settings", exc_info=True)
        return []
    # The bare inbox is scanned flat so other cases' subfolders are not picked up.
    return [(inbox, False), (inbox / case_id, True)]


def _candidate_files(root: Path, recursive: bool) -> list[Path]:
    if not root.is_dir():
        return []
    entries = root.rglob("*") if recursive else root.iterdir()
    out = []
    for p in entries:
        rel_parts = p.relative_to(root).parts
        if any(part.startswith((".", "~$")) for part in rel_parts):
            continue  # hidden files, Office lock files
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            out.append(p)
    return sorted(out)


def discover(
    evidence_dir: Path, case_id: str, *, inbox_dirs: list[Path] | None = None
) -> list[Document]:
    """Hash every supported file under ``evidence_dir`` and the upload inbox.

    ``inbox_dirs`` (scanned recursively) defaults to the settings inbox: loose files in
    ``inbox/`` plus everything in ``inbox/<case_id>/``. Identical files are listed once.
    """
    evidence_dir = Path(evidence_dir)
    if not evidence_dir.is_dir():
        log.warning("evidence dir %s does not exist", evidence_dir)
    roots: list[tuple[Path, bool]] = [(evidence_dir, True)]
    if inbox_dirs is None:
        roots += _default_inbox(case_id)
    else:
        roots += [(Path(d), True) for d in inbox_dirs]

    docs: list[Document] = []
    seen: set[str] = set()
    for root, recursive in roots:
        for path in _candidate_files(root, recursive):
            try:
                sha = _hash_file(path)
            except OSError:
                log.warning("cannot read %s; skipping", path, exc_info=True)
                continue
            if sha in seen:
                continue
            seen.add(sha)
            docs.append(
                Document(
                    id=sha[:16],
                    case_id=case_id,
                    path=str(path.resolve()),
                    filename=path.name,
                    mime=MIME_TYPES[path.suffix.lower()],
                    sha256=sha,
                )
            )
    return docs


# --------------------------------------------------------------------------- parsing


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
    """A PDF has no usable text layer and no OCR engine is available to read it."""


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
        with _docling_lock:  # the converter loads models once; reuse it
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
    """OCR pages that have (almost) no text; raise NeedsOCRError if nothing is readable."""
    import pymupdf

    missing = [i for i, t in enumerate(page_texts) if _meaningful_chars(t) < _MIN_PAGE_CHARS]
    if not missing:
        return page_texts
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        # Only pages carrying an image can be scans; blank or short pages are left alone.
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
            f"needs OCR, no text layer: {total} of {total} pages are scanned images with no "
            "extractable text"
            + (
                f" and OCR ({engine}) returned nothing"
                if engine
                else "; no OCR engine available (install the tesseract-ocr binary, the "
                "rapidocr-onnxruntime or easyocr Python package, or the docling extra)"
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
# Runs of bullets / blocks that redaction tools leave in the text layer.
_REDACTION_RE = re.compile(r"[•·●▪■□█▇▆▓▒░◼◾]{2,}(?:[ \t]*[•·●▪■□█▇▆▓▒░◼◾]+)*")
_PAGE_NUM_RE = re.compile(
    r"^(?:(?:page|pg\.?|p\.)\s*)?[-–—(\[]?\s*\d{1,3}\s*[-–—)\]]?"
    r"(?:\s*(?:of|/)\s*\d{1,4})?$",
    re.IGNORECASE,
)
_LINE_NUM_RE = re.compile(r"^(\d{1,2})(?:\s+|$)")
_EDGE_LINES = 4  # lines at the top / bottom of a page that can be running furniture
_FUZZY_KEY_LEN = 12
_FUZZY_KEY_RATIO = 88
_MIN_LINE_NUMBERED = 8  # rows with a leading 1..30 counter before a page is a transcript


def _clean_redactions(text: str) -> str:
    return _REDACTION_RE.sub(REDACTED, text)


def _strip_line_numbers(text: str) -> str:
    """Drop transcript / pleading-paper margin line numbers (1, 2, 3 ... 25)."""
    lines = text.split("\n")
    nums = [(i, int(m.group(1))) for i, ln in enumerate(lines) if (m := _LINE_NUM_RE.match(ln))]
    nums = [(i, n) for i, n in nums if 1 <= n <= 30]
    if len(nums) < _MIN_LINE_NUMBERED:
        return text
    steps = sum(1 for (_, a), (_, b) in zip(nums, nums[1:], strict=False) if b == a + 1)
    if steps < 0.7 * (len(nums) - 1):
        return text
    out = list(lines)
    for i, _ in nums:
        out[i] = _LINE_NUM_RE.sub("", lines[i], count=1)
    return "\n".join(ln for i, ln in enumerate(out) if ln.strip() or lines[i].strip() == "")


def _furniture_key(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _edge_indices(lines: list[str]) -> list[int]:
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    return sorted(set(nonblank[:_EDGE_LINES] + nonblank[-_EDGE_LINES:]))


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
    edges = [_edge_indices(lines) for lines in pages]
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
    Lines with no letters or digits at all (rules, stray marks) are dropped.
    """
    pages = [
        [ln for ln in t.split("\n") if not ln.strip() or any(c.isalnum() for c in ln)]
        for t in page_texts
    ]
    for _ in range(2):
        pages = _strip_furniture_once(pages)
    return ["\n".join(lines) for lines in pages]


def _clean_pdf_pages(page_texts: list[str]) -> list[str]:
    pages = [_clean_redactions(normalize_text(t)) for t in page_texts]
    pages = [_strip_line_numbers(t) for t in pages]
    return _strip_page_furniture(pages)


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


def parse(doc: Document) -> ParsedDocument:
    """Extract normalized text (and PDF page offsets) from a discovered document."""
    path = Path(doc.path)
    data = path.read_bytes()  # read-only access to evidence
    if _hash_bytes(data) != doc.sha256:
        log.warning("%s changed since discovery (sha256 mismatch)", doc.filename)
    ext = path.suffix.lower()
    pages: list[Page] = []
    if not data:  # zero-byte file: nothing to parse (and not a corrupt PDF/DOCX)
        text = ""
    elif ext == ".pdf":
        page_texts = _parse_pdf_docling(path) or _parse_pdf_pymupdf(data)
        page_texts = _ocr_missing_pages(data, page_texts, doc.filename)
        text, pages = _join_pages(_clean_pdf_pages(page_texts))
        doc = doc.model_copy(update={"num_pages": len(pages)})
    elif ext == ".docx":
        text = normalize_text(_parse_docx(data))
    elif ext == ".eml":
        text = normalize_text(_parse_eml(data))
    elif ext in {".txt", ".md"}:
        text = normalize_text(_decode_text(data))
    else:
        raise ValueError(f"unsupported file type: {doc.filename}")
    if ext != ".pdf":
        text = _clean_redactions(text)  # same length-changing edit as PDFs, before offsets
    if not text.strip():
        log.warning("%s produced no text (scanned PDF without OCR?)", doc.filename)
    return ParsedDocument(doc=doc, text=text, pages=pages)


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
