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
import time
import unicodedata
from collections import Counter
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


_TRUE = {"1", "true", "yes"}


def _docling_enabled() -> bool:
    return os.environ.get("CT_DISABLE_DOCLING", "").lower() not in _TRUE


def _docling_ocr_enabled() -> bool:
    """Docling's own OCR is off unless ``CT_DOCLING_OCR`` is set.

    On scans that carry a text layer it replaced good text-layer lines with garbled OCR, it
    left many image-only pages empty, and it was slow; image-only pages are OCR'd by
    ``_ocr_pages`` instead.
    """
    return os.environ.get("CT_DOCLING_OCR", "").lower() in _TRUE


class NeedsOCRError(RuntimeError):
    """A PDF has no usable text layer and no OCR engine could read it."""


_docling_lock = threading.Lock()
_docling_converter: Any = None


def _new_docling_converter() -> Any:
    from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]

    try:
        from docling.datamodel.base_models import InputFormat  # type: ignore[import-not-found]
        from docling.datamodel.pipeline_options import (  # type: ignore[import-not-found]
            PdfPipelineOptions,
        )
        from docling.document_converter import PdfFormatOption  # type: ignore[import-not-found]

        options = PdfPipelineOptions(do_ocr=_docling_ocr_enabled())
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
    except (ImportError, TypeError, ValueError):  # older/newer Docling API: its defaults
        log.debug("docling pipeline options unavailable; using defaults", exc_info=True)
        return DocumentConverter()


def _parse_pdf_docling(path: Path) -> list[str] | None:
    """Per-page markdown via Docling, else None."""
    global _docling_converter
    if not _docling_enabled():
        return None
    try:
        import docling.document_converter  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return None
    try:
        with _docling_lock:  # the converter loads models once; reuse it
            if _docling_converter is None:
                _docling_converter = _new_docling_converter()
            result = _docling_converter.convert(str(path))
        ddoc = result.document
        n = len(ddoc.pages)
        pages = [ddoc.export_to_markdown(page_no=i) for i in range(1, n + 1)]
    except Exception:
        log.warning("docling failed on %s; falling back to PyMuPDF", path.name, exc_info=True)
        return None
    return pages if any(p.strip() for p in pages) else None


_DOCLING_MIN_AGREEMENT = 0.95  # share of words Docling and the text layer must have in common
_IMAGE_PLACEHOLDER_RE = re.compile(r"<!--\s*image\s*-->")


def _word_counts(text: str) -> Counter[str]:
    return Counter(re.findall(r"[^\W_]{2,}", _IMAGE_PLACEHOLDER_RE.sub(" ", text).lower()))


def _docling_agrees(docling: list[str], layer: list[str]) -> bool:
    """True if Docling kept every text-layer page's words, on the same page, and added none.

    Docling can drop or garble text-layer lines and move a paragraph that runs over a page
    break onto the previous page, which would lose evidence or put quotes on the wrong page.
    Pages without a text layer are not compared.
    """
    if len(docling) != len(layer):
        return False
    for d, t in zip(docling, layer, strict=True):
        if _meaningful_chars(t) < _MIN_PAGE_CHARS:
            continue
        a, b = _word_counts(t), _word_counts(d)
        common = sum((a & b).values())
        need = _DOCLING_MIN_AGREEMENT * max(a.total(), b.total())
        if common < need:
            return False
    return True


def _pdf_page_texts(path: Path, data: bytes) -> list[str]:
    """Per-page text: Docling when it is faithful to the text layer, else PyMuPDF's layout.

    Docling is skipped for PDFs with no text layer at all unless its own OCR is enabled.
    """
    layer = _parse_pdf_pymupdf(data)
    has_layer = any(_meaningful_chars(t) >= _MIN_PAGE_CHARS for t in layer)
    if not (has_layer or _docling_ocr_enabled()):
        return layer
    docling = _parse_pdf_docling(path)
    if not docling:
        return layer
    if _docling_agrees(docling, layer):
        return docling
    log.info("%s: Docling text disagrees with the text layer; using PyMuPDF", path.name)
    if len(docling) != len(layer):
        return layer
    # Keep what Docling read (OCR) on pages that have no text layer.
    return [
        d if _meaningful_chars(t) < _MIN_PAGE_CHARS <= _meaningful_chars(d) else t
        for d, t in zip(docling, layer, strict=True)
    ]


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


# ---- OCR for pages without a (usable) text layer

_MIN_PAGE_CHARS = 20  # fewer alphanumerics than this (after redaction marks): no text layer
_LOW_YIELD_RATIO = 0.25  # an image page yielding under this share of the doc median is re-read
_MIN_OCR_CONFIDENCE = 0.6  # confident, upright characters per character read, below: retry
_OCR_DPI = 300
_OCR_RETRY_DPI = 400
_OCR_ROTATIONS = (90, 180, 270)  # tried, besides upright, on pages that read poorly
_OCR_MAX_PIXELS = 5000  # longest side of a rendered page, whatever the DPI
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
_ocr_lock = threading.Lock()


class OcrText(str):
    """OCR output text carrying a quality ``score``.

    The score counts characters read with confidence in upright (horizontal) lines, so
    the right page orientation scores highest. Plain strings score their character count.
    """

    score: float

    def __new__(cls, text: str, score: float | None = None) -> OcrText:
        obj = super().__new__(cls, text)
        obj.score = float(_meaningful_chars(text) if score is None else score)
        return obj


def _ocr_score(text: str) -> float:
    return text.score if isinstance(text, OcrText) else float(_meaningful_chars(text))


def _boxes_to_text(items: list[tuple[float, float, float, str]]) -> str:
    """Group OCR boxes (x0, y_center, height, text) into rows, top to bottom, left to right."""
    rows: list[tuple[float, float, list[tuple[float, str]]]] = []
    for x0, yc, h, text in sorted(items, key=lambda it: (it[1], it[0])):
        if rows and abs(yc - rows[-1][0]) <= 0.5 * max(h, rows[-1][1]):
            rows[-1][2].append((x0, text))
        else:
            rows.append((yc, h, [(x0, text)]))
    return "\n".join(" ".join(t for _, t in sorted(r[2])) for r in rows)


def _ocr_results(png: bytes, engine: str) -> list[tuple[Any, str, float]]:
    """(box corner points, text, confidence) for each line the engine read."""
    with _ocr_lock:  # models load once and are not documented as thread-safe
        if engine in {"rapidocr_onnxruntime", "rapidocr"}:
            if engine not in _ocr_models:
                mod = __import__(engine, fromlist=["RapidOCR"])
                _ocr_models[engine] = mod.RapidOCR()
            # The line-orientation classifier flips some upright lines and garbles them
            # ("NOTIFICATION, INVESTIGATION, ..." read as "NOT I  I -  A IOSE"); upside-down
            # pages are handled by reading the whole page turned instead.
            try:
                out = _ocr_models[engine](png, use_cls=False)
            except TypeError:  # an engine version without the switch
                out = _ocr_models[engine](png)
            if isinstance(out, tuple):  # rapidocr_onnxruntime: ([[box, text, score], ...], t)
                return [(r[0], str(r[1]), float(r[2])) for r in (out[0] or [])]
            boxes = out.boxes if out.boxes is not None else []  # rapidocr >= 2
            scores = out.scores if out.scores is not None else [1.0] * len(boxes)
            return [
                (b, str(t), float(s))
                for b, t, s in zip(boxes, out.txts or (), scores, strict=False)
            ]
        if engine == "easyocr":
            if engine not in _ocr_models:
                import easyocr  # type: ignore[import-not-found]

                _ocr_models[engine] = easyocr.Reader(["en"], gpu=False, verbose=False)
            return [(r[0], str(r[1]), float(r[2])) for r in _ocr_models[engine].readtext(png)]
    raise ValueError(f"unknown OCR engine {engine}")


def _ocr_png(png: bytes, engine: str) -> str:
    """OCR one rendered page; returns an ``OcrText`` (a str with a quality score)."""
    if engine == "tesseract":
        proc = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", "6"],
            input=png,
            capture_output=True,
            timeout=_OCR_TIMEOUT,
            check=True,
        )
        return proc.stdout.decode("utf-8", "replace")
    items = []
    score = 0.0
    for box, text, conf in _ocr_results(png, engine):
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        items.append((min(xs), (min(ys) + max(ys)) / 2, height, text))
        if width >= height:  # sideways lines mean the page is probably turned
            score += conf * sum(ch.isalnum() for ch in text)
    return OcrText(_boxes_to_text(items), score)


def _render_png(page: Any, dpi: int, rotation: int = 0) -> bytes:
    import pymupdf

    longest = max(page.rect.width, page.rect.height, 1.0) / 72  # inches
    zoom = min(dpi, _OCR_MAX_PIXELS / longest) / 72
    matrix = pymupdf.Matrix(zoom, zoom)
    if rotation:
        matrix = matrix.prerotate(rotation)
    return page.get_pixmap(matrix=matrix).tobytes("png")


def _ocr_page(page: Any, engine: str, dpi: int, rotation: int, name: str) -> str:
    try:
        return _ocr_png(_render_png(page, dpi, rotation), engine)
    except Exception:
        log.warning(
            "OCR (%s) failed on %s page %d (%d dpi, %d deg)",
            engine,
            name,
            page.number + 1,
            dpi,
            rotation,
            exc_info=True,
        )
        return ""


def _reads_poorly(text: str, floor: float) -> bool:
    score = _ocr_score(text)
    return score < floor or score < _MIN_OCR_CONFIDENCE * _meaningful_chars(text)


def _median(values: list[float]) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def _ocr_pages(data: bytes, page_texts: list[str], name: str) -> list[str]:
    """OCR scanned pages whose text is missing or far below the rest of the document.

    A page is a candidate if it carries an image and yields fewer than ``_MIN_PAGE_CHARS``
    characters, or under ``_LOW_YIELD_RATIO`` of the document's median page. Candidates are
    read upright at ``_OCR_DPI``; image-only pages that still read poorly (little or
    low-confidence text, or sideways lines) are read again at ``_OCR_RETRY_DPI`` and turned
    90/180/270 degrees, keeping the best-scoring reading. A page with some text layer keeps
    it unless OCR reads clearly more. Raises ``NeedsOCRError`` if the document has
    no readable text at all.
    """
    import pymupdf

    yields = [_meaningful_chars(t) for t in page_texts]
    text_median = _median([y for y in yields if y >= _MIN_PAGE_CHARS])
    low = max(_MIN_PAGE_CHARS, _LOW_YIELD_RATIO * text_median)
    if all(y >= low for y in yields):
        return page_texts
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        # Only pages carrying an image can be scans; blank or short text pages are left alone.
        todo = [
            i
            for i, y in enumerate(yields)
            if y < low and i < len(pdf) and pdf[i].get_images(full=True)
        ]
        if not todo:
            return page_texts
        engine = ocr_engine()
        out = list(page_texts)
        if engine:
            started = time.monotonic()
            read = {i: _ocr_page(pdf[i], engine, _OCR_DPI, 0, name) for i in todo}
            # Judge each reading against the document's typical page, OCR'd pages included.
            median = _median(
                [_ocr_score(read[i]) if i in read else y for i, y in enumerate(yields)]
            )
            floor = max(_MIN_PAGE_CHARS, _LOW_YIELD_RATIO * median)
            # Text-layer pages were only checked for missing scanned text: one reading.
            retried = [
                i for i in todo if yields[i] < _MIN_PAGE_CHARS and _reads_poorly(read[i], floor)
            ]
            for i in retried:
                tries = [(_OCR_RETRY_DPI, 0)] + [(_OCR_DPI, r) for r in _OCR_ROTATIONS]
                for dpi, rot in tries:
                    text = _ocr_page(pdf[i], engine, dpi, rot, name)
                    if _ocr_score(text) > _ocr_score(read[i]):
                        read[i] = text
            for i in todo:
                # A page with some text layer keeps it unless OCR reads clearly more.
                if yields[i] < _MIN_PAGE_CHARS or _ocr_score(read[i]) > 2 * yields[i] + low:
                    out[i] = str(read[i])
            log.info(
                "%s: OCR (%s) of %d page(s) in %.1f s; %d re-read at %d dpi / turned; "
                "chars per page %s",
                name,
                engine,
                len(todo),
                time.monotonic() - started,
                len(retried),
                _OCR_RETRY_DPI,
                ", ".join(f"p{i + 1}: {yields[i]}->{_meaningful_chars(out[i])}" for i in todo),
            )
        still = [i for i in todo if _meaningful_chars(out[i]) < _MIN_PAGE_CHARS]
    total = len(page_texts)
    if total and sum(_meaningful_chars(t) for t in out) < _MIN_PAGE_CHARS:
        tried = (
            f"OCR ({engine}, {_OCR_DPI} and {_OCR_RETRY_DPI} dpi, 4 orientations) found no "
            "readable text; the pages may be blank, handwritten or too faint"
            if engine
            else "no OCR engine available (install the tesseract-ocr binary, the "
            "rapidocr-onnxruntime or easyocr Python package, or the docling extra)"
        )
        raise NeedsOCRError(
            f"needs OCR, no text layer: {len(still)} of {total} pages are scanned images with "
            f"no extractable text; {tried}"
        )
    if still:
        log.warning(
            "%s: %d of %d pages have no text layer and %s (pages %s)",
            name,
            len(still),
            total,
            "OCR found no readable text on them" if engine else "were not OCR'd",
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
        page_texts = _ocr_pages(data, _pdf_page_texts(path, data), doc.filename)
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
    rf"\b(?:{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{{4}}"
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
    """Best-guess authoring date: EML ``Date`` header, else a date-like line near the top.

    A ``Date:`` line that is the incident date of a where/when block ("Location: ...",
    "Date: ...", "Time: ...") is not the authoring date and is not preferred.
    """
    lines = parsed.text.split("\n")[:_DOC_DATE_SCAN_LINES]
    if parsed.doc.filename.lower().endswith(".eml"):
        for line in lines:
            if line.startswith("Date:"):
                if found := _email_header_date(line[5:]):
                    return found
                break
    # Prefer explicit "Date: ..." lines, then any line with a full calendar date.
    offset = 0
    for line in lines:
        m = _DATE_LINE_RE.match(line)
        if m and not _is_incident_field(parsed.text, offset + m.start(1)):
            if (dm := _DATE_RE.search(m.group(1))) and (found := _parse_date_str(dm.group(0))):
                return found
        offset += len(line) + 1
    for line in lines:
        if dm := _DATE_RE.search(line):
            if found := _parse_date_str(dm.group(0)):
                return found
    return None


# --------------------------------------------------------------------------- incident date
# The date of the occurrence a document is about (accident, incident, loss, offense), as
# given by a header or form field ("Date of Accident: ...", "Date: June 5, 2021, 0940 CDT"
# in a "Location: / Date: / Time:" block), or by an early sentence ("the collision ... on
# June 5, 2021"). This is not the authoring date: an investigation report, statement or
# letter is often written weeks after the occurrence it describes.

_INCIDENT_SCAN_CHARS = 6000
_INCIDENT_NOUN = (
    r"(?:accident|incident|occurrence|crash|collision|mishap|loss|injury|offen[cs]e|event)"
)
_NUM_DATE = r"\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})\b"
_ANY_DATE_RE = re.compile(rf"{_DATE_RE.pattern}|\b{_NUM_DATE}", re.IGNORECASE)
# Form labels that name the occurrence date ("Date of Accident", "Date/Time of Incident
# (local)", "Incident Date", "Date of Loss", "Occurred On").
_INCIDENT_LABEL_RE = re.compile(
    rf"\b(?:date\s*(?:/\s*time)?\s*of\s*(?:the\s+)?{_INCIDENT_NOUN}(?:\s*/\s*{_INCIDENT_NOUN})?"
    rf"|{_INCIDENT_NOUN}\s+(?:date|dt)(?:\s*/\s*time)?"
    r"|(?:inc(?:ident)?\.?\s+)?(?:occurred|happened)(?:\s+(?:on|date|start|from))?"
    r"(?!\s+(?:address|location|place|end|to|until|at)\b))"
    r"(?:\s*\([^)\n]{0,20}\))?\s*[:\-]?",
    re.IGNORECASE,
)
# Another date field: the value of an incident label never extends past one.
_OTHER_DATE_LABEL_RE = re.compile(r"\b(?:date|dt|dated|time\s*of)\b", re.IGNORECASE)
_DATE_FIELD_RE = re.compile(r"(?<![\w/])date\s*(?:/\s*time)?\s*:\s*", re.IGNORECASE)
_LOCATION_FIELD_RE = re.compile(r"\blocation\s*:", re.IGNORECASE)
# A short section heading that names the occurrence ("B. VEHICLE ACCIDENT", "2. INCIDENT").
_INCIDENT_HEADING_RE = re.compile(
    rf"^[\W_]*(?:[A-Z0-9]{{1,3}}[.)]\s+)?[A-Z][A-Z0-9 ,/&'-]{{0,50}}\b{_INCIDENT_NOUN.upper()}\b"
    r"[A-Z0-9 ,/&'-]{0,30}$"
)
# Prose: "the collision involving ... on June 5, 2021", "Incident occurred Saturday, June 5, 2021".
_INCIDENT_PROSE_RE = re.compile(
    r"\b(?:accident|incident|crash|collision|mishap|wreck)\b(?:(?!\n\s*\n)[^.;]){0,100}?"
    r"\b(?:on|of|dated|(?:that\s+|which\s+)?(?:occurred|happened)(?:\s+on)?)\s+"
    r"(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day,?\s+)?(?=\S)",
    re.IGNORECASE,
)


def _is_incident_field(text: str, pos: int) -> bool:
    """The "Date:" value at `pos` sits in a where/when block or under an incident heading."""
    before = text[max(0, pos - 160) : pos]
    if _LOCATION_FIELD_RE.search(before):
        return True
    prev = [ln.strip() for ln in before.split("\n")[:-1] if ln.strip()]
    return bool(prev) and _INCIDENT_HEADING_RE.match(prev[-1]) is not None


def _parse_any_date(s: str) -> date | None:
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})", s.strip())
    if m:  # numeric form fields are month/day/year
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y < 70 else 1900
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    return _parse_date_str(s)


def _leading_date(s: str, max_offset: int = 0) -> date | None:
    """A date at (or within `max_offset` characters of) the start of `s`."""
    s = s.lstrip()
    m = _ANY_DATE_RE.search(s)
    if m is None or m.start() > max_offset:
        return None
    return _parse_any_date(m.group(0))


def incident_date(parsed: ParsedDocument) -> date | None:
    """The date of the occurrence the document is about, from its header, form or opening.

    Looks, in order, for an explicit incident-date form field ("Date of Accident:",
    "Incident Date", "Date/Time of Loss"), then a "Date:" field in a
    "Location: / Date:" block or right under an accident/incident/event heading, then an
    early sentence tying the occurrence to a date ("the collision ... on June 5, 2021"). None
    when the document names no such date; a plain "Date:" header (the authoring date) is
    never taken for it.
    """
    head = parsed.text[:_INCIDENT_SCAN_CHARS]
    for m in _INCIDENT_LABEL_RE.finditer(head):
        window = head[m.end() : m.end() + 160]
        if stop := _OTHER_DATE_LABEL_RE.search(window):
            window = window[: stop.start()]
        if found := _leading_date(window, len(window)):
            return found

    authored = doc_date(parsed)
    for m in _DATE_FIELD_RE.finditer(head):
        if not _is_incident_field(head, m.start()):
            continue
        found = _leading_date(head[m.end() : m.end() + 40])
        if found is not None and found != authored:
            return found

    for m in _INCIDENT_PROSE_RE.finditer(head):
        if found := _leading_date(head[m.end() : m.end() + 40]):
            return found
    return None
