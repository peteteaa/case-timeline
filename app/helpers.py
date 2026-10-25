"""Pure helpers for the web UI: request parsing, safe HTML building, display formatting.

Nothing here touches the database; ``app.app`` wires these into routes and templates.
"""

from __future__ import annotations

import ipaddress
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Any
from urllib.parse import quote, unquote, urlencode

from markupsafe import Markup, escape

from pipeline.schema import (
    CERTAINTIES,
    DOC_KINDS,
    REVIEWS,
    STATUSES,
    EvidenceLink,
    Fact,
    FactQuery,
    ParsedDocument,
)

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
USER_COOKIE = "ct_user"
MAX_NAME_LEN = 60

SORT_KEYS = ("date", "title", "evidence", "tags", "witnesses", "notes", "created", "updated")
SORT_LABELS = {
    "date": "Date",
    "title": "Title",
    "evidence": "Evidence",
    "tags": "Tags",
    "witnesses": "Witnesses",
    "notes": "Notes",
    "created": "Created",
    "updated": "Updated",
}
STATUS_LABELS = {"key_fact": "Key fact", "case_impact": "Case impact", "disputed": "Disputed"}
ALERT_LABELS = {
    "contradiction": "Possible conflict between sources",
    "flagged_witness": "Flagged witness mentioned",
    "possible_deadline": "Possible deadline",
}
FIELD_LABELS = {"date": "Date", "actor": "People involved", "fact": "What happened"}

_FACT_ID_RE = re.compile(r"\bF-\d{4,}\b")
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


# --------------------------------------------------------------------------- network


def host_is_loopback(host: str) -> bool:
    """True for 127.0.0.0/8, ::1 and localhost."""
    host = host.strip().strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def hostname(host_header: str) -> str:
    """The host part of a Host header (no port, IPv6 brackets removed), lowercased."""
    host_header = host_header.strip().lower()
    if host_header.startswith("["):
        return host_header[1 : host_header.find("]")] if "]" in host_header else host_header
    return host_header.rsplit(":", 1)[0] if host_header.count(":") == 1 else host_header


def safe_next(target: str | None, default: str) -> str:
    """A same-site path to redirect to after a form post (never another origin)."""
    if not target or not isinstance(target, str):
        return default
    target = target.strip()
    if (
        not target.startswith("/")
        or target.startswith("//")
        or "\\" in target
        or any(ord(ch) < 32 for ch in target)
    ):
        return default
    return target


# --------------------------------------------------------------------------- files and names


def sanitize_filename(name: str) -> str | None:
    """Reduce an uploaded filename to a safe basename, or None if nothing usable remains."""
    name = unicodedata.normalize("NFKC", name or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    name = re.sub(r"\.{2,}", ".", name)
    if not name or name in {".", ".."}:
        return None
    stem, dot, suffix = name.rpartition(".")
    if dot and len(name) > 150:
        name = stem[: 150 - len(suffix) - 1] + "." + suffix
    return name[:150]


def clean_display_name(raw: str | None) -> str | None:
    """A usable display name (collapsed whitespace, no control chars), or None."""
    if not raw:
        return None
    name = unicodedata.normalize("NFKC", raw)
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = " ".join(name.split())[:MAX_NAME_LEN].strip()
    if not name or name.casefold() == "agent":  # "agent" is reserved for the pipeline
        return None
    return name


def encode_user_cookie(name: str) -> str:
    return quote(name, safe="")


def decode_user_cookie(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return clean_display_name(unquote(value))
    except Exception:  # noqa: BLE001 - a mangled cookie just means "ask again"
        return None


def valid_color(value: str | None) -> str | None:
    """``#rrggbb`` or None; anything else is dropped (colors go into style attributes)."""
    if value and _COLOR_RE.match(value.strip()):
        return value.strip().lower()
    return None


# --------------------------------------------------------------------------- query parsing


class ParamGetter:
    """Uniform access to multi-valued request data (query params, form data or JSON)."""

    def __init__(self, data: Any) -> None:
        self._data = data

    def has(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> str | None:
        if isinstance(self._data, Mapping) and not hasattr(self._data, "getlist"):
            value = self._data.get(key)
            if isinstance(value, list):
                value = value[0] if value else None
            return None if value is None else str(value)
        value = self._data.get(key)
        return None if value is None else str(value)

    def getlist(self, key: str) -> list[str]:
        if hasattr(self._data, "getlist"):
            values = list(self._data.getlist(key))
        else:
            raw = self._data.get(key)
            if raw is None:
                values = []
            elif isinstance(raw, list | tuple):
                values = list(raw)
            else:
                values = [raw]
        out: list[str] = []
        for v in values:
            if v is None:
                continue
            for part in str(v).split(",") if key in _COMMA_LISTS else [str(v)]:
                part = part.strip()
                if part and part not in out:
                    out.append(part)
        return out


_COMMA_LISTS = {"aliases"}


def _parse_date(value: str | None, label: str, errors: list[str]) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        errors.append(f"{label} must be a date like 2024-03-01")
        return None


def parse_filters(params: ParamGetter) -> tuple[dict[str, Any], list[str]]:
    """Normalized list-view filters from query params, plus friendly errors for bad values.

    Returns a dict with: date_from, date_to (date | None), status, tag, witness, doc, kind,
    review (lists), origin, q, sort, dir, view.
    """
    errors: list[str] = []
    f: dict[str, Any] = {
        "date_from": _parse_date(params.get("date_from"), "From date", errors),
        "date_to": _parse_date(params.get("date_to"), "To date", errors),
        "status": [s for s in params.getlist("status") if s in STATUSES],
        "tag": params.getlist("tag"),
        "witness": params.getlist("witness"),
        "doc": params.getlist("doc"),
        "kind": [k for k in params.getlist("kind") if k in DOC_KINDS],
        "review": [r for r in params.getlist("review") if r in REVIEWS] or ["draft", "accepted"],
        "origin": params.get("origin") if params.get("origin") in ("ai", "user") else "",
        "q": (params.get("q") or "").strip()[:500],
        "sort": params.get("sort") if params.get("sort") in SORT_KEYS else "date",
        "dir": "desc" if params.get("dir") == "desc" else "asc",
        "view": "visual" if params.get("view") == "visual" else "list",
    }
    return f, errors


def build_query(f: Mapping[str, Any], fact_ids: list[str] | None = None) -> FactQuery:
    return FactQuery(
        date_from=f["date_from"],
        date_to=f["date_to"],
        statuses=list(f["status"]),
        tag_ids=list(f["tag"]),
        witness_ids=list(f["witness"]),
        doc_ids=list(f["doc"]),
        doc_kinds=list(f["kind"]),
        reviews=list(f["review"]),
        origins=[f["origin"]] if f["origin"] else [],
        fact_ids=fact_ids,
        sort=f["sort"],
        descending=f["dir"] == "desc",
    )


def filter_pairs(f: Mapping[str, Any], *, skip: Iterable[str] = ()) -> list[tuple[str, str]]:
    """The filters as (name, value) pairs for URLs and hidden inputs (defaults omitted)."""
    skip = set(skip)
    pairs: list[tuple[str, str]] = []
    for key in ("date_from", "date_to"):
        if f.get(key) and key not in skip:
            pairs.append((key, f[key].isoformat()))
    for key in ("status", "tag", "witness", "doc", "kind"):
        if key not in skip:
            pairs += [(key, v) for v in f.get(key) or []]
    if "review" not in skip and list(f.get("review") or []) != ["draft", "accepted"]:
        pairs += [("review", v) for v in f["review"]]
    for key in ("origin", "q"):
        if f.get(key) and key not in skip:
            pairs.append((key, f[key]))
    if "sort" not in skip and f.get("sort", "date") != "date":
        pairs.append(("sort", f["sort"]))
    if "dir" not in skip and f.get("dir") == "desc":
        pairs.append(("dir", "desc"))
    if "view" not in skip and f.get("view") == "visual":
        pairs.append(("view", "visual"))
    return pairs


def query_string(pairs: Sequence[tuple[str, str]], **extra: str) -> str:
    items = list(pairs) + [(k, v) for k, v in extra.items() if v is not None]
    return urlencode(items)


def filters_active(f: Mapping[str, Any]) -> bool:
    return bool(filter_pairs(f, skip=("sort", "dir", "view")))


# --------------------------------------------------------------------------- fact edits


_TEXT_FIELDS = (
    "title",
    "details",
    "date_text",
    "date_start",
    "date_end",
    "date_certainty",
    "anchor_fact_id",
)
_LIST_FIELDS = ("statuses", "tag_ids", "witness_ids")


def fact_changes(params: ParamGetter) -> dict[str, Any]:
    """Editable fact fields present in a form or JSON body.

    Checkbox lists can't say "none selected", so a form lists the list fields it carries in
    ``has`` (e.g. ``has=statuses``); JSON bodies just include the key.
    """
    changes: dict[str, Any] = {}
    declared = set(params.getlist("has"))
    for key in _TEXT_FIELDS:
        if params.has(key):
            value = (params.get(key) or "").strip()
            if key == "title":
                changes[key] = value
            elif key == "date_certainty":
                if value:
                    if value not in CERTAINTIES:
                        raise ValueError(f"bad date certainty {value!r}")
                    changes[key] = value
            else:
                changes[key] = value or None
    for key in _LIST_FIELDS:
        if params.has(key) or key in declared:
            changes[key] = params.getlist(key)
    return changes


def parse_int(value: str | None, label: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a whole number") from exc


# --------------------------------------------------------------------------- display


def fmt_date(d: date | None) -> str:
    return d.isoformat() if d else ""


def fact_date(f: Fact) -> str:
    """'2024-03-03', '~2022-03-01 – 2022-03-31', or '' for undated facts."""
    if f.date_start is None:
        return ""
    text = f.date_start.isoformat()
    if f.date_end and f.date_end != f.date_start:
        text += " – " + f.date_end.isoformat()
    if f.date_certainty in ("approximate", "relative"):
        text = "~" + text
    return text


def fmt_dt(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M UTC")


def pct(value: float | None) -> str:
    return "" if value is None else f"{round(value * 100)}%"


def evidence_cite(ev: EvidenceLink) -> str:
    """'p. 3', 'p. 3:12–3:15' for transcripts, or '' when unknown."""
    if ev.page is None and ev.line_start is None:
        return ""
    if ev.line_start is not None:
        page = f"{ev.page}:" if ev.page is not None else "line "
        cite = f"{page}{ev.line_start}"
        if ev.line_end is not None and ev.line_end != ev.line_start:
            cite += f"–{page}{ev.line_end}" if ev.page is not None else f"–{ev.line_end}"
        return f"p. {cite}" if ev.page is not None else cite
    return f"p. {ev.page}"


def history_value(value: Any, names: Mapping[str, str]) -> str:
    if value is None or value == "" or value == []:
        return "—"
    if isinstance(value, list):
        return ", ".join(names.get(str(v), STATUS_LABELS.get(str(v), str(v))) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return names.get(str(value), str(value))


def linkify_fact_ids(text: str, known: Callable[[str], bool]) -> Markup:
    """Escape ``text``, then link each F-#### that names a fact of this case to its card."""
    html = str(escape(text))

    def link(m: re.Match[str]) -> str:
        fid = m.group(0)
        if not known(fid):
            return fid
        return f'<a class="fact-ref" href="/facts/{quote(fid, safe="")}">{fid}</a>'

    return Markup(_FACT_ID_RE.sub(link, html))


def fact_ids_in(text: str) -> list[str]:
    out: list[str] = []
    for fid in _FACT_ID_RE.findall(text or ""):
        if fid not in out:
            out.append(fid)
    return out


# --------------------------------------------------------------------------- source viewer


def source_segments(
    parsed: ParsedDocument, hl_start: int | None = None, hl_end: int | None = None
) -> list[dict[str, Any]]:
    """Split a document into page segments rendered so DOM text offsets equal stored offsets.

    Each segment's ``html`` is one ``<span class="seg" data-start=N>`` whose text content is
    exactly ``text[start:end]``; the highlighted span (if any) is wrapped in ``<mark>`` pieces,
    the first with ``id="hl"``. Everything is HTML-escaped here, in Python, so template
    whitespace can never shift offsets.
    """
    text = parsed.text
    bounds: list[tuple[int | None, int, int]] = []
    pos = 0
    for p in sorted(parsed.pages, key=lambda p: p.char_start):
        start, end = max(p.char_start, pos), min(p.char_end, len(text))
        if start > pos:  # text between pages (rare): its own unnumbered segment
            bounds.append((None, pos, start))
        if end > start:
            bounds.append((p.number, start, end))
            pos = end
    if pos < len(text) or not bounds:
        bounds.append((None, pos, len(text)))

    highlight = hl_start is not None and hl_end is not None and 0 <= hl_start < hl_end <= len(text)
    first_mark = True
    segments: list[dict[str, Any]] = []
    for number, start, end in bounds:
        parts: list[str] = []
        cuts = [start, end]
        if highlight:
            cuts += [c for c in (hl_start, hl_end) if start < c < end]  # type: ignore[operator]
        cuts = sorted(set(cuts))
        for a, b in zip(cuts, cuts[1:], strict=False):
            piece = str(escape(text[a:b]))
            if highlight and hl_start <= a and b <= hl_end:  # type: ignore[operator]
                mark_id = ' id="hl"' if first_mark else ""
                first_mark = False
                parts.append(f'<mark class="hl"{mark_id}>{piece}</mark>')
            else:
                parts.append(piece)
        html = f'<span class="seg" data-start="{start}" data-end="{end}">{"".join(parts)}</span>'
        segments.append({"page": number, "start": start, "end": end, "html": Markup(html)})
    return segments
