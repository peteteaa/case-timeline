"""SQLite (WAL) persistence for the fact model: FTS5 keyword index and sqlite-vec vectors.

One ``Store`` per process; its connection is guarded by a re-entrant lock and every write is a
short ``BEGIN IMMEDIATE`` transaction, so the app server's worker, the UI and the MCP server can
share one database file.

The store enforces SPEC §6.2: writes made ``by=AGENT`` may only create drafts, attach draft
evidence, suggest witnesses, open contradictions and touch job/alert rows. Everything else
raises :class:`PermissionError`. Span evidence is checked verbatim against the stored document
text before it is written.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pipeline.schema import (
    AGENT,
    ALERT_TYPES,
    CERTAINTIES,
    DOC_KINDS,
    STATUSES,
    Alert,
    Contradiction,
    Document,
    EvidenceLink,
    Fact,
    FactQuery,
    Job,
    Line,
    Note,
    Page,
    ParsedDocument,
    ReviewEvent,
    SpanIn,
    Tag,
    Witness,
)

log = logging.getLogger(__name__)

EMBED_DIM = 1024
_CHUNK = 500  # max ids per IN (...) clause

Embedder = Callable[[list[str]], list[list[float]]]

# Fields a person may change with update_fact (SPEC §5.2 PATCH /facts/{id}).
EDITABLE_FACT_FIELDS = (
    "title",
    "details",
    "date_text",
    "date_start",
    "date_end",
    "date_certainty",
    "anchor_fact_id",
    "statuses",
    "tag_ids",
    "witness_ids",
)
_PEOPLE_ONLY_FIELDS = ("statuses", "tag_ids")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS counters (
    prefix TEXT PRIMARY KEY,
    next INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    location TEXT NOT NULL,
    kind TEXT NOT NULL,
    kind_set_by TEXT,
    pages INTEGER,
    parse_status TEXT NOT NULL,
    parse_error TEXT,
    added_at TEXT NOT NULL,
    drafted_at TEXT,
    UNIQUE (case_id, sha256)
);

CREATE TABLE IF NOT EXISTS doc_text (
    doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id),
    text TEXT NOT NULL,
    pages TEXT NOT NULL DEFAULT '[]',
    lines TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS facts (
    pk INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT NOT NULL UNIQUE,
    case_id TEXT NOT NULL,
    title TEXT NOT NULL,
    details TEXT,
    date_text TEXT,
    date_start TEXT,
    date_end TEXT,
    date_certainty TEXT NOT NULL DEFAULT 'unknown',
    anchor_fact_id TEXT,
    statuses TEXT NOT NULL DEFAULT '[]',
    origin TEXT NOT NULL,
    review TEXT NOT NULL,
    confidence REAL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_by TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    vec_dirty INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS facts_case_date ON facts(case_id, date_start);
CREATE INDEX IF NOT EXISTS facts_case_review ON facts(case_id, review);

CREATE TABLE IF NOT EXISTS evidence (
    pk INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id TEXT NOT NULL UNIQUE,
    fact_id TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    doc_id TEXT,
    page INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    line_start INTEGER,
    line_end INTEGER,
    quote TEXT,
    reference TEXT,
    origin TEXT NOT NULL,
    review TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_fact ON evidence(fact_id);
CREATE INDEX IF NOT EXISTS evidence_doc ON evidence(doc_id);

CREATE TABLE IF NOT EXISTS witnesses (
    witness_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    role TEXT,
    notes TEXT,
    flagged INTEGER NOT NULL DEFAULT 0,
    confirmed INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS witnesses_case ON witnesses(case_id);

CREATE TABLE IF NOT EXISTS fact_witnesses (
    fact_id TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    witness_id TEXT NOT NULL REFERENCES witnesses(witness_id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, witness_id)
);
CREATE INDEX IF NOT EXISTS fact_witnesses_w ON fact_witnesses(witness_id);

CREATE TABLE IF NOT EXISTS tags (
    tag_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    name TEXT NOT NULL,
    color TEXT,
    UNIQUE (case_id, name COLLATE NOCASE)
);

CREATE TABLE IF NOT EXISTS fact_tags (
    fact_id TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, tag_id)
);
CREATE INDEX IF NOT EXISTS fact_tags_t ON fact_tags(tag_id);

CREATE TABLE IF NOT EXISTS notes (
    note_id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    author TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_fact ON notes(fact_id);

CREATE TABLE IF NOT EXISTS review_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    evidence_id TEXT,
    action TEXT NOT NULL,
    reason TEXT,
    reviewer TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS review_events_case ON review_events(case_id, at);

CREATE TABLE IF NOT EXISTS fact_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT NOT NULL,
    field TEXT NOT NULL,
    old TEXT,
    new TEXT,
    by TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS fact_history_fact ON fact_history(fact_id, id);

CREATE TABLE IF NOT EXISTS contradictions (
    contradiction_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    key TEXT NOT NULL UNIQUE,
    fact_ids TEXT NOT NULL,
    field TEXT NOT NULL,
    explanation TEXT NOT NULL,
    status TEXT NOT NULL,
    reviewed_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS contradictions_case ON contradictions(case_id, created_at);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    type TEXT NOT NULL,
    fact_ids TEXT NOT NULL DEFAULT '[]',
    doc_ids TEXT NOT NULL DEFAULT '[]',
    job_id TEXT,
    created_at TEXT NOT NULL,
    delivered_at TEXT
);
CREATE INDEX IF NOT EXISTS alerts_case ON alerts(case_id, created_at);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL,
    counts TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    queued_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, queued_at);
CREATE INDEX IF NOT EXISTS jobs_case ON jobs(case_id, queued_at);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    title, details, notes, quotes, tokenize = 'porter unicode61'
);
"""

_VEC_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS fact_vec USING vec0(
    embedding float[{EMBED_DIM}] distance_metric=cosine,
    case_id text
);
"""


# --------------------------------------------------------------------------- helpers


def now() -> datetime:
    return datetime.now(UTC)


def _ts(value: datetime | None) -> str | None:
    """UTC ISO string (naive datetimes are taken as UTC)."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _now_ts() -> str:
    return _ts(now())  # type: ignore[return-value]


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _d(value: date | str | None) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return date.fromisoformat(value[:10]).isoformat()
    return value.isoformat()


def _chunks(ids: Sequence[str]) -> Iterator[list[str]]:
    for i in range(0, len(ids), _CHUNK):
        yield list(ids[i : i + _CHUNK])


def _marks(n: int) -> str:
    return ",".join("?" * n)


def _is_agent(by: str) -> bool:
    return by == AGENT


def _require_person(by: str, what: str) -> None:
    if _is_agent(by):
        raise PermissionError(f"the agent may not {what} (SPEC §6.2)")
    if not by or not by.strip():
        raise ValueError("a display name is required")


def _norm_name(name: str) -> str:
    return " ".join(name.split()).casefold()


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


class Store:
    """Thread-safe store: one connection guarded by a re-entrant lock."""

    def __init__(self, db_path: str | Path, embedder: Embedder | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=30.0, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._vec = self._load_vec()
        self._init()

    # ------------------------------------------------------------------ setup

    def _load_vec(self) -> bool:
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except Exception as exc:  # noqa: BLE001 - degrade to FTS-only on any failure
            log.warning("sqlite-vec unavailable (%s); search is keyword-only", exc)
            return False

    def _init(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(_SCHEMA)
            if self._vec:
                try:
                    self._conn.executescript(_VEC_SCHEMA)
                except sqlite3.Error as exc:
                    log.warning("could not create vec0 table (%s); keyword-only search", exc)
                    self._vec = False

    @property
    def vector_enabled(self) -> bool:
        return self._vec

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """A short write transaction (``BEGIN IMMEDIATE``); nested calls join the outer one."""
        with self._lock:
            if self._conn.in_transaction:
                yield self._conn
                return
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _q(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _q1(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _next_id(self, prefix: str) -> str:
        """Allocate the next short id ("F-0142"); call inside a transaction."""
        row = self._conn.execute(
            """INSERT INTO counters (prefix, next) VALUES (?, 2)
               ON CONFLICT(prefix) DO UPDATE SET next = next + 1
               RETURNING next - 1 AS n""",
            (prefix,),
        ).fetchone()
        return f"{prefix}-{row['n']:04d}"

    # ------------------------------------------------------------------ documents

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            case_id=row["case_id"],
            filename=row["filename"],
            sha256=row["sha256"],
            location=row["location"],
            kind=row["kind"],
            pages=row["pages"],
            parse_status=row["parse_status"],
            added_at=_dt(row["added_at"]),
        )

    def register_document(
        self,
        case_id: str,
        *,
        filename: str,
        rel_path: str,
        sha256: str,
        location: str,
        kind: str = "other",
    ) -> tuple[Document, bool]:
        """Add a document unless one with the same content hash exists; ``(doc, created)``."""
        if kind not in DOC_KINDS:
            raise ValueError(f"unknown document kind {kind!r}")
        with self._tx() as c:
            row = c.execute(
                "SELECT * FROM documents WHERE case_id = ? AND sha256 = ?", (case_id, sha256)
            ).fetchone()
            if row is not None:
                return self._row_to_document(row), False
            doc_id = self._next_id("D")
            c.execute(
                """INSERT INTO documents (doc_id, case_id, filename, rel_path, sha256, location,
                       kind, parse_status, added_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (doc_id, case_id, filename, rel_path, sha256, location, kind, _now_ts()),
            )
            row = c.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._row_to_document(row), True

    def get_document(self, doc_id: str) -> Document | None:
        row = self._q1("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return self._row_to_document(row) if row else None

    def get_document_by_sha(self, case_id: str, sha256: str) -> Document | None:
        row = self._q1(
            "SELECT * FROM documents WHERE case_id = ? AND sha256 = ?", (case_id, sha256)
        )
        return self._row_to_document(row) if row else None

    def list_documents(self, case_id: str) -> list[Document]:
        rows = self._q(
            "SELECT * FROM documents WHERE case_id = ? ORDER BY added_at, doc_id", (case_id,)
        )
        return [self._row_to_document(r) for r in rows]

    def document_source(self, doc_id: str) -> tuple[str, str] | None:
        """``(location, rel_path)`` of a document's file, relative to its root folder."""
        row = self._q1("SELECT location, rel_path FROM documents WHERE doc_id = ?", (doc_id,))
        return (row["location"], row["rel_path"]) if row else None

    def document_info(self, doc_id: str) -> dict[str, Any] | None:
        """Bookkeeping columns not in the Document contract (parse error, drafted time)."""
        row = self._q1(
            "SELECT parse_error, drafted_at, kind_set_by, rel_path FROM documents "
            "WHERE doc_id = ?",
            (doc_id,),
        )
        if row is None:
            return None
        return {
            "parse_error": row["parse_error"],
            "drafted_at": _dt(row["drafted_at"]),
            "kind_set_by": row["kind_set_by"],
            "rel_path": row["rel_path"],
        }

    def documents_to_process(self, case_id: str) -> list[Document]:
        """Documents never parsed, or parsed but not yet drafted (failed ones wait for reprocess)."""
        rows = self._q(
            """SELECT * FROM documents WHERE case_id = ? AND (parse_status = 'pending'
                   OR (parse_status = 'parsed' AND drafted_at IS NULL))
               ORDER BY added_at, doc_id""",
            (case_id,),
        )
        return [self._row_to_document(r) for r in rows]

    def set_parse_result(
        self,
        doc_id: str,
        status: str,
        *,
        pages: int | None = None,
        error: str | None = None,
        kind: str | None = None,
    ) -> None:
        """Record parse status; ``kind`` is applied only if no person has set the kind."""
        if status not in ("pending", "parsed", "failed"):
            raise ValueError(f"bad parse status {status!r}")
        with self._tx() as c:
            c.execute(
                "UPDATE documents SET parse_status = ?, pages = COALESCE(?, pages), "
                "parse_error = ? WHERE doc_id = ?",
                (status, pages, error, doc_id),
            )
            if kind is not None:
                if kind not in DOC_KINDS:
                    raise ValueError(f"unknown document kind {kind!r}")
                c.execute(
                    "UPDATE documents SET kind = ? WHERE doc_id = ? AND kind_set_by IS NULL",
                    (kind, doc_id),
                )

    def set_document_kind(self, doc_id: str, kind: str, by: str) -> Document:
        """A person corrects the document kind (SPEC §6.5)."""
        _require_person(by, "change document kinds")
        if kind not in DOC_KINDS:
            raise ValueError(f"unknown document kind {kind!r}")
        with self._tx() as c:
            cur = c.execute(
                "UPDATE documents SET kind = ?, kind_set_by = ? WHERE doc_id = ?",
                (kind, by, doc_id),
            )
            if cur.rowcount == 0:
                raise KeyError(doc_id)
        return self.get_document(doc_id)  # type: ignore[return-value]

    def save_parsed_text(
        self, doc_id: str, text: str, pages: Sequence[Page] = (), lines: Sequence[Line] = ()
    ) -> None:
        pages_json = json.dumps([Page.model_validate(p).model_dump() for p in pages])
        lines_json = json.dumps([Line.model_validate(x).model_dump() for x in lines])
        with self._tx() as c:
            c.execute(
                """INSERT INTO doc_text (doc_id, text, pages, lines) VALUES (?, ?, ?, ?)
                   ON CONFLICT(doc_id) DO UPDATE SET text = excluded.text,
                       pages = excluded.pages, lines = excluded.lines""",
                (doc_id, text, pages_json, lines_json),
            )

    def get_parsed(self, doc_id: str) -> ParsedDocument | None:
        """Stored text, pages and transcript lines, or None if the document isn't parsed."""
        row = self._q1("SELECT text, pages, lines FROM doc_text WHERE doc_id = ?", (doc_id,))
        doc = self.get_document(doc_id)
        if row is None or doc is None:
            return None
        return ParsedDocument(
            doc=doc,
            text=row["text"],
            pages=[Page.model_validate(p) for p in json.loads(row["pages"])],
            lines=[Line.model_validate(x) for x in json.loads(row["lines"])],
        )

    def mark_drafted(self, doc_id: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE documents SET drafted_at = ? WHERE doc_id = ?", (_now_ts(), doc_id))

    def reset_document(self, doc_id: str) -> int:
        """Prepare a document to be drafted again; returns how many AI drafts were removed.

        Removes draft AI facts whose evidence all comes from this document, and draft AI
        evidence from this document on other facts. Reviewed facts and evidence are kept.
        Open contradictions that referenced a removed fact are removed too.
        """
        with self._tx() as c:
            rows = c.execute(
                """SELECT f.fact_id FROM facts f WHERE f.origin = 'ai' AND f.review = 'draft'
                   AND EXISTS (SELECT 1 FROM evidence e WHERE e.fact_id = f.fact_id
                               AND e.doc_id = ?)
                   AND NOT EXISTS (SELECT 1 FROM evidence e WHERE e.fact_id = f.fact_id
                                   AND (e.doc_id IS NULL OR e.doc_id != ?))""",
                (doc_id, doc_id),
            ).fetchall()
            gone = [r["fact_id"] for r in rows]
            for fid in gone:
                self._delete_fact(c, fid)
            touched = c.execute(
                """SELECT DISTINCT fact_id FROM evidence WHERE doc_id = ? AND origin = 'ai'
                   AND review = 'draft'""",
                (doc_id,),
            ).fetchall()
            c.execute(
                "DELETE FROM evidence WHERE doc_id = ? AND origin = 'ai' AND review = 'draft'",
                (doc_id,),
            )
            for r in touched:
                self._reindex(c, r["fact_id"])
            if gone:
                for row in c.execute(
                    "SELECT contradiction_id, fact_ids FROM contradictions WHERE status = 'open'"
                ).fetchall():
                    if set(json.loads(row["fact_ids"])) & set(gone):
                        c.execute(
                            "DELETE FROM contradictions WHERE contradiction_id = ?",
                            (row["contradiction_id"],),
                        )
            c.execute(
                "UPDATE documents SET parse_status = 'pending', parse_error = NULL, "
                "drafted_at = NULL WHERE doc_id = ?",
                (doc_id,),
            )
        return len(gone)

    def documents_mentioning(self, case_id: str, names: Sequence[str]) -> list[Document]:
        """Parsed documents whose text contains any of ``names`` (case-insensitive)."""
        names = [n for n in (" ".join(x.split()) for x in names) if len(n) >= 3]
        if not names:
            return []
        cond = " OR ".join("instr(lower(t.text), ?) > 0" for _ in names)
        rows = self._q(
            f"""SELECT d.* FROM documents d JOIN doc_text t ON t.doc_id = d.doc_id
                WHERE d.case_id = ? AND ({cond}) ORDER BY d.added_at, d.doc_id""",
            [case_id, *[n.lower() for n in names]],
        )
        return [self._row_to_document(r) for r in rows]

    # ------------------------------------------------------------------ facts: reads

    @staticmethod
    def _row_to_evidence(row: sqlite3.Row) -> EvidenceLink:
        return EvidenceLink(
            evidence_id=row["evidence_id"],
            kind=row["kind"],
            doc_id=row["doc_id"],
            page=row["page"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            quote=row["quote"],
            reference=row["reference"],
            origin=row["origin"],
            review=row["review"],
        )

    def _hydrate(self, rows: Sequence[sqlite3.Row], *, rejected_evidence: bool) -> list[Fact]:
        """Build Fact models for fact rows, batch-loading evidence, tags and witnesses."""
        ids = [r["fact_id"] for r in rows]
        evidence: dict[str, list[EvidenceLink]] = {i: [] for i in ids}
        tags: dict[str, list[str]] = {i: [] for i in ids}
        wits: dict[str, list[str]] = {i: [] for i in ids}
        ev_filter = "" if rejected_evidence else " AND review != 'rejected'"
        for chunk in _chunks(ids):
            m = _marks(len(chunk))
            for e in self._q(
                f"SELECT * FROM evidence WHERE fact_id IN ({m}){ev_filter} ORDER BY pk", chunk
            ):
                evidence[e["fact_id"]].append(self._row_to_evidence(e))
            for t in self._q(
                f"""SELECT ft.fact_id, ft.tag_id FROM fact_tags ft JOIN tags t
                    ON t.tag_id = ft.tag_id WHERE ft.fact_id IN ({m}) ORDER BY t.name""",
                chunk,
            ):
                tags[t["fact_id"]].append(t["tag_id"])
            for w in self._q(
                f"""SELECT fw.fact_id, fw.witness_id FROM fact_witnesses fw JOIN witnesses w
                    ON w.witness_id = fw.witness_id WHERE fw.fact_id IN ({m}) ORDER BY w.name""",
                chunk,
            ):
                wits[w["fact_id"]].append(w["witness_id"])
        return [
            Fact(
                fact_id=r["fact_id"],
                case_id=r["case_id"],
                title=r["title"],
                details=r["details"],
                date_text=r["date_text"],
                date_start=r["date_start"],
                date_end=r["date_end"],
                date_certainty=r["date_certainty"],
                anchor_fact_id=r["anchor_fact_id"],
                statuses=json.loads(r["statuses"]),
                tag_ids=tags[r["fact_id"]],
                witness_ids=wits[r["fact_id"]],
                evidence=evidence[r["fact_id"]],
                origin=r["origin"],
                review=r["review"],
                confidence=r["confidence"],
                created_by=r["created_by"],
                created_at=_dt(r["created_at"]),
                updated_by=r["updated_by"],
                updated_at=_dt(r["updated_at"]),
            )
            for r in rows
        ]

    def get_fact(self, fact_id: str, *, rejected_evidence: bool = False) -> Fact | None:
        """One fact (any review state); rejected evidence is left out unless asked for."""
        row = self._q1("SELECT * FROM facts WHERE fact_id = ?", (fact_id,))
        return self._hydrate([row], rejected_evidence=rejected_evidence)[0] if row else None

    def get_facts(self, fact_ids: Sequence[str]) -> list[Fact]:
        """Facts by id, in the order given (unknown ids skipped)."""
        rows: list[sqlite3.Row] = []
        for chunk in _chunks(list(fact_ids)):
            rows += self._q(f"SELECT * FROM facts WHERE fact_id IN ({_marks(len(chunk))})", chunk)
        by_id = {f.fact_id: f for f in self._hydrate(rows, rejected_evidence=False)}
        return [by_id[i] for i in fact_ids if i in by_id]

    _SORTS = {
        "title": "f.title COLLATE NOCASE",
        "evidence": "(SELECT COUNT(*) FROM evidence e WHERE e.fact_id = f.fact_id "
        "AND e.review != 'rejected')",
        "tags": "(SELECT COUNT(*) FROM fact_tags t WHERE t.fact_id = f.fact_id)",
        "witnesses": "(SELECT COUNT(*) FROM fact_witnesses w WHERE w.fact_id = f.fact_id)",
        "notes": "(SELECT COUNT(*) FROM notes n WHERE n.fact_id = f.fact_id)",
        "created": "f.created_at",
        "updated": "f.updated_at",
        "confidence": "COALESCE(f.confidence, 2.0)",
    }

    def list_facts(self, case_id: str, query: FactQuery | None = None) -> list[Fact]:
        """Facts matching ``query`` (default: drafts and accepted, in date order).

        Date sort keeps undated facts at the end in either direction.
        """
        q = query or FactQuery()
        if q.fact_ids is not None and not q.fact_ids:
            return []
        where = ["f.case_id = ?"]
        params: list[Any] = [case_id]

        def any_of(column: str, values: Sequence[str]) -> None:
            where.append(f"{column} IN ({_marks(len(values))})")
            params.extend(values)

        if q.reviews:
            any_of("f.review", list(q.reviews))
        if q.origins:
            any_of("f.origin", list(q.origins))
        if q.date_from or q.date_to:
            where.append("f.date_start IS NOT NULL")
            if q.date_from:
                where.append("COALESCE(f.date_end, f.date_start) >= ?")
                params.append(q.date_from.isoformat())
            if q.date_to:
                where.append("f.date_start <= ?")
                params.append(q.date_to.isoformat())
        elif not q.include_undated:
            where.append("f.date_start IS NOT NULL")
        if q.statuses:
            where.append(
                "EXISTS (SELECT 1 FROM json_each(f.statuses) s WHERE s.value IN "
                f"({_marks(len(q.statuses))}))"
            )
            params.extend(q.statuses)
        if q.tag_ids:
            where.append(
                "EXISTS (SELECT 1 FROM fact_tags t WHERE t.fact_id = f.fact_id AND t.tag_id IN "
                f"({_marks(len(q.tag_ids))}))"
            )
            params.extend(q.tag_ids)
        if q.witness_ids:
            where.append(
                "EXISTS (SELECT 1 FROM fact_witnesses w WHERE w.fact_id = f.fact_id AND "
                f"w.witness_id IN ({_marks(len(q.witness_ids))}))"
            )
            params.extend(q.witness_ids)
        if q.doc_ids:
            where.append(
                "EXISTS (SELECT 1 FROM evidence e WHERE e.fact_id = f.fact_id AND "
                f"e.review != 'rejected' AND e.doc_id IN ({_marks(len(q.doc_ids))}))"
            )
            params.extend(q.doc_ids)
        if q.doc_kinds:
            where.append(
                "EXISTS (SELECT 1 FROM evidence e JOIN documents d ON d.doc_id = e.doc_id "
                "WHERE e.fact_id = f.fact_id AND e.review != 'rejected' AND d.kind IN "
                f"({_marks(len(q.doc_kinds))}))"
            )
            params.extend(q.doc_kinds)
        if q.fact_ids is not None:
            where.append(
                f"f.fact_id IN (SELECT value FROM json_each(?))"
            )
            params.append(json.dumps(q.fact_ids))

        direction = "DESC" if q.descending else "ASC"
        if q.sort == "date":
            order = (
                f"f.date_start IS NULL, f.date_start {direction}, "
                f"COALESCE(f.date_end, f.date_start) {direction}, f.created_at, f.pk"
            )
        else:
            order = f"{self._SORTS[q.sort]} {direction}, f.date_start IS NULL, f.date_start, f.pk"
        sql = f"SELECT f.* FROM facts f WHERE {' AND '.join(where)} ORDER BY {order}"
        if q.limit is not None:
            sql += " LIMIT ?"
            params.append(int(q.limit))
        return self._hydrate(self._q(sql, params), rejected_evidence=False)

    def fact_counts(self, fact_ids: Sequence[str]) -> dict[str, dict[str, int]]:
        """Per fact: counts of (non-rejected) evidence, tags, witnesses and notes."""
        out = {i: {"evidence": 0, "tags": 0, "witnesses": 0, "notes": 0} for i in fact_ids}
        for chunk in _chunks(list(fact_ids)):
            m = _marks(len(chunk))
            for name, sql in (
                ("evidence", f"SELECT fact_id, COUNT(*) n FROM evidence WHERE fact_id IN ({m}) "
                 "AND review != 'rejected' GROUP BY fact_id"),
                ("tags", f"SELECT fact_id, COUNT(*) n FROM fact_tags WHERE fact_id IN ({m}) "
                 "GROUP BY fact_id"),
                ("witnesses", f"SELECT fact_id, COUNT(*) n FROM fact_witnesses "
                 f"WHERE fact_id IN ({m}) GROUP BY fact_id"),
                ("notes", f"SELECT fact_id, COUNT(*) n FROM notes WHERE fact_id IN ({m}) "
                 "GROUP BY fact_id"),
            ):
                for r in self._q(sql, chunk):
                    out[r["fact_id"]][name] = r["n"]
        return out

    def alleged_fact_ids(self, case_id: str) -> set[str]:
        """Facts whose evidence (non-rejected, at least one item) all comes from pleadings."""
        rows = self._q(
            """SELECT f.fact_id FROM facts f WHERE f.case_id = ?
               AND EXISTS (SELECT 1 FROM evidence e WHERE e.fact_id = f.fact_id
                           AND e.review != 'rejected')
               AND NOT EXISTS (SELECT 1 FROM evidence e LEFT JOIN documents d
                               ON d.doc_id = e.doc_id WHERE e.fact_id = f.fact_id
                               AND e.review != 'rejected'
                               AND (d.kind IS NULL OR d.kind != 'pleading'))""",
            (case_id,),
        )
        return {r["fact_id"] for r in rows}

    def conflicted_fact_ids(self, case_id: str) -> set[str]:
        """Facts linked to an open contradiction (the conflict badge, SPEC §6.4)."""
        out: set[str] = set()
        for c in self.list_contradictions(case_id, status="open"):
            out.update(c.fact_ids)
        return out

    def facts_citing_document(self, doc_id: str, *, include_rejected: bool = False) -> list[Fact]:
        """Facts with evidence from this document (for the source viewer's side panel)."""
        review = "" if include_rejected else " AND f.review != 'rejected'"
        rows = self._q(
            f"""SELECT f.* FROM facts f WHERE EXISTS (SELECT 1 FROM evidence e
                    WHERE e.fact_id = f.fact_id AND e.doc_id = ? AND e.review != 'rejected')
                {review} ORDER BY f.date_start IS NULL, f.date_start, f.pk""",
            (doc_id,),
        )
        return self._hydrate(rows, rejected_evidence=False)

    def get_evidence(self, evidence_id: str) -> tuple[str, EvidenceLink] | None:
        """``(fact_id, evidence)`` for an evidence id."""
        row = self._q1("SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,))
        return (row["fact_id"], self._row_to_evidence(row)) if row else None

    def fact_history(self, fact_id: str) -> list[dict[str, Any]]:
        """Every recorded edit: field, old and new values, who and when (oldest first)."""
        rows = self._q(
            "SELECT field, old, new, by, at FROM fact_history WHERE fact_id = ? ORDER BY id",
            (fact_id,),
        )
        return [
            {
                "field": r["field"],
                "old": json.loads(r["old"]) if r["old"] is not None else None,
                "new": json.loads(r["new"]) if r["new"] is not None else None,
                "by": r["by"],
                "at": _dt(r["at"]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ facts: writes

    def _check_span(self, c: sqlite3.Connection, span: SpanIn) -> SpanIn:
        """Verbatim check against stored text; fills page and transcript lines if missing."""
        row = c.execute(
            "SELECT text, pages, lines FROM doc_text WHERE doc_id = ?", (span.doc_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"document {span.doc_id} has no parsed text")
        text = row["text"]
        if not (0 <= span.char_start < span.char_end <= len(text)):
            raise ValueError("span offsets are out of range")
        if text[span.char_start : span.char_end] != span.quote:
            raise ValueError("quote does not match the source text verbatim")
        page = span.page
        if page is None:
            for p in json.loads(row["pages"]):
                if p["char_start"] <= span.char_start < max(p["char_end"], p["char_start"] + 1):
                    page = p["number"]
                    break
        line_start, line_end = span.line_start, span.line_end
        if line_start is None:
            hits = [
                ln
                for ln in json.loads(row["lines"])
                if ln["char_start"] < span.char_end and ln["char_end"] > span.char_start
            ]
            if hits:
                line_start, line_end = hits[0]["number"], hits[-1]["number"]
                page = page if page is not None else hits[0]["page"]
        return span.model_copy(update={"page": page, "line_start": line_start,
                                       "line_end": line_end})

    def _insert_evidence(
        self,
        c: sqlite3.Connection,
        fact_id: str,
        *,
        kind: str,
        by: str,
        span: SpanIn | None = None,
        doc_id: str | None = None,
        reference: str | None = None,
    ) -> tuple[str, bool]:
        """Insert one evidence row; returns ``(evidence_id, created)`` (duplicates are reused)."""
        origin = "ai" if _is_agent(by) else "user"
        review = "draft" if _is_agent(by) else "accepted"
        if kind == "span":
            if span is None:
                raise ValueError("span evidence needs a span")
            span = self._check_span(c, span)
            dup = c.execute(
                """SELECT evidence_id FROM evidence WHERE fact_id = ? AND kind = 'span'
                   AND doc_id = ? AND char_start = ? AND char_end = ?""",
                (fact_id, span.doc_id, span.char_start, span.char_end),
            ).fetchone()
            values = (span.doc_id, span.page, span.char_start, span.char_end, span.line_start,
                      span.line_end, span.quote, None)
        elif kind == "document":
            if not doc_id or c.execute(
                "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone() is None:
                raise ValueError("document evidence needs an existing doc_id")
            dup = c.execute(
                "SELECT evidence_id FROM evidence WHERE fact_id = ? AND kind = 'document' "
                "AND doc_id = ?",
                (fact_id, doc_id),
            ).fetchone()
            values = (doc_id, None, None, None, None, None, None, None)
        elif kind == "external":
            reference = (reference or "").strip()
            if not reference:
                raise ValueError("external evidence needs a reference")
            dup = c.execute(
                "SELECT evidence_id FROM evidence WHERE fact_id = ? AND kind = 'external' "
                "AND reference = ?",
                (fact_id, reference),
            ).fetchone()
            values = (None, None, None, None, None, None, None, reference)
        else:
            raise ValueError(f"unknown evidence kind {kind!r}")
        if dup is not None:
            return dup["evidence_id"], False
        evidence_id = self._next_id("E")
        c.execute(
            """INSERT INTO evidence (evidence_id, fact_id, kind, doc_id, page, char_start,
                   char_end, line_start, line_end, quote, reference, origin, review, created_by,
                   created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (evidence_id, fact_id, kind, *values, origin, review, by, _now_ts()),
        )
        return evidence_id, True

    def _validate_fields(self, c: sqlite3.Connection, case_id: str, fields: dict[str, Any]) -> None:
        if "title" in fields:
            if not isinstance(fields["title"], str) or not fields["title"].strip():
                raise ValueError("a fact needs a title")
            fields["title"] = fields["title"].strip()
        if "date_certainty" in fields and fields["date_certainty"] not in CERTAINTIES:
            raise ValueError(f"bad date_certainty {fields['date_certainty']!r}")
        for key in ("date_start", "date_end"):
            if key in fields:
                fields[key] = _d(fields[key])
        if fields.get("date_start") and fields.get("date_end"):
            if fields["date_end"] < fields["date_start"]:
                raise ValueError("date_end is before date_start")
        if "statuses" in fields:
            bad = set(fields["statuses"]) - set(STATUSES)
            if bad:
                raise ValueError(f"unknown statuses {sorted(bad)}")
            fields["statuses"] = [s for s in STATUSES if s in fields["statuses"]]
        for key, table, col in (("tag_ids", "tags", "tag_id"),
                                ("witness_ids", "witnesses", "witness_id")):
            if key in fields:
                ids = _dedupe(fields[key])
                for chunk in _chunks(ids):
                    found = {
                        r[0]
                        for r in c.execute(
                            f"SELECT {col} FROM {table} WHERE case_id = ? AND {col} IN "
                            f"({_marks(len(chunk))})",
                            [case_id, *chunk],
                        )
                    }
                    if missing := set(chunk) - found:
                        raise ValueError(f"unknown {key}: {sorted(missing)}")
                fields[key] = ids
        if fields.get("anchor_fact_id"):
            if c.execute(
                "SELECT 1 FROM facts WHERE fact_id = ?", (fields["anchor_fact_id"],)
            ).fetchone() is None:
                raise ValueError("unknown anchor_fact_id")

    def add_fact(
        self,
        case_id: str,
        title: str,
        *,
        by: str,
        details: str | None = None,
        date_text: str | None = None,
        date_start: date | str | None = None,
        date_end: date | str | None = None,
        date_certainty: str = "unknown",
        anchor_fact_id: str | None = None,
        statuses: Sequence[str] = (),
        tag_ids: Sequence[str] = (),
        witness_ids: Sequence[str] = (),
        confidence: float | None = None,
        spans: Sequence[SpanIn] = (),
        doc_ids: Sequence[str] = (),
        references: Sequence[str] = (),
    ) -> Fact:
        """Create a fact. The agent's facts are drafts; a person's start accepted (SPEC §6.3).

        Evidence may be given as verified spans, whole documents, or external references.
        """
        agent = _is_agent(by)
        if agent and (statuses or tag_ids):
            raise PermissionError("the agent may not set statuses or tags (SPEC §6.2)")
        if not agent:
            _require_person(by, "create facts")
        fields: dict[str, Any] = {
            "title": title,
            "date_certainty": date_certainty,
            "date_start": date_start,
            "date_end": date_end,
            "statuses": list(statuses),
            "tag_ids": list(tag_ids),
            "witness_ids": list(witness_ids),
            "anchor_fact_id": anchor_fact_id,
        }
        stamp = _now_ts()
        with self._tx() as c:
            self._validate_fields(c, case_id, fields)
            fact_id = self._next_id("F")
            c.execute(
                """INSERT INTO facts (fact_id, case_id, title, details, date_text, date_start,
                       date_end, date_certainty, anchor_fact_id, statuses, origin, review,
                       confidence, created_by, created_at, updated_by, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact_id,
                    case_id,
                    fields["title"],
                    (details or None),
                    (date_text or None),
                    fields["date_start"],
                    fields["date_end"],
                    fields["date_certainty"],
                    fields["anchor_fact_id"],
                    json.dumps(fields["statuses"]),
                    "ai" if agent else "user",
                    "draft" if agent else "accepted",
                    confidence if agent else None,
                    by,
                    stamp,
                    by,
                    stamp,
                ),
            )
            self._set_links(c, fact_id, "fact_tags", "tag_id", fields["tag_ids"])
            self._set_links(c, fact_id, "fact_witnesses", "witness_id", fields["witness_ids"])
            for span in spans:
                self._insert_evidence(c, fact_id, kind="span", by=by, span=span)
            for doc_id in doc_ids:
                self._insert_evidence(c, fact_id, kind="document", by=by, doc_id=doc_id)
            for ref in references:
                self._insert_evidence(c, fact_id, kind="external", by=by, reference=ref)
            self._reindex(c, fact_id)
        return self.get_fact(fact_id)  # type: ignore[return-value]

    @staticmethod
    def _set_links(
        c: sqlite3.Connection, fact_id: str, table: str, column: str, ids: Sequence[str]
    ) -> None:
        c.execute(f"DELETE FROM {table} WHERE fact_id = ?", (fact_id,))
        c.executemany(
            f"INSERT INTO {table} (fact_id, {column}) VALUES (?, ?)",
            [(fact_id, i) for i in ids],
        )

    def _history(
        self, c: sqlite3.Connection, fact_id: str, field: str, old: Any, new: Any, by: str
    ) -> None:
        c.execute(
            "INSERT INTO fact_history (fact_id, field, old, new, by, at) VALUES (?, ?, ?, ?, ?, ?)",
            (fact_id, field, json.dumps(old), json.dumps(new), by, _now_ts()),
        )

    def _review_event(
        self,
        c: sqlite3.Connection,
        case_id: str,
        fact_id: str,
        action: str,
        reviewer: str,
        reason: str | None,
        evidence_id: str | None = None,
    ) -> None:
        c.execute(
            """INSERT INTO review_events (case_id, fact_id, evidence_id, action, reason,
                   reviewer, at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (case_id, fact_id, evidence_id, action, (reason or None), reviewer, _now_ts()),
        )

    def update_fact(self, fact_id: str, changes: dict[str, Any], *, by: str) -> Fact:
        """Edit fact fields; records history. A person editing a draft accepts it (SPEC §6.1).

        The agent may only edit its own unreviewed drafts and never statuses or tags.
        """
        unknown = set(changes) - set(EDITABLE_FACT_FIELDS)
        if unknown:
            raise ValueError(f"fields not editable: {sorted(unknown)}")
        current = self.get_fact(fact_id)
        if current is None:
            raise KeyError(fact_id)
        agent = _is_agent(by)
        if agent:
            if current.review != "draft":
                raise PermissionError("the agent may not edit a reviewed fact (SPEC §6.2)")
            if any(k in changes for k in _PEOPLE_ONLY_FIELDS):
                raise PermissionError("the agent may not set statuses or tags (SPEC §6.2)")
        else:
            _require_person(by, "edit facts")
        fields = dict(changes)
        old = current.model_dump(mode="json")
        with self._tx() as c:
            self._validate_fields(c, current.case_id, fields)
            new_start = fields.get("date_start", old["date_start"])
            new_end = fields.get("date_end", old["date_end"])
            if new_start and new_end and new_end < new_start:
                raise ValueError("date_end is before date_start")
            changed = {k: v for k, v in fields.items() if old.get(k) != v}
            if fields.get("anchor_fact_id") == fact_id:
                raise ValueError("a fact cannot anchor itself")
            columns = {k: v for k, v in changed.items() if k not in ("tag_ids", "witness_ids")}
            if "statuses" in columns:
                columns["statuses"] = json.dumps(columns["statuses"])
            accept = not agent and current.review == "draft"
            if accept:
                columns["review"] = "accepted"
            stamp = _now_ts()
            if changed or accept:
                sets = ", ".join(f"{k} = ?" for k in columns)
                sets = (sets + ", " if sets else "") + "updated_by = ?, updated_at = ?"
                if any(k in changed for k in ("title", "details")):
                    sets += ", vec_dirty = 1"
                c.execute(
                    f"UPDATE facts SET {sets} WHERE fact_id = ?",
                    [*columns.values(), by, stamp, fact_id],
                )
            if "tag_ids" in changed:
                self._set_links(c, fact_id, "fact_tags", "tag_id", changed["tag_ids"])
            if "witness_ids" in changed:
                self._set_links(c, fact_id, "fact_witnesses", "witness_id", changed["witness_ids"])
            for k, v in changed.items():
                self._history(c, fact_id, k, old.get(k), v, by)
            if accept:
                self._accept_draft_evidence(c, fact_id)
                self._history(c, fact_id, "review", "draft", "accepted", by)
                self._review_event(c, current.case_id, fact_id, "edit", by, None)
            self._reindex(c, fact_id)
        return self.get_fact(fact_id)  # type: ignore[return-value]

    def _accept_draft_evidence(self, c: sqlite3.Connection, fact_id: str) -> None:
        c.execute(
            "UPDATE evidence SET review = 'accepted' WHERE fact_id = ? AND review = 'draft'",
            (fact_id,),
        )

    def review_fact(
        self,
        fact_id: str,
        action: str,
        *,
        reviewer: str,
        reason: str | None = None,
        changes: dict[str, Any] | None = None,
    ) -> Fact:
        """Accept, edit (accepts with changes) or reject a fact; logs a ReviewEvent.

        Accepting a fact also accepts the draft evidence on it at that moment.
        """
        _require_person(reviewer, "review facts")
        if action == "edit":
            if not changes:
                raise ValueError("an edit needs changes")
            fact = self.get_fact(fact_id)
            if fact is None:
                raise KeyError(fact_id)
            was_draft = fact.review == "draft"
            updated = self.update_fact(fact_id, changes, by=reviewer)
            if not was_draft:  # update_fact logs the event only when it accepts a draft
                with self._tx() as c:
                    self._review_event(c, fact.case_id, fact_id, "edit", reviewer, reason)
            elif reason:
                with self._tx() as c:
                    c.execute(
                        "UPDATE review_events SET reason = ? WHERE id = (SELECT MAX(id) FROM "
                        "review_events WHERE fact_id = ? AND action = 'edit')",
                        (reason, fact_id),
                    )
            return updated
        if action not in ("accept", "reject"):
            raise ValueError(f"unknown review action {action!r}")
        new = "accepted" if action == "accept" else "rejected"
        with self._tx() as c:
            row = c.execute(
                "SELECT case_id, review FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                raise KeyError(fact_id)
            c.execute(
                "UPDATE facts SET review = ?, updated_by = ?, updated_at = ? WHERE fact_id = ?",
                (new, reviewer, _now_ts(), fact_id),
            )
            if action == "accept":
                self._accept_draft_evidence(c, fact_id)
            if row["review"] != new:
                self._history(c, fact_id, "review", row["review"], new, reviewer)
            self._review_event(c, row["case_id"], fact_id, action, reviewer, reason)
        return self.get_fact(fact_id)  # type: ignore[return-value]

    def add_evidence(
        self,
        fact_id: str,
        *,
        kind: str,
        by: str,
        span: SpanIn | None = None,
        doc_id: str | None = None,
        reference: str | None = None,
    ) -> EvidenceLink:
        """Link evidence to a fact. The agent's evidence starts as a draft (SPEC §6.2)."""
        if not _is_agent(by):
            _require_person(by, "link evidence")
        with self._tx() as c:
            row = c.execute("SELECT 1 FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
            if row is None:
                raise KeyError(fact_id)
            evidence_id, created = self._insert_evidence(
                c, fact_id, kind=kind, by=by, span=span, doc_id=doc_id, reference=reference
            )
            if created:
                if not _is_agent(by):
                    self._history(c, fact_id, "evidence", None, evidence_id, by)
                c.execute(
                    "UPDATE facts SET updated_by = ?, updated_at = ?, vec_dirty = 1 "
                    "WHERE fact_id = ?",
                    (by, _now_ts(), fact_id),
                )
                self._reindex(c, fact_id)
        return self.get_evidence(evidence_id)[1]  # type: ignore[index]

    def review_evidence(
        self, evidence_id: str, action: str, *, reviewer: str, reason: str | None = None
    ) -> EvidenceLink:
        """Accept or reject one piece of evidence the AI attached."""
        _require_person(reviewer, "review evidence")
        if action not in ("accept", "reject"):
            raise ValueError(f"unknown review action {action!r}")
        new = "accepted" if action == "accept" else "rejected"
        with self._tx() as c:
            row = c.execute(
                """SELECT e.fact_id, e.review, f.case_id FROM evidence e JOIN facts f
                   ON f.fact_id = e.fact_id WHERE e.evidence_id = ?""",
                (evidence_id,),
            ).fetchone()
            if row is None:
                raise KeyError(evidence_id)
            c.execute("UPDATE evidence SET review = ? WHERE evidence_id = ?", (new, evidence_id))
            self._history(c, row["fact_id"], f"evidence {evidence_id}", row["review"], new,
                          reviewer)
            self._review_event(
                c, row["case_id"], row["fact_id"], action, reviewer, reason, evidence_id
            )
            c.execute(
                "UPDATE facts SET updated_by = ?, updated_at = ?, vec_dirty = 1 WHERE fact_id = ?",
                (reviewer, _now_ts(), row["fact_id"]),
            )
            self._reindex(c, row["fact_id"])
        return self.get_evidence(evidence_id)[1]  # type: ignore[index]

    def _delete_fact(self, c: sqlite3.Connection, fact_id: str) -> None:
        row = c.execute("SELECT pk FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
        if row is None:
            return
        c.execute("DELETE FROM facts_fts WHERE rowid = ?", (row["pk"],))
        if self._vec:
            c.execute("DELETE FROM fact_vec WHERE rowid = ?", (row["pk"],))
        c.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))

    def list_review_events(self, case_id: str) -> list[ReviewEvent]:
        rows = self._q("SELECT * FROM review_events WHERE case_id = ? ORDER BY id", (case_id,))
        return [
            ReviewEvent(
                fact_id=r["fact_id"],
                evidence_id=r["evidence_id"],
                action=r["action"],
                reason=r["reason"],
                reviewer=r["reviewer"],
                at=_dt(r["at"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ notes

    def add_note(self, fact_id: str, *, author: str, body: str) -> Note:
        _require_person(author, "write notes")
        body = body.strip()
        if not body:
            raise ValueError("a note needs a body")
        stamp = _now_ts()
        with self._tx() as c:
            if c.execute("SELECT 1 FROM facts WHERE fact_id = ?", (fact_id,)).fetchone() is None:
                raise KeyError(fact_id)
            note_id = self._next_id("N")
            c.execute(
                "INSERT INTO notes (note_id, fact_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (note_id, fact_id, author, body, stamp),
            )
            c.execute(
                "UPDATE facts SET updated_by = ?, updated_at = ? WHERE fact_id = ?",
                (author, stamp, fact_id),
            )
            self._history(c, fact_id, "note", None, note_id, author)
            self._reindex(c, fact_id)
        return Note(note_id=note_id, fact_id=fact_id, author=author, body=body,
                    created_at=_dt(stamp))

    def list_notes(self, fact_id: str) -> list[Note]:
        rows = self._q("SELECT * FROM notes WHERE fact_id = ? ORDER BY created_at", (fact_id,))
        return [
            Note(note_id=r["note_id"], fact_id=r["fact_id"], author=r["author"], body=r["body"],
                 created_at=_dt(r["created_at"]))
            for r in rows
        ]

    # ------------------------------------------------------------------ witnesses

    @staticmethod
    def _row_to_witness(row: sqlite3.Row) -> Witness:
        return Witness(
            witness_id=row["witness_id"],
            case_id=row["case_id"],
            name=row["name"],
            aliases=json.loads(row["aliases"]),
            role=row["role"],
            notes=row["notes"],
            flagged=bool(row["flagged"]),
            confirmed=bool(row["confirmed"]),
        )

    def add_witness(
        self,
        case_id: str,
        name: str,
        *,
        by: str,
        aliases: Sequence[str] = (),
        role: str | None = None,
        notes: str | None = None,
        flagged: bool = False,
    ) -> Witness:
        """Add a witness. The agent's witnesses are suggestions (``confirmed=False``)."""
        name = " ".join(name.split())
        if not name:
            raise ValueError("a witness needs a name")
        agent = _is_agent(by)
        if not agent:
            _require_person(by, "add witnesses")
        stamp = _now_ts()
        aliases = [a for a in _dedupe(" ".join(a.split()) for a in aliases) if a != name]
        with self._tx() as c:
            witness_id = self._next_id("W")
            c.execute(
                """INSERT INTO witnesses (witness_id, case_id, name, aliases, role, notes,
                       flagged, confirmed, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (witness_id, case_id, name, json.dumps(aliases), role, notes,
                 int(flagged and not agent), int(not agent), stamp, stamp),
            )
        return self.get_witness(witness_id)  # type: ignore[return-value]

    def get_witness(self, witness_id: str) -> Witness | None:
        row = self._q1("SELECT * FROM witnesses WHERE witness_id = ?", (witness_id,))
        return self._row_to_witness(row) if row else None

    def list_witnesses(self, case_id: str, *, confirmed: bool | None = None) -> list[Witness]:
        sql = "SELECT * FROM witnesses WHERE case_id = ?"
        params: list[Any] = [case_id]
        if confirmed is not None:
            sql += " AND confirmed = ?"
            params.append(int(confirmed))
        rows = self._q(sql + " ORDER BY name COLLATE NOCASE", params)
        return [self._row_to_witness(r) for r in rows]

    def find_witness(self, case_id: str, name: str) -> Witness | None:
        """Exact (case- and space-insensitive) match on name or any alias; confirmed first."""
        target = _norm_name(name)
        if not target:
            return None
        for w in sorted(self.list_witnesses(case_id), key=lambda w: not w.confirmed):
            if target in {_norm_name(n) for n in [w.name, *w.aliases]}:
                return w
        return None

    def update_witness(self, witness_id: str, changes: dict[str, Any], *, by: str) -> Witness:
        """A person edits a witness: name, aliases, role, notes, flagged, confirmed."""
        _require_person(by, "edit witnesses")
        allowed = {"name", "aliases", "role", "notes", "flagged", "confirmed"}
        if unknown := set(changes) - allowed:
            raise ValueError(f"fields not editable: {sorted(unknown)}")
        current = self.get_witness(witness_id)
        if current is None:
            raise KeyError(witness_id)
        cols: dict[str, Any] = {}
        name = " ".join(changes.get("name", current.name).split())
        if not name:
            raise ValueError("a witness needs a name")
        if "name" in changes:
            cols["name"] = name
        if "aliases" in changes:
            cols["aliases"] = json.dumps(
                [a for a in _dedupe(" ".join(a.split()) for a in changes["aliases"]) if a != name]
            )
        for key in ("role", "notes"):
            if key in changes:
                cols[key] = (changes[key] or None)
        for key in ("flagged", "confirmed"):
            if key in changes:
                cols[key] = int(bool(changes[key]))
        if cols:
            sets = ", ".join(f"{k} = ?" for k in cols)
            with self._tx() as c:
                c.execute(
                    f"UPDATE witnesses SET {sets}, updated_at = ? WHERE witness_id = ?",
                    [*cols.values(), _now_ts(), witness_id],
                )
        return self.get_witness(witness_id)  # type: ignore[return-value]

    def merge_witnesses(self, source_id: str, into_id: str, *, by: str) -> Witness:
        """Fold ``source`` into ``into``: its name becomes an alias and its facts move over."""
        _require_person(by, "merge witnesses")
        src, dst = self.get_witness(source_id), self.get_witness(into_id)
        if src is None or dst is None:
            raise KeyError(source_id if src is None else into_id)
        if source_id == into_id:
            raise ValueError("cannot merge a witness into itself")
        aliases = [a for a in _dedupe([*dst.aliases, src.name, *src.aliases]) if a != dst.name]
        with self._tx() as c:
            c.execute(
                """INSERT OR IGNORE INTO fact_witnesses (fact_id, witness_id)
                   SELECT fact_id, ? FROM fact_witnesses WHERE witness_id = ?""",
                (into_id, source_id),
            )
            c.execute("DELETE FROM witnesses WHERE witness_id = ?", (source_id,))
            c.execute(
                "UPDATE witnesses SET aliases = ?, flagged = MAX(flagged, ?), updated_at = ? "
                "WHERE witness_id = ?",
                (json.dumps(aliases), int(src.flagged), _now_ts(), into_id),
            )
        return self.get_witness(into_id)  # type: ignore[return-value]

    def delete_witness(self, witness_id: str, *, by: str) -> None:
        """Dismiss a witness (typically an AI suggestion); unlinks it from facts."""
        _require_person(by, "dismiss witnesses")
        with self._tx() as c:
            c.execute("DELETE FROM witnesses WHERE witness_id = ?", (witness_id,))

    def link_witnesses(self, fact_id: str, witness_ids: Sequence[str], *, by: str) -> None:
        """Add witnesses to a fact without removing any (the agent may only touch drafts)."""
        fact = self.get_fact(fact_id)
        if fact is None:
            raise KeyError(fact_id)
        merged = _dedupe([*fact.witness_ids, *witness_ids])
        if merged != fact.witness_ids:
            self.update_fact(fact_id, {"witness_ids": merged}, by=by)

    def witness_facts(self, witness_id: str) -> list[Fact]:
        """The witness's (non-rejected) facts in date order."""
        rows = self._q(
            """SELECT f.* FROM facts f JOIN fact_witnesses fw ON fw.fact_id = f.fact_id
               WHERE fw.witness_id = ? AND f.review != 'rejected'
               ORDER BY f.date_start IS NULL, f.date_start, f.pk""",
            (witness_id,),
        )
        return self._hydrate(rows, rejected_evidence=False)

    def witness_documents(self, witness_id: str) -> list[Document]:
        """Documents cited by the witness's facts plus documents whose text names them."""
        w = self.get_witness(witness_id)
        if w is None:
            return []
        cited = self._q(
            """SELECT DISTINCT d.* FROM documents d JOIN evidence e ON e.doc_id = d.doc_id
               JOIN fact_witnesses fw ON fw.fact_id = e.fact_id
               WHERE fw.witness_id = ? AND e.review != 'rejected'""",
            (witness_id,),
        )
        docs = {r["doc_id"]: self._row_to_document(r) for r in cited}
        for d in self.documents_mentioning(w.case_id, [w.name, *w.aliases]):
            docs.setdefault(d.doc_id, d)
        return sorted(docs.values(), key=lambda d: (d.added_at, d.doc_id))

    # ------------------------------------------------------------------ tags

    def add_tag(self, case_id: str, name: str, *, by: str, color: str | None = None) -> Tag:
        _require_person(by, "create tags")
        name = " ".join(name.split())
        if not name:
            raise ValueError("a tag needs a name")
        with self._tx() as c:
            existing = c.execute(
                "SELECT * FROM tags WHERE case_id = ? AND name = ? COLLATE NOCASE",
                (case_id, name),
            ).fetchone()
            if existing is not None:
                return Tag(**dict(existing))
            tag_id = self._next_id("T")
            c.execute(
                "INSERT INTO tags (tag_id, case_id, name, color) VALUES (?, ?, ?, ?)",
                (tag_id, case_id, name, color or None),
            )
        return Tag(tag_id=tag_id, case_id=case_id, name=name, color=color or None)

    def list_tags(self, case_id: str) -> list[Tag]:
        rows = self._q("SELECT * FROM tags WHERE case_id = ? ORDER BY name COLLATE NOCASE",
                       (case_id,))
        return [Tag(**dict(r)) for r in rows]

    def get_tag(self, tag_id: str) -> Tag | None:
        row = self._q1("SELECT * FROM tags WHERE tag_id = ?", (tag_id,))
        return Tag(**dict(row)) if row else None

    def update_tag(self, tag_id: str, *, by: str, name: str | None = None,
                   color: str | None = None) -> Tag:
        _require_person(by, "edit tags")
        with self._tx() as c:
            if name is not None:
                name = " ".join(name.split())
                if not name:
                    raise ValueError("a tag needs a name")
                c.execute("UPDATE tags SET name = ? WHERE tag_id = ?", (name, tag_id))
            if color is not None:
                c.execute("UPDATE tags SET color = ? WHERE tag_id = ?", (color or None, tag_id))
        tag = self.get_tag(tag_id)
        if tag is None:
            raise KeyError(tag_id)
        return tag

    def delete_tag(self, tag_id: str, *, by: str) -> None:
        _require_person(by, "delete tags")
        with self._tx() as c:
            c.execute("DELETE FROM tags WHERE tag_id = ?", (tag_id,))

    # ------------------------------------------------------------------ contradictions

    @staticmethod
    def _row_to_contradiction(row: sqlite3.Row) -> Contradiction:
        return Contradiction(
            contradiction_id=row["contradiction_id"],
            fact_ids=json.loads(row["fact_ids"]),
            field=row["field"],
            explanation=row["explanation"],
            status=row["status"],
        )

    def add_contradiction(
        self, case_id: str, fact_ids: Sequence[str], field: str, explanation: str
    ) -> tuple[Contradiction, bool]:
        """Open a contradiction unless the same facts already conflict on ``field``."""
        ids = sorted(set(fact_ids))
        if len(ids) < 2:
            raise ValueError("a contradiction needs at least two facts")
        if field not in ("date", "actor", "fact"):
            raise ValueError(f"bad field {field!r}")
        key = f"{case_id}|{'|'.join(ids)}#{field}"
        stamp = _now_ts()
        with self._tx() as c:
            row = c.execute("SELECT * FROM contradictions WHERE key = ?", (key,)).fetchone()
            if row is not None:
                return self._row_to_contradiction(row), False
            cid = self._next_id("C")
            c.execute(
                """INSERT INTO contradictions (contradiction_id, case_id, key, fact_ids, field,
                       explanation, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)""",
                (cid, case_id, key, json.dumps(ids), field, explanation.strip(), stamp, stamp),
            )
            row = c.execute(
                "SELECT * FROM contradictions WHERE contradiction_id = ?", (cid,)
            ).fetchone()
        return self._row_to_contradiction(row), True

    def get_contradiction(self, contradiction_id: str) -> Contradiction | None:
        row = self._q1(
            "SELECT * FROM contradictions WHERE contradiction_id = ?", (contradiction_id,)
        )
        return self._row_to_contradiction(row) if row else None

    def contradiction_case(self, contradiction_id: str) -> str | None:
        row = self._q1(
            "SELECT case_id FROM contradictions WHERE contradiction_id = ?", (contradiction_id,)
        )
        return row["case_id"] if row else None

    def list_contradictions(
        self, case_id: str, status: str | None = None, *, fact_id: str | None = None
    ) -> list[Contradiction]:
        sql = "SELECT * FROM contradictions WHERE case_id = ?"
        params: list[Any] = [case_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        if fact_id is not None:
            sql += " AND EXISTS (SELECT 1 FROM json_each(fact_ids) j WHERE j.value = ?)"
            params.append(fact_id)
        rows = self._q(sql + " ORDER BY created_at, contradiction_id", params)
        return [self._row_to_contradiction(r) for r in rows]

    def review_contradiction(
        self, contradiction_id: str, action: str, *, reviewer: str
    ) -> Contradiction:
        """Confirm (linked facts become disputed, SPEC §6.4) or dismiss a contradiction."""
        _require_person(reviewer, "confirm or dismiss contradictions")
        if action not in ("confirm", "dismiss", "reopen"):
            raise ValueError(f"unknown action {action!r}")
        status = {"confirm": "confirmed", "dismiss": "dismissed", "reopen": "open"}[action]
        current = self.get_contradiction(contradiction_id)
        if current is None:
            raise KeyError(contradiction_id)
        with self._tx() as c:
            c.execute(
                "UPDATE contradictions SET status = ?, reviewed_by = ?, updated_at = ? "
                "WHERE contradiction_id = ?",
                (status, reviewer, _now_ts(), contradiction_id),
            )
        if action == "confirm":
            for fact in self.get_facts(current.fact_ids):
                if "disputed" not in fact.statuses:
                    self.update_fact(
                        fact.fact_id, {"statuses": [*fact.statuses, "disputed"]}, by=reviewer
                    )
        return self.get_contradiction(contradiction_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------ alerts

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> Alert:
        return Alert(
            alert_id=row["alert_id"],
            case_id=row["case_id"],
            type=row["type"],
            fact_ids=json.loads(row["fact_ids"]),
            doc_ids=json.loads(row["doc_ids"]),
            created_at=_dt(row["created_at"]),
            delivered_at=_dt(row["delivered_at"]),
        )

    def add_alert(
        self,
        case_id: str,
        type: str,
        *,
        fact_ids: Sequence[str] = (),
        doc_ids: Sequence[str] = (),
        job_id: str | None = None,
    ) -> Alert:
        if type not in ALERT_TYPES:
            raise ValueError(f"unknown alert type {type!r}")
        with self._tx() as c:
            alert_id = self._next_id("A")
            c.execute(
                """INSERT INTO alerts (alert_id, case_id, type, fact_ids, doc_ids, job_id,
                       created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (alert_id, case_id, type, json.dumps(_dedupe(fact_ids)),
                 json.dumps(_dedupe(doc_ids)), job_id, _now_ts()),
            )
        return self.get_alert(alert_id)  # type: ignore[return-value]

    def get_alert(self, alert_id: str) -> Alert | None:
        row = self._q1("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
        return self._row_to_alert(row) if row else None

    def list_alerts(self, case_id: str, *, pending_only: bool = False) -> list[Alert]:
        """Newest first; ``pending_only`` keeps alerts not yet delivered by the agent."""
        sql = "SELECT * FROM alerts WHERE case_id = ?"
        if pending_only:
            sql += " AND delivered_at IS NULL"
        rows = self._q(sql + " ORDER BY created_at DESC, alert_id DESC", (case_id,))
        return [self._row_to_alert(r) for r in rows]

    def mark_alerts_delivered(self, alert_ids: Sequence[str]) -> int:
        """Set ``delivered_at`` on undelivered alerts; returns how many changed."""
        ids = _dedupe(alert_ids)
        if not ids:
            return 0
        n = 0
        with self._tx() as c:
            for chunk in _chunks(ids):
                cur = c.execute(
                    f"UPDATE alerts SET delivered_at = ? WHERE delivered_at IS NULL AND "
                    f"alert_id IN ({_marks(len(chunk))})",
                    [_now_ts(), *chunk],
                )
                n += cur.rowcount
        return n

    # ------------------------------------------------------------------ jobs

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            job_id=row["job_id"],
            case_id=row["case_id"],
            trigger=row["trigger"],
            status=row["status"],
            counts=json.loads(row["counts"]),
            error=row["error"],
            queued_at=_dt(row["queued_at"]),
            started_at=_dt(row["started_at"]),
            finished_at=_dt(row["finished_at"]),
        )

    def request_job(self, case_id: str, trigger: str) -> tuple[Job, bool]:
        """Queue a scan; if one is already queued for the case, return it. ``(job, created)``."""
        if trigger not in ("agent", "ui", "upload"):
            raise ValueError(f"bad trigger {trigger!r}")
        with self._tx() as c:
            row = c.execute(
                "SELECT * FROM jobs WHERE case_id = ? AND status = 'queued' "
                "ORDER BY queued_at LIMIT 1",
                (case_id,),
            ).fetchone()
            if row is not None:
                return self._row_to_job(row), False
            job_id = self._next_id("J")
            c.execute(
                "INSERT INTO jobs (job_id, case_id, trigger, status, queued_at) "
                "VALUES (?, ?, ?, 'queued', ?)",
                (job_id, case_id, trigger, _now_ts()),
            )
            row = c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row), True

    def claim_next_job(self) -> Job | None:
        """Mark the oldest queued job running, skipping cases that already have one running."""
        with self._tx() as c:
            row = c.execute(
                """SELECT * FROM jobs j WHERE status = 'queued' AND NOT EXISTS (
                       SELECT 1 FROM jobs r WHERE r.case_id = j.case_id AND r.status = 'running')
                   ORDER BY queued_at, job_id LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            c.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE job_id = ?",
                (_now_ts(), row["job_id"]),
            )
            row = c.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
        return self._row_to_job(row)

    def update_job_counts(self, job_id: str, counts: dict[str, int]) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE jobs SET counts = ? WHERE job_id = ?",
                (json.dumps({k: int(v) for k, v in counts.items()}), job_id),
            )

    def finish_job(
        self, job_id: str, status: str, *, error: str | None = None,
        counts: dict[str, int] | None = None,
    ) -> Job:
        if status not in ("done", "failed"):
            raise ValueError(f"bad final status {status!r}")
        with self._tx() as c:
            if counts is not None:
                c.execute(
                    "UPDATE jobs SET counts = ? WHERE job_id = ?",
                    (json.dumps({k: int(v) for k, v in counts.items()}), job_id),
                )
            c.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE job_id = ?",
                (status, error, _now_ts(), job_id),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def requeue_running(self) -> int:
        """On worker start-up: jobs left 'running' by a crash go back to the queue."""
        with self._tx() as c:
            cur = c.execute(
                "UPDATE jobs SET status = 'queued', started_at = NULL WHERE status = 'running'"
            )
        return cur.rowcount

    def get_job(self, job_id: str) -> Job | None:
        row = self._q1("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        return self._row_to_job(row) if row else None

    def latest_job(self, case_id: str) -> Job | None:
        row = self._q1(
            "SELECT * FROM jobs WHERE case_id = ? ORDER BY queued_at DESC, job_id DESC LIMIT 1",
            (case_id,),
        )
        return self._row_to_job(row) if row else None

    def list_jobs(self, case_id: str, limit: int = 20) -> list[Job]:
        rows = self._q(
            "SELECT * FROM jobs WHERE case_id = ? ORDER BY queued_at DESC, job_id DESC LIMIT ?",
            (case_id, limit),
        )
        return [self._row_to_job(r) for r in rows]

    # ------------------------------------------------------------------ search indexes

    def _reindex(self, c: sqlite3.Connection, fact_id: str) -> None:
        """Rebuild the FTS row for one fact (title, details, notes, evidence quotes)."""
        row = c.execute(
            "SELECT pk, title, details FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            return
        notes = " \n".join(
            r["body"] for r in c.execute("SELECT body FROM notes WHERE fact_id = ?", (fact_id,))
        )
        quotes = " \n".join(
            r["quote"]
            for r in c.execute(
                "SELECT quote FROM evidence WHERE fact_id = ? AND quote IS NOT NULL "
                "AND review != 'rejected'",
                (fact_id,),
            )
        )
        c.execute("DELETE FROM facts_fts WHERE rowid = ?", (row["pk"],))
        c.execute(
            "INSERT INTO facts_fts (rowid, title, details, notes, quotes) VALUES (?, ?, ?, ?, ?)",
            (row["pk"], row["title"], row["details"] or "", notes, quotes),
        )

    def fts_fact_ids(self, case_id: str, match: str, limit: int = 1000) -> list[str]:
        """Fact ids (non-rejected) matching an FTS5 MATCH expression, best first (bm25)."""
        rows = self._q(
            """SELECT f.fact_id FROM facts_fts JOIN facts f ON f.pk = facts_fts.rowid
               WHERE facts_fts MATCH ? AND f.case_id = ? AND f.review != 'rejected'
               ORDER BY bm25(facts_fts) LIMIT ?""",
            (match, case_id, limit),
        )
        return [r["fact_id"] for r in rows]

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if not texts:
            return []
        try:
            if self._embedder is None:
                from pipeline import llm

                self._embedder = llm.embed
            vectors = self._embedder(texts)
        except Exception as exc:  # noqa: BLE001 - vectors are optional
            log.warning("embedding failed (%s); skipping vector index", exc)
            return None
        if any(len(v) != EMBED_DIM for v in vectors):
            log.warning("embedding dim mismatch (expected %d); skipping vectors", EMBED_DIM)
            return None
        return vectors

    def refresh_vectors(self, case_id: str, batch: int = 64) -> int:
        """Embed facts whose text changed since their last vector; returns how many."""
        if not self._vec:
            return 0
        import sqlite_vec

        done = 0
        while True:
            rows = self._q(
                "SELECT pk, fact_id, title, details FROM facts WHERE case_id = ? AND vec_dirty = 1 "
                "ORDER BY pk LIMIT ?",
                (case_id, batch),
            )
            if not rows:
                return done
            texts = []
            for r in rows:
                quotes = [
                    q["quote"]
                    for q in self._q(
                        "SELECT quote FROM evidence WHERE fact_id = ? AND quote IS NOT NULL "
                        "AND review != 'rejected' ORDER BY pk LIMIT 3",
                        (r["fact_id"],),
                    )
                ]
                texts.append(" | ".join([r["title"], r["details"] or "", *quotes]))
            vectors = self._embed(texts)
            if vectors is None:
                return done
            with self._tx() as c:
                for r, vec in zip(rows, vectors, strict=True):
                    c.execute("DELETE FROM fact_vec WHERE rowid = ?", (r["pk"],))
                    c.execute(
                        "INSERT INTO fact_vec (rowid, embedding, case_id) VALUES (?, ?, ?)",
                        (r["pk"], sqlite_vec.serialize_float32(list(vec)), case_id),
                    )
                    c.execute("UPDATE facts SET vec_dirty = 0 WHERE pk = ?", (r["pk"],))
            done += len(rows)

    def vector_fact_ids(
        self, case_id: str, text: str, k: int = 20, max_distance: float = 0.95
    ) -> list[str]:
        """Nearest non-rejected facts to ``text`` by cosine distance; [] without sqlite-vec."""
        if not self._vec or not text.strip():
            return []
        import sqlite_vec

        qvec = self._embed([text])
        if not qvec or not any(qvec[0]):
            return []
        rows = self._q(
            """SELECT rowid, distance FROM fact_vec WHERE embedding MATCH ? AND k = ?
               AND case_id = ? ORDER BY distance""",
            (sqlite_vec.serialize_float32(qvec[0]), min(max(k * 2, k), 4096), case_id),
        )
        pks = [r["rowid"] for r in rows if r["distance"] is not None
               and r["distance"] < max_distance]
        if not pks:
            return []
        found = {
            r["pk"]: r["fact_id"]
            for r in self._q(
                f"SELECT pk, fact_id FROM facts WHERE review != 'rejected' AND pk IN "
                f"({_marks(len(pks))})",
                pks,
            )
        }
        return [found[pk] for pk in pks if pk in found][:k]

    # ------------------------------------------------------------------ activity (digest)

    def activity_since(self, case_id: str, since: datetime) -> dict[str, Any]:
        """What changed after ``since``: new facts, reviews, contradictions, alerts, documents."""
        s = _ts(since)
        facts = self._q(
            "SELECT * FROM facts WHERE case_id = ? AND created_at > ? AND review != 'rejected' "
            "ORDER BY created_at, pk",
            (case_id, s),
        )
        reviews = self._q(
            "SELECT action, COUNT(*) n FROM review_events WHERE case_id = ? AND at > ? "
            "GROUP BY action",
            (case_id, s),
        )
        contradictions = self._q(
            "SELECT * FROM contradictions WHERE case_id = ? AND created_at > ? ORDER BY created_at",
            (case_id, s),
        )
        alerts = self._q(
            "SELECT * FROM alerts WHERE case_id = ? AND created_at > ? ORDER BY created_at",
            (case_id, s),
        )
        docs = self._q(
            "SELECT * FROM documents WHERE case_id = ? AND added_at > ? ORDER BY added_at",
            (case_id, s),
        )
        return {
            "facts": self._hydrate(facts, rejected_evidence=False),
            "reviews": {r["action"]: r["n"] for r in reviews},
            "contradictions": [self._row_to_contradiction(r) for r in contradictions],
            "alerts": [self._row_to_alert(r) for r in alerts],
            "documents": [self._row_to_document(r) for r in docs],
        }

    def review_counts(self, case_id: str) -> dict[str, int]:
        """Fact counts by review state."""
        rows = self._q(
            "SELECT review, COUNT(*) n FROM facts WHERE case_id = ? GROUP BY review", (case_id,)
        )
        out = {"draft": 0, "accepted": 0, "rejected": 0}
        out.update({r["review"]: r["n"] for r in rows})
        return out
