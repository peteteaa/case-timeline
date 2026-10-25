"""SQLite (WAL) persistence with FTS5 keyword search and sqlite-vec vector search."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.contracts import (
    Alert,
    Changes,
    Contradiction,
    ContradictionStatus,
    Document,
    Event,
    EventStatus,
    Job,
    JobState,
    Page,
    ParsedDocument,
)

log = logging.getLogger(__name__)

EMBED_DIM = 1024
RRF_K = 60
_MAX_VEC_DISTANCE = 0.95  # cosine distance; drops orthogonal (unrelated) neighbours
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_HUMAN_STATUSES = (EventStatus.ACCEPTED.value, EventStatus.REJECTED.value)
# Event.importance at or above this marks a "key event" (the timeline's default view).
KEY_EVENT_THRESHOLD = 0.6
DEFAULT_IMPORTANCE = 0.5  # Event.importance default; also the value migrated rows get


def is_key_event(event: Event, threshold: float = KEY_EVENT_THRESHOLD) -> bool:
    """True when the event's importance reaches the key-event threshold."""
    return event.importance >= threshold


Embedder = Callable[[list[str]], list[list[float]]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    path TEXT NOT NULL,
    filename TEXT NOT NULL,
    mime TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    num_pages INTEGER,
    ingested_at TEXT,
    processed_at TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_case ON documents(case_id);

CREATE TABLE IF NOT EXISTS doc_text (
    doc_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    pages TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    pk INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    case_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    date_text TEXT NOT NULL,
    date_start TEXT,
    date_end TEXT,
    certainty TEXT NOT NULL,
    anchor_event_id TEXT,
    actors TEXT NOT NULL DEFAULT '[]',
    description TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    deadline_relevant INTEGER NOT NULL DEFAULT 0,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_case_date ON events(case_id, date_start);
CREATE INDEX IF NOT EXISTS events_case_created ON events(case_id, created_at);
CREATE INDEX IF NOT EXISTS events_doc ON events(doc_id);

CREATE TABLE IF NOT EXISTS event_dups (
    event_id TEXT PRIMARY KEY,
    dup_of TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS event_dups_of ON event_dups(dup_of);

CREATE TABLE IF NOT EXISTS contradictions (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    event_ids TEXT NOT NULL,
    field TEXT NOT NULL,
    explanation TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS contradictions_case ON contradictions(case_id, created_at);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    type TEXT NOT NULL,
    alert_created_at TEXT NOT NULL,
    dashboard_url TEXT NOT NULL,
    ref_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_case ON alerts(case_id, created_at);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    state TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    docs_total INTEGER NOT NULL DEFAULT 0,
    docs_done INTEGER NOT NULL DEFAULT 0,
    events_added INTEGER NOT NULL DEFAULT 0,
    contradictions_added INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_case ON jobs(case_id, created_at);

CREATE TABLE IF NOT EXISTS cursors (
    consumer TEXT NOT NULL,
    case_id TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    PRIMARY KEY (consumer, case_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    description, actors, quote, tokenize = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS events_fts_ai AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, description, actors, quote) VALUES (
        new.pk, new.description,
        (SELECT group_concat(value, ' ') FROM json_each(new.actors)),
        json_extract(new.source, '$.quote'));
END;
CREATE TRIGGER IF NOT EXISTS events_fts_ad AFTER DELETE ON events BEGIN
    DELETE FROM events_fts WHERE rowid = old.pk;
END;
CREATE TRIGGER IF NOT EXISTS events_fts_au AFTER UPDATE OF description, actors, source ON events
BEGIN
    DELETE FROM events_fts WHERE rowid = old.pk;
    INSERT INTO events_fts(rowid, description, actors, quote) VALUES (
        new.pk, new.description,
        (SELECT group_concat(value, ' ') FROM json_each(new.actors)),
        json_extract(new.source, '$.quote'));
END;
"""

_VEC_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS event_vec USING vec0(
    embedding float[{EMBED_DIM}] distance_metric=cosine,
    case_id text
);
"""

_JOB_FIELDS = tuple(Job.model_fields)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _ts(value: datetime | None) -> str | None:
    """Normalize a datetime to the UTC ISO string format used in the db (naive = UTC)."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _importance(row: sqlite3.Row) -> float:
    """The row's importance, clamped to [0, 1]; the default for rows that predate the column."""
    value = row["importance"] if "importance" in row.keys() else None
    if value is None:
        return DEFAULT_IMPORTANCE
    return min(1.0, max(0.0, float(value)))


def _embed_text(event: Event) -> str:
    return " | ".join([event.description, ", ".join(event.actors), event.source.quote])


def _fts_query(query: str) -> str | None:
    tokens = _TOKEN_RE.findall(query)
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else None


class Store:
    """Thread-safe store: one connection guarded by a re-entrant lock."""

    def __init__(self, db_path: str | Path, embedder: Embedder | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._vec = self._load_vec()
        self._initialized = False
        self.init()

    # --- setup ----------------------------------------------------------------------------

    def _load_vec(self) -> bool:
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except Exception as exc:  # noqa: BLE001 - degrade to FTS-only on any failure
            log.warning("sqlite-vec unavailable (%s); search_events will be FTS-only", exc)
            return False

    @property
    def vector_enabled(self) -> bool:
        return self._vec

    def init(self) -> None:
        """Create tables, indexes, FTS5 triggers and the vec0 table (idempotent)."""
        with self._lock:
            if self._initialized:
                return
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._migrate()
            if self._vec:
                try:
                    self._conn.executescript(_VEC_SCHEMA)
                except sqlite3.Error as exc:
                    log.warning("could not create vec0 table (%s); FTS-only search", exc)
                    self._vec = False
            self._initialized = True

    def _migrate(self) -> None:
        """Bring databases created by older versions up to the current schema (idempotent)."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(events)")}
        if "importance" not in cols:
            self._conn.execute(
                "ALTER TABLE events ADD COLUMN importance REAL NOT NULL "
                f"DEFAULT {DEFAULT_IMPORTANCE}"
            )
            self._conn.commit()
            log.info("migrated events table: added importance (default %s)", DEFAULT_IMPORTANCE)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _write(self) -> Iterator[str]:
        """Write transaction holding SQLite's write lock; yields the ``created_at`` stamp.

        The stamp is taken *after* the lock is acquired, and :meth:`get_changes` takes the
        same lock before choosing its upper bound, so a row can never be committed with a
        ``created_at`` at or below a cursor that has already moved past it.
        """
        with self._lock:
            if self._conn.in_transaction:
                self._conn.commit()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield _now()
            except BaseException:
                self._conn.rollback()
                raise
            self._conn.commit()

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

    # --- documents ------------------------------------------------------------------------

    def upsert_document(self, doc: Document) -> None:
        now = _now()
        ingested = _ts(doc.ingested_at) or now
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO documents (id, case_id, path, filename, mime, sha256, num_pages,
                       ingested_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET case_id=excluded.case_id, path=excluded.path,
                       filename=excluded.filename, mime=excluded.mime, sha256=excluded.sha256,
                       num_pages=COALESCE(excluded.num_pages, documents.num_pages)""",
                (
                    doc.id,
                    doc.case_id,
                    doc.path,
                    doc.filename,
                    doc.mime,
                    doc.sha256,
                    doc.num_pages,
                    ingested,
                    now,
                ),
            )

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            case_id=row["case_id"],
            path=row["path"],
            filename=row["filename"],
            mime=row["mime"],
            sha256=row["sha256"],
            num_pages=row["num_pages"],
            ingested_at=_dt(row["ingested_at"]),
        )

    def get_document(self, doc_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self, case_id: str) -> list[Document]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE case_id = ? ORDER BY filename, id", (case_id,)
            ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def add_parsed_text(self, doc_id: str, text: str, pages: Sequence[Page] = ()) -> None:
        pages_json = json.dumps([Page.model_validate(p).model_dump() for p in pages])
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO doc_text (doc_id, text, pages, created_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT(doc_id) DO UPDATE SET text=excluded.text, pages=excluded.pages""",
                (doc_id, text, pages_json, _now()),
            )

    def get_parsed_text(self, doc_id: str) -> ParsedDocument | None:
        """The stored normalized text + pages, or None if the doc or its text is missing."""
        with self._lock:
            row = self._conn.execute(
                "SELECT text, pages FROM doc_text WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        doc = self.get_document(doc_id)
        if row is None or doc is None:
            return None
        pages = [Page.model_validate(p) for p in json.loads(row["pages"])]
        return ParsedDocument(doc=doc, text=row["text"], pages=pages)

    def mark_processed(self, doc_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE documents SET processed_at = ? WHERE id = ?", (_now(), doc_id)
            )

    def processed_doc_ids(self, case_id: str) -> set[str]:
        """Docs marked processed or whose parsed text is stored."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT id FROM documents d WHERE case_id = ? AND (processed_at IS NOT NULL
                       OR EXISTS (SELECT 1 FROM doc_text t WHERE t.doc_id = d.id))""",
                (case_id,),
            ).fetchall()
        return {r["id"] for r in rows}

    # --- events ---------------------------------------------------------------------------

    def add_events(
        self, events: Iterable[Event], embeddings: Sequence[Sequence[float]] | None = None
    ) -> int:
        """Insert or update events (human ACCEPTED/REJECTED status is preserved).

        Embeddings are computed via ``llm.embed`` (or the injected embedder) for events whose
        text changed, unless supplied. Returns the number of events written.
        """
        events = list(events)
        if not events:
            return 0
        vectors: dict[str, Sequence[float]] = {}
        if self._vec:
            if embeddings is not None:
                if len(embeddings) != len(events):
                    raise ValueError("embeddings must align with events")
                vectors = {e.id: v for e, v in zip(events, embeddings, strict=True)}
            else:
                stale = self._events_needing_vectors(events)
                computed = self._embed([_embed_text(e) for e in stale])
                if computed:
                    vectors = {e.id: v for e, v in zip(stale, computed, strict=True)}

        with self._write() as now:
            for e in events:
                self._conn.execute(
                    """INSERT INTO events (id, case_id, doc_id, date_text, date_start, date_end,
                           certainty, anchor_event_id, actors, description, source, confidence,
                           status, deadline_relevant, importance, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET case_id=excluded.case_id,
                           doc_id=excluded.doc_id, date_text=excluded.date_text,
                           date_start=excluded.date_start, date_end=excluded.date_end,
                           certainty=excluded.certainty, anchor_event_id=excluded.anchor_event_id,
                           actors=excluded.actors, description=excluded.description,
                           source=excluded.source, confidence=excluded.confidence,
                           status=CASE WHEN events.status IN (?, ?) THEN events.status
                                       ELSE excluded.status END,
                           deadline_relevant=excluded.deadline_relevant,
                           importance=excluded.importance,
                           updated_at=excluded.updated_at""",
                    (
                        e.id,
                        e.case_id,
                        e.source.doc_id,
                        e.date_text,
                        e.date_start.isoformat() if e.date_start else None,
                        e.date_end.isoformat() if e.date_end else None,
                        e.certainty.value,
                        e.anchor_event_id,
                        json.dumps(e.actors),
                        e.description,
                        e.source.model_dump_json(),
                        e.confidence,
                        e.status.value,
                        int(e.deadline_relevant),
                        e.importance,
                        now,
                        now,
                        *_HUMAN_STATUSES,
                    ),
                )
            if vectors:
                self._write_vectors(vectors)
        return len(events)

    def _events_needing_vectors(self, events: list[Event]) -> list[Event]:
        """Events that are new, or whose embedded text changed, or that lack a vector."""
        ids = [e.id for e in events]
        marks = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT e.id, e.description, e.actors, e.source,
                           EXISTS (SELECT 1 FROM event_vec v WHERE v.rowid = e.pk) AS has_vec
                    FROM events e WHERE e.id IN ({marks})""",
                ids,
            ).fetchall()
        current = {
            r["id"]: (r["description"], json.loads(r["actors"]), json.loads(r["source"])["quote"])
            for r in rows
            if r["has_vec"]
        }
        return [e for e in events if current.get(e.id) != (e.description, e.actors, e.source.quote)]

    def _write_vectors(self, vectors: dict[str, Sequence[float]]) -> None:
        import sqlite_vec

        for event_id, vec in vectors.items():
            row = self._conn.execute(
                "SELECT pk, case_id FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            if row is None:
                continue
            self._conn.execute("DELETE FROM event_vec WHERE rowid = ?", (row["pk"],))
            self._conn.execute(
                "INSERT INTO event_vec (rowid, embedding, case_id) VALUES (?, ?, ?)",
                (row["pk"], sqlite_vec.serialize_float32(list(vec)), row["case_id"]),
            )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"],
            case_id=row["case_id"],
            date_text=row["date_text"],
            date_start=row["date_start"],
            date_end=row["date_end"],
            certainty=row["certainty"],
            anchor_event_id=row["anchor_event_id"],
            actors=json.loads(row["actors"]),
            description=row["description"],
            source=json.loads(row["source"]),
            confidence=row["confidence"],
            status=row["status"],
            deadline_relevant=bool(row["deadline_relevant"]),
            importance=_importance(row),
        )

    def get_event(self, event_id: str) -> Event | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return self._row_to_event(row) if row else None

    def list_events(
        self,
        case_id: str,
        status: EventStatus | str | None = None,
        *,
        include_duplicates: bool = False,
        min_importance: float | None = None,
    ) -> list[Event]:
        """Chronological events (undated last); linked duplicates hidden by default.

        ``min_importance`` keeps only events rated at least that important (e.g.
        ``KEY_EVENT_THRESHOLD`` for key events).
        """
        sql = "SELECT * FROM events e WHERE e.case_id = ?"
        params: list[Any] = [case_id]
        if status is not None:
            sql += " AND e.status = ?"
            params.append(EventStatus(status).value)
        if min_importance is not None:
            sql += " AND e.importance >= ?"
            params.append(float(min_importance))
        if not include_duplicates:
            sql += " AND NOT EXISTS (SELECT 1 FROM event_dups d WHERE d.event_id = e.id)"
        sql += " ORDER BY e.date_start IS NULL, e.date_start, e.date_end, e.created_at, e.pk"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def set_event_status(self, event_id: str, status: EventStatus | str) -> bool:
        """Set review status; returns False if the event does not exist."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE events SET status = ?, updated_at = ? WHERE id = ?",
                (EventStatus(status).value, _now(), event_id),
            )
        return cur.rowcount > 0

    def link_duplicate(self, event_id: str, dup_of: str) -> None:
        """Record that ``event_id`` duplicates ``dup_of`` (both sources are kept)."""
        if event_id == dup_of:
            raise ValueError("an event cannot duplicate itself")
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO event_dups (event_id, dup_of, created_at) VALUES (?, ?, ?)
                   ON CONFLICT(event_id) DO UPDATE SET dup_of = excluded.dup_of""",
                (event_id, dup_of, _now()),
            )

    def duplicates_of(self, event_id: str) -> list[str]:
        """Ids of events linked as duplicates of ``event_id``."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT event_id FROM event_dups WHERE dup_of = ? ORDER BY created_at",
                (event_id,),
            ).fetchall()
        return [r["event_id"] for r in rows]

    def search_events(self, case_id: str, query: str, k: int = 10) -> list[Event]:
        """Hybrid search: reciprocal-rank fusion of FTS5 (bm25) and vector KNN."""
        if k <= 0 or not query.strip():
            return []
        pool = max(k * 4, 20)
        rankings: list[list[int]] = []
        match = _fts_query(query)
        if match:
            with self._lock:
                rows = self._conn.execute(
                    """SELECT e.pk FROM events_fts f JOIN events e ON e.pk = f.rowid
                       WHERE events_fts MATCH ? AND e.case_id = ?
                       ORDER BY bm25(events_fts) LIMIT ?""",
                    (match, case_id, pool),
                ).fetchall()
            rankings.append([r["pk"] for r in rows])
        if self._vec:
            qvec = self._embed([query])
            # A zero vector (e.g. a punctuation-only query) has no cosine distance.
            if qvec and any(qvec[0]):
                import sqlite_vec

                with self._lock:
                    rows = self._conn.execute(
                        """SELECT rowid, distance FROM event_vec
                           WHERE embedding MATCH ? AND k = ? AND case_id = ?
                           ORDER BY distance""",
                        (sqlite_vec.serialize_float32(qvec[0]), min(pool, 4096), case_id),
                    ).fetchall()
                rankings.append(
                    [
                        r["rowid"]
                        for r in rows
                        if r["distance"] is not None and r["distance"] < _MAX_VEC_DISTANCE
                    ]
                )

        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, pk in enumerate(ranking):
                scores[pk] = scores.get(pk, 0.0) + 1.0 / (RRF_K + rank + 1)
        if not scores:
            return []
        ordered = sorted(scores, key=lambda pk: (-scores[pk], pk))
        marks = ",".join("?" * len(ordered))
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM events e WHERE e.pk IN ({marks})
                    AND NOT EXISTS (SELECT 1 FROM event_dups d WHERE d.event_id = e.id)""",
                ordered,
            ).fetchall()
        by_pk = {r["pk"]: r for r in rows}
        return [self._row_to_event(by_pk[pk]) for pk in ordered if pk in by_pk][:k]

    # --- contradictions & alerts ----------------------------------------------------------

    def add_contradictions(self, contradictions: Iterable[Contradiction]) -> int:
        """Insert or update; an existing reviewed status (dismissed/confirmed) is preserved."""
        n = 0
        with self._write() as now:
            for c in contradictions:
                self._conn.execute(
                    """INSERT INTO contradictions (id, case_id, event_ids, field, explanation,
                           status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET event_ids=excluded.event_ids,
                           field=excluded.field, explanation=excluded.explanation,
                           updated_at=excluded.updated_at""",
                    (
                        c.id,
                        c.case_id,
                        json.dumps(c.event_ids),
                        c.field.value,
                        c.explanation,
                        c.status.value,
                        now,
                        now,
                    ),
                )
                n += 1
        return n

    @staticmethod
    def _row_to_contradiction(row: sqlite3.Row) -> Contradiction:
        return Contradiction(
            id=row["id"],
            case_id=row["case_id"],
            event_ids=json.loads(row["event_ids"]),
            field=row["field"],
            explanation=row["explanation"],
            status=row["status"],
        )

    def list_contradictions(
        self, case_id: str, status: ContradictionStatus | str | None = None
    ) -> list[Contradiction]:
        sql = "SELECT * FROM contradictions WHERE case_id = ?"
        params: list[Any] = [case_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(ContradictionStatus(status).value)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY created_at, id", params).fetchall()
        return [self._row_to_contradiction(r) for r in rows]

    def set_contradiction_status(
        self, contradiction_id: str, status: ContradictionStatus | str
    ) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE contradictions SET status = ?, updated_at = ? WHERE id = ?",
                (ContradictionStatus(status).value, _now(), contradiction_id),
            )
        return cur.rowcount > 0

    def add_alerts(self, alerts: Iterable[Alert]) -> int:
        """Insert new alerts (existing ids are left unchanged); returns the number inserted."""
        n = 0
        with self._write() as now:
            for a in alerts:
                cur = self._conn.execute(
                    """INSERT OR IGNORE INTO alerts (id, case_id, type, alert_created_at,
                           dashboard_url, ref_ids, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        a.id,
                        a.case_id,
                        a.type.value,
                        _ts(a.created_at),
                        a.dashboard_url,
                        json.dumps(a.ref_ids),
                        now,
                    ),
                )
                n += cur.rowcount
        return n

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> Alert:
        return Alert(
            id=row["id"],
            case_id=row["case_id"],
            type=row["type"],
            created_at=row["alert_created_at"],
            dashboard_url=row["dashboard_url"],
            ref_ids=json.loads(row["ref_ids"]),
        )

    def list_alerts(self, case_id: str) -> list[Alert]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM alerts WHERE case_id = ? ORDER BY created_at DESC, id", (case_id,)
            ).fetchall()
        return [self._row_to_alert(r) for r in rows]

    # --- jobs -----------------------------------------------------------------------------

    def create_job(self, case_id: str, docs_total: int = 0) -> Job:
        job = Job(
            id=uuid.uuid4().hex, case_id=case_id, state=JobState.QUEUED, docs_total=docs_total
        )
        self._save_job(job, created=True)
        return job

    def _save_job(self, job: Job, *, created: bool = False) -> None:
        cols = list(_JOB_FIELDS)
        values = [
            _ts(v) if isinstance(v, datetime) else getattr(v, "value", v)
            for v in (getattr(job, c) for c in cols)
        ]
        with self._lock, self._conn:
            if created:
                self._conn.execute(
                    f"INSERT INTO jobs ({', '.join(cols)}, created_at) "
                    f"VALUES ({', '.join('?' * (len(cols) + 1))})",
                    [*values, _now()],
                )
            else:
                sets = ", ".join(f"{c} = ?" for c in cols if c != "id")
                self._conn.execute(
                    f"UPDATE jobs SET {sets} WHERE id = ?",
                    [v for c, v in zip(cols, values, strict=True) if c != "id"] + [job.id],
                )

    def update_job(self, job: Job | str, **fields: Any) -> Job:
        """Apply field updates (e.g. ``state=JobState.DONE``) to a job and persist it."""
        unknown = set(fields) - (set(_JOB_FIELDS) - {"id", "case_id"})
        if unknown:
            raise ValueError(f"cannot update job fields: {sorted(unknown)}")
        current = self.get_job(job if isinstance(job, str) else job.id)
        if current is None:
            if isinstance(job, str):
                raise KeyError(f"no such job: {job}")
            self._save_job(job, created=True)
            current = job
        updated = Job.model_validate({**current.model_dump(), **fields})
        self._save_job(updated)
        return updated

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job.model_validate({c: row[c] for c in _JOB_FIELDS})

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def latest_job(self, case_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE case_id = ? ORDER BY created_at DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    # --- change feed ----------------------------------------------------------------------

    def changes_since(
        self, case_id: str, since: datetime | None, until: datetime | None = None
    ) -> Changes:
        """Events, contradictions and alerts first stored after ``since`` (all if None)."""
        lo = _ts(since) or ""
        hi = _ts(until) or "9999"
        params = (case_id, lo, hi)
        cond = "case_id = ? AND created_at > ? AND created_at <= ?"
        with self._lock:
            ev = self._conn.execute(
                f"SELECT * FROM events WHERE {cond} ORDER BY created_at, pk", params
            ).fetchall()
            co = self._conn.execute(
                f"SELECT * FROM contradictions WHERE {cond} ORDER BY created_at, id", params
            ).fetchall()
            al = self._conn.execute(
                f"SELECT * FROM alerts WHERE {cond} ORDER BY created_at, id", params
            ).fetchall()
        return Changes(
            case_id=case_id,
            since=since,
            new_events=[self._row_to_event(r) for r in ev],
            new_contradictions=[self._row_to_contradiction(r) for r in co],
            new_alerts=[self._row_to_alert(r) for r in al],
        )

    def get_cursor(self, consumer: str, case_id: str) -> datetime | None:
        """The ``last_seen`` timestamp for a consumer on a case (None if never advanced)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT last_seen FROM cursors WHERE consumer = ? AND case_id = ?",
                (consumer, case_id),
            ).fetchone()
        return _dt(row["last_seen"]) if row else None

    def advance_cursor(self, consumer: str, case_id: str, to: datetime | None = None) -> datetime:
        """Move a consumer's cursor to ``to`` (default: now); returns the new value."""
        stamp = _ts(to) or _now()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO cursors (consumer, case_id, last_seen) VALUES (?, ?, ?)
                   ON CONFLICT(consumer, case_id) DO UPDATE SET last_seen = excluded.last_seen""",
                (consumer, case_id, stamp),
            )
        return datetime.fromisoformat(stamp)

    def set_cursor(self, consumer: str, case_id: str, ts: datetime) -> None:
        """Alias of :meth:`advance_cursor` with an explicit timestamp."""
        self.advance_cursor(consumer, case_id, ts)

    def get_changes(self, case_id: str, consumer: str = "default") -> Changes:
        """Changes since this consumer's cursor, then advance the cursor past them."""
        with self._write() as stamp:
            since = self.get_cursor(consumer, case_id)
            until = datetime.fromisoformat(stamp)
            changes = self.changes_since(case_id, since, until=until)
            self._conn.execute(
                """INSERT INTO cursors (consumer, case_id, last_seen) VALUES (?, ?, ?)
                   ON CONFLICT(consumer, case_id) DO UPDATE SET last_seen = excluded.last_seen""",
                (consumer, case_id, stamp),
            )
        return changes
