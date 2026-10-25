"""Frozen day-0 contracts. Change only by team agreement; every module imports from here."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Certainty(StrEnum):
    EXACT = "exact"  # full calendar date stated ("March 3, 2024")
    APPROXIMATE = "approximate"  # "early March 2024", "around 3pm"
    RELATIVE = "relative"  # "two days after the meeting" (resolved via anchor)
    RANGE = "range"  # "between May and June 2024"
    UNKNOWN = "unknown"  # no usable date


class EventStatus(StrEnum):
    PROPOSED = "proposed"  # extracted, awaiting human review
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"  # quote could not be located verbatim in source


class ConflictField(StrEnum):
    DATE = "date"
    ACTOR = "actor"
    FACT = "fact"


class ContradictionStatus(StrEnum):
    OPEN = "open"
    DISMISSED = "dismissed"
    CONFIRMED = "confirmed"


class SourceRef(BaseModel):
    doc_id: str
    page: int | None = None  # 1-based; None for formats without pages
    char_start: int  # offsets into the document's normalized full text
    char_end: int
    quote: str  # verbatim slice of the source text


class Event(BaseModel):
    id: str
    case_id: str
    date_text: str  # date as written in the source
    date_start: date | None = None  # resolved
    date_end: date | None = None  # == date_start unless a range
    certainty: Certainty = Certainty.UNKNOWN
    anchor_event_id: str | None = None  # for RELATIVE dates
    actors: list[str] = Field(default_factory=list)
    description: str
    source: SourceRef
    confidence: float = Field(ge=0.0, le=1.0)
    status: EventStatus = EventStatus.PROPOSED
    deadline_relevant: bool = False  # always surfaced as "needs attorney review"


class Contradiction(BaseModel):
    id: str
    case_id: str
    event_ids: list[str]  # >= 2
    field: ConflictField
    explanation: str
    status: ContradictionStatus = ContradictionStatus.OPEN


class Document(BaseModel):
    id: str  # sha256 of file bytes (first 16 hex chars)
    case_id: str
    path: str  # absolute path in the read-only evidence folder
    filename: str
    mime: str
    sha256: str
    num_pages: int | None = None
    ingested_at: datetime | None = None


class Page(BaseModel):
    number: int  # 1-based
    char_start: int  # offsets into ParsedDocument.text
    char_end: int


class ParsedDocument(BaseModel):
    doc: Document
    text: str  # normalized full text; SourceRef offsets index into this
    pages: list[Page] = Field(default_factory=list)


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job(BaseModel):
    id: str
    case_id: str
    state: JobState
    started_at: datetime | None = None
    finished_at: datetime | None = None
    docs_total: int = 0
    docs_done: int = 0
    events_added: int = 0
    contradictions_added: int = 0
    error: str | None = None


class AlertType(StrEnum):
    NEW_CONTRADICTION = "new_contradiction"
    FLAGGED_WITNESS = "flagged_witness"
    DEADLINE_DATE = "deadline_date"


class Alert(BaseModel):
    """Messaging-safe: contains NO case content — only case name, type, and local link."""

    id: str
    case_id: str
    type: AlertType
    created_at: datetime
    dashboard_url: str  # e.g. http://127.0.0.1:8080/cases/<case>/alerts/<id>
    ref_ids: list[str] = Field(default_factory=list)  # event/contradiction ids (local only)


class Changes(BaseModel):
    case_id: str
    since: datetime | None
    new_events: list[Event] = Field(default_factory=list)
    new_contradictions: list[Contradiction] = Field(default_factory=list)
    new_alerts: list[Alert] = Field(default_factory=list)
