"""Frozen day-0 contracts (SPEC §8.1). Change only by team agreement; every module imports here.

The first block is the spec's data model, verbatim in shape. The second block holds
pipeline-internal types (parsed text, drafts, queries) that never cross the MCP boundary.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, get_args

from pydantic import BaseModel, Field

Status = Literal["key_fact", "case_impact", "disputed"]
Review = Literal["draft", "accepted", "rejected"]
Origin = Literal["ai", "user"]
DocKind = Literal[
    "pleading", "motion", "order", "exhibit", "transcript", "email", "letter", "other"
]
Certainty = Literal["exact", "approximate", "relative", "unknown"]
ConflictField = Literal["date", "actor", "fact"]
ContradictionStatus = Literal["open", "confirmed", "dismissed"]
JobTrigger = Literal["agent", "ui", "upload"]
JobStatus = Literal["queued", "running", "done", "failed"]
AlertType = Literal["contradiction", "flagged_witness", "possible_deadline"]
EvidenceKind = Literal["span", "document", "external"]
ReviewAction = Literal["accept", "edit", "reject"]

STATUSES: tuple[str, ...] = get_args(Status)
REVIEWS: tuple[str, ...] = get_args(Review)
DOC_KINDS: tuple[str, ...] = get_args(DocKind)
CERTAINTIES: tuple[str, ...] = get_args(Certainty)
ALERT_TYPES: tuple[str, ...] = get_args(AlertType)

AGENT = "agent"  # created_by / updated_by for everything the pipeline writes


# --------------------------------------------------------------------------- SPEC §8.1


class Document(BaseModel):
    doc_id: str  # "D-0007"
    case_id: str
    filename: str
    sha256: str  # dedupe: renamed copies aren't reprocessed
    location: Literal["evidence", "inbox"]
    kind: DocKind
    pages: int | None
    parse_status: Literal["pending", "parsed", "failed"]
    added_at: datetime


class EvidenceLink(BaseModel):
    evidence_id: str  # "E-0311"
    kind: EvidenceKind
    doc_id: str | None = None  # span and document
    page: int | None = None  # span
    char_start: int | None = None
    char_end: int | None = None
    line_start: int | None = None  # transcripts cite page:line
    line_end: int | None = None
    quote: str | None = None  # span: must match the source verbatim
    reference: str | None = None  # external: URL or citation, never fetched
    origin: Origin
    review: Review  # evidence the AI attaches starts as a draft


class Fact(BaseModel):
    fact_id: str  # "F-0142": short, so chat answers can cite it
    case_id: str
    title: str  # the only required field
    details: str | None = None
    date_text: str | None = None  # as written in the source
    date_start: date | None = None  # None means undated
    date_end: date | None = None  # for ranges
    date_certainty: Certainty = "unknown"
    anchor_fact_id: str | None = None  # for relative dates
    statuses: list[Status] = []  # set by people only
    tag_ids: list[str] = []
    witness_ids: list[str] = []
    evidence: list[EvidenceLink] = []
    origin: Origin
    review: Review  # people's facts start as accepted
    confidence: float | None = None  # AI drafts only
    created_by: str  # display name, or "agent"
    created_at: datetime
    updated_by: str
    updated_at: datetime


class Witness(BaseModel):
    witness_id: str
    case_id: str
    name: str
    aliases: list[str] = []
    role: str | None = None
    notes: str | None = None
    flagged: bool = False  # alert when a new document mentions them
    confirmed: bool  # False for witnesses the AI suggested


class Tag(BaseModel):  # an issue
    tag_id: str
    case_id: str
    name: str
    color: str | None = None


class Note(BaseModel):
    note_id: str
    fact_id: str
    author: str
    body: str
    created_at: datetime


class ReviewEvent(BaseModel):  # the feedback trail, used by eval
    fact_id: str
    evidence_id: str | None
    action: ReviewAction
    reason: str | None
    reviewer: str
    at: datetime


class Contradiction(BaseModel):
    contradiction_id: str
    fact_ids: list[str]
    field: ConflictField
    explanation: str
    status: ContradictionStatus


class Job(BaseModel):
    job_id: str
    case_id: str
    trigger: JobTrigger
    status: JobStatus
    counts: dict[str, int]  # docs seen, parsed, failed; drafts; quotes rejected
    error: str | None
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class Alert(BaseModel):
    alert_id: str
    case_id: str
    type: AlertType
    fact_ids: list[str]
    doc_ids: list[str]
    created_at: datetime
    delivered_at: datetime | None  # set once the agent has announced it


# --------------------------------------------------------------------------- pipeline-internal


class Page(BaseModel):
    """A page of a parsed document: ``[char_start, char_end)`` in ParsedDocument.text."""

    number: int  # 1-based
    char_start: int
    char_end: int


class Line(BaseModel):
    """A printed transcript line number (page:line citations)."""

    page: int
    number: int
    char_start: int
    char_end: int


class ParsedDocument(BaseModel):
    doc: Document
    text: str  # normalized full text; evidence offsets index into this
    pages: list[Page] = Field(default_factory=list)
    lines: list[Line] = Field(default_factory=list)  # transcripts only


class SpanIn(BaseModel):
    """A located quote: what ``verify`` produces and ``Store.add_evidence`` stores."""

    doc_id: str
    page: int | None = None
    char_start: int
    char_end: int
    line_start: int | None = None
    line_end: int | None = None
    quote: str


class DraftFact(BaseModel):
    """A fact the extractor proposes, before verification, merging and storage.

    ``key`` is unique within one document's drafts; ``anchor_key`` points at another draft's
    key for relative dates, and ``anchor_fact_id`` at a stored fact.
    """

    key: str
    title: str
    details: str | None = None
    date_text: str | None = None
    date_start: date | None = None
    date_end: date | None = None
    date_certainty: Certainty = "unknown"
    anchor_key: str | None = None
    anchor_fact_id: str | None = None
    people: list[str] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)  # as the model wrote them
    spans: list[SpanIn] = Field(default_factory=list)  # filled by verify
    confidence: float = 0.5
    deadline_language: bool = False


SortKey = Literal[
    "date", "title", "evidence", "tags", "witnesses", "notes", "created", "updated", "confidence"
]


class FactQuery(BaseModel):
    """Filters and sort for list views, exports and MCP reads. Unset fields don't filter."""

    date_from: date | None = None
    date_to: date | None = None
    include_undated: bool = True  # when a date range is set, undated facts are dropped anyway
    statuses: list[Status] = Field(default_factory=list)  # any of
    tag_ids: list[str] = Field(default_factory=list)  # any of
    witness_ids: list[str] = Field(default_factory=list)  # any of
    doc_ids: list[str] = Field(default_factory=list)  # cites any of (non-rejected evidence)
    doc_kinds: list[DocKind] = Field(default_factory=list)  # cites a document of any kind
    reviews: list[Review] = Field(default_factory=lambda: ["draft", "accepted"])
    origins: list[Origin] = Field(default_factory=list)
    fact_ids: list[str] | None = None  # restrict to these (e.g. search hits); [] matches none
    sort: SortKey = "date"
    descending: bool = False
    limit: int | None = None
