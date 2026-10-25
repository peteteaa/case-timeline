"""Fake runner/store for MCP server tests (loaded in the server subprocess via CT_MCP_BACKEND)."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

from mcp_server.server import Backend
from pipeline.contracts import (
    Alert,
    AlertType,
    Changes,
    ConflictField,
    Contradiction,
    Document,
    Event,
    EventStatus,
    Job,
    JobState,
    Page,
    ParsedDocument,
    SourceRef,
)

TEXT = (
    "Page one intro. On March 3, 2024 Alice met Bob at the warehouse.\n"
    "Page two. Bob says the meeting was on March 5, 2024. The filing deadline is June 1, 2024."
)


def _ref(quote: str) -> SourceRef:
    start = TEXT.index(quote)
    page = 1 if start < TEXT.index("Page two") else 2
    return SourceRef(
        doc_id="doc1", page=page, char_start=start, char_end=start + len(quote), quote=quote
    )


EVENTS = [
    Event(
        id="e2",
        case_id="demo",
        date_text="March 5, 2024",
        date_start=date(2024, 3, 5),
        date_end=date(2024, 3, 5),
        certainty="exact",
        actors=["Bob"],
        description="Bob says meeting was March 5",
        source=_ref("the meeting was on March 5, 2024"),
        confidence=0.8,
    ),
    Event(
        id="e1",
        case_id="demo",
        date_text="March 3, 2024",
        date_start=date(2024, 3, 3),
        date_end=date(2024, 3, 3),
        certainty="exact",
        actors=["Alice", "Bob"],
        description="Alice met Bob at the warehouse",
        source=_ref("On March 3, 2024 Alice met Bob at the warehouse."),
        confidence=0.9,
        status=EventStatus.ACCEPTED,
    ),
    Event(
        id="e3",
        case_id="demo",
        date_text="June 1, 2024",
        date_start=date(2024, 6, 1),
        actors=[],
        description="Filing deadline",
        source=_ref("The filing deadline is June 1, 2024."),
        confidence=0.95,
        deadline_relevant=True,
    ),
    Event(
        id="e4",
        case_id="demo",
        date_text="",
        description="Undated remark",
        source=_ref("Page one intro."),
        confidence=0.5,
    ),
]

CONTRADICTION = Contradiction(
    id="c1",
    case_id="demo",
    event_ids=["e1", "e2"],
    field=ConflictField.DATE,
    explanation="Meeting dated March 3 vs March 5.",
)

ALERT = Alert(
    id="a1",
    case_id="demo",
    type=AlertType.NEW_CONTRADICTION,
    created_at=datetime(2024, 1, 1, tzinfo=UTC),
    dashboard_url="http://127.0.0.1:8080/alerts",
    ref_ids=["c1"],
)


class FakeStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list_events(self, case_id: str, status: EventStatus | None = None) -> list[Event]:
        return [e for e in EVENTS if status is None or e.status == status]

    def get_event(self, event_id: str) -> Event | None:
        return next((e for e in EVENTS if e.id == event_id), None)

    def list_contradictions(self, case_id: str, status=None) -> list[Contradiction]:
        return [CONTRADICTION] if status in (None, CONTRADICTION.status) else []

    def search_events(self, case_id: str, query: str, k: int) -> list[Event]:
        q = query.lower()
        return [e for e in EVENTS if q in e.description.lower()][:k]

    def changes_since(self, case_id: str, since: datetime | None) -> Changes:
        if since is not None:
            return Changes(case_id=case_id, since=since)
        return Changes(
            case_id=case_id,
            since=None,
            new_events=EVENTS,
            new_contradictions=[CONTRADICTION],
            new_alerts=[ALERT],
        )

    def get_parsed_text(self, doc_id: str) -> ParsedDocument | None:
        if doc_id != "doc1":
            return None
        split = TEXT.index("Page two")
        doc = Document(
            id="doc1",
            case_id="demo",
            path="/evidence/memo.pdf",
            filename="memo.pdf",
            mime="application/pdf",
            sha256="x" * 64,
            num_pages=2,
        )
        return ParsedDocument(
            doc=doc,
            text=TEXT,
            pages=[
                Page(number=1, char_start=0, char_end=split),
                Page(number=2, char_start=split, char_end=len(TEXT)),
            ],
        )


def make_backend() -> Backend:
    store = FakeStore()

    def scan_case(case_id: str) -> Job:
        print("NOISE FROM A LIBRARY THAT PRINTS")  # must not reach the JSON-RPC stream
        job = Job(id="job-1", case_id=case_id, state=JobState.QUEUED)
        store.jobs[job.id] = job.model_copy(update={"state": JobState.DONE, "docs_total": 1})
        return job

    def daily_digest(case_id: str) -> dict:
        print("MORE NOISE")
        return {"case_id": case_id, "new_events": 4, "open_contradictions": 1}

    return Backend(
        store=store,
        scan_case=scan_case,
        daily_digest=daily_digest,
        case_id="demo",
        data_dir=Path(os.environ["CT_DATA_DIR"]),
    )
