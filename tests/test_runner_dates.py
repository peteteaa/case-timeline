"""The runner dates each document against the case: incident-date consensus and time zone."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from pipeline import conflicts, extract, llm, runner
from pipeline.config import get_settings
from pipeline.contracts import Document, Event, JobState, ParsedDocument, SourceRef
from pipeline.store import Store

REPORT_A = "Police Report\nDate of Accident: 06/05/2021\n\nThe truck overturned.\n"
REPORT_B = "Adjuster Notes\nDate of Loss: June 5, 2021\n\nThe trailer was damaged.\n"
REPORT_C = "Tow Log\nDate of Incident: 06/06/2021\n\nTowed the trailer.\n"
INTERVIEW = (
    "Interview about the truck accident\n"
    "September 2, 2021\n\n"
    "A. At 0931 the truck tipped over.\n"
    "> At 0931 | At 0931 the truck tipped over.\n"
    "A. The radio log says June 6, 2021, 0130 UTC for the tow call.\n"
    "> June 6, 2021, 0130 UTC | June 6, 2021, 0130 UTC for the tow call.\n"
)


def _parsed(text: str, doc_id: str = "d") -> ParsedDocument:
    doc = Document(
        id=doc_id, case_id="c", path="/x", filename=f"{doc_id}.txt", mime="x", sha256="x"
    )
    return ParsedDocument(doc=doc, text=text)


def test_case_incident_date_is_the_documents_consensus() -> None:
    docs = [_parsed(REPORT_A, "a"), _parsed(REPORT_B, "b"), _parsed(REPORT_C, "c")]
    assert runner.case_incident_date(docs) == date(2021, 6, 5)
    assert runner.case_incident_date(docs + [_parsed(INTERVIEW, "i")]) == date(2021, 6, 5)


def test_case_incident_date_needs_a_clear_winner() -> None:
    assert runner.case_incident_date([_parsed(REPORT_A, "a"), _parsed(REPORT_C, "c")]) is None
    assert runner.case_incident_date([_parsed(INTERVIEW, "i")]) is None
    assert runner.case_incident_date([]) is None


# --------------------------------------------------------------------------- a real scan


def _fake_extract(parsed: ParsedDocument) -> list[Event]:
    """One event per "> date_text | quote" line of the document."""
    events = []
    for line in parsed.text.splitlines():
        if not line.startswith("> "):
            continue
        date_text, quote = (part.strip() for part in line[2:].split("|", 1))
        eid = hashlib.sha256(f"{parsed.doc.id}{quote}".encode()).hexdigest()[:16]
        events.append(
            Event(
                id=eid,
                case_id=parsed.doc.case_id,
                date_text=date_text,
                description=quote,
                source=SourceRef(doc_id=parsed.doc.id, char_start=0, char_end=0, quote=quote),
                confidence=0.9,
            )
        )
    return events


@pytest.fixture
def case_dir(monkeypatch, tmp_path: Path) -> Path:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "a_report.txt").write_text(REPORT_A)
    (evidence / "b_notes.txt").write_text(REPORT_B)
    monkeypatch.setenv("CT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CT_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("CT_CASE_ID", "c")
    monkeypatch.setenv("CT_DISABLE_DOCLING", "1")
    monkeypatch.delenv("CT_DB_PATH", raising=False)
    monkeypatch.delenv("CT_CASE_TZ", raising=False)
    monkeypatch.delenv("CT_CASE_INCIDENT_DATE", raising=False)
    monkeypatch.setattr(extract, "extract_events", _fake_extract)
    monkeypatch.setattr(llm, "embed", lambda texts: [llm.hashed_embedding(t) for t in texts])
    monkeypatch.setattr(conflicts, "find_contradictions", lambda *a, **k: [])
    get_settings.cache_clear()
    yield evidence
    get_settings.cache_clear()


def _dates() -> dict[str, date | None]:
    with Store(get_settings().db_path) as store:
        return {e.date_text: e.date_start for e in store.list_events("c", include_duplicates=True)}


def test_scan_dates_an_interview_against_the_case_incident(case_dir: Path, monkeypatch) -> None:
    monkeypatch.setenv("CT_CASE_TZ", "America/Chicago")
    get_settings.cache_clear()
    assert runner.scan_case_sync("c").state == JobState.DONE  # the reports alone
    # The interview arrives later: the consensus comes from the stored reports.
    (case_dir / "c_interview.txt").write_text(INTERVIEW)
    job = runner.scan_case_sync("c")
    assert job.state == JobState.DONE and job.docs_total == 1, job.error
    found = _dates()
    assert found["At 0931"] == date(2021, 6, 5)  # not the interview date, September 2
    assert found["June 6, 2021, 0130 UTC"] == date(2021, 6, 5)  # 20:30 CDT on June 5


def test_scan_without_case_zone_keeps_utc_dates(case_dir: Path) -> None:
    (case_dir / "c_interview.txt").write_text(INTERVIEW)
    assert runner.scan_case_sync("c").state == JobState.DONE
    found = _dates()
    assert found["June 6, 2021, 0130 UTC"] == date(2021, 6, 6)
    assert found["At 0931"] == date(2021, 6, 5)


def test_configured_case_incident_date_wins(case_dir: Path, monkeypatch) -> None:
    monkeypatch.setenv("CT_CASE_INCIDENT_DATE", "2021-06-04")
    get_settings.cache_clear()
    (case_dir / "c_interview.txt").write_text(INTERVIEW)
    assert runner.scan_case_sync("c").state == JobState.DONE
    assert _dates()["At 0931"] == date(2021, 6, 4)


def test_settings_reject_an_unknown_time_zone(monkeypatch) -> None:
    from pipeline.config import Settings

    with pytest.raises(ValueError, match="time zone"):
        Settings(case_tz="Not/AZone")
    assert Settings(case_tz=" ").case_tz is None
    assert Settings(case_tz="UTC").case_tz == "UTC"
