import importlib
import sys
import types

import pytest

from pipeline import extract, prompts
from pipeline.config import get_settings
from pipeline.contracts import Document, EventStatus, Page, ParsedDocument
from pipeline.extract import (
    ExtractedEvent,
    ExtractionResult,
    chunk_text,
    event_id,
    extract_events,
)

TEXT = (
    "Acme Corp. hired John Smith on March 3, 2024.\n\n"
    "Two days after the meeting, Smith emailed Jane Doe about the invoice.\n\n"
    "The answer is due no later than April 1, 2024."
)


def make_parsed(text: str = TEXT, pages: list[Page] | None = None) -> ParsedDocument:
    doc = Document(
        id="doc0000000000001",
        case_id="demo",
        path="/evidence/memo.txt",
        filename="memo.txt",
        mime="text/plain",
        sha256="0" * 64,
    )
    return ParsedDocument(doc=doc, text=text, pages=pages or [])


@pytest.fixture(autouse=True)
def _sequential_llm(monkeypatch: pytest.MonkeyPatch):
    """The fakes below answer in call order, so chunks are sent one at a time here.
    Concurrent extraction is covered by tests/test_concurrency.py."""
    monkeypatch.setenv("CT_LLM_CONCURRENCY", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch):
    """Patch pipeline.llm.chat_json (or install a stub module if llm.py is unavailable)."""
    calls: list[tuple[str, str, type]] = []
    responses: list[object] = []

    def chat_json(system, user, schema, **kw):
        calls.append((system, user, schema))
        item = responses.pop(0) if responses else ExtractionResult()
        if isinstance(item, Exception):
            raise item
        return item

    try:
        llm = importlib.import_module("pipeline.llm")
    except Exception:
        llm = types.ModuleType("pipeline.llm")
        monkeypatch.setitem(sys.modules, "pipeline.llm", llm)
    monkeypatch.setattr(llm, "chat_json", chat_json, raising=False)
    return calls, responses


def test_extract_converts_to_events(fake_llm) -> None:
    calls, responses = fake_llm
    responses.append(
        ExtractionResult(
            events=[
                ExtractedEvent(
                    quote="Acme Corp. hired John Smith on March 3, 2024.",
                    date_text="March 3, 2024",
                    actors=["Acme Corp.", "John Smith", "John Smith", " "],
                    description="Acme Corp. hired John Smith.",
                    confidence=0.9,
                ),
                ExtractedEvent(
                    quote="The answer is due no later than April 1, 2024.",
                    date_text="no later than April 1, 2024",
                    description="Answer deadline.",
                    deadline_relevant=True,
                ),
            ]
        )
    )
    parsed = make_parsed(
        pages=[
            Page(number=1, char_start=0, char_end=60),
            Page(number=2, char_start=60, char_end=len(TEXT)),
        ]
    )
    events = extract_events(parsed)

    assert len(calls) == 1
    system, user, schema = calls[0]
    assert schema is ExtractionResult
    assert "verbatim" in system.lower()
    assert "memo.txt" in user and "Acme Corp. hired" in user

    first, second = events
    assert first.id == event_id(parsed.doc.id, first.source.quote, "March 3, 2024")
    assert len(first.id) == 16 and int(first.id, 16) >= 0
    assert first.case_id == "demo"
    assert first.actors == ["Acme Corp.", "John Smith"]
    assert first.status == EventStatus.PROPOSED
    assert first.date_start is None  # dates are resolved later
    assert TEXT[first.source.char_start : first.source.char_end] == first.source.quote
    assert first.source.page == 1
    assert second.deadline_relevant is True
    assert second.source.page == 2
    assert TEXT[second.source.char_start : second.source.char_end] == second.source.quote


def test_ids_are_deterministic_and_whitespace_insensitive() -> None:
    a = event_id("d", "Smith  emailed\nDoe.", "May 1, 2024")
    b = event_id("d", "Smith emailed Doe.", " May 1, 2024 ")
    assert a == b
    assert a != event_id("other", "Smith emailed Doe.", "May 1, 2024")


def test_unlocated_quote_points_at_chunk(fake_llm) -> None:
    _, responses = fake_llm
    responses.append(
        ExtractionResult(
            events=[
                ExtractedEvent(
                    quote="Smith e-mailed Doe regarding the invoice",
                    date_text="two days after the meeting",
                    description="Smith emailed Doe.",
                )
            ]
        )
    )
    (event,) = extract_events(make_parsed())
    assert (event.source.char_start, event.source.char_end) == (0, len(TEXT))
    assert event.source.page is None


def test_empty_quote_or_description_dropped(fake_llm) -> None:
    _, responses = fake_llm
    responses.append(
        ExtractionResult(
            events=[
                ExtractedEvent(quote="  ", date_text="March 3, 2024", description="x"),
                ExtractedEvent(quote="Acme Corp.", date_text="March 3, 2024", description=" "),
            ]
        )
    )
    assert extract_events(make_parsed()) == []


def test_overlapping_chunks_are_deduped(fake_llm, monkeypatch: pytest.MonkeyPatch) -> None:
    calls, responses = fake_llm
    para = "Filler sentence about nothing in particular. " * 30
    key = "On May 5, 2024 the court held a hearing."
    text = f"{para}\n\n{key}\n\n{para}"
    monkeypatch.setattr(extract, "CHUNK_SIZE", 1500)
    monkeypatch.setattr(extract, "CHUNK_OVERLAP", 400)
    spans = chunk_text(text, 1500, 400)
    in_chunks = [s for s in spans if key in text[s[0] : s[1]]]
    assert len(in_chunks) >= 2  # the key paragraph sits in the overlap

    def result(conf: float, actors: list[str], deadline: bool) -> ExtractionResult:
        return ExtractionResult(
            events=[
                ExtractedEvent(
                    quote=key,
                    date_text="May 5, 2024",
                    actors=actors,
                    description="Court hearing held.",
                    confidence=conf,
                    deadline_relevant=deadline,
                )
            ]
        )

    sightings = 0
    for s in spans:
        if key in text[s[0] : s[1]]:
            sightings += 1
            first = sightings == 1
            actors = ["Court"] if first else ["Court", "Smith"]
            responses.append(result(0.6 if first else 0.8, actors, first))
        else:
            responses.append(ExtractionResult())

    events = extract_events(make_parsed(text))
    assert len(calls) == len(spans)
    assert len(events) == 1
    (event,) = events
    assert event.confidence == 0.8
    assert event.actors == ["Court", "Smith"]
    assert event.deadline_relevant is True
    assert text[event.source.char_start : event.source.char_end] == key


def test_one_failed_chunk_is_skipped(fake_llm, monkeypatch: pytest.MonkeyPatch) -> None:
    _, responses = fake_llm
    text = "\n\n".join(f"Paragraph {i} " + "word " * 300 for i in range(4))
    monkeypatch.setattr(extract, "CHUNK_SIZE", 2000)
    monkeypatch.setattr(extract, "CHUNK_OVERLAP", 200)
    n = len(chunk_text(text, 2000, 200))
    assert n > 1
    responses.append(RuntimeError("model hiccup"))
    responses.extend(ExtractionResult() for _ in range(n - 1))
    assert extract_events(make_parsed(text)) == []


def test_all_chunks_failing_raises(fake_llm) -> None:
    _, responses = fake_llm
    responses.append(ConnectionError("vLLM down"))
    with pytest.raises(ConnectionError):
        extract_events(make_parsed())


def test_empty_document_makes_no_calls(fake_llm) -> None:
    calls, _ = fake_llm
    assert extract_events(make_parsed("")) == []
    assert calls == []


# ---------------------------------------------------------------------------- chunking


def test_chunk_small_text_is_single_chunk() -> None:
    assert chunk_text("hello world") == [(0, 11)]


def test_chunks_cover_text_overlap_and_respect_paragraphs() -> None:
    paras = [f"Paragraph {i}. " + "lorem ipsum dolor " * 20 for i in range(80)]
    text = "\n\n".join(paras)
    spans = chunk_text(text, 6000, 500)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (_s1, e1), (s2, e2) in zip(spans, spans[1:], strict=False):
        assert s2 < e1 <= e2  # overlapping and progressing
        assert e1 - s2 <= 500
        assert text[s2 - 2 : s2] == "\n\n"  # next chunk starts at a paragraph
        assert text[e1 - 2 : e1] == "\n\n"  # chunk ends at a paragraph
    for s, e in spans:
        assert e - s <= 6000


def test_chunk_text_without_breaks_still_progresses() -> None:
    text = "x" * 20000
    spans = chunk_text(text, 6000, 500)
    assert spans[0] == (0, 6000)
    assert spans[-1][1] == 20000
    assert all(b[0] > a[0] for a, b in zip(spans, spans[1:], strict=False))


def test_bad_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)


def test_computed_date_text_lowers_confidence(fake_llm) -> None:
    _, responses = fake_llm
    responses.append(
        ExtractionResult(
            events=[
                ExtractedEvent(
                    quote="Two days after the meeting, Smith emailed Jane Doe about the invoice.",
                    date_text="March 5, 2024",  # computed by the model, not in the text
                    description="Smith emailed Jane Doe.",
                    confidence=1.0,
                ),
                ExtractedEvent(
                    quote="Acme Corp. hired John Smith on March 3, 2024.",
                    date_text="march  3, 2024",  # copied (case/whitespace differences are fine)
                    description="Acme Corp. hired John Smith.",
                    confidence=1.0,
                ),
            ]
        )
    )
    by_desc = {e.description: e for e in extract_events(make_parsed())}
    computed, copied = by_desc["Smith emailed Jane Doe."], by_desc["Acme Corp. hired John Smith."]
    assert computed.confidence == pytest.approx(extract.UNCOPIED_DATE_PENALTY)
    assert copied.confidence == 1.0


def test_later_chunks_get_document_head_as_context(
    fake_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = fake_llm
    head = "WITNESS STATEMENT OF JANE ROE\nDate: April 2, 2024\n\n"
    text = head + ("Filler sentence about nothing in particular. " * 60)
    monkeypatch.setattr(extract, "CHUNK_SIZE", 1500)
    monkeypatch.setattr(extract, "CHUNK_OVERLAP", 200)
    extract_events(make_parsed(text))
    assert len(calls) >= 2
    first_user, second_user = calls[0][1], calls[1][1]
    assert "<document_start>" not in first_user  # the first chunk already has the head
    assert "<document_start>" in second_user and "Date: April 2, 2024" in second_user


def test_prompts_keep_contracted_names() -> None:
    from pipeline import prompts

    for name in ("EXTRACTION_SYSTEM", "EXTRACTION_USER", "CONFLICT_SYSTEM", "CONFLICT_USER"):
        assert isinstance(getattr(prompts, name), str)
    user = prompts.extraction_user("f.txt", "chunk text", 1, 1)
    assert "chunk text" in user and "f.txt" in user and "{" not in user


def test_gleaning_adds_only_new_events(fake_llm, monkeypatch: pytest.MonkeyPatch) -> None:
    calls, responses = fake_llm
    monkeypatch.setattr(extract, "GLEAN_PASSES", 1)
    hired = ExtractedEvent(
        quote="Acme Corp. hired John Smith on March 3, 2024.",
        date_text="March 3, 2024",
        description="Acme Corp. hired John Smith.",
    )
    due = ExtractedEvent(
        quote="The answer is due no later than April 1, 2024.",
        date_text="no later than April 1, 2024",
        description="Answer due.",
    )
    responses.append(ExtractionResult(events=[hired]))
    responses.append(ExtractionResult(events=[hired, due]))  # repeats one, adds one
    events = extract_events(make_parsed())
    assert len(calls) == 2
    assert "Acme Corp. hired John Smith." in calls[1][1]  # first-pass events are listed
    assert [e.date_text for e in events] == ["March 3, 2024", "no later than April 1, 2024"]


def test_gleaning_failure_keeps_first_pass(fake_llm, monkeypatch: pytest.MonkeyPatch) -> None:
    _, responses = fake_llm
    monkeypatch.setattr(extract, "GLEAN_PASSES", 1)
    responses.append(
        ExtractionResult(
            events=[
                ExtractedEvent(
                    quote="Acme Corp. hired John Smith on March 3, 2024.",
                    date_text="March 3, 2024",
                    description="Acme Corp. hired John Smith.",
                )
            ]
        )
    )
    responses.append(RuntimeError("model unavailable"))
    assert len(extract_events(make_parsed())) == 1


@pytest.mark.llm
def test_live_extraction_quotes_are_verbatim() -> None:
    events = extract_events(make_parsed())
    assert events
    for e in events:
        assert e.source.quote in TEXT or e.source.char_end - e.source.char_start == len(TEXT)


def test_importance_is_passed_through_and_deadlines_stay_key(fake_llm) -> None:
    _, responses = fake_llm
    responses.append(
        ExtractionResult(
            events=[
                ExtractedEvent(
                    quote="Acme Corp. hired John Smith on March 3, 2024.",
                    date_text="March 3, 2024",
                    description="Acme Corp. hired John Smith.",
                    importance=0.2,
                ),
                ExtractedEvent(
                    quote="The answer is due no later than April 1, 2024.",
                    date_text="no later than April 1, 2024",
                    description="Answer deadline.",
                    deadline_relevant=True,
                    importance=0.1,
                ),
                ExtractedEvent(
                    quote="Two days after the meeting, Smith emailed Jane Doe about the invoice.",
                    date_text="Two days after the meeting",
                    description="Smith emailed Jane Doe about the invoice.",
                ),
            ]
        )
    )
    hired, emailed, deadline = extract_events(make_parsed())
    assert hired.importance == 0.2
    assert emailed.importance == 0.5  # not rated by the model: the contract default
    assert deadline.importance == extract.DEADLINE_MIN_IMPORTANCE  # never hidden as minor


def test_overlap_duplicates_keep_the_higher_importance() -> None:
    parsed = make_parsed()
    quote = "Acme Corp. hired John Smith on March 3, 2024."
    start = TEXT.index(quote)
    item = ExtractedEvent(quote=quote, date_text="March 3, 2024", description="Hired.")
    low = extract._to_event(parsed, item.model_copy(update={"importance": 0.3}), 0, len(TEXT))
    high = extract._to_event(parsed, item.model_copy(update={"importance": 0.8}), start, len(TEXT))
    assert extract._merge_dup(low, high).importance == 0.8


def test_prompt_asks_for_importance_and_the_central_occurrence() -> None:
    system = prompts.EXTRACTION_SYSTEM
    assert '"importance"' in system and "central occurrence" in system
    assert "importance" in ExtractedEvent.model_json_schema()["properties"]
