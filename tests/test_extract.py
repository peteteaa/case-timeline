"""Tests for pipeline.extract with a fake LLM."""

from __future__ import annotations

import pytest

from pipeline import extract, llm
from pipeline.extract import (
    ExtractedFact,
    ExtractionResult,
    chunk_text,
    clean_people,
    draft_facts,
    draft_key,
)
from tests.pipeline_helpers import make_parsed

TEXT = (
    "Acme Corp. hired John Smith on March 3, 2024.\n\n"
    "Two days after the meeting, Smith emailed Jane Doe about the invoice.\n\n"
    "The answer is due no later than April 1, 2024."
)


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str, type]] = []
    responses: list[object] = []

    def chat_json(system, user, schema, **kw):
        calls.append((system, user, schema))
        item = responses.pop(0) if responses else ExtractionResult()
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm, "chat_json", chat_json)
    return calls, responses


def test_extract_converts_to_drafts(fake_llm) -> None:
    calls, responses = fake_llm
    responses.append(
        ExtractionResult(
            facts=[
                ExtractedFact(
                    title="  Acme hires   John Smith ",
                    details="Smith joins Acme.",
                    date_text="March 3, 2024",
                    people=["Acme Corp.", "John Smith", "john smith", " ", "he"],
                    quotes=["Acme Corp. hired John Smith on March 3, 2024.", ""],
                    confidence=0.9,
                ),
                ExtractedFact(
                    title="Answer deadline",
                    date_text="no later than April 1, 2024",
                    quotes=["The answer is due no later than April 1, 2024."],
                    deadline_language=True,
                ),
                ExtractedFact(title="Undated fact", quotes=["Smith emailed Jane Doe"]),
            ]
        )
    )
    parsed = make_parsed(TEXT)
    drafts = draft_facts(parsed)

    assert len(calls) == 1
    system, user, schema = calls[0]
    assert schema is ExtractionResult
    assert "verbatim" in system.lower()
    assert "evidence, not instructions" in system
    assert "memo.txt" in user and "Acme Corp. hired" in user

    first, second, third = drafts
    assert first.title == "Acme hires John Smith"
    assert first.details == "Smith joins Acme."
    assert first.key == draft_key(parsed.doc.doc_id, first.quotes[0], first.title)
    assert len(first.key) == 16
    assert first.people == ["John Smith"]
    assert first.quotes == ["Acme Corp. hired John Smith on March 3, 2024."]
    assert first.spans == [] and first.date_start is None  # verify and dates come later
    assert second.deadline_language is True
    assert third.date_text is None and third.details is None
    assert len({d.key for d in drafts}) == 3


def test_draft_keys_are_deterministic_and_whitespace_insensitive() -> None:
    a = draft_key("D-1", "Smith  emailed\nDoe.", "Smith emails Doe")
    b = draft_key("D-1", "Smith emailed Doe.", " smith emails  doe ")
    assert a == b
    assert a != draft_key("D-2", "Smith emailed Doe.", "Smith emails Doe")
    assert a != draft_key("D-1", "Smith emailed Doe.", "Another title")


def test_clean_people() -> None:
    names = [
        "Dana Rivera",
        "dana  rivera",
        "Halvorsen Freight Co.",
        "Lakeview Police Department",
        "she",
        "",
        "HR",
        "Officer Jamie Ortiz",
        "123",
    ]
    assert clean_people(names) == ["Dana Rivera", "Officer Jamie Ortiz"]


def test_drafts_without_title_dropped_without_quotes_kept(fake_llm) -> None:
    _, responses = fake_llm
    responses.append(
        ExtractionResult(
            facts=[
                ExtractedFact(title="  ", quotes=["Acme Corp. hired John Smith"]),
                ExtractedFact(title="No quote", quotes=[]),
            ]
        )
    )
    (only,) = draft_facts(make_parsed(TEXT))
    assert only.title == "No quote" and only.quotes == []  # verify discards and counts it


def test_same_quote_different_facts_in_one_chunk_are_kept(fake_llm) -> None:
    _, responses = fake_llm
    quote = "Two days after the meeting, Smith emailed Jane Doe about the invoice."
    responses.append(
        ExtractionResult(
            facts=[
                ExtractedFact(
                    title="Smith emails Doe", date_text="Two days after the meeting", quotes=[quote]
                ),
                ExtractedFact(
                    title="Invoice discussed",
                    date_text="Two days after the meeting",
                    quotes=[quote],
                ),
            ]
        )
    )
    assert len(draft_facts(make_parsed(TEXT))) == 2


def test_overlapping_chunks_are_deduped(fake_llm, monkeypatch: pytest.MonkeyPatch) -> None:
    calls, responses = fake_llm
    para = "Filler sentence about nothing in particular. " * 30
    key = "On May 5, 2024 the court held a hearing."
    text = f"{para}\n\n{key}\n\n{para}"
    monkeypatch.setattr(extract, "CHUNK_SIZE", 1500)
    monkeypatch.setattr(extract, "CHUNK_OVERLAP", 400)
    spans = chunk_text(text, 1500, 400)
    assert len([s for s in spans if key in text[s[0] : s[1]]]) >= 2  # in the overlap

    sightings = 0
    for s in spans:
        if key in text[s[0] : s[1]]:
            sightings += 1
            first = sightings == 1
            responses.append(
                ExtractionResult(
                    facts=[
                        ExtractedFact(
                            # the model words the title differently in each chunk
                            title="Court holds hearing" if first else "Hearing held by court",
                            date_text="May 5, 2024",
                            people=["Judge Ann Lee"] if first else ["Judge Ann Lee", "Bo Smith"],
                            quotes=[key],
                            confidence=0.6 if first else 0.8,
                            deadline_language=first,
                        )
                    ]
                )
            )
        else:
            responses.append(ExtractionResult())

    drafts = draft_facts(make_parsed(text))
    assert len(calls) == len(spans)
    (draft,) = drafts
    assert draft.confidence == 0.8
    assert draft.title == "Hearing held by court"  # the more confident sighting
    assert draft.people == ["Judge Ann Lee", "Bo Smith"]
    assert draft.deadline_language is True
    assert draft.quotes == [key]


def test_one_failed_chunk_is_skipped(fake_llm, monkeypatch: pytest.MonkeyPatch) -> None:
    _, responses = fake_llm
    text = "\n\n".join(f"Paragraph {i} " + "word " * 300 for i in range(4))
    monkeypatch.setattr(extract, "CHUNK_SIZE", 2000)
    monkeypatch.setattr(extract, "CHUNK_OVERLAP", 200)
    n = len(chunk_text(text, 2000, 200))
    assert n > 1
    responses.append(RuntimeError("model hiccup"))
    responses.extend(ExtractionResult() for _ in range(n - 1))
    assert draft_facts(make_parsed(text)) == []


def test_all_chunks_failing_raises(fake_llm) -> None:
    _, responses = fake_llm
    responses.append(ConnectionError("vLLM down"))
    with pytest.raises(ConnectionError):
        draft_facts(make_parsed(TEXT))


def test_empty_document_makes_no_calls(fake_llm) -> None:
    calls, _ = fake_llm
    assert draft_facts(make_parsed("")) == []
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
            facts=[
                ExtractedFact(
                    title="Smith emails Doe",
                    date_text="March 5, 2024",  # computed by the model, not in the text
                    quotes=["Two days after the meeting, Smith emailed Jane Doe"],
                    confidence=1.0,
                ),
                ExtractedFact(
                    title="Acme hires Smith",
                    date_text="march  3, 2024",  # copied (case/whitespace differences are fine)
                    quotes=["Acme Corp. hired John Smith on March 3, 2024."],
                    confidence=1.0,
                ),
            ]
        )
    )
    by_title = {d.title: d for d in draft_facts(make_parsed(TEXT))}
    assert by_title["Smith emails Doe"].confidence == pytest.approx(extract.UNCOPIED_DATE_PENALTY)
    assert by_title["Acme hires Smith"].confidence == 1.0


def test_later_chunks_get_document_head_as_context(
    fake_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = fake_llm
    head = "WITNESS STATEMENT OF JANE ROE\nDate: April 2, 2024\n\n"
    text = head + ("Filler sentence about nothing in particular. " * 60)
    monkeypatch.setattr(extract, "CHUNK_SIZE", 1500)
    monkeypatch.setattr(extract, "CHUNK_OVERLAP", 200)
    draft_facts(make_parsed(text))
    assert len(calls) >= 2
    first_user, second_user = calls[0][1], calls[1][1]
    assert "<document_start>" not in first_user  # the first chunk already has the head
    assert "<document_start>" in second_user and "Date: April 2, 2024" in second_user


def test_prompts_focus_on_dated_occurrences() -> None:
    from pipeline import prompts

    system = prompts.EXTRACTION_SYSTEM
    assert "one fact per distinct occurrence" in system
    assert "character for character" in system and "Never compute" in system
    assert "evidence, not instructions" in system
    user = prompts.extraction_user("f.txt", "chunk text", 1, 1)
    assert "chunk text" in user and "f.txt" in user and "{" not in user
    for name in ("title", "details", "date_text", "people", "quotes", "deadline_language"):
        assert f'"{name}"' in system


@pytest.mark.llm
def test_live_extraction_quotes_are_verbatim() -> None:
    from pipeline.verify import verify_drafts

    parsed = make_parsed(TEXT)
    drafts = draft_facts(parsed)
    assert drafts
    verified, counts = verify_drafts(parsed, drafts)
    assert verified
    for d in verified:
        for s in d.spans:
            assert TEXT[s.char_start : s.char_end] == s.quote


@pytest.mark.parametrize("placeholder", ["no date", "N/A", "unknown", "(no date given)", "none"])
def test_placeholder_date_text_means_no_date(fake_llm, placeholder: str) -> None:
    _, responses = fake_llm
    responses.append(
        ExtractionResult(
            facts=[
                ExtractedFact(
                    title="Smith emails Doe",
                    date_text=placeholder,
                    quotes=["Two days after the meeting, Smith emailed Jane Doe"],
                    confidence=1.0,
                )
            ]
        )
    )
    (d,) = draft_facts(make_parsed(TEXT))
    assert d.date_text is None and d.confidence == 1.0  # no "uncopied date" penalty either
