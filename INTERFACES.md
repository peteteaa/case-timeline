# Module interfaces (day 0)

All data types live in `pipeline/contracts.py`, and they are frozen. Only `pipeline/llm.py` talks to the model. The UI and the MCP server import `pipeline.*` and never call the model or SQL directly.

## Configuration: `pipeline/config.py`
`Settings` (pydantic-settings, env prefix `CT_`):
- `evidence_dir`: default `./tests/fixtures/case_demo`. Read-only; never written to.
- `data_dir`: default `./data`. Holds the SQLite database, caches, and logs.
- `db_path`: default `data_dir/case.db`.
- `case_id`: default `demo`.
- `llm_base_url`: `http://172.17.0.1:8000/v1` (docker0 bridge, so the NemoClaw sandbox can reach it). `llm_allowed_hosts` lists hosts trusted to be this machine; `llm.py` refuses anything else.
- `llm_model`: `qwen3.6-35b-a3b`.
- `embed_model`: `BAAI/bge-m3`.
- `ui_host`: `127.0.0.1`. `ui_port`: `8080`.

Get the settings with `get_settings()`.

## `pipeline/llm.py`: the only model client
- `chat_json(system: str, user: str, schema: type[BaseModel], *, temperature=0.0) -> BaseModel` uses vLLM structured output (`response_format` json_schema / `extra_body={"structured_outputs": {"json": ...}}`). Thinking is disabled through `chat_template_kwargs={"enable_thinking": False}`.
- `chat(messages: list[dict], **kw) -> str`
- `embed(texts: list[str]) -> list[list[float]]` runs a local sentence-transformers BGE-M3 model, loaded lazily, on CPU or GPU.
- `health() -> dict`

## `pipeline/parse.py`
- `discover(evidence_dir: Path, case_id: str) -> list[Document]` hashes the files.
- `parse(doc: Document) -> ParsedDocument` uses Docling and falls back to PyMuPDF for PDFs and plain reads for .txt/.md/.eml. It handles PDF, DOCX, TXT, MD, and EML. It also records page offsets.

## `pipeline/extract.py`
- `extract_events(parsed: ParsedDocument) -> list[Event]` splits the text into overlapping chunks and calls `llm.chat_json` with a Pydantic schema. The model returns the quote, date_text, actors, and description. Offsets are computed later, not by the LLM.

## `pipeline/verify.py`
- `locate_quote(text: str, quote: str) -> tuple[int, int] | None` finds the quote exactly first, then with whitespace/quote-normalized matching, then by fuzzy matching (rapidfuzz, ≥ 90).
- `verify_events(parsed, events) -> list[Event]` sets `SourceRef` offsets and the page. Events whose quote can't be located get `status=UNVERIFIED` and confidence × 0.5.

## `pipeline/dates.py`
- `resolve_dates(events: list[Event], doc_date: date | None) -> list[Event]` uses dateparser and dateutil to fill `date_start`/`date_end`/`certainty`. Relative dates resolve against `anchor_event_id` or the document date. It never invents precision.

## `pipeline/merge.py`
- `merge_events(new: list[Event], existing: list[Event]) -> tuple[list[Event], list[Event]]` returns `(to_insert, duplicates)`. Two events are duplicates when they share a date and have similar descriptions and actors (embedding cosine ≥ 0.9 or rapidfuzz). Duplicates keep both sources through the dedup link stored by `store`.

## `pipeline/conflicts.py`
- `find_contradictions(events: list[Event], case_id) -> list[Contradiction]` first applies cheap candidate rules (same actors/topic, different dates; same event, different actor), then asks the LLM to confirm each candidate pair.
- `alerts_for(contradictions, events, case_id, base_url) -> list[Alert]`. Deadline-relevant events and flagged witnesses also produce alerts.

## `pipeline/store.py`: SQLite + FTS5 + sqlite-vec
- `Store(db_path)` provides `init()`, `upsert_document`, `add_parsed_text(doc_id, text, pages)`, `get_parsed_text(doc_id)`, `add_events`, `list_events(case_id, status=None)`, `get_event`, `add_contradictions`, `list_contradictions(case_id, status=None)`, `add_alerts`, `list_alerts`, `create_job`/`update_job`/`get_job`, `changes_since(case_id, since) -> Changes`, `search_events(case_id, query, k) -> list[Event]` (hybrid FTS5 + vector), `processed_doc_ids(case_id) -> set[str]`, `set_event_status(id, status)`, and `link_duplicate(event_id, dup_of)`.
- It also tracks a `last_seen` cursor per consumer for `get_changes`.

## `pipeline/runner.py`: orchestration
- `scan_case(case_id) -> Job` processes only new or changed docs: parse → extract → verify → dates → merge → conflicts → store. It runs in a background thread, is idempotent, and holds a file lock so only one scan runs at a time.
- `daily_digest(case_id) -> dict` returns counts only, and contains no content when used for messaging.

## `mcp_server/server.py`: FastMCP, stdio, never prints to stdout
The tools are `scan_case`, `job_status`, `get_changes`, `get_timeline`, `search_events`, `get_source`, `list_contradictions`, and `daily_digest`. They are thin wrappers over `pipeline.runner` and `pipeline.store`.

## `ui/app.py`: FastAPI + HTMX + vis-timeline (vendored)
Routes are `/`, `/upload`, `/timeline`, `/source/{doc_id}?start=&end=` (highlights the quote), `/contradictions`, `/alerts`, `/chat`, and `POST /process`. The app binds to 127.0.0.1:8080.

`/upload` copies files into a separate `data_dir/inbox` that `discover` also scans, because the evidence folder is read-only.

## `eval/`
`make eval` compares the stored events against `eval/gold/<case>.json` (a list of `{date, actors, description}`). It reports event precision/recall (a description fuzzy match plus a date match) and date accuracy.
