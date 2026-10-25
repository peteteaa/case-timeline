# Module interfaces (Spec 2)

The data contract is `pipeline/schema.py` (SPEC §8.1), frozen on day 0. Every module imports its
types from there. Only `pipeline/llm.py` talks to the model. The app server and the MCP server
call `pipeline.*` and never the model. Everything the pipeline writes is stamped
`by="agent"` (`schema.AGENT`), and the store enforces what the agent may do (SPEC §6.2).

Status as of this writing: every stage module below is implemented except
`pipeline/digest.py`, which is still a stub with its signature fixed.

## `pipeline/schema.py`: contracts

- **SPEC §8.1 models:** `Document`, `EvidenceLink`, `Fact`, `Witness`, `Tag`, `Note`,
  `ReviewEvent`, `Contradiction`, `Job`, `Alert`, plus the literals `Status`, `Review`, `Origin`,
  `DocKind`, `Certainty`, `AlertType`, and so on, with tuples `STATUSES`, `REVIEWS`,
  `DOC_KINDS`, `CERTAINTIES`, `ALERT_TYPES`.
- **Pipeline-internal types** (never cross the MCP boundary):
  - `Page(number, char_start, char_end)` and `Line(page, number, char_start, char_end)` are
    transcript page:line offsets.
  - `ParsedDocument(doc, text, pages, lines)`.
  - `SpanIn(doc_id, page?, char_start, char_end, line_start?, line_end?, quote)` is a located,
    verbatim quote.
  - `DraftFact(key, title, details?, date_text?, date_start?, date_end?, date_certainty,
    anchor_key?, anchor_fact_id?, people, quotes, spans, confidence, deadline_language)`.
  - `FactQuery(date_from?, date_to?, include_undated, statuses, tag_ids, witness_ids, doc_ids,
    doc_kinds, reviews=["draft","accepted"], origins, fact_ids?, sort, descending, limit?)`
    holds the filters and sort shared by list views, exports, and MCP reads.

## `pipeline/config.py`

- `Settings` reads env prefix `CT_` and `.env`. Fields: `evidence_dir`, `data_dir`, `db_path`,
  `case_id`, `llm_base_url` (`http://172.17.0.1:8000/v1`), `llm_allowed_hosts`, `llm_model`,
  `llm_timeout`, `embed_model`, `embed_dim`, `ui_host`, `ui_port`, `max_drafts_per_doc`,
  `worker_poll_seconds`, `run_worker`.
- Properties: `inbox` (`data_dir/inbox`, created on access) and `ui_base_url`. Methods:
  `check_layout()` refuses a db, data dir, or inbox inside the evidence folder;
  `ensure_dirs()`.
- `get_settings() -> Settings` is cached. `valid_case_id(case_id) -> str` raises `ValueError` for
  anything path-like.

## `pipeline/store.py`: SQLite (WAL) + FTS5 + sqlite-vec

`Store(db_path, embedder=None)` is thread-safe, with one connection behind a lock and short
`BEGIN IMMEDIATE` writes. It works as a context manager. Writes take `by=` (a display name or
`"agent"`).

- **Documents:**
  - `register_document(case_id, *, filename, rel_path, sha256, location, kind="other") -> (Document, created)`
    deduplicates by content hash.
  - `get_document`, `get_document_by_sha`, `list_documents`, `document_source`,
    `document_info`, and `documents_to_process(case_id)`.
  - `set_parse_result(doc_id, status, *, pages?, error?, kind?)` and `set_document_kind(doc_id, kind, by)`.
  - `save_parsed_text(doc_id, text, pages, lines)`, `get_parsed(doc_id) -> ParsedDocument | None`,
    `mark_drafted`, `reset_document(doc_id) -> int`, and `documents_mentioning(case_id, names)`.
- **Facts (reads):**
  - `get_fact(fact_id, *, rejected_evidence=False)` returns a fact in any review state.
  - `get_facts(ids)` returns facts in any review state, in the order given.
  - `list_facts(case_id, FactQuery | None)` excludes rejected facts by default.
  - Also `fact_counts`, `alleged_fact_ids`, `conflicted_fact_ids`, `facts_citing_document`,
    `get_evidence`, and `fact_history`.
- **Facts (writes):**
  - `add_fact(case_id, title, *, by, details?, date_text?, date_start?, date_end?, date_certainty, anchor_fact_id?, statuses, tag_ids, witness_ids, confidence?, spans, doc_ids, references) -> Fact`.
    Agent facts start as drafts; a person's start accepted.
  - `update_fact(fact_id, changes, *, by)`. A person editing a draft accepts it.
  - `review_fact(fact_id, "accept"|"edit"|"reject", *, reviewer, reason?, changes?)`.
  - `add_evidence(fact_id, *, kind, by, span?|doc_id?|reference?)`. Span quotes are checked
    verbatim.
  - `review_evidence(evidence_id, action, *, reviewer, reason?)` and `list_review_events(case_id)`.
- **Notes:** `add_note(fact_id, *, author, body)` and `list_notes(fact_id)`.
- **Witnesses:**
  - `add_witness(case_id, name, *, by, aliases, role?, notes?, flagged)`. Agent witnesses are
    unconfirmed suggestions.
  - `get_witness`, `list_witnesses(case_id, *, confirmed?)`, and
    `find_witness(case_id, name)`, which matches the exact name or an alias.
  - `update_witness`, `merge_witnesses(source_id, into_id, *, by)`, `delete_witness`, and
    `link_witnesses`.
  - `witness_facts(witness_id)` excludes rejected facts. Also `witness_documents(witness_id)`.
- **Tags:** `add_tag(case_id, name, *, by, color?)`, `list_tags`, `get_tag`, `update_tag`, and
  `delete_tag`.
- **Contradictions:**
  - `add_contradiction(case_id, fact_ids, field, explanation) -> (Contradiction, created)`.
  - `get_contradiction`, `contradiction_case`, and
    `list_contradictions(case_id, status?, *, fact_id?)`.
  - `review_contradiction(id, "confirm"|"dismiss"|"reopen", *, reviewer)`. Confirming gives the
    linked facts the disputed status.
- **Alerts:**
  - `add_alert(case_id, type, *, fact_ids, doc_ids, job_id?)`.
  - `get_alert`, and `list_alerts(case_id, *, pending_only=False)`, newest first.
  - `mark_alerts_delivered(alert_ids) -> int`, which is idempotent.
- **Jobs:**
  - `request_job(case_id, trigger) -> (Job, created)` reuses an already-queued job.
  - `claim_next_job()`, `update_job_counts`, `finish_job(job_id, status, *, error?, counts?)`,
    and `requeue_running()`.
  - `get_job`, `latest_job`, and `list_jobs(case_id, limit=20)`.
- **Search indexes:** `fts_fact_ids(case_id, match, limit)`, `refresh_vectors(case_id)`,
  `vector_fact_ids(case_id, text, k)`, and `vector_enabled`.
- **Activity:** `activity_since(case_id, since) -> dict` and `review_counts(case_id)`.

## `pipeline/jobs.py`: queue and worker (SPEC §5.1)

- `request_scan(store, case_id, trigger) -> Job` only queues a job, and never blocks.
- `run_job(store, job, settings=None) -> Job` never raises. It runs discover → parse → extract
  → verify → cap → dates → witnesses → merge → store, then refreshes vectors, runs conflicts,
  and computes alerts. Counts go in `COUNT_KEYS`: `docs_seen`, `docs_new`, `docs_parsed`,
  `docs_failed`, `drafts`, `quotes_checked`, `quotes_rejected`, `drafts_discarded`,
  `facts_created`, `evidence_attached`, `contradictions`, `alerts`.
- `Worker(settings=None, *, store=None, poll_seconds=None)` has `start()` (which re-queues jobs
  left running), `wake()`, `stop()`, and `run_pending() -> int`.
- `run_now(case_id=None, *, trigger="ui", settings=None) -> Job` queues a job and runs it
  inline. This is the path for `make scan` and `python -m pipeline.jobs scan --case ID`.

## Stages

- **`pipeline/parse.py`:**
  - `discover(evidence_dir, inbox_dir) -> list[FoundFile]` hashes files recursively and handles
    `.pdf`, `.docx`, `.txt`, `.md`, and `.eml`.
  - `parse_file(path, doc) -> ParsedDocument` uses Docling and PyMuPDF, plus the text and email
    readers, and records page and transcript line offsets.
  - Also `detect_kind(filename, text) -> DocKind`, `doc_date(parsed) -> date | None`,
    `looks_like_transcript`, `transcript_lines`, `normalize_text`, and `hash_file`.
- **`pipeline/extract.py`:**
  - `draft_facts(parsed) -> list[DraftFact]` chunks the text and calls `llm.chat_json`.
  - Also `chunk_text`, `draft_key`, and `clean_people`. Prompts live in `pipeline/prompts.py`.
- **`pipeline/verify.py`** (no LLM):
  - `verify_drafts(parsed, drafts) -> (drafts_with_spans, counts)`, where `counts` has
    `quotes_checked`, `quotes_rejected`, and `drafts_discarded`.
  - Also `locate_quote(text, quote) -> (start, end) | None`, `make_span`, `page_for_offset`, and
    `lines_for_span`.
- **`pipeline/dates.py`:**
  - `resolve_drafts(drafts, doc_date, existing_facts=None) -> list[DraftFact]` uses dateparser
    and dateutil. A month or year alone becomes an approximate range. Relative dates are
    resolved against anchors.
  - Also `parse_absolute` and `parse_relative`.
- **`pipeline/witnesses.py`:**
  - `match_people(store, case_id, names) -> list[witness_id]` creates suggested witnesses as
    needed.
  - Also `find(witnesses, name)`, `tokens`, and `display_name`.
- **`pipeline/merge.py`:**
  - `plan(drafts, existing, *, witness_ids=None) -> {draft_key: fact_id | None}` returns
    `fact_id` when a draft should attach its evidence to an existing fact.
  - Also `similarity`, `dates_compatible`, `people_compatible`, `gap_days`, and `interval`.
- **`pipeline/conflicts.py`:**
  - `detect(store, case_id, fact_ids) -> list[Contradiction]` returns the newly opened
    contradictions. It runs the cheap rules in `classify` and `candidate_pairs`, then LLM
    `confirm`.
- **`pipeline/alerts.py`:**
  - `compute(store, case_id, *, job_id, new_fact_ids, contradictions, doc_ids, today=None) -> list[Alert]`
    applies the SPEC §8.4 rules with no LLM. Also `has_deadline_language(text)`.
- **`pipeline/search.py`:**
  - `search_facts(store, case_id, query, k=20, *, semantic=True) -> list[fact_id]` fuses FTS5
    and vector results.
  - Also `to_fts5(query)` (Westlaw subset → FTS5; raises `QueryError`), `plain_fts5`,
    `keyword_fact_ids`, and `SYNTAX_HELP`.
- **`pipeline/digest.py` (stub):**
  - `daily_digest(store, case_id, *, now=None) -> {"case_id", "since", "text", "counts", "fact_ids"}`
    covers the last 24 hours. It backs both MCP `daily_digest` and the UI's "Today" panel.
- **`pipeline/export.py`:**
  - `to_csv` and `to_xlsx(store, case_id, facts, columns=DEFAULT_COLUMNS) -> bytes`.
  - `to_svg(store, case_id, facts, zoom="month") -> str` and `to_png(...) -> bytes` (via
    cairosvg). Also `format_date(fact)`.
- **`pipeline/llm.py`:**
  - `chat_json(system, user, schema, *, temperature=0.0, max_tokens=None)` and
    `chat(messages, **kw)`.
  - `embed(texts)` uses local BGE-M3 and falls back to `hashed_embedding`.
  - Also `embed_backend()`, `health()`, and `LLMError`. It refuses non-local endpoints.

## `app/`: app server (web UI + job worker)

`create_app(settings=None, *, store=None) -> FastAPI` starts one `jobs.Worker` when
`settings.run_worker` is set. `python -m app` (`make app`) serves on `ui_host:ui_port` and
refuses non-loopback hosts. Its routes follow SPEC §5.2: `/cases/{id}` (dashboard),
`/cases/{id}/facts`, `/cases/{id}/review`, `/facts/{id}` (card, `PATCH`, `/review`,
`/evidence`, `/notes`), `/documents/{id}/view?evidence=`, `/cases/{id}/documents` (upload to
the inbox), `/cases/{id}/process`, `/jobs/{id}`, `/cases/{id}/witnesses`, `/witnesses/{id}`,
`/cases/{id}/tags`, `/cases/{id}/contradictions`, `/contradictions/{id}`, `/cases/{id}/alerts`,
`/cases/{id}/export?format=xlsx|csv|svg|png`, and `/cases/{id}/chat`.

## `mcp_server/server.py`: agent tools (SPEC §8.2), stdio

`build_server() -> MCPServer` (official `mcp` SDK, `mcp.server.mcpserver`) creates the server
lazily. `main()` logs to stderr and `data/mcp.log`, and keeps stray prints off the wire. The
backend is `Backend(store, case_id, dashboard)`. `set_backend()` is a test hook, and
`CT_MCP_BACKEND=module:factory` overrides the backend. `case_id` is optional everywhere and is
checked by `valid_case_id`. A bad argument comes back as a tool error with its message.

| Tool | Returns |
| --- | --- |
| `scan_case(case_id?)` | `{job_id, case_id, status, queued_at}`. Calls `jobs.request_scan(..., "agent")` only |
| `job_status(job_id)` | The `Job` fields plus `found`, and a `hint` if the job is stuck in the queue |
| `get_timeline(case_id?, start?, end?, witness?, tag?, status?, limit=200)` | `{facts, total, truncated, filters}` |
| `search_facts(query, case_id?, k=20)` | `{facts}`, or `{error, syntax_help}` |
| `get_fact(fact_id)` | A fact view plus, for each quote, `before`, `after`, and `highlighted`; also `notes`, `possible_conflicts`, and `anchor` |
| `get_witness(name, case_id?)` | `{witness, facts, documents}`, matched by id, name or alias, then fuzzy |
| `list_contradictions(case_id?, open_only=true)` | Each contradiction with `label` (possible conflict, disputed, or dismissed) and both facts with their quotes |
| `get_pending_alerts(case_id?)` | `{alerts: [{alert_id, type, label, needs_attorney_review, dashboard_url, message, internal: {do_not_relay, fact_ids, doc_ids}}], relay_policy}` |
| `mark_alerts_delivered(alert_ids)` | `{requested, marked}`. Idempotent |
| `daily_digest(case_id?)` | `digest.daily_digest` output plus `dashboard_url`, a content-free `message`, and `relay_policy` |

A fact view has these fields: `fact_id`, `title`, `details`, `review`, `origin`,
`confidence`, `date_text`, `date_start`, `date_end`, `date_certainty`, `anchor_fact_id`,
`statuses`, `disputed`, `tags` (names), `witnesses` (names), `alleged`, `possible_conflict`,
and `evidence`. Each evidence item has `evidence_id`, `kind`, `review`, `origin`, `cite`,
`doc_id`, `filename`, `doc_kind`, `page`, `line_start`, `line_end`, `quote`, and
`source_link`. `cite` looks like `memo.pdf p. 3` or `depo.txt 14:8-9`. No tool returns a
rejected fact, including inside contradictions, witness profiles, alerts, and the digest.

## `eval/`

`python -m eval --case ID [--gold] [--disputes] [--db] [--out]` (`make eval`) runs
`run_eval.evaluate(store, case_id, gold, disputes) -> dict`. The metrics are described in the
module docstring and SPEC §13. Gold lives in `eval/gold/<case>_timeline.json` and
`<case>_disputes.json` for the demo, and in `eval/gold_timeline.json` and
`eval/gold_disputes.json` for the public set. `scripts/fetch_eval.py` (`make fetch-eval`)
downloads `eval/docs/manifest.json` into `eval/docs/<group>/`, verifying each sha256.
