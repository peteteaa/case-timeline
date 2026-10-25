# Case Timeline Agent — Spec 2

**Status:** Draft · **Replaces:** Initial Setup Spec v0.2 · **Last updated:** October 25, 2026

## What changed from v0.2

- **Facts replace events.** The core unit now follows DISCO Timelines: a fact needs only a title, and dates, evidence, statuses, issue tags, witnesses, and notes are optional and editable (§8.1).
- **AI output is a draft.** Extracted facts wait in a review queue until a person accepts, edits, or rejects them in the source viewer (§6).
- **A real workspace.** List and visual timelines, sorting and filters, Westlaw-style search, witness profiles, and export (§7).
- **People confirm disputes.** A detected contradiction becomes DISCO's "disputed" status only after a reviewer confirms it (§6.4).
- **Five fixes to v0.2.** Alerts are computed in the core library; uploads go to a writable inbox instead of the read-only evidence folder; demo mode has no messaging channel by default; jobs are stored in SQLite and run in the app server; alert delivery is tracked in the database, so the agent needs no memory between runs (§5.1, §8).
- **Two new milestones.** M5 delivers the DISCO-equivalent workspace and M6 our differentiators (§14).

## 1. Purpose

This spec does two jobs. First, as in v0.2, it gets the GB10, the NVIDIA stack, the repo, and the team's workflow to the point where one document flows end to end and the always-on loop works (M0–M4). Second, it defines the product model (facts, review, and views) precisely enough to freeze the schema on day 0, because every role builds against it. Extraction quality and UI polish still come after setup.

## 2. Product in one paragraph

A private, always-on case chronology workspace modeled on DISCO Timelines. Facts, not documents, are the unit: each fact has a title and can carry a date or date range, linked evidence, statuses, issue tags, witnesses, and notes. The system watches a case folder, drafts facts from every new document, and ties each one to the exact source sentence. Lawyers review the drafts, organize the timeline as a list or a visual timeline, and see where sources contradict each other. Everything runs on one GB10, and no case data leaves the machine.

## 3. Positioning against DISCO Timelines

DISCO Timelines, part of DISCO Case Builder, is our reference product. We copy its fact model and review workflow, adapt its AI drafting to run continuously, and differ deliberately where its design conflicts with our constraints.

| DISCO capability | Our version | Verdict |
| --- | --- | --- |
| Facts are the unit; only a title is required | Same (§8.1) | Adopt |
| Optional date or date range; undated facts allowed | Same, plus date certainty and relative dates resolved against an anchor fact | Adopt |
| Statuses: key fact, case impact, disputed | Same three statuses, set only by people | Adopt |
| Issue tags, witnesses with profiles, and notes on each fact | Same; witness profiles list the witness's facts and documents | Adopt |
| Evidence linked from anywhere, several items per fact | Quoted spans of case documents, whole documents, or external references | Adopt |
| AI drafts a timeline from an uploaded document | Every new document in the case folder is drafted automatically | Adapt |
| Reviewers check AI work against the exact source sentences | Same, and every quote must also pass a verbatim check in code before anyone sees it | Adopt, stronger |
| Feedback on AI output from the document viewer | Accept, edit, or reject, with an optional reason kept for evaluation | Adopt |
| AI adds facts from more documents to an existing timeline | The default behavior; facts about the same event gain evidence instead of being duplicated | Adopt, extended |
| List view and visual view, with key facts highlighted | Same; undated facts get their own group and lane | Adopt |
| Sorting by date, title, counts, and created or updated time | Same | Adopt |
| Westlaw-style search plus filters by evidence, tag, or witness | A Westlaw-style subset translated to SQLite FTS5, plus the same filters and a few more (§8.5) | Adopt |
| Spreadsheet and image export, scoped to the current search or filter | XLSX, CSV, SVG, and PNG, with the same scoping | Adopt |
| Creating facts while reading a document | "Add fact" from a text selection in our source viewer | Adapt |
| Cloud collaboration from any device | One shared copy on the GB10, reached through an SSH tunnel | Differ: local only |
| (none) | Always-on monitoring, alerts, cross-document contradiction detection, fully offline operation | Ours only |

Two rules apply to this alignment. We match workflows, not branding: no DISCO or Cecilia names, screenshots, or assets appear in our product or demo. And DISCO features missing from this table are out of scope for the hackathon.

## 4. Constraints

| Constraint | Detail |
| --- | --- |
| Hardware | One NVIDIA GB10 (DGX Spark class), Linux on Arm64, 128 GB unified memory |
| Required stack | Exactly one of NemoClaw / OpenClaw / OpenShell. Assumption: NemoClaw, pending organizer confirmation (fallback in §15) |
| Always-on | Must run continuously and react to new documents without manual triggers |
| Local only | All inference, parsing, and storage on the box. No case content leaves it, alerts included |
| Human review | The AI drafts facts and proposes evidence, witnesses, and disputes. Only people accept or edit facts, set statuses and tags, or confirm disputes. Original documents are never edited or deleted |
| Network exposure | Nothing listens on the network except SSH. The app server and model server bind to 127.0.0.1 only, and the MCP server uses stdio, so it has no port. People reach the UI through an SSH tunnel |
| Agent permissions | The agent's shell access is turned off; it can only call our MCP tools (OpenClaw tool policy, verify). None of those tools can change facts |

## 5. Architecture

```
GB10 host
├── vLLM server (NemoClaw-managed) ◄── inference.local ──┐
├── Evidence folder (read-only) ───────── read ──────────┤
└── OpenShell sandbox                                    │
    ├── OpenClaw agent + case-timeline skill + cron jobs │
    │     └── MCP tool calls ──► MCP server (stdio subprocess)
    │                                └── read ──► SQLite (WAL)
    ├── App server: web UI + job worker, 127.0.0.1 only ◄── SSH tunnel ── Laptop browser
    │     ├── Core library: parse · extract · verify · dates · witnesses · merge
    │     │                 · conflicts · alerts · search · export ──► vLLM (chat)
    │     ├── SQLite (WAL): facts · evidence · witnesses · tags · notes · docs · jobs · alerts
    │     └── Inbox (UI uploads, writable) ◄── save uploads
```

| Component | Runs where | Responsibility |
| --- | --- | --- |
| vLLM server | Host, managed by NemoClaw | Serves the local LLM (and optionally embeddings) |
| OpenClaw agent | Sandbox | Always-on triggers (cron), alert delivery, chat Q&A |
| Skill (SKILL.md) | Sandbox | Tells the agent when and how to call our tools and how to answer |
| MCP server | Sandbox, stdio subprocess of OpenClaw | Thin wrapper exposing a few coarse tools; queues jobs but never runs them |
| App server | Sandbox, bound to 127.0.0.1 | The web UI plus the job worker that runs the pipeline |
| Core library (`pipeline/`) | Sandbox | All real logic, from parsing to export (§10). The only code that talks to the model |
| SQLite (WAL mode) | Sandbox | Facts, evidence, witnesses, tags, notes, documents, jobs, alerts, contradictions, full-text and vector search |
| Inbox | Sandbox data directory, writable | Files uploaded through the UI |
| Evidence folder | Host, mounted read-only | The case documents |

### 5.1 Processes and jobs

Two long-lived processes run in the sandbox: OpenClaw, with our MCP server as its subprocess, and the app server. All processing happens in jobs:

- Anything can request a job: OpenClaw cron through `scan_case`, the UI's "Process now" button, or an upload. Requesting a job only adds a row to the jobs table.
- The app server's worker runs jobs, one at a time per case. Job state, progress, and errors live in SQLite, so they survive a restart of either process. On startup, the worker re-queues any job left marked as running.
- A scan compares content hashes across the evidence folder and the inbox, so renamed copies and duplicates aren't processed twice.
- Each job ends by computing alerts with the rules in §8.4 and writing them to the database. The agent only delivers them.
- SQLite runs in WAL mode with short transactions, so the UI and the MCP server can read while a job writes.

### 5.2 How people and the agent reach the system

There are two entry points, both thin layers over the core library:

| Entry point | Used by | Protocol | Reachable from |
| --- | --- | --- | --- |
| Web UI (app server) | Lawyers, demo audience | HTTP | 127.0.0.1 on the GB10; laptops through an SSH tunnel |
| MCP server | OpenClaw agent only | MCP over stdio | Nothing; it runs as a subprocess of OpenClaw |

| Action | What happens |
| --- | --- |
| Add a file | Copy it into the evidence folder, or upload it in the UI, which saves it to the inbox. Either way, the next scan picks it up |
| Process new files | OpenClaw cron calls `scan_case` every few minutes; "Process now" in the UI queues the same job |
| Review drafts | The "Needs review" view lists AI drafts by confidence; each one opens in the source viewer |
| Organize the timeline | People edit facts, statuses, tags, witnesses, and notes in the UI; the agent isn't involved |
| Ask a question | The UI's chat box forwards the message to OpenClaw, which answers with MCP tools and cites fact IDs (verify how OpenClaw accepts messages from a local app, e.g. its gateway) |
| Alerts | Written by the core library after each job and shown in the UI; the agent can also announce them (§8.3) |

All access goes through SSH, so everyone who reaches the UI is a trusted teammate. The UI asks for a display name once and keeps it in a cookie, for authorship on facts, notes, and reviews. There are no accounts or permissions.

Web UI routes (internal to the box, not a public API):

| Route | Purpose |
| --- | --- |
| `POST /cases/{id}/documents` | Upload a file (saved to the inbox; queues a job) |
| `POST /cases/{id}/process` | "Process now" (queues a job) |
| `GET /jobs/{id}` | Job progress |
| `GET /cases/{id}/facts` | Data for the list and visual views; filters, sort, and search as query parameters |
| `POST /cases/{id}/facts` | Create a fact |
| `PATCH /facts/{id}` | Edit title, details, dates, statuses, tags, or witnesses |
| `POST /facts/{id}/review` | Accept, edit, or reject a draft, with an optional reason |
| `POST /facts/{id}/evidence` | Link evidence: a span from the viewer, a whole document, or an external reference |
| `POST /evidence/{id}/review` | Accept or reject evidence the AI attached |
| `POST /facts/{id}/notes` | Add a note |
| `GET /documents/{id}/view` | Source viewer; `?evidence=` highlights a quote |
| `POST /documents/{id}/reprocess` | Draft facts from this document again |
| `GET, POST /cases/{id}/witnesses` · `GET, PATCH /witnesses/{id}` | Witness list and profiles |
| `GET, POST /cases/{id}/tags` | Issue tags |
| `GET /cases/{id}/contradictions` · `POST /contradictions/{id}` | Review conflicts: confirm or dismiss |
| `GET /cases/{id}/alerts` | Alerts |
| `GET /cases/{id}/export` | `format` is xlsx, csv, svg, or png; takes the same filters as the views, plus `zoom` for images |
| `POST /cases/{id}/chat` | Forward a question to the agent |

Reaching the UI from a laptop:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@gb10-host
# then open http://localhost:8080 on the laptop
```

Two rules: all real logic lives in the core library, and neither the MCP server nor the model server is ever put on the network.

## 6. Review workflow

### 6.1 Life of an AI-drafted fact

1. **Draft.** The extractor proposes a fact with a title, details, date fields, the people mentioned, and one or more quotes.
2. **Verify.** `verify.py` checks every quote against the parsed page text, without the LLM. Drafts whose quotes don't match are discarded and counted in the job report.
3. **Match witnesses.** `witnesses.py` matches the people mentioned to witness records by name and alias. Unmatched names become suggested witnesses for a person to confirm, merge, or dismiss.
4. **Merge.** `merge.py` compares the draft with existing facts. If it describes the same event, its quote is attached to the existing fact as new evidence instead of creating a duplicate. If the sources disagree on the date or the people involved, `conflicts.py` opens a contradiction.
5. **Review.** The draft appears in "Needs review." A reviewer opens it in the source viewer, which highlights the quoted sentence, and accepts it, edits it, or rejects it with an optional reason. Editing a draft accepts it.
6. **Keep the trail.** Rejected facts are hidden from every view but kept, with their reasons, for evaluation.

### 6.2 What the AI may and may not change

| The AI may | The AI may not |
| --- | --- |
| Create draft facts | Accept, edit, or delete a fact after a person has reviewed it |
| Attach new evidence to any fact (marked as new until a person accepts it) | Set or remove statuses (key fact, case impact, disputed) |
| Suggest witnesses | Apply or remove tags |
| Open contradictions | Edit notes or touch original documents |

### 6.3 Facts people create

- **From the source viewer:** select text and click "Add fact." The selection becomes a verified evidence span, and the person writes the title.
- **From the timeline:** "New fact" needs only a title. Evidence can be added later as a document span, a whole document, or an external reference (a URL or citation that's stored but never fetched).

Facts people create start as accepted.

### 6.4 Disputes

In DISCO, "disputed" is a person's judgment. We keep it that way and add detection. `conflicts.py` opens a contradiction when sources disagree on a fact's date, the people involved, or what happened, and the affected facts show a conflict badge. A reviewer then confirms the contradiction, which gives the linked facts the disputed status, or dismisses it.

### 6.5 Allegations and evidence

Every document has a kind: pleading, motion, order, exhibit, transcript, email, letter, or other. A filename and first-page heuristic sets it, and anyone can correct it in the UI. A fact whose only evidence comes from pleadings is labeled "alleged" in every view, because a complaint's statements are allegations, not established facts.

## 7. Views and tools

| Feature | Behavior |
| --- | --- |
| List view | A table of facts: date, title, statuses, and counts of evidence, tags, witnesses, and notes. Undated facts are grouped at the end |
| Visual view | vis-timeline: points for dates, bars for ranges, key facts in a highlight color, undated facts in their own lane. Zoom from days to years. Clicking a fact opens its card |
| Fact card | Title, details, dates, statuses, tags, witnesses, notes, and evidence. Every edit records who made it and when |
| Sorting | Date, title, counts of evidence, tags, witnesses, or notes, created time, updated time |
| Filters | Date range, status, tag, witness, document, document kind, review state, and origin (AI or person) |
| Search | Fact titles, details, notes, and evidence quotes (§8.5). Search and filters combine |
| Source viewer | The document page with the cited sentence highlighted; a side panel listing the facts that cite this document; "Add fact" from a selection; accept and reject buttons for drafts |
| Witness profile | Name, aliases, role, notes, and a flag for alerts; the witness's facts in date order; documents that mention them |
| Contradictions | Open conflicts with both sources side by side; confirm or dismiss |
| Alerts | Newest first; each links to the facts and documents involved |
| Export | XLSX or CSV of the current list, with columns in on-screen order; SVG or PNG of the visual view at a chosen zoom. Exports respect the current search and filters |
| Bulk review | Select several drafts and accept or reject them together |

## 8. Contracts (freeze on day 0)

### 8.1 Data model

```python
from datetime import date, datetime
from typing import Literal
from pydantic import BaseModel

Status = Literal["key_fact", "case_impact", "disputed"]
Review = Literal["draft", "accepted", "rejected"]
Origin = Literal["ai", "user"]

class Document(BaseModel):
    doc_id: str                  # "D-0007"
    case_id: str
    filename: str
    sha256: str                  # dedupe: renamed copies aren't reprocessed
    location: Literal["evidence", "inbox"]
    kind: Literal["pleading", "motion", "order", "exhibit",
                  "transcript", "email", "letter", "other"]
    pages: int | None
    parse_status: Literal["pending", "parsed", "failed"]
    added_at: datetime

class EvidenceLink(BaseModel):
    evidence_id: str             # "E-0311"
    kind: Literal["span", "document", "external"]
    doc_id: str | None = None    # span and document
    page: int | None = None      # span
    char_start: int | None = None
    char_end: int | None = None
    line_start: int | None = None    # transcripts cite page:line
    line_end: int | None = None
    quote: str | None = None     # span: must match the source verbatim
    reference: str | None = None     # external: URL or citation, never fetched
    origin: Origin
    review: Review               # evidence the AI attaches starts as a draft

class Fact(BaseModel):
    fact_id: str                 # "F-0142": short, so chat answers can cite it
    case_id: str
    title: str                   # the only required field
    details: str | None = None
    date_text: str | None = None         # as written in the source
    date_start: date | None = None       # None means undated
    date_end: date | None = None         # for ranges
    date_certainty: Literal["exact", "approximate", "relative", "unknown"] = "unknown"
    anchor_fact_id: str | None = None    # for relative dates
    statuses: list[Status] = []          # set by people only
    tag_ids: list[str] = []
    witness_ids: list[str] = []
    evidence: list[EvidenceLink] = []
    origin: Origin
    review: Review               # people's facts start as accepted
    confidence: float | None = None      # AI drafts only
    created_by: str              # display name, or "agent"
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
    flagged: bool = False        # alert when a new document mentions them
    confirmed: bool              # False for witnesses the AI suggested

class Tag(BaseModel):            # an issue
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

class ReviewEvent(BaseModel):    # the feedback trail, used by eval
    fact_id: str
    evidence_id: str | None
    action: Literal["accept", "edit", "reject"]
    reason: str | None
    reviewer: str
    at: datetime

class Contradiction(BaseModel):
    contradiction_id: str
    fact_ids: list[str]
    field: Literal["date", "actor", "fact"]
    explanation: str
    status: Literal["open", "confirmed", "dismissed"]

class Job(BaseModel):
    job_id: str
    case_id: str
    trigger: Literal["agent", "ui", "upload"]
    status: Literal["queued", "running", "done", "failed"]
    counts: dict[str, int]       # docs seen, parsed, failed; drafts; quotes rejected
    error: str | None
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

class Alert(BaseModel):
    alert_id: str
    case_id: str
    type: Literal["contradiction", "flagged_witness", "possible_deadline"]
    fact_ids: list[str]
    doc_ids: list[str]
    created_at: datetime
    delivered_at: datetime | None    # set once the agent has announced it
```

### 8.2 MCP tools (agent-facing)

Keep the agent's tool list small and coarse; the individual pipeline stages stay internal.

| Tool | Purpose | Returns |
| --- | --- | --- |
| `scan_case(case_id)` | Queue a scan for new or changed documents | job_id (never blocks) |
| `job_status(job_id)` | Progress of a job | status, counts |
| `get_timeline(case_id, start?, end?, witness?, tag?, status?)` | Facts in date order | facts, each with its review state |
| `search_facts(case_id, query)` | Westlaw-style keyword search plus semantic search | facts |
| `get_fact(fact_id)` | One fact with each quote, its document and page, and the surrounding text | fact + evidence context |
| `get_witness(case_id, name)` | A witness profile, matched by name or alias | witness + facts |
| `list_contradictions(case_id, open_only)` | Conflicts between sources | contradictions |
| `get_pending_alerts(case_id)` | Alerts not yet delivered | alerts |
| `mark_alerts_delivered(alert_ids)` | Record that alerts were announced | count |
| `daily_digest(case_id)` | Summary of the day's changes; the same function behind the UI's "Today" panel | text with fact IDs |

Tools never return rejected facts. All tools are read-only except `scan_case` and `mark_alerts_delivered`, which write only job and alert rows. The agent can't create, edit, or review facts; people do that in the UI.

### 8.3 Skill outline (SKILL.md)

- **Scheduled scan:** call `scan_case`, poll `job_status` until it finishes, then call `get_pending_alerts`. If a messaging channel is configured, announce each alert and call `mark_alerts_delivered`. Without a channel, stop there; the UI shows alerts on its own.
- **Content-free alerts:** anything sent outside the UI contains only the case name, the alert type, and a link to the local dashboard.
- **Citations:** every factual answer cites fact IDs and evidence (document and page). Prefer accepted facts, and say so when citing an unreviewed draft. If no fact supports an answer, say that instead of answering.
- **Disputes:** call a fact disputed only if it carries the disputed status. An open contradiction is a "possible conflict."

### 8.4 Alert rules (`pipeline/alerts.py`)

Computed at the end of each job. None of these rules uses the LLM.

| Alert | Fires when |
| --- | --- |
| Contradiction | The job opened a new contradiction |
| Flagged witness | A newly processed document mentions a witness marked as flagged |
| Possible deadline | A new fact's date is in the future, or its text matches deadline language ("no later than," "due," "bar date," "deadline"). Always labeled "needs attorney review" |

### 8.5 Search syntax (`pipeline/search.py`)

Search covers fact titles, details, notes, and evidence quotes. `search.py` translates a Westlaw-style subset into SQLite FTS5 queries:

| Westlaw-style | Meaning | FTS5 |
| --- | --- | --- |
| `term1 term2` | Either term (Westlaw treats a space as OR) | `term1 OR term2` |
| `&` | Both terms | `AND` |
| `"exact phrase"` | Phrase | `"exact phrase"` |
| `/n`, e.g. `/5` | Within n words | `NEAR(a b, 5)` |
| `/s`, `/p` | Same sentence, same paragraph | `NEAR` at about 20 and 60 words (an approximation) |
| `%` | But not | `NOT` |
| `!` | Root expander | Prefix query (`term*`) |

`search_facts` merges these keyword results with sqlite-vec similarity, so natural-language questions in chat also work. The UI shows a short syntax help next to the search box, since space-as-OR surprises people who don't use Westlaw.

## 9. Framework options

Recommendations are in **bold**. Anything marked *verify* hasn't been confirmed on our hardware yet.

### 9.1 Agent runtime (fixed by the hackathon)

| Option | Pros | Cons |
| --- | --- | --- |
| **NemoClaw (OpenClaw inside OpenShell)** | Always-on agent + enforced sandbox + local vLLM, one installer | Must confirm it counts as "one tool" |
| Bare OpenClaw | Clearly satisfies the rule | No enforced sandbox; privacy relies on firewall rules |

### 9.2 How our code plugs into OpenClaw

| Option | Use it for | Verdict |
| --- | --- | --- |
| **MCP server** | Exposing pipeline actions as tools | Yes: Python, testable without OpenClaw, laptop-friendly |
| **Skill (SKILL.md folder)** | Workflow instructions and answer rules | Yes, alongside the MCP server |
| Plugin (OpenClaw SDK) | Custom channels, hooks, providers | Skip unless MCP + skill can't do something |

### 9.3 MCP server framework

| Option | Notes |
| --- | --- |
| **FastMCP 4.x** | Built on the official SDK v2; serves the current and older MCP handshakes from one server, which protects us if OpenClaw's client lags behind |
| Official MCP Python SDK v2 (`MCPServer`) | Fine too. Note that v2 renamed the old FastMCP class to `MCPServer`, so older tutorials won't match |
| TypeScript SDK | Only if the team prefers TS; the rest of our stack is Python |

Pin the exact version in `pyproject.toml`. Use stdio transport. Never print to stdout in a stdio MCP server; log to stderr.

### 9.4 Model serving

| Option | Notes |
| --- | --- |
| **NemoClaw-managed vLLM** | Default on DGX Spark; OpenAI-compatible; supports JSON-schema structured output |
| Ollama | Fallback if vLLM misbehaves; simplest setup |
| Managed llama.cpp | Experimental Nemotron recipe on DGX Spark; skip unless needed |

### 9.5 Models

| Role | Starting choice | Notes |
| --- | --- | --- |
| Main LLM | **nvidia/Qwen3.6-35B-A3B-NVFP4** | NemoClaw's validated default on DGX Spark |
| Alternative LLM | Nemotron 3 Nano 30B-A3B | Has a NemoClaw llama.cpp recipe; compare on our test set |
| Embeddings | A small open embedding model (e.g. BGE-M3) | *Verify* Arm64 install; can be served by vLLM or run in-process |

### 9.6 Document parsing

| Option | Notes |
| --- | --- |
| **Docling** | Arm64 supported; keeps page numbers and bounding boxes for every element; parses .eml/.msg emails; has OCR and a VLM pipeline for scans |
| PyMuPDF | Very fast for PDFs with a text layer; good fallback and the first step of the cascade |
| Marker / MinerU | Alternatives if Docling struggles on our documents |

For transcripts, keep the printed line numbers from the text layer so evidence can cite page and line (*verify* on a real deposition).

### 9.7 Structured extraction

| Option | Notes |
| --- | --- |
| **vLLM JSON-schema output + Pydantic models** | Schema generated from our Pydantic models; output always parses |
| Instructor | Nice retries/validation layer; optional |
| Free-text + regex | No |

### 9.8 Pipeline orchestration

| Option | Notes |
| --- | --- |
| **Plain Python modules** | OpenClaw is already our agent; a second agent framework adds confusion and debugging pain |
| LangGraph / LlamaIndex | Only if we hit a real need |

### 9.9 Storage and search

| Option | Notes |
| --- | --- |
| **SQLite (WAL mode) + FTS5 + sqlite-vec** | One file, zero services, keyword + vector search; WAL lets the UI and MCP server read while a job writes |
| LanceDB | Good embedded vector store if sqlite-vec is awkward |
| Postgres + pgvector | Overkill for the hackathon |

### 9.10 Dates

dateparser + python-dateutil for normalization. A month or year on its own becomes a range with approximate certainty ("March 2022" covers March 1–31). Relative dates ("two weeks after the meeting") are resolved in code against an anchor fact; unresolved ones stay undated and appear in the undated group.

### 9.11 Web UI

| Option | Notes |
| --- | --- |
| **FastAPI + HTMX + vis-timeline** | Fastest to build; one language on the backend; bind to 127.0.0.1 |
| FastAPI + React (Vite) + vis-timeline | More polish, more work |
| Streamlit | Quickest prototype, least control over the timeline view |

Serve all JavaScript and CSS (HTMX, vis-timeline) from the repo rather than a CDN, so the UI works with no internet connection at all.

Exports: openpyxl for XLSX. vis-timeline renders HTML rather than SVG, so `export.py` draws the SVG for image exports server-side from the facts; PNG comes from that SVG via cairosvg (*verify* Arm64).

### 9.12 Always-on scheduling

| Option | Notes |
| --- | --- |
| **OpenClaw cron** | Triggers `scan_case` every few minutes, plus the daily digest; keeps the loop inside the required stack |
| File watcher (watchdog) in the app server | Instant reaction; queues the same job. Optional, if cron intervals feel slow |
| systemd timer | Outside the agent; avoid |

### 9.13 Tooling

Python 3.12, uv, ruff, pytest, pre-commit.

## 10. Repository layout

```
case-timeline/
├── pyproject.toml          # pinned deps (uv)
├── .env.example            # LLM_BASE_URL, MODEL_NAME, EVIDENCE_DIR, DATA_DIR, DB_PATH
├── Makefile                # test, eval, fetch-eval, serve-mcp, app, lint
├── pipeline/
│   ├── schema.py           # Pydantic models (the contract, §8.1)
│   ├── parse.py            # Docling wrapper; page, char, and line offsets; document kind
│   ├── extract.py          # LLM fact drafting (JSON schema)
│   ├── verify.py           # verbatim quote check (no LLM)
│   ├── dates.py            # normalization + relative dates
│   ├── witnesses.py        # name and alias matching, suggested witnesses
│   ├── merge.py            # attach evidence to matching facts across documents
│   ├── conflicts.py        # contradiction detection
│   ├── alerts.py           # alert rules (§8.4)
│   ├── search.py           # Westlaw-style to FTS5 translation + semantic search
│   ├── export.py           # XLSX, CSV, SVG, PNG
│   ├── jobs.py             # job queue and worker
│   └── store.py            # SQLite access (WAL)
├── mcp_server/
│   └── server.py           # agent-facing tools (§8.2)
├── skill/
│   └── SKILL.md            # workflow + answer rules for OpenClaw
├── app/                    # FastAPI app: routes, templates, vendored JS/CSS, worker startup; binds 127.0.0.1
├── eval/
│   ├── docs/               # ~20 public test documents + manifest (URL, retrieval date, sha256)
│   ├── gold_timeline.json  # hand-made expected facts
│   ├── gold_disputes.json  # disputed/undisputed labels from Rule 56.1 pairs
│   └── run_eval.py         # metrics in §13
└── docs/
    └── SPEC.md             # this spec
```

## 11. GB10 setup steps

Do steps 1–4 before the event if the rules allow.

1. **Preflight.** `uname -m` should print `aarch64`. Check free disk space (`df -h`) and confirm internet access for the initial downloads.
2. **Install NemoClaw.**
   ```bash
   curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
   ```
3. **Choose local inference during onboarding.** Pick the managed vLLM option (shown by default on DGX Spark), or pass it explicitly: `... | bash -s -- --local-model-runtime=vllm` (*verify* exact form). Keep the default model to start.
4. **Verify the stack.**
   ```bash
   nemoclaw <name> status
   openshell sandbox list
   openclaw tui        # send a test prompt
   ```
   Then disconnect the network and confirm the model still answers.
5. **Plan two sandbox policies** (*verify* the policy format in the NemoClaw/OpenShell docs):
   - *Setup mode:* egress allowed to package indexes and GitHub so we can install dependencies, clone the repo, and download the public test documents.
   - *Demo mode:* egress only to `inference.local`; evidence folder read-only; writes only to our data directory (database and inbox). If we add a messaging channel, its single endpoint is the only other allowed destination.
6. **Clone the repo and install dependencies** inside the sandbox with `uv sync`, then run `make fetch-eval` while setup mode still allows downloads.
7. **Register our MCP server** with OpenClaw (`openclaw mcp add …`, *verify* syntax), then check it with `openclaw mcp list` and `openclaw mcp doctor --probe`.
8. **Install the skill** from the repo folder: `openclaw skills install ./skill --as case-timeline` (*verify*), then restart the gateway if required.
9. **Schedule the always-on jobs** with OpenClaw cron: `scan_case` every few minutes and a daily digest (*verify* syntax).
10. **Start the app server** (web UI + job worker) bound to 127.0.0.1 and forward its port from the sandbox to the host's loopback interface only (*verify* how OpenShell exposes a sandbox port to the host and keeps a long-lived process running).
11. **Set up SSH access.** Enable the SSH server, add each teammate's public key, and turn off password login (`PasswordAuthentication no` in `sshd_config`). From a laptop, open the UI through the tunnel (§5.2).
12. **Turn off the agent's shell tool** so it can only call our MCP tools (OpenClaw tool policy, *verify* the setting name).
13. **Switch to the demo-mode policy** and run the smoke test (§14, M4).

Do not run `openshell self-update`, `npm update -g openshell`, or recreate the gateway by hand. NVIDIA's docs warn this breaks the NemoClaw-managed setup.

## 12. Team workflow

- One repo, short-lived branches, small PRs, merge several times a day.
- One person owns the box. Only they change installs, policies, or the sandbox.
- Laptops develop against the GB10's model through an SSH tunnel (e.g. `ssh -N -L 8000:127.0.0.1:8000 user@gb10-host`, then `LLM_BASE_URL=http://localhost:8000/v1` in `.env`), so nobody needs a local GPU and the model server never listens on the network. *Verify* which host port the NemoClaw-managed vLLM container uses.
- Everyone tests against `eval/`. `make eval` reports the metrics in §13; that's also our number for the judges.
- Review drafts in the real UI while testing. Those review events feed the acceptance-rate metric.
- Run the full pipeline end to end at least twice before the demo.

### Suggested roles

| Role | Owns |
| --- | --- |
| Infra | GB10, NemoClaw, policies, MCP registration, cron, the app server process, `jobs.py` |
| Parsing | `parse.py`, `verify.py`, page, char, and line offsets, document kinds, `search.py` |
| Extraction | `extract.py`, `dates.py`, `witnesses.py`, `merge.py`, `conflicts.py`, `alerts.py`, prompts |
| UI + demo | `app/`, list and visual views, source viewer, review flow, `export.py`, demo data and script |

## 13. Evaluation and test data

`make eval` reports:

| Metric | What it measures |
| --- | --- |
| Fact precision and recall | Drafted facts against the gold timeline |
| Date accuracy | Exact match on normalized dates; range overlap for approximate dates |
| Quote verification rate | Share of drafted quotes that pass `verify.py` |
| Merge accuracy | Whether facts about the same event from different documents end up as one fact |
| Contradiction precision and recall | Against the gold dispute labels |
| Review acceptance rate | From review events during testing; the closest number to "would a lawyer keep this?" |
| Throughput | Documents per hour on the stress set; measure by M3 |

The eval set uses public documents only: the Twitter v. Musk letters from Musk's Schedule 13D amendments on EDGAR plus the pleadings; the overlapping FTX complaints from the SEC and CFTC; two Rule 56.1 statement-and-response pairs, which supply the gold dispute labels; one Enron email thread; and one deposition excerpt. A separate stress set of 1,000–5,000 documents, sampled from the EDRM Enron set or the FTX bankruptcy docket, measures throughput and failure rates; it has no gold labels. Don't use the Epstein files: the public release contains unredacted victim information.

## 14. Milestones and definition of done

| Milestone | Done when |
| --- | --- |
| M0 Box ready | NemoClaw installed; local model answers with the network disconnected |
| M1 Repo ready | Skeleton, linting and tests running, §8 contracts merged, test set and gold files committed |
| M2 Wiring | MCP server with stub tools registered; inside the sandbox the agent calls `get_timeline` and gets fixture facts back; the app server's worker runs a queued no-op job |
| M3 Thin slice | One real document becomes draft facts with verified quotes, shown in the list view; clicking one opens the source viewer with the sentence highlighted; accept and reject work |
| M4 Always-on loop | Dropping a file into the evidence folder, or uploading it in the UI, produces new drafts and any alerts the rules call for, with no manual step; an outbound connection attempt is blocked; a port scan from another machine shows only SSH open |
| M5 Fact workspace | Visual view with key-fact highlighting and an undated lane; editing of statuses, tags, witnesses, and notes; sorting and filters; Westlaw-style search; "Add fact" from a selection; witness profiles; XLSX, CSV, and SVG export that respects filters |
| M6 Differentiators | Cross-document merging attaches evidence to existing facts; contradictions can be confirmed into the disputed status; flagged-witness and possible-deadline alerts fire |

Setup is complete at M4. M5 brings the DISCO-equivalent workspace, and M6 is what DISCO doesn't do. If time runs short, trim M5's lower-value items (PNG export, bulk review, witness profile pages) before touching M6, because M6 is what sets us apart.

## 15. Risks and open questions

| Risk / question | Mitigation |
| --- | --- |
| Does NemoClaw count as "one tool"? | Ask organizers on day 0. Fallback: bare OpenClaw + firewall rules allowing only the local model |
| Can we pre-install on the GB10 before the event? | Ask; if not, script steps 1–9 so they run unattended |
| Local inference is marked "caveated" in NemoClaw docs | Test M0 early; Ollama as fallback |
| Missing Arm64 wheels for a dependency | Install everything on day 0; Docling is Arm64-supported; verify cairosvg and sqlite-vec |
| MCP protocol mismatch with OpenClaw's client | FastMCP 4 serves old and new handshakes; pin versions |
| stdout writes corrupt the stdio MCP server | Log to stderr only; add a test |
| Messaging channel leaks case content | No channel by default; if one is added, content-free alerts only (§8.3) |
| Sandbox needs internet during setup | Separate setup-mode and demo-mode policies (§11, step 5) |
| Throughput on large document sets | Cascade: hash dedupe and cheap rules first, OCR only when there's no text layer, the LLM only where needed; measure docs/hour by M3 |
| Event Wi-Fi blocks device-to-device traffic (kills the SSH tunnel) | Test on day 0. Fallbacks: a monitor and keyboard plugged into the GB10, or a laptop on a direct Ethernet cable |
| The sandbox's "localhost" isn't the host's | Verify OpenShell port forwarding early (§11, step 10) |
| The app server isn't kept running inside the sandbox | Verify how OpenShell keeps a long-lived process alive; queued jobs wait safely until it's back |
| OpenClaw's MCP integration won't cooperate | Fallback: skill + a small CLI over the same core library that prints JSON. The agent then needs shell access, so restrict it to that one command if possible |
| SQLite contention between the app server and the MCP server | WAL mode, short transactions, one job per case at a time |
| Too many AI drafts for reviewers | "Needs review" sorted by confidence, bulk accept and reject, and a per-document cap on drafts (tune) |
| Name variants for one witness ("J. Smith," "John Smith," "Mr. Smith") | Alias lists; unmatched names become suggested witnesses that a person confirms or merges |
| vis-timeline can't export SVG | Draw export SVGs server-side from the facts (§9.11) |
| Chasing DISCO parity eats the schedule | The §3 table is the cap; anything else is out of scope |
| The product looks like a DISCO copy | Match workflows, not branding; no DISCO or Cecilia names or assets (§3) |

## 16. References

- NVIDIA NemoClaw docs: https://docs.nvidia.com/nemoclaw/
- NemoClaw local inference options: https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/inference/local-inference/choose-local-inference-server
- NemoClaw on GitHub: https://github.com/NVIDIA/NemoClaw
- OpenClaw docs (tools, skills, plugins): https://docs.openclaw.ai
- FastMCP guide: https://www.mcpbundles.com/guides/fastmcp
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- SQLite FTS5: https://sqlite.org/fts5.html
- DISCO Timelines (reference product): https://csdisco.com/offerings/timelines
- DISCO: getting started with Timelines: https://cbsupport.csdisco.com/hc/en-us/articles/13962936043149
- DISCO: visual timelines: https://cbsupport.csdisco.com/hc/en-us/articles/24261522864525-Visual-timelines
- DISCO: Cecilia Auto Timelines: https://cbsupport.csdisco.com/hc/en-us/articles/21196890780429-Cecilia-Auto-Timelines
- DISCO: building timelines from Ediscovery documents: https://support.csdisco.com/hc/en-us/articles/22506571741069-Building-Case-Builder-timelines-using-Ediscovery-documents
- DISCO blog, fact-based timelines: https://csdisco.com/blog/fact-based-legal-timelines
