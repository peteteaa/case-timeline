# Case Timeline Agent

A private, always-on case chronology workspace for one NVIDIA GB10, modeled on DISCO Timelines
(see [docs/SPEC.md](docs/SPEC.md), Spec 2). **Facts**, not documents, are the unit. The system
watches a case folder and drafts facts from every new document, tying each one to the exact
source sentence. Lawyers then review the drafts, organize the timeline, and see where sources
contradict each other. Everything runs on the box, and no case data leaves it.

## Quickstart

```bash
cp .env.example .env    # optional; every setting has a default (CT_ prefix)
make preflight          # aarch64, disk, GPU, uv, openclaw
make setup              # uv sync (needs internet once)
make llm                # vLLM on 172.17.0.1:8000 (docker0 bridge); wait for /v1/models
make fixtures           # rebuild the fictional demo case in tests/fixtures/case_demo
make app                # web UI + job worker on http://127.0.0.1:8080
                        #   "Process now" in the UI queues a scan; the worker runs it
                        #   if 8080 is taken (e.g. by the OpenShell gateway): CT_UI_PORT=8091
make scan CASE=demo     # or: queue a scan and run it inline, without the app server
make eval CASE=demo     # SPEC §13 metrics; report in data/eval_demo.json
```

To wire up the agent (SPEC §11 steps 7–9), run `scripts/register_openclaw.sh`, which
registers the stdio MCP server and installs `skill/` as `case-timeline`. Then run
`scripts/openclaw_cron.sh`, which schedules a scan every 5 minutes and a daily digest. Demo mode
has no messaging channel, so the cron jobs use `--no-deliver` and alerts stay in the UI. Set
`CT_ALERT_CHANNEL` (and `CT_ALERT_TO`) to announce content-free alerts instead.

Other targets:

| Target | What it does |
| --- | --- |
| `make test` | Unit tests; no model needed |
| `make test-llm` | Tests against the live model |
| `make lint` / `make fmt` | ruff check + format check, or auto-fix |
| `make serve-mcp` (alias `mcp`) | MCP server over stdio; OpenClaw normally spawns it |
| `make fetch-eval` | Download the public eval documents (setup mode only) |
| `make ports` | Fails if anything but sshd (and vLLM on the bridge) listens off loopback |

If vLLM fails to initialise with the default MoE backend, run
`MOE_BACKEND=flashinfer_cutlass make llm`.

## Architecture

```
GB10 host
├── vLLM (172.17.0.1:8000, docker0 bridge) ◄──────────────────────────┐
├── Evidence folder (read-only) ─────────────── read ─────────────────┤
└── Sandbox                                                           │
    ├── OpenClaw agent + skill/ + cron                                │
    │     └── MCP tool calls ──► mcp_server (stdio subprocess)        │
    │                              ├── reads SQLite                   │
    │                              └── queues jobs (never runs them)  │
    ├── App server (app/, 127.0.0.1:8080) ◄── SSH tunnel ── browser   │
    │     ├── web UI: list + visual timeline, review, search, export  │
    │     └── job worker: parse → extract → verify → dates →          │
    │                     witnesses → merge → conflicts → alerts ─────┘
    ├── SQLite (WAL) data/case.db: facts, evidence, witnesses, tags,
    │     notes, documents, jobs, alerts, contradictions, FTS5, sqlite-vec
    └── data/inbox (UI uploads, writable)
```

- **Two long-lived processes.** The app server and OpenClaw, with the MCP server as
  OpenClaw's subprocess. Anything can *request* a job (`scan_case`, "Process now", an upload),
  but requesting only adds a row to the jobs table. The app server's worker runs jobs one at a
  time per case and re-queues jobs left running after a crash. Job state lives in SQLite, so it
  survives restarts of either process.
- **Facts are drafts until a person reviews them.** The extractor proposes facts with quotes.
  `verify.py` checks every quote verbatim, without the LLM, and discards drafts whose quotes
  don't match. Witnesses are matched by name and alias, and unmatched names become suggested
  witnesses. Drafts about an existing event attach their quote as new evidence instead of
  creating a duplicate. Disagreeing sources open a contradiction. People accept, edit, or
  reject drafts in the source viewer. Rejected facts are hidden everywhere but kept for eval.
- **People own judgments.** Only people set statuses (key fact, case impact, disputed), apply
  tags, write notes, or confirm a contradiction, which makes its facts *disputed*. The store
  enforces this: writes made as `agent` may only create drafts, attach draft evidence, suggest
  witnesses, open contradictions, and touch job and alert rows.
- **Allegations.** A fact whose only evidence comes from pleadings is labeled *alleged*.
- **Alerts** (contradiction, flagged witness, possible deadline) are computed in
  `pipeline/alerts.py` at the end of each job and stored in the database. The agent only
  delivers them and records delivery (`mark_alerts_delivered`), so it needs no memory between
  runs.

All real logic lives in `pipeline/`. Only `pipeline/llm.py` talks to the model. The UI and
the MCP server are thin layers over the store. Module signatures are listed in
[INTERFACES.md](INTERFACES.md).

### MCP tools (SPEC §8.2)

`scan_case`, `job_status`, `get_timeline`, `search_facts`, `get_fact`, `get_witness`,
`list_contradictions`, `get_pending_alerts`, `mark_alerts_delivered`, `daily_digest`. They
never return rejected facts. Only `scan_case` and `mark_alerts_delivered` write, and only job
and alert rows. The agent cannot create, edit, or review facts.

## Security model

- **Loopback only.** The app server binds `127.0.0.1:8080` and refuses a non-loopback host.
  vLLM binds the docker0 bridge (`172.17.0.1:8000`) so the NemoClaw sandbox can reach it, and
  is not reachable from Wi-Fi or Tailscale. Nothing binds `0.0.0.0`. `make ports` checks this.
- **SSH tunnel for people.** Use `ssh -N -L 8080:127.0.0.1:8080 user@gb10-host`, then open
  http://localhost:8080. Everyone who reaches the UI is a trusted teammate who gives a display
  name once. There are no accounts.
- **MCP over stdio.** The server has no port. Stdout carries JSON-RPC only: logs go to stderr
  and `data/mcp.log`, and stray prints are redirected.
- **Local model only.** `pipeline/llm.py` refuses non-local endpoints and ignores proxy
  variables.
- **Evidence is read-only.** Uploads go to `data/inbox`. The settings refuse a data dir,
  database, or inbox inside the evidence folder. Case ids are validated so they can't act as
  paths.
- **Content-free alerts.** Anything sent outside the UI carries only the case name, the alert
  type, and a dashboard link. Possible deadlines are always marked "needs attorney review".
  Demo mode has no messaging channel.
- **Verbatim citations.** Every AI quote must match the source text exactly before anyone sees
  it. Transcripts cite page:line.

## Evaluation

`make eval CASE=<id>` scores the stored facts after a scan. It reports fact precision, recall,
and F1; date accuracy; quote verification rate; merge accuracy; contradiction precision and
recall; review acceptance rate; and throughput. See `eval/run_eval.py`. The fictional demo case
has gold labels in `eval/gold/demo_timeline.json` and `eval/gold/demo_disputes.json`. The public
eval set is listed in `eval/docs/manifest.json` (EDGAR, SEC, CFTC, CourtListener, and DOJ URLs,
each verified) and is fetched by `make fetch-eval`. Its gold files, `eval/gold_timeline.json`
and `eval/gold_disputes.json`, still need to be labeled by hand.

## Milestones (SPEC §14)

- [ ] **M0 Box ready:** NemoClaw installed; the local model answers with the network
  disconnected.
- [ ] **M1 Repo ready:** skeleton, linting, and tests running; §8 contracts merged; test set and
  gold files committed.
- [ ] **M2 Wiring:** MCP server registered with stub tools; inside the sandbox the agent calls
  `get_timeline` and gets fixture facts back; the app server's worker runs a queued no-op job.
- [ ] **M3 Thin slice:** one real document becomes draft facts with verified quotes in the list
  view; clicking one opens the source viewer with the sentence highlighted; accept and reject
  work.
- [ ] **M4 Always-on loop:** dropping a file into the evidence folder, or uploading it, produces
  new drafts and any alerts the rules call for, with no manual step; an outbound connection
  attempt is blocked; a port scan from another machine shows only SSH open.
- [ ] **M5 Fact workspace:** visual view with key-fact highlighting and an undated lane; editing
  of statuses, tags, witnesses, and notes; sorting and filters; Westlaw-style search; "Add fact"
  from a selection; witness profiles; XLSX, CSV, and SVG export that respects filters.
- [ ] **M6 Differentiators:** cross-document merging attaches evidence to existing facts;
  contradictions can be confirmed into the disputed status; flagged-witness and
  possible-deadline alerts fire.
