# Case Timeline Agent

An always-on local assistant for an NVIDIA GB10. It watches a case evidence folder, turns the documents into a chronology with source citations, flags contradictions and uncertain dates, and answers questions with citations.

**Local only.** Case data never leaves the machine. The model, embeddings, database, and UI all run on the box. People review every proposed event. The system never edits or deletes original evidence.

## Quickstart

```bash
make preflight          # checks aarch64, disk, GPU, uv, and openclaw
make setup              # uv sync (needs internet once)
make llm                # starts vLLM on 127.0.0.1:8000; wait for /v1/models to answer
make fixtures           # builds the demo case in tests/fixtures/case_demo
make scan CASE=demo     # parses, extracts, verifies, dates, merges, and checks for conflicts
make ui                 # dashboard at http://127.0.0.1:8080
make eval CASE=demo     # scores against eval/gold/demo.json; report in data/eval_demo.json
```

Other targets: `make test` (unit tests; no model needed), `make test-llm` (live-model tests), `make lint`, `make mcp` (MCP server over stdio), and `make ports` (fails if anything other than ssh is exposed).

Configuration uses environment variables with the `CT_` prefix. See `pipeline/config.py`. The common ones are `CT_EVIDENCE_DIR`, `CT_DATA_DIR`, and `CT_CASE_ID`.

If vLLM fails to initialise with the default MoE backend, run `MOE_BACKEND=flashinfer_cutlass make llm`.

## Architecture

```
 evidence dir (read-only)   data/inbox (uploads)
            \                 /
             v               v
   parse (Docling / PyMuPDF / text) -> ParsedDocument + page offsets
             |
   extract  -- pipeline/llm.py --> vLLM Qwen3.6 @127.0.0.1:8000 (JSON schema)
             |
   verify (locate quote -> char offsets)  -> dates (resolve, certainty)
             |
   merge (dedup: rapidfuzz / BGE-M3)  -> conflicts (rules + LLM confirm) -> alerts
             |
   store: SQLite + FTS5 + sqlite-vec  (data/case.db)
        /                 |                       \
  ui (FastAPI+HTMX)   mcp_server (stdio)      eval (gold vs stored)
  127.0.0.1:8080      scan/timeline/search    precision/recall/date acc
```

Only `pipeline/llm.py` talks to the model. The UI and the MCP server go through `pipeline.*`. Contracts are frozen in `pipeline/contracts.py`, and module signatures are in `INTERFACES.md`.

## Security model

- Every service binds to `127.0.0.1`: vLLM on :8000 and the UI on :8080. Nothing binds to `0.0.0.0`. `make ports` fails if any listener other than sshd:22 uses a non-loopback address.
- For remote access, use an SSH tunnel: `ssh -L 8080:127.0.0.1:8080 dell@<gb10>`, then open http://127.0.0.1:8080.
- The evidence folder is read-only to the pipeline. Uploads go to `data/inbox`.
- Alerts carry no case content, only the case name, the alert type, and a local dashboard link. That makes them safe to relay through messaging. `daily_digest` returns counts only.
- The MCP server uses stdio only and never writes to stdout outside the protocol.
- Every event cites a verbatim quote with document offsets. Events whose quote can't be found are marked `unverified`.

## Milestones

- [ ] **M0 Scaffold:** frozen contracts, config, Makefile, scripts, preflight, port check, and an eval harness.
- [ ] **M1 Ingest and extract:** discover/parse PDF/DOCX/TXT/MD/EML, LLM extraction, quote verification, and date resolution, stored in SQLite.
- [ ] **M2 Timeline and review:** merge/dedup, a UI timeline with source highlighting, accept/reject, and search (FTS5 + vectors).
- [ ] **M3 Contradictions and alerts:** candidate rules, LLM confirmation, deadline and witness alerts, and the daily digest.
- [ ] **M4 Always-on:** folder watching and incremental scans, MCP tools, openclaw messaging, and an eval baseline on the demo case.
