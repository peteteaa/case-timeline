# Handoff: merging onto Spec 2 (from dell-98, 2026-10-25 ~22:20 UTC)

This note is for Leo and Leo's Claude, plus anyone else who merges into `main`.

## State of `main`

| Commit | What |
|---|---|
| `6f101bd` | Leo's Spec 2. The team adopted it as the base (user decision, 21:59). |
| `4a77604` | `deploy/serve-qwen36.sh` synced to the live script (cutlass default, `--max-num-seqs 8`). Only touches `deploy/`. |
| this commit | This handoff note plus `.claude/workflows/` (dell-1b's ct-* workflows, already updated for Spec 2). |

To merge your new frontend: `git fetch && git rebase origin/main` (or merge). Nothing here touches `app/`, `pipeline/` or `tests/`, so there should be no conflicts.

## Lost in the Spec 2 merge, still to port

The merge kept the Spec 2 versions of these files, which dropped:

1. **Date fixes** (`pipeline/dates.py`, `runner` → now `pipeline/jobs.py`):
   - Time-only mentions ("At 1816:19") inherit the day from the narrative or the incident/header date; `parse.incident_date` survives in `parse.py`, but dates and jobs no longer use it.
   - A case-level incident date for documents that never state it (e.g. a pilot interview held weeks later).
   - A per-case timezone (UTC/Zulu → local before taking the date).
   - Year-only mentions as an APPROXIMATE year range, never Jan 1.
   - Measured live on the old architecture: NTSB date accuracy 0.54 → 0.92.
2. **Contradiction precision rules** (`pipeline/conflicts.py`): date_relation / range containment, plan vs. outcome, message date vs. referenced date, knowledge vs. actor, time-zone shift and clock times within 90 s count as compatible, and the ACTOR exclusion cue must be local to its own side.
3. **llm_concurrency fan-out:** `llm.map_ordered` survives, but extract/conflicts/jobs use it in fewer places.
4. **OpenClaw scripts:** `scripts/register_openclaw.sh` and `scripts/openclaw_cron.sh` lost `--agent case-timeline` and the ntsb defaults.

Reference implementation: branch `wip/ntsb-accuracy` (`cc59d06`, old architecture; don't merge it) and commits `57f6ca4`, `764c1d7`, `4bea4cd` in history.

## Gotchas

- **Eval gold path:** fixed. The NTSB gold is now `eval/gold/ntsb_timeline.json` (renamed, content unchanged), so `make eval CASE=ntsb` and the app's **Eval** tab (`/cases/ntsb/eval`) find it without `--gold`.
- **Data policy (admin):** live scans and evals use the real NTSB case (`CT_CASE_ID=ntsb`, `CT_EVIDENCE_DIR=/home/dell/case-data/ntsb_cen10fa443/evidence`; Leo's copy is `/srv/case-timeline/case-data/ntsb_cen10fa443`). Unit tests stay on `tests/fixtures`.
- **Scan:** `python -m pipeline.jobs scan --case ntsb` with its own `CT_DATA_DIR`. Never use `data/` itself: an OpenClaw cron job (currently paused) scans into it.

## The shared model server (please read)

- vLLM serves Qwen3.6 at `http://172.17.0.1:8000/v1` (`qwen3.6-35b-a3b`). `localhost:8000` doesn't work.
- **It crashes under heavy parallel load.** At 21:51 it hit a Triton illegal memory access in the GDN linear-attention kernel (`fused_sigmoid_gating_delta_rule_update`), which runs on mixed prefill+decode batches at about 10–16 concurrent requests. The fix in the script is `--max-num-seqs 8`, but the server running now hasn't been restarted with it yet. Keep `CT_LLM_CONCURRENCY` at 4 or lower, and run one scan at a time.
- If it's down, restart it with `nohup ~/models/serve-qwen36.sh > ~/models/vllm-qwen36.log 2>&1 &` as user dell (about 3 min), and log it in `/srv/case-timeline/MACHINE_STATUS.md`.

## Demo UI

`ct-ui-8090` (owned by dell-c9) serves the OLD ui at http://127.0.0.1:8090 from the worktree `/home/dell/case-timeline-demo` (`4bea4cd`), with NTSB data copied to `/home/dell/ct-demo/ntsb`. To switch it to Spec 2: `python -m app` from a clean checkout of `main`, with the same env (`CT_UI_PORT=8090`, `CT_CASE_ID=ntsb`, `CT_EVIDENCE_DIR`, `CT_DATA_DIR` pointed at a Spec 2 scan copy, `CT_CHAT_BACKEND=local` if supported). Ask dell-c9 to do the restart.

## Status of the port (leo-zhang's Claude, 2026-10-25 ~22:35 UTC)

Branch `port/handoff` (on top of `4004c9d`). 655 tests green, ruff clean.

| Handoff item | State |
|---|---|
| 1. Date fixes | Ported to `pipeline/dates.py` + `pipeline/jobs.py`: bare times inherit the day (same sentence → exact; else the narrative's last day, the incident date, the document date → approximate); `parse.incident_date` is used again; a case-level incident date (consensus of the documents, or `CT_CASE_INCIDENT_DATE`) for documents that never state it; heading dates never anchor the account; `CT_CASE_TZ` converts UTC/Zulu times; "since 2006" is the whole year. Year-only was already a full-year range on Spec 2. |
| 2. Contradiction precision rules | Ported to `pipeline/conflicts.py` (date relation / containment / slack, message vs. referenced date, plan vs. outcome, knowledge vs. actor, time-zone and 90 s clock compatibility, local exclusion cue, date notes in the prompt). |
| 3. llm_concurrency fan-out | Extraction chunks, documents in a job, and contradiction confirmations all go through `llm.map_ordered`; stored results stay in document/candidate order. Default `llm_concurrency` lowered to 4. |
| 4. OpenClaw scripts | `--agent` (CT_OPENCLAW_AGENT, default case-timeline) in both scripts; `openclaw_cron.sh` defaults to the ntsb case. |
| Gotcha: eval gold path | Fixed twice over: dell-98 renamed the gold to `ntsb_timeline.json`; `default_paths` also accepts `eval/gold/<case>.json`, warns on any fallback, and never scores a case's contradictions against the demo disputes. |

Live NTSB scan on the port (one scan, concurrency 4, `CT_CASE_TZ=America/Chicago`, 252 s): **date accuracy 1.000 (21/21)**, recall 0.778 (21/27), 0 contradictions from 28 candidates (the model judged all of them distinct occurrences; there is no ntsb disputes gold to score them).

Recall fix (follow-up, same evening): the extraction prompt now gives each step of a recounted occurrence (a pilot's account of the flight in a later interview) its own fact, tied to the occurrence ("during the accident flight") rather than the interview's date, and asks for an incident's whole sequence (departure through response). The per-document cap scales with length (`CT_MAX_DRAFTS_PER_PAGE=8`, floor `CT_MAX_DRAFTS_PER_DOC=40`). A model-written authoring date on a recounted step moves to the incident date, and placeholder dates ("no date", "N/A") become empty. Second live scan (351 s): **recall 0.926 (25/27)**, date accuracy 0.920 as stored, **0.96 (24/25) with the authoring-date guard** (re-resolved offline on the same drafts), 202 facts, 8 contradictions from 39 candidates (unscored: no ntsb disputes gold). Extraction varies run to run (vLLM batching), so single-run recall is noisy (the interview's G14/G15 appear in some runs and not others).

Still open:
- The FAA Controller Statements PDF needs OCR (install the extras: `uv sync --all-extras`).
- Not ported from `cc59d06` (WIP, unverified): the `importance` key-event flag, low-yield re-OCR, `_refine_from_quote`.
