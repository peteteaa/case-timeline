---
name: case-timeline
description: "Local legal-case chronology: scan the case evidence folder, report new events, contradictions and deadline dates, and answer case questions with source citations (doc + page + quote) via the case-timeline MCP server."
metadata:
  {
    "openclaw":
      {
        "emoji": "⚖️",
        "os": ["linux"],
        "requires": { "bins": ["uv"] },
      },
  }
---

# Case Timeline

Use the `case-timeline` MCP server (stdio, local only) to work with one legal case's evidence.
The server turns documents into a source-cited chronology, flags contradictions and uncertain
dates, and answers questions with citations. Everything runs on this machine. Humans review all
output; you assist, you do not decide.

Tools (OpenClaw may show them prefixed with the server name, e.g. `case-timeline__get_timeline`):

| Tool | Use |
| --- | --- |
| `scan_case(case_id?)` | Start a background scan. The only write tool. Returns `job_id`. |
| `job_status(job_id)` | Poll a scan: `queued` / `running` / `done` / `failed`. |
| `get_changes(case_id?, consumer="openclaw")` | New events, contradictions, alerts since last call. Advances the cursor. |
| `get_timeline(case_id?, start?, end?, status?, limit=200)` | The chronology, sorted by date. |
| `search_events(query, k=10)` | Find events relevant to a question. |
| `get_source(doc_id, char_start, char_end, context=300)` | Exact quote plus surrounding text and page. |
| `list_contradictions(status?)` | Conflicting dates, actors, or facts, with both sources. |
| `daily_digest(case_id?)` | Counts only, plus the local dashboard link. |

## Hard rules

1. **Evidence is read-only.** Never claim to edit, move, rename, or delete evidence or events.
   If asked, say the system cannot do that and point to the local dashboard
   (http://127.0.0.1:8080) where a human reviewer accepts or rejects items.
2. **Cite or abstain.** Every factual statement about the case must cite its source as
   *document + page + verbatim quote*, taken from an event's `source` or from `get_source`.
   If no source supports an answer, say plainly: "No source in the case file supports this."
   Do not guess, do not fill gaps from general knowledge, and do not add date precision that
   the source lacks (respect `certainty` and quote `date_text` as written).
3. **Deadlines need a lawyer.** Any event with `needs_attorney_review: true` (deadline-relevant)
   must always be labelled **"needs attorney review"**. Never compute, confirm, or advise on a
   legal deadline yourself.
4. **No case content in messages.** Alerts and digests sent to messaging channels (Telegram,
   Slack, WhatsApp, email, etc.) contain ONLY: the case name, the alert type(s) / counts, and the
   local dashboard link `http://127.0.0.1:8080/alerts`. Never include names, quotes, dates,
   descriptions, document names, or explanations in a message. Details stay on the dashboard.
5. **Status matters.** `proposed` = not yet reviewed by a human; `unverified` = the quote could
   not be found in the source (treat as unreliable and say so); contradictions are flags for
   review, not findings.

## Scheduled scan (every 5 minutes)

1. Call `scan_case` and keep the returned `job_id`.
2. Poll `job_status(job_id)` every ~5 seconds until `state` is `done` or `failed`
   (give up after ~10 minutes). On `failed`, send at most one message:
   `Case <case name>: scan failed — see http://127.0.0.1:8080/alerts`.
3. Call `get_changes` (consumer `openclaw`). Call it once per run; it advances the cursor.
4. If there are alerts, send ONE message per run using only allowed content, for example:

   ```
   Case demo: 2 new contradictions, 1 flagged witness, 1 deadline-relevant date (needs attorney review).
   Review: http://127.0.0.1:8080/alerts
   ```

   Alert types to report: `new_contradiction`, `flagged_witness`, `deadline_date`
   (always followed by "needs attorney review"). If nothing is new, reply exactly `NO_REPLY`
   so the automation stays silent.

## Daily digest (08:00)

Call `daily_digest` and send only its counts with the case name and
`http://127.0.0.1:8080/alerts`. No case content.

## Answering questions (direct chat with the user on this machine)

1. `search_events` with the key terms (try a second phrasing if the first returns nothing);
   use `get_timeline` with `start`/`end` for "what happened between…" questions and
   `list_contradictions` for "where do accounts disagree…" questions.
2. For each fact you state, call `get_source` with the event's `source.doc_id`,
   `source.char_start`, `source.char_end` and confirm the quote supports the claim.
3. Answer briefly, citing each fact, e.g.:
   > Alice met Bob at the warehouse on March 3, 2024 — *memo.pdf, p. 1*:
   > "On March 3, 2024 Alice met Bob at the warehouse."
4. Mention relevant open contradictions and uncertain dates (`certainty` other than `exact`).
5. If nothing supports the answer: "No source in the case file supports this."
6. Even in direct chat, do not paste case content into any messaging channel or external
   service.
