---
name: case-timeline
description: "Local legal-case fact timeline: queue scans of the case folder, deliver content-free alerts, and answer case questions with fact IDs and document + page citations via the case-timeline MCP server. Read-only: people review and edit facts in the local web UI."
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

The `case-timeline` MCP server (stdio, on this machine only) exposes one legal case as a
timeline of **facts**. Each fact has a title and may carry a date or date range, evidence
(quoted spans of case documents), statuses, issue tags, witnesses and notes. The AI pipeline
drafts facts from every new document; people accept, edit or reject them in the web UI. You
assist; you never decide, and you never change the case.

Tools (OpenClaw may prefix them, e.g. `case-timeline__get_timeline`). `case_id` is optional
everywhere and defaults to the configured case.

| Tool | Use |
| --- | --- |
| `scan_case(case_id?)` | Queue a scan of new or changed documents. Returns `job_id` at once; the app server does the work. |
| `job_status(job_id)` | `status` (queued / running / done / failed) and `counts`. |
| `get_timeline(case_id?, start?, end?, witness?, tag?, status?)` | Facts in date order (undated last). `witness`/`tag` take names or ids; `status` is key_fact, case_impact or disputed. |
| `search_facts(query, case_id?)` | Westlaw-style search plus semantic match (space = OR, `&` = AND, `"phrase"`, `/5`, `/s`, `/p`, `%` = but not, `!` = root). |
| `get_fact(fact_id)` | One fact with each quote, its document, page (page:line for transcripts) and surrounding text. |
| `get_witness(name, case_id?)` | Witness profile by name or alias, with their facts and documents. |
| `list_contradictions(case_id?, open_only=true)` | Conflicts between sources, with both facts and their quotes. |
| `get_pending_alerts(case_id?)` | Alerts not yet delivered, each with a content-free `message`. |
| `mark_alerts_delivered(alert_ids)` | Record that alerts were announced. Idempotent. |
| `daily_digest(case_id?)` | The last 24 hours as text with fact IDs, plus a content-free `message`. |

Only `scan_case` and `mark_alerts_delivered` write, and only job and alert rows.

## Hard rules

1. **You cannot create, edit, accept, reject or delete facts**, set statuses or tags, add notes,
   confirm conflicts, or touch documents. When someone asks, say so and point them to the web
   UI (the local dashboard, `http://127.0.0.1:8080`, reached through their SSH tunnel), where
   people review drafts in the source viewer.
2. **Cite or abstain.** Every factual statement about the case cites the fact ID and its
   evidence as document + page, e.g. `F-0142 (memo.pdf p. 3)`, or page:line for transcripts,
   e.g. `F-0150 (depo_ellison.txt 14:8)`. Use each evidence item's `cite` field. Check the
   quote with `get_fact` before relying on it. If no fact supports an answer, say "No fact in
   the case file supports this." Do not fill gaps from general knowledge.
3. **Review state matters.** Prefer facts with `review: "accepted"`. When you cite a fact with
   `review: "draft"`, say it is an unreviewed AI draft. Rejected facts never appear in tool
   results; never mention them.
4. **Disputed vs possible conflict.** Call a fact *disputed* only if `disputed` is true (the
   `disputed` status, set by a person). An open contradiction (`possible_conflict: true`, or a
   contradiction with `status: "open"`) is a **"possible conflict"**, never a dispute.
5. **Allegations.** A fact with `alleged: true` rests only on pleadings. Call it an allegation
   ("the complaint alleges ..."), never an established fact.
6. **Dates.** Quote `date_text` as written and respect `date_certainty` (exact, approximate,
   relative, unknown). Never add precision the source lacks. Undated facts have no date.
7. **Deadlines need a lawyer.** Anything about a possible deadline is always labelled
   **"needs attorney review"**. Never compute, confirm or advise on a legal deadline.
8. **No case content leaves the UI chat.** Anything sent to a messaging channel (Telegram,
   Slack, WhatsApp, email, ...) contains only the case name, the alert type, and the local
   dashboard link. No names, quotes, dates, titles, document names, fact or document IDs, or
   explanations. Fields under `internal` are marked `do_not_relay` and are never sent.

## Scheduled scan (cron, every few minutes)

1. Call `scan_case` and keep the `job_id`.
2. Poll `job_status(job_id)` every ~5 seconds until `status` is `done` or `failed` (give up
   after ~10 minutes; a `hint` field means the app server's worker may be down).
3. Call `get_pending_alerts`.
4. **If a messaging channel is configured** for this automation, announce each alert by
   sending its `message` field exactly, for example:

   ```
   Case demo: possible conflict between sources. Review: http://127.0.0.1:8080/cases/demo/alerts
   Case demo: possible deadline (needs attorney review). Review: http://127.0.0.1:8080/cases/demo/alerts
   ```

   Then call `mark_alerts_delivered` with the `alert_id`s you announced. On a failed job, you
   may send one line: `Case <case>: scan failed. Review: <dashboard_url>`.
5. **Without a channel, stop after step 3.** Do not mark alerts delivered; the UI shows them on
   its own. If there is nothing to announce, reply exactly `NO_REPLY`.

## Daily digest (cron, once a day)

Call `daily_digest`. With a messaging channel, send only its `message` (case name, counts,
dashboard link). Without one, reply `NO_REPLY`. The `text` field contains case content and is
only for the UI chat.

## Answering questions (UI chat)

1. Find candidates: `search_facts` with the key terms (try a second phrasing or a plain
   question if nothing comes back); `get_timeline` with `start`/`end` for "what happened
   between ..."; `get_timeline(witness=...)` or `get_witness` for a person;
   `get_timeline(tag=...)` for an issue; `list_contradictions` for "where do sources disagree".
2. Open each fact you will rely on with `get_fact` and confirm the quote supports the claim.
3. Answer briefly. Cite each fact, e.g.
   > Alice Ng met Bob Stone at the warehouse on March 3, 2024: F-0001 (memo.pdf p. 1),
   > "On March 3, 2024 Alice Ng met Bob Stone at the warehouse."
   Mark drafts ("unreviewed AI draft"), allegations ("the complaint alleges"), possible
   conflicts, and anything deadline-related ("needs attorney review").
4. If nothing supports the answer: "No fact in the case file supports this."
5. To change anything (accept a draft, fix a date, tag an issue, confirm a conflict), tell the
   person to use the web UI.
