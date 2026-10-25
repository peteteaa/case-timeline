export const meta = {
  name: 'ct-eval-diagnose',
  description: 'Diagnose a case-timeline eval run: group errors, verify each root cause in code, rank fixes by impact (read-only)',
  whenToUse: 'After a scan+eval run, to decide what to fix next. args: {case: "ntsb"|"demo", runDir: "data/<run>"} (runDir defaults to the newest eval for the case)',
  phases: [
    { title: 'Analyze', detail: 'one agent per error category' },
    { title: 'Verify', detail: 'a skeptic re-checks each root cause in the code' },
    { title: 'Rank', detail: 'prioritized fix plan with expected metric gains' },
  ],
}

const ROOT = '/home/dell/case-timeline'
const A = args || {}
const CASE = A.case || 'ntsb'
const RUN = A.runDir || ''

const CONTEXT = `Project: ${ROOT} (case-timeline Spec 2: documents -> dated facts -> timeline workspace; see docs/SPEC.md, INTERFACES.md, pipeline/schema.py).
Eval report: ${RUN ? `${ROOT}/${RUN}/eval_${CASE}.json` : `the newest ${ROOT}/data/*/eval_${CASE}.json by mtime (say which one you used)`}. Its SQLite db is case.db in the same folder. Gold: ${ROOT}/eval/gold/${CASE}_timeline.json if it exists, else ${ROOT}/eval/gold/${CASE}.json. The scorer is eval/run_eval.py; without --gold it silently falls back to eval/gold_timeline.json. If the report's counts don't match the gold you're using, it was probably scored against the wrong file. Say so and stop.
Spec 2 reports have matches (per-match rows with gold/pred dates and date correctness), unmatched_facts, missed, false_positives, precision/recall/f1, strict_* and date_accuracy. Older pre-Spec-2 reports use unmatched_gold/unmatched_pred/per_doc instead. Read run_eval.py to confirm which fields you have.
READ-ONLY: do not edit files, run scans, or call the LLM server. Open SQLite with ?mode=ro. Reading source documents is fine.`

const CATEGORIES = [
  { key: 'dates', prompt: 'Date errors: every match whose date is wrong. Group them by (gold date -> predicted date) pattern, find the source date text/quote for each, and trace which code (pipeline/dates.py, pipeline/jobs.py, extract) produced the wrong date.' },
  { key: 'missed', prompt: 'Missed facts: every gold item with no match (missed / unmatched gold). For each one, find the passage in the source document and work out why nothing matched: not extracted, wrong date, merged away, low description similarity, or a parse/OCR gap.' },
  { key: 'noise', prompt: 'Noise: predicted facts with no gold match (false_positives / unmatched_facts), broken down by source document. Sample the noisiest documents and classify the extras as duplicate, real-but-minor, wrong/hallucinated, or not-a-fact. Point to the extract/merge code that could filter each class.' },
  { key: 'parsing', prompt: 'Parsing/OCR: documents with few predictions or 0 matches. Check how each one parsed (text yield per page, whether OCR ran) and whether text is missing for some pages.' },
]

const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          evidence: { type: 'string', description: 'concrete examples: gold idx / pred idx / document / text' },
          root_cause: { type: 'string', description: 'file:function and why' },
          fix: { type: 'string' },
          affected: { type: 'integer', description: 'eval items this would change' },
        },
        required: ['title', 'evidence', 'root_cause', 'fix', 'affected'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT = {
  type: 'object',
  properties: {
    holds: { type: 'boolean' },
    reason: { type: 'string' },
    corrected_root_cause: { type: 'string' },
  },
  required: ['holds', 'reason'],
}

const results = await pipeline(
  CATEGORIES,
  c => agent(`${CONTEXT}\n\nYour category: ${c.prompt}\nReturn at most 2 findings, largest affected count first.`,
    { label: `analyze:${c.key}`, phase: 'Analyze', schema: FINDINGS }),
  (r, c) => parallel((r?.findings || []).map(f => () =>
    agent(`${CONTEXT}\n\nA colleague claims the root cause below. Try to REFUTE it by reading the code and data yourself. If you can't confirm it, set holds=false.\nTitle: ${f.title}\nEvidence: ${f.evidence}\nRoot cause: ${f.root_cause}\nProposed fix: ${f.fix}`,
      { label: `verify:${c.key}`, phase: 'Verify', schema: VERDICT })
      .then(v => ({ ...f, category: c.key, verdict: v })))),
)

const all = results.filter(Boolean).flat().filter(Boolean)
const kept = all.filter(f => f.verdict?.holds)
const dropped = all.filter(f => !f.verdict?.holds).map(f => ({ title: f.title, why: f.verdict?.reason }))
log(`${all.length} findings, ${kept.length} survived verification`)
if (!kept.length) return { case: CASE, report: 'No root cause survived verification.', kept, dropped }

phase('Rank')
const report = await agent(`${CONTEXT}\n\nVerified findings (JSON):\n${JSON.stringify(kept, null, 1)}\n\nWrite a prioritized fix plan for the case-timeline owner. Rank by expected metric gain (recall, date_accuracy, precision) per unit of effort, and merge overlapping findings. For each fix give: what to change (file/function), the expected effect with numbers from this eval, and a regression test to add. Flag anything that needs a human; for example, gold additions are human review only, and agents never edit eval/gold. Under 400 words.`,
  { label: 'rank', phase: 'Rank' })

return { case: CASE, report, kept, dropped }
