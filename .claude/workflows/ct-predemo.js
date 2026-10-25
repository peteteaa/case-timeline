export const meta = {
  name: 'ct-predemo',
  description: 'Pre-demo go/no-go for case-timeline: lint+tests, model + sandbox relay, UI smoke test on a scratch port, git state, eval trend',
  whenToUse: 'Before showing the demo, or before a commit you want as the known-good point. No args.',
  phases: [
    { title: 'Checks', detail: 'independent checks in parallel' },
    { title: 'Verdict', detail: 'go/no-go with the fixes needed' },
  ],
}

const ROOT = '/home/dell/case-timeline'
const VENV = `${ROOT}/.venv`
const NOTE = 'Read-only unless a step says otherwise. Never stop or restart shared services (vLLM, the relay, OpenClaw, NemoClaw, or the admin UI unit ct-ui-8090).'

const CHECKS = [
  { key: 'tests', prompt: `In ${ROOT}: run ${VENV}/bin/ruff check . ; ${VENV}/bin/ruff format --check . ; ${VENV}/bin/python -m pytest -q. Report pass/fail counts and the first failure for each failing test. If \`git status --short\` shows uncommitted changes, say the results include work in progress.` },
  { key: 'model', prompt: `Check the model path. Run curl -s -m5 http://172.17.0.1:8000/v1/models (must list qwen3.6-35b-a3b). Send one tiny chat completion (max_tokens 20) and report its latency. Check the sandbox relay with curl -s -m5 http://172.18.0.1:8000/v1/models. Read the last 10 lines of \`sg team -c 'cat /srv/case-timeline/MACHINE_STATUS.md'\` for planned vLLM downtime.` },
  { key: 'ui', prompt: `Smoke-test the Spec 2 app on scratch port 8095. Never use 8090; it's the admin's UI. Pick the newest ${ROOT}/data/*/case.db written by Spec 2 code: its schema must match pipeline/store.py, and pre-Spec-2 dbs won't load. Copy it into a mktemp dir with sqlite3's .backup command, so no live results get touched. If there's no Spec 2 db, use an empty mktemp dir. Start the app in the background from ${ROOT} with the job worker OFF, so it can't pick up jobs or call the model: CT_RUN_WORKER=false CT_UI_PORT=8095 CT_DATA_DIR=<tmpdir> CT_CASE_ID=<case in that db, else demo> ${VENV}/bin/python -m app. Wait until it answers, then GET /, /cases/<case>, /cases/<case>/facts, /cases/<case>/review and /cases/<case>/documents. Report the status codes and the number of facts shown. ALWAYS kill the process you started and confirm that port 8095 is free again.` },
  { key: 'git', prompt: `In ${ROOT}: report the branch, \`git status --short\`, commits not pushed (git log @{u}..), whether HEAD matches origin, and any wf/* branches not merged into HEAD. Don't fetch, pull, push, or change anything.` },
  { key: 'eval', prompt: `Compare eval trends in ${ROOT}/data: for cases demo and ntsb, find the two newest eval_<case>.json files by mtime. Report precision, recall, f1 and date_accuracy for each (newest vs previous, plus the run dir), and flag any regression of more than 0.03. Spec 2 reports (they have unmatched_facts and strict_* fields) and pre-Spec-2 reports (unmatched_gold/per_doc) aren't directly comparable. Say so instead of treating a format change as a regression. Also check that n_gold matches the case's gold file (eval/gold/<case>_timeline.json or eval/gold/<case>.json). Spec 2's eval without --gold silently scores against eval/gold_timeline.json, so a report whose n_gold doesn't match is invalid and should be flagged.` },
]

phase('Checks')
const results = await parallel(CHECKS.map(c => () =>
  agent(`${NOTE}\n\n${c.prompt}`, { label: `check:${c.key}`, phase: 'Checks' }).then(r => ({ key: c.key, r }))))

phase('Verdict')
const got = results.filter(Boolean)
const missing = CHECKS.map(c => c.key).filter(k => !got.some(x => x.key === k))
if (missing.length) log(`checks with no result: ${missing.join(', ')}`)

const verdict = await agent(`Pre-demo check results for case-timeline:\n${got.map(x => `## ${x.key}\n${x.r}`).join('\n\n')}\n${missing.length ? `\nMissing checks (count them as unknown, not passed): ${missing.join(', ')}` : ''}\n\nGive a GO / NO-GO for demoing now. List the blockers, each with the exact command or owner to fix it. Then list the non-blocking risks (e.g. planned vLLM downtime, uncommitted work, NTSB noise). Under 250 words.`,
  { label: 'verdict', phase: 'Verdict' })

return { verdict, checks: Object.fromEntries(got.map(x => [x.key, x.r])), missing }
