export const meta = {
  name: 'ct-fix',
  description: 'Implement case-timeline fixes in parallel git worktrees (tests + lint offline), review each branch, optionally live-eval them one at a time',
  whenToUse: 'To land several independent fixes at once. args: {tasks: [{slug, goal}], liveEval: {case: "ntsb", baselineRun: "data/<run>"} | null}',
  phases: [
    { title: 'Implement', detail: 'one worktree + branch wf/<slug> per task' },
    { title: 'Review', detail: 'independent review of each branch diff' },
    { title: 'Live eval', detail: 'serial: one scan+eval at a time on the shared vLLM' },
  ],
}

const ROOT = '/home/dell/case-timeline'
const VENV = `${ROOT}/.venv`
const A = args || {}
const TASKS = (A.tasks || []).filter(t => t && t.slug && t.goal)
if (!TASKS.length) return { error: 'args.tasks is empty: pass [{slug, goal}, ...]' }

const EVIDENCE = {
  demo: `${ROOT}/tests/fixtures/case_demo`,
  ntsb: '/home/dell/case-data/ntsb_cen10fa443/evidence',
}

const RULES = `Rules:
- Don't touch eval/gold/*, eval/docs/* or the evidence folders. Gold additions are human review only.
- Keep the Spec 2 contracts (pipeline/schema.py, documented in docs/SPEC.md and INTERFACES.md). If your goal really requires changing one, update the docs in the same commit and say so.
- Offline only: never call the LLM server and never run tests marked llm. Don't start servers, and don't bind ports 8080, 8081-8084 or 8090.
- Use the main venv; don't create a new one or run uv sync: ${VENV}/bin/ruff check . && ${VENV}/bin/ruff format --check . && ${VENV}/bin/python -m pytest -q (run from the repo root you are in; the project is not installed, so imports resolve to your checkout).
- Never push. Never modify ${ROOT} outside your own worktree.`

const IMPL = {
  type: 'object',
  properties: {
    branch: { type: 'string', description: 'wf/<slug>, or empty if nothing was committed' },
    commit: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
    tests: { type: 'string', description: 'pytest summary line, plus lint result' },
    summary: { type: 'string' },
    risks: { type: 'string' },
  },
  required: ['branch', 'commit', 'files', 'tests', 'summary', 'risks'],
}

const REVIEW = {
  type: 'object',
  properties: {
    approve: { type: 'boolean' },
    blocking: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['approve', 'blocking', 'notes'],
}

const done = await pipeline(
  TASKS,
  t => agent(`You are working in a fresh git worktree of ${ROOT} (based on its HEAD; uncommitted work in the main tree is NOT included).
Goal: ${t.goal}
1. Run \`git switch -c wf/${t.slug}\`.
2. Add or adjust a test that fails before your change.
3. Make the smallest change that achieves the goal.
4. Run lint and the offline tests until they're green.
5. Commit with a clear message.
${RULES}
Return the branch, commit sha, changed files, the test summary, and the risks.`,
    { label: `impl:${t.slug}`, phase: 'Implement', isolation: 'worktree', schema: IMPL }),
  (impl, t) => {
    if (!impl || !impl.branch) return { task: t, impl, review: null }
    return agent(`Review branch ${impl.branch} in ${ROOT}: run \`git -C ${ROOT} diff HEAD...${impl.branch}\` and \`git -C ${ROOT} log HEAD..${impl.branch}\`. Read-only: don't check it out in the main tree or edit anything.
Goal it claims to implement: ${t.goal}
Check: does it achieve the goal? Are there regressions or unhandled edge cases? Does it keep the INTERFACES.md contracts? Does a test cover it, and would that test fail without the change? Are there any edits to eval/gold or evidence (blocking)? Does it break loopback-only binding or llm_allowed_hosts (blocking)?
Author's summary: ${impl.summary}\nAuthor's risks: ${impl.risks}`,
      { label: `review:${t.slug}`, phase: 'Review', schema: REVIEW })
      .then(review => ({ task: t, impl, review }))
  },
)

const items = done.filter(Boolean)
const approved = items.filter(x => x.review?.approve)
log(`${approved.length}/${TASKS.length} branches approved`)

const live = A.liveEval
const evals = []
if (live && live.case && approved.length) {
  phase('Live eval')
  const ev = EVIDENCE[live.case]
  if (!ev) log(`no evidence path known for case ${live.case}; live eval skipped`)
  // Serial on purpose: vLLM is shared by the whole team. Only one live scan runs at a time.
  for (const x of ev ? approved : []) {
    const slug = x.task.slug
    const r = await agent(`Live scan + eval of branch ${x.impl.branch} for case "${live.case}" on the SHARED model server. Be a good neighbour:
1. Preflight: \`curl -s -m5 http://172.17.0.1:8000/v1/models\` must list qwen3.6-35b-a3b. If it doesn't, stop and report "skipped: model down". Run \`pgrep -af 'pipeline.jobs scan|make scan'\`. If another live scan is running, re-check every 60s for up to 10 min, then skip. Read \`sg team -c 'tail -n 15 /srv/case-timeline/MACHINE_STATUS.md'\`. If a GPU window or vLLM restart is announced as in progress, skip.
2. Announce: \`sg team -c "printf '%s\\n' '- <HH:MM>: ct-fix live eval of ${x.impl.branch} (case ${live.case}) started; heavy use of :8000' >> /srv/case-timeline/MACHINE_STATUS.md"\`
3. WT=$(mktemp -d)/wt; git -C ${ROOT} worktree add "$WT" ${x.impl.branch}; cd "$WT"
4. Run with env CT_EVIDENCE_DIR=${ev} CT_DATA_DIR=${ROOT}/data/wf-${slug} CT_CASE_ID=${live.case} (plus CT_CASE_TZ / CT_CASE_INCIDENT_DATE only if pipeline/config.py defines them and the case README or MACHINE_STATUS says what dell-98 uses): \`${VENV}/bin/python -m pipeline.jobs scan --case ${live.case}\`, then score with an EXPLICIT gold file. Without one, the eval silently falls back to eval/gold_timeline.json, which is the wrong gold: use eval/gold/${live.case}_timeline.json if it exists in the branch, otherwise eval/gold/${live.case}.json, i.e. \`${VENV}/bin/python -m eval --case ${live.case} --gold <that file>\`. Report which gold file you used.
5. Always clean up, even on failure: \`git -C ${ROOT} worktree remove --force "$WT"\`, and log "finished" in MACHINE_STATUS the same way as step 2.
6. Compare ${ROOT}/data/wf-${slug}/eval_${live.case}.json with the baseline ${live.baselineRun ? `${ROOT}/${live.baselineRun}/eval_${live.case}.json` : `newest other ${ROOT}/data/*/eval_${live.case}.json`}: precision, recall, f1, date_accuracy, before -> after.`,
      { label: `eval:${slug}`, phase: 'Live eval' })
    evals.push({ slug, result: r })
  }
}

return {
  branches: items.map(x => ({
    slug: x.task.slug,
    branch: x.impl?.branch || null,
    commit: x.impl?.commit || null,
    tests: x.impl?.tests || null,
    approved: !!x.review?.approve,
    blocking: x.review?.blocking || [],
    notes: x.review?.notes || '',
  })),
  liveEvals: evals,
  next: 'Merge approved wf/* branches yourself (or ask the project owner). Nothing was pushed.',
}
