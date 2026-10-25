export const meta = {
  name: 'ct-review',
  description: 'Review case-timeline changes across correctness, contracts/tests, locality/security and shared-machine safety; verify each finding',
  whenToUse: 'Before committing or merging. args: {target: "working-tree" | "<branch>" | "<rev>..<rev>"} (default working-tree)',
  phases: [
    { title: 'Find', detail: 'one reviewer per dimension' },
    { title: 'Verify', detail: 'a skeptic tries to refute each finding' },
  ],
}

const ROOT = '/home/dell/case-timeline'
const TARGET = (args && args.target) || 'working-tree'
const DIFF = TARGET === 'working-tree'
  ? `git -C ${ROOT} diff HEAD (plus \`git -C ${ROOT} status --short\` for new files)`
  : TARGET.includes('..') ? `git -C ${ROOT} diff ${TARGET}` : `git -C ${ROOT} diff HEAD...${TARGET}`

const BASE = `Review this case-timeline change: get it with ${DIFF}. Read surrounding code as needed. Read-only: don't edit, check out, or run servers. You may run the offline tests: ${ROOT}/.venv/bin/python -m pytest -q.`

const DIMENSIONS = [
  { key: 'correctness', prompt: 'Correctness and regressions: logic errors, off-by-one/date/timezone mistakes, broken edge cases (empty docs, scanned PDFs, year-only dates), error handling, behaviour changes callers depend on.' },
  { key: 'contracts', prompt: 'Contracts and tests: does it keep the Spec 2 contracts (pipeline/schema.py, docs/SPEC.md, INTERFACES.md) and the config fields? Is each behaviour change covered by a test that would fail without it? Are fixtures and gold untouched (eval/gold edits are forbidden)?' },
  { key: 'locality', prompt: 'Locality and security: the app (python -m app) and the MCP server must bind loopback only (ui_host); the LLM base URL must stay within llm_allowed_hosts; the MCP server never prints to stdout; case_id/doc_id/path validation must prevent traversal, including document upload and view routes; evidence_dir is never written; no secrets in code or logs.' },
  { key: 'shared', prompt: 'Shared-machine safety: the GB10 is shared by several people and agents. Check LLM concurrency and timeouts against the single shared vLLM at 172.17.0.1:8000, port choices (8080 = OpenShell gateway, 8081-8084 = teammates, 8090 = admin UI), anything that starts or stops shared services, and large downloads or GPU use.' },
]

const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: { type: 'string' },
          line: { type: 'integer' },
          severity: { type: 'string', enum: ['blocking', 'major', 'minor'] },
          issue: { type: 'string' },
          scenario: { type: 'string', description: 'concrete input/state -> wrong result' },
        },
        required: ['file', 'severity', 'issue', 'scenario'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT = {
  type: 'object',
  properties: { real: { type: 'boolean' }, reason: { type: 'string' } },
  required: ['real', 'reason'],
}

const results = await pipeline(
  DIMENSIONS,
  d => agent(`${BASE}\n\nYour dimension: ${d.prompt}\nReport only issues introduced or exposed by this change. Return at most 3, worst first; an empty list is fine.`,
    { label: `find:${d.key}`, phase: 'Find', schema: FINDINGS }),
  (r, d) => parallel((r?.findings || []).map(f => () =>
    agent(`${BASE}\n\nTry to REFUTE this review finding: check the code and, if you can, reproduce the scenario. If you can't show it's real, set real=false.\n${JSON.stringify(f)}`,
      { label: `verify:${d.key}`, phase: 'Verify', schema: VERDICT })
      .then(v => ({ ...f, dimension: d.key, verdict: v })))),
)

const all = results.filter(Boolean).flat().filter(Boolean)
const confirmed = all.filter(f => f.verdict?.real)
const order = { blocking: 0, major: 1, minor: 2 }
confirmed.sort((a, b) => order[a.severity] - order[b.severity])
log(`${confirmed.length} of ${all.length} findings confirmed`)

return {
  target: TARGET,
  confirmed: confirmed.map(f => ({ severity: f.severity, where: `${f.file}${f.line ? ':' + f.line : ''}`, issue: f.issue, scenario: f.scenario, dimension: f.dimension })),
  refuted: all.filter(f => !f.verdict?.real).map(f => ({ issue: f.issue, why: f.verdict?.reason })),
}
