#!/usr/bin/env bash
# Register the case-timeline stdio MCP server and skill with OpenClaw (idempotent).
#
#   scripts/register_openclaw.sh            # register + install skill + verify
#   scripts/register_openclaw.sh --dry-run  # print the commands only
#
# Env overrides: OPENCLAW (CLI path), CT_DATA_DIR, CT_EVIDENCE_DIR, CT_CASE_ID,
# CT_SKILL_SCOPE=global (install into the shared skills dir instead of the agent workspace),
# CT_OPENCLAW_AGENT (agent whose workspace gets the skill; default case-timeline).
# Everything stays local: the server is spawned over stdio and binds no network port. The MCP
# server only reads the database and queues jobs; the app server (`make app`) runs them.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW="${OPENCLAW:-$HOME/.openclaw/bin/openclaw}"
SERVER_NAME="case-timeline"
SKILL_DIR="$PROJECT/skill"
DATA_DIR="${CT_DATA_DIR:-$PROJECT/data}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

run() {
  if (( DRY_RUN )); then
    printf '+'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

[[ -x "$OPENCLAW" ]] || { echo "openclaw CLI not found at $OPENCLAW" >&2; exit 1; }
# Use the project venv directly: `uv run` would re-sync (and drop optional extras) on every spawn.
PYTHON="$PROJECT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "missing $PYTHON; run 'make setup' first" >&2; exit 1; }
[[ -f "$SKILL_DIR/SKILL.md" ]] || { echo "missing $SKILL_DIR/SKILL.md" >&2; exit 1; }

# Server definition. `mcp set` replaces any existing entry, so re-running is safe.
SERVER_JSON="$(
  PYTHON="$PYTHON" PROJECT="$PROJECT" DATA_DIR="$DATA_DIR" python3 - <<'PY'
import json, os
env = {"CT_DATA_DIR": os.environ["DATA_DIR"]}
for key in ("CT_EVIDENCE_DIR", "CT_CASE_ID"):
    if os.environ.get(key):
        env[key] = os.environ[key]
print(json.dumps({
    "command": os.environ["PYTHON"],
    "args": ["-m", "mcp_server"],
    "cwd": os.environ["PROJECT"],
    "env": env,
}))
PY
)"

echo "==> Registering MCP server '$SERVER_NAME'"
run "$OPENCLAW" mcp set "$SERVER_NAME" "$SERVER_JSON"

echo "==> Installing skill from $SKILL_DIR"
SKILL_ARGS=(skills install "$SKILL_DIR" --as case-timeline --force --agent "${CT_OPENCLAW_AGENT:-case-timeline}")
[[ "${CT_SKILL_SCOPE:-}" == "global" ]] && SKILL_ARGS+=(--global)
run "$OPENCLAW" "${SKILL_ARGS[@]}"

echo "==> Verifying"
run "$OPENCLAW" mcp list
run "$OPENCLAW" mcp doctor --probe
