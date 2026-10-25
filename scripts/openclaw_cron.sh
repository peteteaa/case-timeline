#!/usr/bin/env bash
# Schedule case-timeline automations in OpenClaw (idempotent via --declaration-key):
#   - scan every 5 minutes: scan_case -> poll job_status -> get_changes -> content-free alert
#   - daily digest at 08:00 (host local time unless CT_TZ is set)
#
#   scripts/openclaw_cron.sh            # create/update the two automations
#   scripts/openclaw_cron.sh --dry-run  # print the commands only
#
# Env overrides: OPENCLAW, CT_CASE_ID (default ntsb), CT_TZ (IANA tz),
# CT_OPENCLAW_AGENT (default case-timeline: the shell-less agent),
# CT_ALERT_CHANNEL (unset = no delivery; alerts stay on the dashboard), CT_ALERT_TO (optional).
set -euo pipefail

OPENCLAW="${OPENCLAW:-$HOME/.openclaw/bin/openclaw}"
CASE_ID="${CT_CASE_ID:-ntsb}"
AGENT="${CT_OPENCLAW_AGENT:-case-timeline}"
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

if [[ -n "${CT_ALERT_CHANNEL:-}" ]]; then
  DELIVERY=(--announce --best-effort-deliver --channel "$CT_ALERT_CHANNEL")
  [[ -n "${CT_ALERT_TO:-}" ]] && DELIVERY+=(--to "$CT_ALERT_TO")
else
  DELIVERY=(--no-deliver)
fi
TZ_ARGS=()
[[ -n "${CT_TZ:-}" ]] && TZ_ARGS=(--tz "$CT_TZ")

SCAN_MSG="Use the case-timeline skill. Scheduled scan for case '$CASE_ID': call scan_case, \
poll job_status every 5 seconds until done or failed, then call get_changes once. \
If there are new alerts, reply with ONE line containing only the case name, the alert types \
and counts (deadline dates always marked 'needs attorney review'), and \
http://127.0.0.1:8090/alerts. Include NO case content: no names, quotes, dates, documents or \
descriptions. If nothing is new, reply exactly NO_REPLY."

DIGEST_MSG="Use the case-timeline skill. Daily digest for case '$CASE_ID': call daily_digest and \
reply with only the case name, the counts, and http://127.0.0.1:8090/alerts. \
Include NO case content."

echo "==> Scan every 5 minutes"
run "$OPENCLAW" cron add \
  --agent "$AGENT" \
  --name "case-timeline scan ($CASE_ID)" \
  --declaration-key "case-timeline:scan:$CASE_ID" \
  --description "Scan the case evidence folder and alert on new contradictions/deadlines" \
  --every 5m \
  --session isolated \
  --light-context \
  --timeout-seconds 900 \
  --message "$SCAN_MSG" \
  "${DELIVERY[@]}"

echo "==> Daily digest at 08:00"
run "$OPENCLAW" cron add \
  --agent "$AGENT" \
  --name "case-timeline digest ($CASE_ID)" \
  --declaration-key "case-timeline:digest:$CASE_ID" \
  --description "Counts-only daily digest for the case" \
  --cron "0 8 * * *" \
  --exact \
  "${TZ_ARGS[@]}" \
  --session isolated \
  --light-context \
  --message "$DIGEST_MSG" \
  "${DELIVERY[@]}"

run "$OPENCLAW" cron list --query case-timeline
