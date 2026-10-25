#!/usr/bin/env bash
# Schedule case-timeline automations in OpenClaw (idempotent via --declaration-key):
#   - scan every 5 minutes: scan_case -> poll job_status -> get_pending_alerts
#     (-> announce content-free alerts + mark_alerts_delivered, only with a channel)
#   - daily digest at 08:00 (host local time unless CT_TZ is set)
#
#   scripts/openclaw_cron.sh            # create/update the two automations
#   scripts/openclaw_cron.sh --dry-run  # print the commands only
#
# Demo mode has NO messaging channel (SPEC §1, §15): by default nothing is announced
# (--no-deliver) and alerts stay in the web UI. Set CT_ALERT_CHANNEL (e.g. telegram) and
# optionally CT_ALERT_TO to announce content-free alerts on that channel; the sandbox policy
# must then allow that one endpoint.
#
# Env overrides: OPENCLAW, CT_CASE_ID (default ntsb, the live case), CT_TZ (IANA tz), CT_SCAN_EVERY (default 5m),
# CT_OPENCLAW_AGENT (default case-timeline: the shell-less agent),
# CT_ALERT_CHANNEL (unset = no channel), CT_ALERT_TO (channel destination, optional).
set -euo pipefail

OPENCLAW="${OPENCLAW:-$HOME/.openclaw/bin/openclaw}"
CASE_ID="${CT_CASE_ID:-ntsb}"
EVERY="${CT_SCAN_EVERY:-5m}"
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

[[ -x "$OPENCLAW" ]] || (( DRY_RUN )) || { echo "openclaw CLI not found at $OPENCLAW" >&2; exit 1; }

if [[ -n "${CT_ALERT_CHANNEL:-}" ]]; then
  DELIVERY=(--announce --channel "$CT_ALERT_CHANNEL")
  [[ -n "${CT_ALERT_TO:-}" ]] && DELIVERY+=(--to "$CT_ALERT_TO")
  echo "Alerts will be announced on channel '$CT_ALERT_CHANNEL' (content-free only)."
  SCAN_MSG="Use the case-timeline skill. Scheduled scan for case '$CASE_ID' WITH a messaging \
channel: call scan_case, poll job_status every 5 seconds until status is done or failed, then \
call get_pending_alerts. Reply with each alert's 'message' field exactly, one per line, and \
nothing else, then call mark_alerts_delivered with those alert_ids. Never include case content \
or anything under 'internal'. If there are no alerts, reply exactly NO_REPLY."
  DIGEST_MSG="Use the case-timeline skill. Daily digest for case '$CASE_ID' WITH a messaging \
channel: call daily_digest and reply with its 'message' field exactly and nothing else. Never \
include the digest text or any case content."
else
  DELIVERY=(--no-deliver)
  echo "No messaging channel (demo mode): alerts stay in the web UI."
  SCAN_MSG="Use the case-timeline skill. Scheduled scan for case '$CASE_ID' with NO messaging \
channel: call scan_case, poll job_status every 5 seconds until status is done or failed, then \
call get_pending_alerts and stop. Do not call mark_alerts_delivered; the web UI shows alerts. \
Reply exactly NO_REPLY."
  DIGEST_MSG="Use the case-timeline skill. Daily digest for case '$CASE_ID' with NO messaging \
channel: call daily_digest and reply exactly NO_REPLY."
fi
TZ_ARGS=()
[[ -n "${CT_TZ:-}" ]] && TZ_ARGS=(--tz "$CT_TZ")

echo "==> Scan every $EVERY"
run "$OPENCLAW" cron add \
  --agent "$AGENT" \
  --name "case-timeline scan ($CASE_ID)" \
  --declaration-key "case-timeline:scan:$CASE_ID" \
  --description "Queue a scan of the case folder and check for new alerts" \
  --every "$EVERY" \
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
  --description "Daily digest for the case (content-free when announced)" \
  --cron "0 8 * * *" \
  --exact \
  "${TZ_ARGS[@]}" \
  --session isolated \
  --light-context \
  --message "$DIGEST_MSG" \
  "${DELIVERY[@]}"

run "$OPENCLAW" cron list --query case-timeline
