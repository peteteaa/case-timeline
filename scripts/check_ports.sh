#!/usr/bin/env bash
# Fail if any TCP listener other than SSH (port 22) is bound to a non-loopback address.
# Exceptions (exact address:port pairs only, nothing broader):
#   - vLLM on the docker0 bridge (172.17.0.1:8000), reachable only from the host and sandbox;
#   - the NemoClaw socat relay on the OpenShell docker bridge (172.18.0.1:8000), which only
#     forwards to 172.17.0.1:8000 so the OpenShell sandbox can reach the same vLLM.
set -euo pipefail

echo "Listening TCP sockets:"
ss -Hltn | awk '{print "  " $4}'

bad=0
while read -r local; do
  [[ -z "${local}" ]] && continue
  addr="${local%:*}"
  port="${local##*:}"
  addr="${addr#[}"; addr="${addr%]}"; addr="${addr%%%*}"
  case "${addr}" in
    127.*|::1|::ffff:127.*) continue ;;
  esac
  if [[ "${port}" == "22" ]]; then
    continue
  fi
  if [[ "${port}" == "8000" && ( "${addr}" == "172.17.0.1" || "${addr}" == "172.18.0.1" ) ]]; then
    continue
  fi
  echo "FAIL: ${local} is listening on a non-loopback address" >&2
  bad=1
done < <(ss -Hltn | awk '{print $4}')

if [[ "${bad}" -ne 0 ]]; then
  echo "Bind services to 127.0.0.1 and reach them over an SSH tunnel." >&2
  exit 1
fi
echo "OK: only loopback listeners (plus ssh:22, vLLM on 172.17.0.1:8000 and its socat relay on 172.18.0.1:8000)."
