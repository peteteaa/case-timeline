#!/usr/bin/env bash
# Fail if any TCP listener other than SSH (port 22) is bound to a non-loopback address.
# Exception: vLLM on the docker0 bridge (172.17.0.1:8000), reachable only from the host and sandbox.
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
  if [[ "${addr}" == "172.17.0.1" && "${port}" == "8000" ]]; then
    continue
  fi
  echo "FAIL: ${local} is listening on a non-loopback address" >&2
  bad=1
done < <(ss -Hltn | awk '{print $4}')

if [[ "${bad}" -ne 0 ]]; then
  echo "Bind services to 127.0.0.1 and reach them over an SSH tunnel." >&2
  exit 1
fi
echo "OK: only loopback listeners (plus ssh:22 and vLLM on 172.17.0.1:8000)."
