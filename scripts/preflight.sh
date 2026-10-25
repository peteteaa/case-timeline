#!/usr/bin/env bash
# Environment checks for the GB10 host. Exits non-zero if a required check fails.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; fail=1; }

echo "Preflight:"

arch="$(uname -m)"
[[ "${arch}" == "aarch64" ]] && ok "arch ${arch}" || bad "arch ${arch} (expected aarch64)"

free_gb="$(df -BG --output=avail "${ROOT}" | tail -1 | tr -dc '0-9')"
if (( free_gb >= 50 )); then ok "disk ${free_gb}G free"
elif (( free_gb >= 10 )); then warn "disk ${free_gb}G free (low)"
else bad "disk ${free_gb}G free"; fi

# Internet is only needed for setup (uv sync, model downloads); case data never leaves the box.
if curl -fsS --max-time 5 -o /dev/null https://pypi.org/simple/ 2>/dev/null; then
  ok "internet reachable (setup only)"
else
  warn "no internet (fine once deps and models are installed)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
  [[ -n "${gpu}" ]] && ok "GPU ${gpu}" || bad "nvidia-smi present but no GPU reported"
else
  bad "nvidia-smi not found"
fi

if command -v uv >/dev/null 2>&1; then ok "$(uv --version)"; else bad "uv not found"; fi

if command -v openclaw >/dev/null 2>&1 || [[ -x "${HOME}/.openclaw/bin/openclaw" ]]; then
  ok "openclaw present"
else
  warn "openclaw not found (only needed for messaging alerts)"
fi

[[ -x /home/dell/.venvs/vllm/bin/vllm ]] && ok "vllm binary" || warn "vllm binary missing"
if curl -fsS --max-time 2 http://172.17.0.1:8000/v1/models >/dev/null 2>&1; then
  ok "LLM serving on 172.17.0.1:8000"
else
  warn "LLM not running (make llm)"
fi

exit "${fail}"
