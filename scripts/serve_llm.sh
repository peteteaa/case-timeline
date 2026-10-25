#!/usr/bin/env bash
# Start the local vLLM server (OpenAI-compatible) at http://172.17.0.1:8000/v1.
# Binds the docker0 bridge only, so the host and the NemoClaw sandbox can reach it but Wi-Fi and
# Tailscale cannot. Never change --host to 0.0.0.0. Use an SSH tunnel for remote access.
#
# Env overrides:
#   MOE_BACKEND   default flashinfer_b12x (needs CUTE_DSL_ARCH=sm_121a on GB10).
#                 Fallback if the engine fails to initialise: MOE_BACKEND=flashinfer_cutlass
#   MODEL_DIR     default /home/dell/models/Qwen3.6-35B-A3B-NVFP4-Fast
#   VLLM_BIN      default /home/dell/.venvs/vllm/bin/vllm
#   MAX_MODEL_LEN default 65536
#   FOREGROUND=1  run in the foreground instead of backgrounding with nohup
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/dell/models/Qwen3.6-35B-A3B-NVFP4-Fast}"
VLLM_BIN="${VLLM_BIN:-/home/dell/.venvs/vllm/bin/vllm}"
SERVED_NAME="${SERVED_NAME:-qwen3.6-35b-a3b}"
HOST="${HOST:-172.17.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MOE_BACKEND="${MOE_BACKEND:-flashinfer_b12x}"
LOG="${ROOT}/data/vllm.log"

export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
# FlashInfer JIT-compiles some kernels on first start: needs nvcc and ninja on PATH.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$(dirname "${VLLM_BIN}"):${CUDA_HOME}/bin:${PATH}"

if curl -fsS --max-time 2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "vLLM already serving on ${HOST}:${PORT}:"
  curl -fsS --max-time 2 "http://${HOST}:${PORT}/v1/models"; echo
  exit 0
fi
if ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
  echo "ERROR: port ${PORT} is in use but not answering /v1/models (still loading?)." >&2
  echo "Check ${LOG} or 'ss -ltnp sport = :${PORT}'." >&2
  exit 1
fi
[[ -x "${VLLM_BIN}" ]] || { echo "ERROR: vllm not found at ${VLLM_BIN}" >&2; exit 1; }
[[ -d "${MODEL_DIR}" ]] || { echo "ERROR: model dir not found: ${MODEL_DIR}" >&2; exit 1; }

mkdir -p "${ROOT}/data"
CMD=(
  "${VLLM_BIN}" serve "${MODEL_DIR}"
  --served-model-name "${SERVED_NAME}"
  --host "${HOST}" --port "${PORT}"
  --moe-backend "${MOE_BACKEND}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization 0.6
  --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
)

echo "Starting vLLM (moe-backend=${MOE_BACKEND}); log: ${LOG}"
if [[ "${FOREGROUND:-0}" == "1" ]]; then
  "${CMD[@]}" 2>&1 | tee -a "${LOG}"
  exit "${PIPESTATUS[0]}"
fi
nohup "${CMD[@]}" >>"${LOG}" 2>&1 &
echo "pid $!  (first load can take several minutes)"
echo "wait:  until curl -fsS http://${HOST}:${PORT}/v1/models; do sleep 5; done"
echo "If it fails to start, retry with: MOE_BACKEND=flashinfer_cutlass $0"
