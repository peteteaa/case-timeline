#!/bin/bash
# Serve Qwen3.6-35B-A3B-NVFP4-Fast locally with vLLM (OpenAI-compatible API at http://172.17.0.1:8000/v1, docker0 bridge only: reachable from the host and NemoClaw sandbox, not from Wi-Fi/Tailscale).
# DGX Spark flags per the model README; parsers per Qwen's recommendations for tool calling.
export CUTE_DSL_ARCH=sm_121a
# FlashInfer JIT-compiles some kernels on first start: needs nvcc and ninja.
export CUDA_HOME=/usr/local/cuda
export PATH=/home/dell/.venvs/vllm/bin:$CUDA_HOME/bin:$PATH
exec /home/dell/.venvs/vllm/bin/vllm serve /home/dell/models/Qwen3.6-35B-A3B-NVFP4-Fast \
    --served-model-name qwen3.6-35b-a3b \
    --host 172.17.0.1 --port 8000 \
    --moe-backend "${MOE_BACKEND:-flashinfer_b12x}" \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.6 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
