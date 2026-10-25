#!/bin/bash
# Serve Qwen3.6-35B-A3B-NVFP4-Fast locally with vLLM (OpenAI-compatible API at http://172.17.0.1:8000/v1, docker0 bridge only: reachable from the host and NemoClaw sandbox, not from Wi-Fi/Tailscale).
# DGX Spark flags per the model README; parsers per Qwen's recommendations for tool calling.
# Qwen3.6 GDN linear-attention Triton kernel (fused_sigmoid_gating_delta_rule_update) hit an illegal
# memory access at ~10-16 concurrent requests (21:51 on 2026-10-25), so batching is capped (MAX_NUM_SEQS).
# flashinfer_b12x crashes (illegal memory access during CUDA-graph profiling) with vllm 0.32.0 +
# flashinfer 0.7.0.post1 on GB10: reproduced 19:16 and 21:09 on 2026-10-25. Default is cutlass.
export CUTE_DSL_ARCH=sm_121a
# FlashInfer JIT-compiles some kernels on first start: needs nvcc and ninja.
export CUDA_HOME=/usr/local/cuda
export PATH=/home/dell/.venvs/vllm/bin:$CUDA_HOME/bin:$PATH
exec /home/dell/.venvs/vllm/bin/vllm serve /home/dell/models/Qwen3.6-35B-A3B-NVFP4-Fast \
    --served-model-name qwen3.6-35b-a3b \
    --host 172.17.0.1 --port 8000 \
    --moe-backend "${MOE_BACKEND:-flashinfer_cutlass}" \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.6 \
    --max-num-seqs "${MAX_NUM_SEQS:-8}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
