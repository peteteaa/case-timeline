#!/bin/bash
# Serve Qwen3.8-Flash-Next-FP4-Local with the community Spark vLLM container
# (OpenAI-compatible API at http://127.0.0.1:8001/v1, loopback only).
# Flags and env follow the model README's "NVIDIA DGX Spark" recipe; the checkpoint's
# compressed NVFP4 scales need this container until upstream vLLM supports them.
# Needs ~97 GB GPU memory: stop the Qwen3.6 server first.
set -e
mkdir -p $HOME/.cache/vllm $HOME/.cache/flashinfer $HOME/.triton $HOME/.cache/b12x
exec docker run --rm --name qwen38-vllm \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -v /home/dell/models/Qwen3.8-Flash-Next-FP4-Local:/models/qwen38:ro \
  -v $HOME/.cache/vllm:/root/.cache/vllm \
  -v $HOME/.cache/flashinfer:/root/.cache/flashinfer \
  -v $HOME/.triton:/root/.triton \
  -v $HOME/.cache/b12x:/root/.cache/b12x \
  -e VLLM_PLE_TABLE_MEMORY=disk \
  -e B12X_DISK_BACKEND=io_uring \
  -e CUTE_DSL_ARCH=sm_121a \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_POLICY_MODE=auto \
  -e VLLM_MXFP8_LM_HEAD=1 \
  -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e B12X_W4A16_A4_PREFILL=1 \
  --entrypoint vllm \
  eugr/spark-vllm-b12x:latest \
  serve /models/qwen38 \
    --served-model-name qwen3.8-flash-next \
    --host 127.0.0.1 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --quantization modelopt_mixed \
    --block-size 16 \
    --load-format b12x \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
    --gdn-decode-kernel b12x \
    --linear-backend b12x \
    --moe-backend b12x \
    --no-enable-flashinfer-autotune \
    --mm-encoder-tp-mode data \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
    --gpu-memory-utilization 0.8
