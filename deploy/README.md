# GB10 host setup scripts

Snapshots of the host-side scripts used to set up this DGX Spark (GB10) for Case Timeline.
They live outside the repo on the box (`~/models/`, `~/`); these copies are for reference
and reuse. They hard-code `/home/dell` paths.

| Script | On the box | What it does |
|---|---|---|
| `serve-qwen36.sh` | `~/models/serve-qwen36.sh` | Serves Qwen3.6-35B-A3B-NVFP4-Fast with vLLM as `qwen3.6-35b-a3b` on `172.17.0.1:8000` (docker0 only). `scripts/serve_llm.sh` is the repo's managed equivalent with health checks. |
| `serve-qwen38.sh` | `~/models/serve-qwen38.sh` | Serves Qwen3.8-Flash-Next in the `eugr/spark-vllm-b12x` container on `127.0.0.1:8001`. Needs ~97 GB GPU memory: stop the Qwen3.6 server first. |
| `nemoclaw-install.sh` | `~/nemoclaw-install.sh` | Runs the NemoClaw installer with docker group access, Express mode off and a fresh onboarding session, so it uses the existing vLLM instead of downloading its own. In onboarding, pick "Other OpenAI-compatible endpoint": `http://172.17.0.1:8000/v1`, model `qwen3.6-35b-a3b`, any API key. |
| `sandbox-forward-8000.sh` | `~/models/sandbox-forward-8000.sh` | **Obsolete.** socat forward from `172.17.0.1:8000` to a loopback-only vLLM. vLLM now binds `172.17.0.1` directly, so running this would clash with it. |

Only one model server fits in memory at a time. Never run two vLLM instances.
