#!/usr/bin/env bash
# Run a command in the pinned eval image with ~/mpb and ~/models mounted.
# usage: dock.sh NAME CMD...
set -euo pipefail
name=$1
shift
extra=()
# optional patched vLLM model file (per-layer expert counts for Qwen3-MoE)
if [ -n "${VLLM_QWEN3_PATCH:-}" ]; then
  extra=(-v "$VLLM_QWEN3_PATCH:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py:ro")
fi
exec docker run --rm --name "$name" --gpus all --ipc=host --network host "${extra[@]}" \
  -v ~/mpb:~/mpb -v ~/models:~/models \
  -v ~/mpb/hfcache:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" -e TOKENIZERS_PARALLELISM=false \
  -w ~/mpb "${IMAGE:-mpb-eval:v0.34.0}" "$@"
