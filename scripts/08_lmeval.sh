#!/usr/bin/env bash
# Run the pinned lm-eval suite on one arm inside the mpb-eval image (vLLM 0.34.0, lm-eval 0.4.13).
# usage: [SKIP_GSM=1] 08_lmeval.sh MODEL_DIR OUT_DIR
# Loglikelihood tasks: arc_challenge, arc_easy (0-shot), mmlu (0-shot), no chat template.
# Generative task: gsm8k_cot_llama (8-shot CoT, chat template, multiturn fewshot), greedy.
set -euo pipefail
M=$1
O=$2
mkdir -p "$O"
MA="pretrained=$M,dtype=auto,gpu_memory_utilization=0.80,max_model_len=4096,enable_prefix_caching=True,seed=0"
lm_eval --model vllm --model_args "$MA" \
  --tasks arc_challenge,arc_easy,mmlu --num_fewshot 0 --batch_size auto \
  --log_samples --output_path "$O/ll" --seed 0 > "$O/ll.log" 2>&1
[ "${SKIP_GSM:-0}" = 1 ] && { echo done; exit 0; }
lm_eval --model vllm --model_args "$MA" \
  --tasks gsm8k_cot_llama --apply_chat_template --fewshot_as_multiturn --batch_size auto \
  --gen_kwargs "temperature=0,max_gen_toks=2048" \
  --log_samples --output_path "$O/gsm" --seed 0 > "$O/gsm.log" 2>&1
echo done
