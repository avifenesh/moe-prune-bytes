#!/usr/bin/env bash
# Phase A suite for one arm, in the pinned image (vLLM 0.34.0, lm-eval 0.4.13).
# Generation with Qwen3-30B-A3B-Instruct-2507's recommended sampling
# (temperature 0.7, top_p 0.8, top_k 20), seed 0, one sample per item, chat template:
#   code: humaneval_instruct (164), mbpp_plus_instruct_fix (378, fixed extractor, lm_tasks/), executed (unsafe code allowed)
#   math: minerva_math500 (500), gsm8k_cot_llama (1,319)
#   instruction following: ifeval (541)
#   tool use: BFCL v3 simple/multiple/parallel/irrelevance (1,040), scripts/26_bfcl.py
# Multilingual, loglikelihood: global_mmlu_full_he (all 14,042), plus ar, ja, ru, de, es at
#   20 questions per subject (1,140 each).
# usage: 27_gen_suite.sh MODEL_DIR OUT_DIR
set -uo pipefail
M=$1
O=$2
mkdir -p "$O"
export HF_ALLOW_CODE_EVAL=1
MA="pretrained=$M,dtype=auto,gpu_memory_utilization=0.80,max_model_len=8192,enable_prefix_caching=True,seed=0"
GK="do_sample=True,temperature=0.7,top_p=0.8,top_k=20,max_gen_toks=4096"
LIM=()
# smoke tests only: LIMIT=n caps every task at n items
if [ -n "${LIMIT:-}" ]; then LIM=(--limit "$LIMIT"); fi
[ -d "$O/gen" ] || lm_eval --model vllm --model_args "$MA" \
  --tasks humaneval_instruct,mbpp_plus_instruct_fix,minerva_math500,gsm8k_cot_llama,ifeval --include_path ~/mpb/lm_tasks \
  --apply_chat_template --fewshot_as_multiturn --batch_size auto --gen_kwargs "$GK" \
  --confirm_run_unsafe_code --log_samples --output_path "$O/gen" --seed 0 "${LIM[@]}" > "$O/gen.log" 2>&1
echo "gen $?"
[ -d "$O/he" ] || lm_eval --model vllm --model_args "$MA" --tasks global_mmlu_full_he --num_fewshot 0 \
  --batch_size auto --log_samples --output_path "$O/he" --seed 0 "${LIM[@]}" > "$O/he.log" 2>&1
echo "he $?"
[ -d "$O/ml" ] || lm_eval --model vllm --model_args "$MA" \
  --tasks global_mmlu_full_ar,global_mmlu_full_ja,global_mmlu_full_ru,global_mmlu_full_de,global_mmlu_full_es \
  --num_fewshot 0 --limit 20 --batch_size auto --log_samples --output_path "$O/ml" --seed 0 > "$O/ml.log" 2>&1
echo "ml $?"
[ -f "$O/bfcl.jsonl" ] || BFCL_LIMIT="${LIMIT:-}" python3 ~/mpb/scripts/26_bfcl.py "$M" ~/mpb/bfcl "$O/bfcl.jsonl" > "$O/bfcl.log" 2>&1
echo "bfcl $?"
echo SUITE_DONE
