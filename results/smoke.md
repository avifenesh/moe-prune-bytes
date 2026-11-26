# Smoke test, 2026-11-23

Tiny random Qwen3-MoE (2 layers, 128 experts, hidden 512, real tokenizer). Plumbing only: the weights are random, so no number here says anything about compression quality.

Full receipts (logs, KL json, speed json, lm-eval samples, REAP report) are kept outside the repo.

| Step | Exit | Note |
|---|---|---|
| REAP report, prune 64 / 72, NVFP4, FP8_DYNAMIC | 0 | see `smoke_build.log` |
| hidden dump BF16 / NVFP4 / FP8 in vLLM 0.34.0 | 0 | first attempt of the BF16 run crashed in a second in-process engine (spawn guard); fixed by moving the self-check to its own call |
| KL vs source | 0 | lm_head identity check passed |
| speed (NVFP4 arm, N=2) | 0 | torch peak memory is meaningful: vLLM 0.34's extensible KV cache does not pre-allocate |
| lm-eval vllm backend, arc_challenge, arc_easy, mmlu, gsm8k_cot_llama, limit 4 | 0 | 60 sample files written; MMLU datasets now cached on the Spark |
| paired bootstrap over lm-eval samples | 0 | |
| hidden-state self-check vs vLLM prompt logprobs | 0 | 4 sequences: mean NLL matches to 1e-5, max per-token |diff| 0.004 |
