# moe-prune-bytes

Fewer experts or fewer bits? A matched-bytes study of MoE compression on Qwen3-30B-A3B.

Status: finished. Six main arms, then follow-ups on expert selection (F/G/H), healing, merging (M) and softmax layer budgets (R/R08), then Phase A: a wider evaluation, calibration on a domain mix and four heals (HB, HQ, SDb, SDa). Results below and in `results/STATUS.md`.

## Result

At the same weight bytes, fewer bits beat fewer experts by a wide margin.

All 128 experts in NVFP4 (B, 18.09 GB) stay close to the BF16 source. KL is 0.066 and MMLU drops 1.2 points. Pruning to 72 experts and storing them in FP8 (D, 18.50 GB, same expert bytes) costs 14.0 MMLU points, and its KL is 6 times higher. B beats D on MMLU by 12.8 points, 95% CI [12.1, 13.5], paired over 14,042 questions.

GSM8K hides all of this. Every arm, including 64 experts in NVFP4 at 9.9 GB, scores within 1 point of the source on GSM8K CoT, and no difference vs S is significant. The same pruned models lose 10 to 21 points on MMLU and ARC. A GSM8K-only check calls REAP 50% lossless. It is not.

| Arm | Experts | Precision | Weight GB | KL vs S [95% CI] | Top-1 agree | ARC-C | ARC-E | MMLU | GSM8K | Prefill tok/s | Decode tok/s | Peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S | 128 | BF16 | 61.06 | 0 | 1 | 0.6271 | 0.8384 | 0.8023 | 0.9416 | 4339 | 32.9 | 57.6 |
| B | 128 | NVFP4 | 18.09 | 0.066 [0.064, 0.068] | 0.900 | 0.6015 (-2.6 [-4.3, -0.9]) | 0.8295 (-0.9 [-1.9, +0.1]) | 0.7903 (-1.2 [-1.6, -0.8]) | 0.9469 (+0.5 [-0.4, +1.4]) | 10485 | 72.6 | 17.6 |
| D | 72 | FP8 | 18.50 | 0.407 [0.388, 0.427] | 0.777 | 0.5145 (-11.3 [-13.8, -8.8]) | 0.7416 (-9.7 [-11.2, -8.2]) | 0.6621 (-14.0 [-14.7, -13.3]) | 0.9424 (+0.1 [-1.0, +1.2]) | 7820 | 55.3 | 17.9 |
| E | 96 | NVFP4 | 14.00 | 0.185 [0.174, 0.196] | 0.852 | 0.5853 (-4.2 [-6.3, -2.0]) | 0.8220 (-1.6 [-2.7, -0.5]) | 0.7343 (-6.8 [-7.4, -6.2]) | 0.9401 (-0.2 [-1.2, +0.9]) | 10849 | 73.1 | 13.8 |
| A | 64 | BF16 | 32.06 | 0.574 [0.552, 0.597] | 0.729 | 0.5222 (-10.5 [-13.1, -7.9]) | 0.7058 (-13.3 [-14.9, -11.7]) | 0.6161 (-18.6 [-19.4, -17.8]) | 0.9409 (-0.1 [-1.2, +1.1]) | 5640 | 32.9 | 30.5 |
| C | 64 | NVFP4 | 9.92 | 0.638 [0.616, 0.661] | 0.709 | 0.4923 (-13.5 [-16.2, -10.8]) | 0.6818 (-15.7 [-17.4, -13.9]) | 0.5949 (-20.7 [-21.6, -19.9]) | 0.9356 (-0.6 [-1.9, +0.6]) | 12146 | 75.2 | 10.0 |

Task cells: accuracy, then the difference vs S in points with its paired-bootstrap 95% CI. ARC uses acc_norm, MMLU acc (micro average), GSM8K strict match. KL is the mean per-token KL(S || arm) in nats over 256 x 2047 next-token positions, with a bootstrap CI over sequences. Weight GB is exact safetensors bytes (10^9). Speed is the median of 10 samples (two runs of 5, ABBA order), batch 1, on one DGX Spark (GB10). Peak GiB is torch peak memory in the vLLM engine. KL by held-out source and every other number: `results/table.md`, `results/*/`.

B vs D, paired, D as reference: MMLU +12.8 [+12.1, +13.5], ARC-C +8.7 [+6.1, +11.4], ARC-E +8.8 [+7.3, +10.3], GSM8K +0.5 [-0.6, +1.5] (`results/bootstrap_B_vs_D.json`).

What is conclusive:
- At matched expert bytes, 128 NVFP4 experts beat 72 FP8 experts on KL, MMLU, ARC-C and ARC-E. Every CI excludes zero. B is also faster: 34% more prefill and 31% more decode tok/s.
- One-shot REAP pruning, calibrated on chat data, hurts knowledge tasks a lot even at 25% (E vs B: -5.6 MMLU points from removing 32 experts per layer). Quantization error is small next to it: NVFP4 adds 0.06 KL to the 64-expert model (A to C), and 2.1 MMLU points.
- Damage grows with the number of experts removed. In NVFP4, KL is 0.066 at 128 experts, 0.185 at 96 and 0.638 at 64. D, at 72 experts in FP8, sits between them at 0.407.
- Pruning hurts more off the calibration distribution: KL on wikitext exceeds KL on ultrachat for every pruned arm. For B, the full-expert arm, it is the other way round.

What is not conclusive:
- GSM8K. No arm differs from S. The CIs are about plus or minus 1.2 points, so GSM8K cannot separate these arms.
- B vs S on ARC-E: the CI touches zero.
- Speed for B decode: its two runs differ by 4.5% (71.0 and 74.2). All other arms agree within 1.5%.

## Follow-up: which experts to drop

A common intuition says the most-used and the rarely-used experts matter and the medium-used ones can go. We tested it at E's footing (96 experts, NVFP4, 14.00 GB), with the same calibration counts.

| Arm | Kept | Rule | KL vs S | KL wikitext | MMLU vs E | ARC-C vs E | ARC-E vs E |
|---|---|---|---|---|---|---|---|
| E | 96 | REAP saliency | 0.185 [0.174, 0.196] | 0.208 | 0.7343 | 0.5853 | 0.8220 |
| F | 96 | drop the 32 medium-frequency experts | 3.238 [3.179, 3.298] | 3.577 | -49.5 [-50.4, -48.4] | -32.5 [-36.0, -29.1] | -50.6 [-52.8, -48.4] |
| G | 96 | REAP, but never drop the 13 rarest | 0.252 [0.239, 0.266] | 0.289 | -1.4 [-1.8, -0.9] | -1.4 [-3.4, +0.6] | -2.4 [-3.6, -1.3] |
| H | 112 | drop the 16 medium-frequency experts | 0.762 [0.706, 0.822] | 0.735 | -1.7 [-2.4, -1.1] | -7.1 [-9.6, -4.5] | -9.4 [-11.1, -7.7] |

F falls to chance (MMLU 0.240). Removing just 16 medium-frequency experts (H) hurts more than REAP removing 64 (C, KL 0.638). Protecting rare experts (G) costs accuracy and helps no MMLU category. Frequency and REAP saliency barely agree: per-layer Spearman 0.29 on average, and F's drop set overlaps REAP's by 7.5 of 32 experts, about what chance gives. The removed routing mass does not explain the damage either: G removes 15.0% of calibration routing mass and stays near E (10.4%), while F removes 14.0% and collapses. Medium-used experts carry large router-weighted outputs. Saliency sees that; frequency does not. Details: `results/STATUS.md`.

## Follow-up: merge instead of drop

Arm M keeps E's 96 REAP experts as centroids and drops only the 45 experts with zero calibration tokens. It merges the other 1,491 into their most similar centroid, REAM-style: gate-logit plus gated-output similarity, Hungarian neuron alignment, softmax-of-saliency weights. Same 14.00 GB as E.

| Arm | KL vs S | KL wikitext | MMLU vs E | ARC-C vs E | ARC-E vs E |
|---|---|---|---|---|---|
| E, drop | 0.185 [0.174, 0.196] | 0.208 | 0.7343 | 0.5853 | 0.8220 |
| M, merge | 0.336 [0.318, 0.357] | 0.369 | -3.9 [-4.4, -3.3] | +0.6 [-1.5, +2.7] | -1.3 [-2.5, -0.1] |

Merging is worse: KL nearly doubles, and every MMLU category drops. Averaging a low expert into a strong one distorts the strong one more than it adds. This matches REAP's case for pruning over merging.

## Follow-up: healing after pruning

Order: prune, heal in BF16, then NVFP4. E is healed by full-vocabulary distillation from the BF16 source on 6.16M tokens of decontaminated fineweb-edu and ultrachat. Two variants share the same data and budget. The full-parameter heal uses FSDP2 across two DGX Sparks with a factored, stochastic-rounding optimizer. The LoRA heal runs on one Spark: router plus rank-16 attention LoRA plus rank-8 LoRA on every expert, via grouped GEMMs. Each was run twice and reproduced (final validation KL 0.0755/0.0753 full, 0.0780/0.0784 LoRA).

| Arm, all 14.00 GB except B | KL vs S | MMLU | vs E | vs B |
|---|---|---|---|---|
| E, unhealed | 0.185 [0.174, 0.196] | 0.7343 | | -5.6 |
| EF, full heal | 0.144 [0.136, 0.154] | 0.7444 | +1.0 [+0.6, +1.4] | -4.6 [-5.1, -4.1] |
| EL, LoRA heal | 0.149 [0.141, 0.159] | 0.7445 | +1.0 [+0.6, +1.4] | -4.6 [-5.1, -4.0] |
| B, 18.09 GB | 0.066 [0.064, 0.068] | 0.7903 | | |

Healing cuts KL by about 20% and adds 1 MMLU point. LoRA gets nearly all of it (tie on MMLU, 1.1 points behind the full heal on ARC-E). It closes about a fifth of the MMLU gap to B; after the full heal, ARC-C and ARC-E match B. ARC and all CIs: `results/STATUS.md`.

## Follow-up: REAP with a softmax (non-uniform layer budgets)

Per layer, a softmax of REAP saliency (tau = 1x or 0.8x the layer median) turns scores into shares. All experts are ranked globally by share, and E's 1,536 lowest are removed, keeping 64 to 128 per layer. The first and last layers shrink to 64 and the middle keeps up to 122. Same bytes as E.

| Arm | KL vs S | MMLU vs uniform 96 | ARC-C vs uniform | ARC-E vs uniform |
|---|---|---|---|---|
| R, tau = median | 0.316 [0.298, 0.337] | +0.8 [+0.2, +1.3] | +1.5 [-0.7, +3.7] | -1.0 [-2.2, +0.3] |
| R08, tau = 0.8 x median | 0.324 [0.307, 0.345] | +0.9 [+0.4, +1.5] | -1.1 [-3.2, +1.1] | -2.0 [-3.4, -0.7] |

Mixed: about 1 MMLU point more than uniform REAP, but 1.7x the KL and a loss on ARC-E. The uniform control is B pruned to 96 by the same route (KL 0.184, as E).

## Phase A: wider evaluation and calibration domain

The suite adds per-domain KL on 576 new held-out sequences, covering code, math, tool use, instruction following, philosophy, long reasoning and Wikipedia in six languages. It also adds generation tasks with the model card's sampling: HumanEval, MBPP+, MATH-500, GSM8K, IFEval, BFCL v3 and Global-MMLU in Hebrew (full) and Arabic, Japanese, Russian, German and Spanish. E-mix is E with REAP calibrated on a domain mix instead of chat only, with the same token count, the same 96 experts per layer and the same 14.00 GB.

| Arm | KL, non-English | KL, code | Global-MMLU he | ja | ar | IFEval | MATH-500 |
|---|---|---|---|---|---|---|---|
| S, BF16 | | | 0.612 | 0.733 | 0.691 | 0.841 | 0.934 |
| B, 128 experts NVFP4 | 0.061 | 0.040 | 0.588 | 0.704 | 0.679 | 0.837 | 0.910 |
| E, 96 experts, chat calibration | 1.668 | 0.190 | 0.394 | 0.499 | 0.493 | 0.797 | 0.898 |
| E-mix, 96 experts, mixed calibration | 0.256 | 0.120 | 0.502 | 0.587 | 0.576 | 0.826 | 0.900 |

Pruning with chat-only calibration erases most non-English knowledge: E loses 9 to 23 Global-MMLU points outside English, against 5.6 on English MMLU. The mixed calibration set wins back about half of that (Hebrew +10.8 [+9.9, +11.7], Japanese +8.8, Arabic +8.3, German +7.0). It also gains IFEval (+3.0) and GSM8K (+1.4). It still trails B by 8 to 12 points in every non-English language. Code and tool-use generation do not separate the arms. The old English chat KL check favors E (0.185 vs 0.226), because that is E's own calibration domain. 438 of 4,608 kept experts change. All tables and CIs: `results/STATUS.md`.

Four heals of E-mix use the same 376-step LoRA budget, each with a different data source or numerics. Per-domain KL vs S, E-mix unhealed 0.170:

| Heal | Data | KL | vs HQ |
|---|---|---|---|
| HB | mixed corpus, BF16, quantized after | 0.136 | +0.0097 |
| HQ | mixed corpus, through the NVFP4 quantizer (QAD) | 0.127 | |
| SDb | the student's own answers, 3 rounds, QAD | 0.134 | +0.0076 |
| SDa | the source's own answers, QAD | 0.134 | +0.0071 |

Training through the deployed quantizer beats healing in BF16 in every domain. Self-distillation data, generated once or each round, does not beat the fixed mixed corpus. On the generation suite the heals are mostly within noise of each other; SDb gains MATH-500 over HQ (+2.8 [+0.6, +5.0]). No heal closes much of the 5 to 10 point non-English gap to B.

## Question

At a fixed weight budget, is it better to keep all 128 experts at 4 bits, or fewer experts at 8 bits? Prior work reports pruning and quantization separately. This study puts them on one byte axis, with real expert removal, real low-precision kernels, and accuracy measured against the BF16 source.

## Arms

Source: `Qwen/Qwen3-30B-A3B-Instruct-2507`, BF16. 48 layers, 128 experts per layer, 8 active.

| Arm | Experts | Precision of linear layers | Purpose |
|---|---|---|---|
| S | 128 | BF16 | source |
| B | 128 | NVFP4 | fewer bits |
| A | 64 (REAP 50%) | BF16 | fewer experts |
| C | 64 (REAP 50%) | NVFP4 | both |
| D | 72 (REAP 43.75%) | FP8 (W8A8 dynamic) | matched bytes vs B |
| E | 96 (REAP 25%) | NVFP4 | gentler prune plus bits |

Why 72 experts for D: NVFP4 stores 0.5 B per weight plus one FP8 scale per 16 weights, so 0.5625 B per weight. FP8 with per-channel scales stores about 1.001 B per weight. 128 x 0.5625 / 1.001 = 71.9, so 72 FP8 experts hold the same expert bytes as 128 NVFP4 experts. Exact bytes per arm come from the safetensors headers (`scripts/07_bytes.py`).

In every arm lm_head, the router gates and the embeddings stay BF16.

## Method

- Calibration: 512 conversations from `HuggingFaceH4/ultrachat_200k` train_sft, shuffled with seed 42, chat template applied, truncated to 2048 tokens. The same set drives REAP saliency and NVFP4 calibration.
- REAP: one calibration pass with `REAPPruningModifier(sparsity=0.0, report_path=...)` writes per-expert saliency. `ExpertPruner` then removes the lowest-saliency experts uniformly per layer straight from the checkpoint: expert tensors are dropped, router rows sliced, `num_experts` rewritten. The result loads as a plain Qwen3-MoE with fewer experts. No masking.
- Quantization: llm-compressor `QuantizationModifier`, scheme `NVFP4` (every expert sees calibration data) or `FP8_DYNAMIC`. Pruning first, then quantization.
- Serving and kernels: vLLM 0.34.0 on a DGX Spark (GB10, sm_121), compressed-tensors checkpoints, native NVFP4 and FP8 paths.
- KL vs source: 256 held-out sequences of 2048 tokens, disjoint from calibration. 128 packed from ultrachat test_sft, 128 from wikitext-103 test. vLLM dumps the final hidden states of each arm. Since lm_head is unchanged and bit-identical in all arms (checked), full-vocabulary logits are hidden @ lm_head.T in fp32. Reported: mean KL(S || arm), next-token NLL, top-1 agreement, with 95% bootstrap CIs over sequences, overall and per source.
- Tasks: lm-eval-harness 0.4.13 with the vLLM backend. arc_challenge and arc_easy (acc_norm, 0-shot), mmlu (acc, 0-shot, micro average), gsm8k_cot_llama (8-shot CoT, chat template, greedy, strict match). Full test sets.
- Statistics: paired bootstrap (10,000 resamples) of the per-item accuracy difference vs S, 95% CI.
- Speed and memory: batch 1 on the Spark. Prefill tok/s on a 2048-token prompt, decode tok/s from 512 generated tokens. 3 warmups, N=5 per run, runs ordered ABBA across arms, medians reported. Memory from the vLLM engine in-process.

## Versions

| Component | Version |
|---|---|
| llm-compressor | 0.15.0 (PyPI) |
| compressed-tensors (build) | 0.20.1 |
| torch (build) | 2.13.0+cu130 |
| transformers (build) | 5.21.0 |
| vLLM image | vllm/vllm-openai:v0.34.0 (torch 2.14.0+cu130, transformers 5.22.2) |
| lm-eval | 0.4.13 (`docker/Dockerfile`) |

Data digests (sha256): `calib.jsonl` ca2dae8d6a6f8da5a7360d9c0a893d6d6cb31f97748941a12771781e78280a96, `heldout.jsonl` c5ef983acf11060e9532cc6d164ff37ce67faeeaf48e37f2afc4709e5ab04e5c.

## Layout

```
scripts/00_prepare_data.py   calibration and held-out sets
scripts/01_reap_report.py    REAP saliency pass (no pruning)
scripts/02_prune.py          real expert removal from the checkpoint
scripts/03_quant.py          NVFP4 / FP8_DYNAMIC
scripts/04_hidden.py         vLLM final hidden states (+ self-check vs vLLM prompt logprobs)
scripts/05_kl.py             full-vocab KL, NLL, top-1 vs source
scripts/06_speed.py          batch-1 prefill and decode, memory
scripts/07_bytes.py          exact weight bytes by tensor group
scripts/08_lmeval.sh         pinned lm-eval suite
scripts/09_bootstrap.py      paired bootstrap vs source
scripts/10_table.py ..       follow-ups: selection (11, 17, 25), heal (13-16, 29, 35), merge (18),
  37_kl_paired.py            softmax budgets (19-21), Phase A data, suite and self-distillation (22, 26-34, 37)
scripts/build_arms.sh        main arms end to end; run_eval.sh and finalize.sh evaluate and tabulate them
scripts/phaseA_baselines.sh  Phase A per-domain KL and generation suite for S, B, E
scripts/heal_arm_A*.sh       one Phase A heal arm (A: BF16 heal, A2: QAD with NVFP4 export); sd_a.sh runs SDa
scripts/heal_lora.sh, heal_full_launch.sh   the LoRA and FSDP2 full heals of E
scripts/nvfp4_fq.py          NVFP4 fake quantization matching the deployed kernels (QAD)
scripts/check_*.py           QAD scale and export checks against the served checkpoint
scripts/dock.sh              runs a step in the pinned image
docker/, lm_tasks/           eval images and the MBPP+ task fix
```

Large outputs (checkpoints, hidden-state dumps, full lm-eval samples) are not in the repo.

## Status

See `results/STATUS.md`.

## Kernels used (vLLM 0.34.0 on GB10)

- NVFP4 arms: `FlashInferCutlassNvFp4LinearKernel` for dense layers, the `FLASHINFER_CUTLASS` NVFP4 MoE backend for experts.
- FP8 arm: the `TRITON` FP8 MoE backend.
- BF16 arms: the default fused MoE path.

## Checks

- Download: all 16 source shards match the hub's sha256 and the index total of 61,064,245,248 bytes (`scripts/verify_download.py`).
- Hidden-state KL: on 4 source sequences, NLL recomputed from the dumped hidden states matches vLLM's own prompt logprobs. Mean NLL agrees within 0.1%. The largest per-token gap is 0.17 nats, from vLLM computing logits in BF16 where this study uses fp32 (`results/S/hidden_check.log`).
- lm_head is bit-identical to the source in every arm (asserted in `05_kl.py`).
- Byte match: B holds 16,307,601,408 expert bytes and D 16,332,226,560 (+0.15%). Totals are 18.09 and 18.50 GB because attention is FP4 in B and FP8 in D.

## Caveats

- REAP calibration is chat data. The ultrachat half of the KL set shares that distribution, so it flatters pruning. The wikitext half is the out-of-distribution check; report both.
- Uniform pruning per layer. Non-uniform budgets may do better and are not tested here.
- No healing after pruning. This measures one-shot compression.
- Speed is batch 1 on one GB10. It says nothing about throughput at high concurrency. The Spark's clock-cap service was active, as in normal use there.
- 45 of 6,144 experts got no calibration tokens. Their REAP saliency is 0, so they are pruned first.
- MMLU and ARC run 0-shot without a chat template. GSM8K runs 8-shot with the chat template, greedy.
- One seed and one calibration set. Calibration-set variance of REAP is not measured.
