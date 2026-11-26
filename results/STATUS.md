# Status

All six arms are built and measured. The results and their interpretation are in the README. `table.md` and `table.json` hold every number, including KL per held-out source.

## Timeline

- 2026-11-23 16:47: source download verified, sha256 on all 16 shards.
- 16:50: first REAP pass OOM-killed while linearizing experts. It was rerun with `load_context()` offloaded loading and finished at 17:30 (40 min, 579,870 calibration tokens).
- 17:40: three parallel prunes overloaded unified memory and the system OOM killer stopped the 72-expert one. It was rerun alone. Lesson: one heavy process at a time on a Spark.
- 18:05 to 22:20: FP8 build (6 min), NVFP4 builds (35 to 60 min each, 512 calibration rows).
- 18:07 to 00:07: evaluation, about 25 to 45 min per arm, then the reverse speed pass.

## Per-arm receipts

`results/<arm>/`: `bytes.json`, `kl_summary.json`, `speed_fwd.json`, `speed_rev.json`, `lmeval_ll_results.json`, `lmeval_gsm_results.json`. Full lm-eval samples and logs are kept outside the repo.

## Not done

- MMLU-Pro, not needed because full MMLU fit in the budget.
- Seed or calibration-set repeats of REAP.
- Non-uniform pruning and post-prune healing.

## Follow-up 1 (2026-11-24): expert selection by routing frequency (arms F, G, H)

All NVFP4, same calibration counts as E (from `reap_report.json`), real removal, router renormalized over kept experts by the model's top-k normalization, as in E. Selection reports: `selection/report_{F,G,H}.json` (built by `scripts/11_select_reports.py`). Kept sets verified against the source routers in all 48 layers (`scripts/17_verify_selection.py`).

| Arm | Kept | Rule | GB | KL all | KL ultrachat | KL wikitext | Top-1 | ARC-C | ARC-E | MMLU |
|---|---|---|---|---|---|---|---|---|---|---|
| E | 96 | REAP saliency | 14.00 | 0.185 [0.174, 0.196] | 0.162 | 0.208 | 0.852 | 0.5853 | 0.8220 | 0.7343 |
| F | 96 | drop 32 medium-frequency | 14.00 | 3.238 [3.179, 3.298] | 2.899 | 3.577 | 0.353 | 0.2602 (-32.5 [-36.0, -29.1]) | 0.3157 (-50.6 [-52.8, -48.4]) | 0.2398 (-49.5 [-50.4, -48.4]) |
| G | 96 | REAP, 13 rarest protected | 14.00 | 0.252 [0.239, 0.266] | 0.215 | 0.289 | 0.827 | 0.5717 (-1.4 [-3.4, +0.6]) | 0.7976 (-2.4 [-3.6, -1.3]) | 0.7207 (-1.4 [-1.8, -0.9]) |
| H | 112 | drop 16 medium-frequency | 16.05 | 0.762 [0.706, 0.822] | 0.790 | 0.735 | 0.692 | 0.5145 (-7.1 [-9.6, -4.5]) | 0.7281 (-9.4 [-11.1, -7.7]) | 0.7170 (-1.7 [-2.4, -1.1]) |

Task deltas are vs E, in points, paired bootstrap 95% CI (`bootstrap_fgh_vs_E.json`; vs S in `bootstrap_fgh_vs_S.json`). ARC acc_norm. GSM8K not run.

Frequency vs REAP saliency (`selection/selection_stats.json`): mean per-layer Spearman 0.29 (0.05 to 0.51); F's drop set shares 7.5 of 32 experts per layer with E's (chance: 8); E's dropped experts have mean frequency rank 39.5 of 127. Calibration routing mass removed: E 10.4%, F 14.0%, G 15.0%, H 6.8%.

MMLU by category vs E (`mmlu_subjects_vs_E.json`): G stem -1.2 [-2.2, -0.3], humanities -0.5 [-1.4, +0.3], social -1.2 [-2.1, -0.3], other -2.9 [-3.9, -1.8]. Keeping rare experts protects no category.

Verdict: the frequency rule is refuted. Dropping medium-used experts is the most damaging choice tested; protecting rare ones costs accuracy.

## Follow-up 3 (2026-11-24): merge the low experts instead of dropping them (arm M)

Same footing as E: 96 experts per layer, NVFP4, 14.00 GB, same calibration set. Script `scripts/18_merge.py`, per-layer stats `M/merge_stats.json`.

Method, settled against the literature (REAM, arXiv 2604.04356, which builds on REAP; M-SMoE/MC-SMoE; HC-SMoE):
- Centroids are the 96 highest-REAP experts (E's kept set). Router rows of the centroids are kept unchanged, as in REAP and REAM. A top-k linear router cannot send a removed expert's tokens to its centroid without duplicate expert slots, which would cost the same bytes as keeping the expert.
- Experts with zero calibration tokens are dropped: 45 in total, 0 to 6 per layer.
- Every other non-centroid (1,491 in total, 31.1 per layer) is merged into its most similar centroid. Similarity = cosine of gate logits over tokens + mean cosine of gated expert outputs (REAM Eq. 7-8), on 8,192 calibration tokens, at most 16 members per centroid (REAM's C for 96 experts).
- Intermediate neurons are aligned with the Hungarian algorithm on activation cost + weight cost, each normalized by its mean, before averaging gate/up/down.
- Merge weights are a softmax of REAP saliency within the group, with temperature equal to the layer's median nonzero saliency (0.11 to 9.74 across layers). The centroid gets 0.54 of the weight on average (0.46 to 0.60).
- Groups: 19.9 per layer (12 to 26); size including the centroid: 664 pairs, 170 triples, 58 of 4, up to 12; the largest group per layer averages 5.9.

| Arm | KL all | KL ultrachat | KL wikitext | Top-1 | ARC-C | ARC-E | MMLU |
|---|---|---|---|---|---|---|---|
| E (REAP drop) | 0.185 [0.174, 0.196] | 0.162 | 0.208 | 0.852 | 0.5853 | 0.8220 | 0.7343 |
| M (drop zero-use, merge the rest) | 0.336 [0.318, 0.357] | 0.303 | 0.369 | 0.800 | 0.5913 (+0.6 [-1.5, +2.7]) | 0.8093 (-1.3 [-2.5, -0.1]) | 0.6958 (-3.9 [-4.4, -3.3]) |

Deltas vs E in points, paired bootstrap 95% CI (`bootstrap_M_vs_E.json`). MMLU by category vs E: stem -3.2 [-4.5, -2.0], humanities -3.1 [-4.2, -2.0], social -3.5 [-4.6, -2.4], other -6.0 [-7.2, -4.7].

Verdict: merging the low experts into their centroids is worse than dropping them. KL nearly doubles and MMLU loses 3.9 points, on both held-out corpora and in every MMLU category. This matches REAP's finding that merging collapses the functional output of the centroid, even with similarity grouping, neuron alignment and saliency weights. A second variant (M2) was not run: the literature shows no clearly better merge for this case. REAM reports +1.2 GEN points over REAP at 96 experts on this model, and its saliency-proportional weights give the centroid more weight, which would likely land between M and E.

## Follow-up 2 (2026-11-24): healing E (prune, then heal, then NVFP4)

E's pruned BF16 model (96 experts) is healed by distillation from the BF16 source, then quantized to NVFP4 with the same recipe as every arm, so all three arms below are 14.00 GB. The best 96-expert arm among E, F and G is E, so E is the one healed; H was not close.

Setup, fixed before launch:
- Objective: forward KL(source || student) over the full vocabulary, T = 1. Teacher logits = source final hidden states @ the shared, frozen lm_head (`14_teacher_hidden.py`).
- Data: 3,072 sequences of 2,048 tokens: half fineweb-edu, half ultrachat train_sft without the calibration rows (`13_heal_data.py`). Documents sharing any 13-gram with MMLU, ARC, GSM8K or the held-out KL set were dropped (21 fineweb-edu and 2,252 ultrachat documents; the ultrachat drops are mostly shared assistant boilerplate). 64 sequences are held out for validation.
- Budget, the same for both variants: 376 steps x 16,384 tokens = 6.16M tokens, warmup 3%, linear decay, grad clip 1.0.
- Full-parameter heal (`16_heal_full.py`): FSDP2 across both Sparks over the QSFP link (NCCL over RoCE). Every decoder parameter is trained; embeddings and lm_head are frozen. Optimizer: factored second moment, bf16 momentum and bf16 stochastic rounding, about 2 B of state per parameter. Full AdamW does not fit 2 x 119 GB, and torchao's 8-bit AdamW fails to compile with torch 2.13 here. LR 1e-5. A first try at 5e-5 (the full-FT LR in the 2026 EMNLP post-compression study on this model) raised the batch KL above its start and was stopped at step 41. About 650 tok/s; 2.6 h.
- LoRA heal (`15_heal.py`): one Spark. Router weights trained fully, LoRA rank 16 on attention, rank 8 per expert on every expert, run through grouped GEMMs. PEFT's 3D-parameter LoRA materializes W + BA for every expert on every call, and was about 7x slower here. LR 1e-4 (about 10x the full-parameter LR). About 520 tok/s; 3.4 h.
- Both runs were repeated with the same seed and data. Final validation KL: full 0.0755 and 0.0753, LoRA 0.0780 and 0.0784; it starts at 0.1045. The first run of each lost its weights to an out-of-memory kill during saving. Saving now streams tensor by tensor, with the adapters written first.

| Arm | KL all | KL ultrachat | KL wikitext | Top-1 | ARC-C | ARC-E | MMLU |
|---|---|---|---|---|---|---|---|
| E, unhealed | 0.185 [0.174, 0.196] | 0.162 | 0.208 | 0.852 | 0.5853 | 0.8220 | 0.7343 |
| EF, full heal | 0.144 [0.136, 0.154] | 0.115 | 0.173 | 0.863 | 0.6015 | 0.8262 | 0.7444 |
| EL, LoRA heal | 0.149 [0.141, 0.159] | 0.123 | 0.176 | 0.862 | 0.5939 | 0.8152 | 0.7445 |
| B, 128 experts | 0.066 [0.064, 0.068] | 0.073 | 0.059 | 0.900 | 0.6015 | 0.8295 | 0.7903 |

Paired bootstrap, points, 95% CI:

| Comparison | MMLU | ARC-C | ARC-E |
|---|---|---|---|
| EF vs E | +1.0 [+0.6, +1.4] | +1.6 [-0.2, +3.4] | +0.4 [-0.6, +1.4] |
| EL vs E | +1.0 [+0.6, +1.4] | +0.9 [-0.9, +2.7] | -0.7 [-1.6, +0.3] |
| EL vs EF | +0.0 [-0.4, +0.4] | -0.8 [-2.5, +0.9] | -1.1 [-2.0, -0.2] |
| EF vs B | -4.6 [-5.1, -4.1] | 0.0 [-2.1, +2.1] | -0.3 [-1.5, +0.8] |
| EL vs B | -4.6 [-5.1, -4.0] | -0.8 [-2.9, +1.3] | -1.4 [-2.6, -0.3] |

In BF16, before quantization, KL vs S is 0.123 for E and 0.096 for the full-healed model; NVFP4 adds about 0.05 to both.

Verdict: healing helps, and LoRA gets nearly all of it. Both heals cut KL by about 20% and gain 1.0 MMLU point; they tie on MMLU, and the full heal is 1.1 points better on ARC-E. Healing does not close the gap to B at 4 GB fewer: MMLU stays 4.6 points behind, about a fifth of the 5.6-point gap closed. For the full heal, ARC-C and ARC-E reach B's level within the CIs; the LoRA heal is still 1.4 points behind on ARC-E. With 6M tokens this is a small heal; the curves were still falling slowly at the end.

## Follow-up 4 (2026-11-24): REAP with a softmax, as non-uniform layer budgets (arms R, R08)

A softmax inside a layer keeps the ranking, so on its own it equals E. The version that changes something: per layer, share = softmax(saliency / tau), tau = k x that layer's median nonzero saliency; rank all 6,144 experts globally by share; remove the lowest 1,536 (E's total), keeping 64 to 128 per layer (`scripts/19_select_R.py`). k = 1 (R) and k = 0.8 (R08). Counts: `selection/R_counts.json`, `selection/R08_counts.json`; for comparison, an absolute tau of 0.8 (counts only, not built) is in `selection/Rabs08_counts.json`.

- R kept per layer: 64 to 121; 8 layers at the 64 floor (1-3, 16, 24, 45-47); mean by depth quarter 94.1 / 103.5 / 100.2 / 86.2. R08 is close: 64 to 122, 9 at the floor, 95.2 / 104.3 / 98.8 / 85.7. At k = 0.5, 38 of 48 layers would change by up to 35 experts, so the allocation is temperature-sensitive per layer even though the depth profile is not. An absolute tau of 0.8 keeps early layers almost whole and cuts layers 37-47 to the floor (111.5 / 114.3 / 93.2 / 65.0), because saliency grows about 100x with depth.
- Build: B's NVFP4 checkpoint pruned per layer from safetensors (`scripts/20_prune_from_nvfp4.py`); transformers and llm-compressor assume one expert count for all layers. Control E2: B pruned uniformly to E's 96 the same way. E2 matches E (KL 0.184 vs 0.185, MMLU +0.2 [-0.1, +0.6]), so building from B is neutral. vLLM 0.34 has one global expert count; a two-line patch (`scripts/patch_vllm_qwen3_per_layer.py`, mounted via `dock.sh`) reads `num_experts_per_layer` from the config. With a uniform model the patched file changes nothing (E2).
- Bytes: R, R08, E2 and E are all exactly 14,004,266,496.

| Arm | KL all | KL ultrachat | KL wikitext | Top-1 | ARC-C | ARC-E | MMLU |
|---|---|---|---|---|---|---|---|
| E2, uniform 96 | 0.184 [0.174, 0.195] | 0.161 | 0.208 | 0.852 | 0.5981 | 0.8249 | 0.7366 |
| R, k = 1 | 0.316 [0.298, 0.337] | 0.265 | 0.366 | 0.805 | 0.6126 | 0.8152 | 0.7445 |
| R08, k = 0.8 | 0.324 [0.307, 0.345] | 0.273 | 0.376 | 0.802 | 0.5870 | 0.8047 | 0.7459 |

Paired bootstrap, points, 95% CI:

| Comparison | MMLU | ARC-C | ARC-E |
|---|---|---|---|
| R vs E2 | +0.8 [+0.2, +1.3] | +1.5 [-0.7, +3.7] | -1.0 [-2.2, +0.3] |
| R08 vs E2 | +0.9 [+0.4, +1.5] | -1.1 [-3.2, +1.1] | -2.0 [-3.4, -0.7] |
| R vs E | +1.0 [+0.5, +1.6] | +2.7 [+0.6, +4.9] | -0.7 [-1.9, +0.6] |
| R08 vs E | +1.2 [+0.6, +1.7] | +0.2 [-2.0, +2.4] | -1.7 [-3.0, -0.4] |
| R vs B | -4.6 [-5.1, -4.1] | +1.1 [-1.1, +3.4] | -1.4 [-2.7, -0.2] |
| R08 vs B | -4.4 [-5.0, -3.9] | -1.5 [-3.8, +0.9] | -2.5 [-3.8, -1.2] |

Verdict: mixed. Softmax budgets buy about 1 MMLU point over uniform REAP at the same bytes (significant for both temperatures), but they cost ARC-E (significant at k = 0.8) and they 1.7x the KL vs the source on both corpora. The budget moves experts out of the first and last layers into the middle. Knowledge recall (MMLU) likes that; the full next-token distribution does not. The ARC-C gain at k = 1 vs E does not survive the matched control E2. The two temperatures give nearly the same model; temperature matters less than the floor, which 8-9 layers hit.

## Phase A (2026-11-25): wider evaluation, mixed calibration, quantization-aware and self-distilled heals

In progress. Sections fill in as arms land.

### Evaluation suite (S, B, E, and every Phase A arm)
- Per-domain KL vs S on 576 new held-out sequences (`22_phaseA_data.py`): 64 each of code, math, tool use, instruction following, philosophy and long reasoning; 32 each of Wikipedia in Hebrew, Arabic, Russian, Japanese, German and Spanish. 13-gram decontaminated against all 35,830 eval items.
- Generation, one sample per item with the model card's sampling (T 0.7, top-p 0.8, top-k 20, up to 4,096 new tokens), lm-eval 0.4.13: HumanEval (instruct), MBPP+ (instruct, with a fixed code extractor in `lm_tasks/`), MATH-500 (math_verify), GSM8K (CoT, strict match), IFEval (prompt-level strict). Global-MMLU: Hebrew in full; Arabic, Japanese, Russian, German and Spanish at 20 items per subject. BFCL v3 simple, multiple, parallel and irrelevance with AST scoring (`26_bfcl.py`).
- Paired bootstrap over items (`33_suiteA.py`), 4,000 resamples. The old KL set and MMLU/ARC are rerun for continuity (`extras_A.sh`).
- lm_head, embeddings and the router stay BF16 in every quantized arm.

### E-mix: REAP on a domain-mixed calibration set
- Same 580k-token budget as E's chat-only calibration, mixed across the domains above (`calib_mix.jsonl`, 817 rows). REAP saliency as before, prune to 96 per layer, NVFP4 with the mixed set as quantization calibration.
- Expert diff vs E (`results/phaseA/emix_diff.json`): 438 of 4,608 kept slots change (9.5%, 5 to 15 per layer). All 45 experts that saw zero tokens under E's calibration get traffic under the mix; 24 of them are kept. The experts E-mix gains mostly serve code (119), Japanese (91), math (59), Russian (36) and Hebrew (26); the ones it drops mostly serve math (88), code (75), tool use (50) and philosophy (48). The swap moves capacity toward non-English text and code at some cost to math and tool experts.

Suite results (NVFP4 arms B, E, E-mix; S is BF16). Accuracy, then paired differences in points with 95% CI (`results/phaseA/suite_ref{S,E,B}.json`). Global-MMLU ar/ja/ru/de/es use 20 items per subject:

| Benchmark | n | S | B | E | E-mix | B vs S | E vs S | E-mix vs E | E-mix vs B |
|---|---|---|---|---|---|---|---|---|---|
| HumanEval | 164 | 0.945 | 0.939 | 0.933 | 0.939 | -0.6 [-3.7, +1.8] | -1.2 [-3.7, +1.2] | +0.6 [-3.0, +4.9] | +0.0 [-3.7, +3.0] |
| MBPP+ | 378 | 0.921 | 0.931 | 0.913 | 0.913 | +1.1 [-1.3, +3.2] | -0.8 [-3.4, +1.6] | +0.0 [-2.4, +2.4] | -1.9 [-4.2, +0.3] |
| MATH-500 | 500 | 0.934 | 0.910 | 0.898 | 0.900 | -2.4 [-4.2, -0.6] | -3.6 [-5.8, -1.4] | +0.2 [-2.2, +2.6] | -1.0 [-3.4, +1.4] |
| GSM8K | 1319 | 0.942 | 0.944 | 0.940 | 0.954 | +0.2 [-0.8, +1.1] | -0.2 [-1.2, +0.8] | +1.4 [+0.4, +2.4] | +1.0 [+0.0, +2.0] |
| IFEval | 541 | 0.841 | 0.837 | 0.797 | 0.826 | -0.4 [-2.4, +1.7] | -4.4 [-7.2, -1.5] | +3.0 [+0.2, +5.7] | -1.1 [-3.7, +1.3] |
| BFCL simple | 400 | 0.940 | 0.943 | 0.940 | 0.932 | +0.2 [-0.5, +1.2] | +0.0 [-1.5, +1.5] | -0.8 [-2.2, +0.8] | -1.0 [-2.5, +0.2] |
| BFCL multiple | 200 | 0.900 | 0.895 | 0.920 | 0.915 | -0.5 [-3.0, +1.5] | +2.0 [+0.0, +4.5] | -0.5 [-2.5, +1.0] | +2.0 [+0.0, +4.5] |
| BFCL parallel | 200 | 0.905 | 0.910 | 0.925 | 0.905 | +0.5 [-2.5, +3.5] | +2.0 [-0.5, +5.0] | -2.0 [-4.0, -0.5] | -0.5 [-2.5, +1.5] |
| BFCL irrelevance | 240 | 0.829 | 0.817 | 0.812 | 0.825 | -1.2 [-4.2, +1.7] | -1.7 [-4.2, +0.8] | +1.2 [-1.7, +4.2] | +0.8 [-2.5, +4.2] |
| Global-MMLU he | 14042 | 0.612 | 0.588 | 0.394 | 0.502 | -2.4 [-3.0, -1.9] | -21.8 [-22.8, -20.9] | +10.8 [+9.9, +11.7] | -8.6 [-9.4, -7.8] |
| Global-MMLU ar | 1140 | 0.691 | 0.679 | 0.493 | 0.576 | -1.2 [-2.8, +0.4] | -19.8 [-22.9, -16.7] | +8.3 [+5.3, +11.3] | -10.3 [-12.9, -7.7] |
| Global-MMLU ja | 1140 | 0.733 | 0.704 | 0.499 | 0.587 | -3.0 [-4.6, -1.2] | -23.4 [-26.4, -20.3] | +8.8 [+5.6, +11.9] | -11.7 [-14.4, -8.9] |
| Global-MMLU ru | 1140 | 0.734 | 0.721 | 0.614 | 0.640 | -1.3 [-2.8, +0.3] | -12.0 [-14.8, -9.2] | +2.6 [+0.2, +5.1] | -8.1 [-10.4, -5.7] |
| Global-MMLU de | 1140 | 0.740 | 0.732 | 0.576 | 0.646 | -0.9 [-2.3, +0.5] | -16.4 [-19.3, -13.2] | +7.0 [+3.9, +10.1] | -8.5 [-10.8, -6.1] |
| Global-MMLU es | 1140 | 0.748 | 0.752 | 0.661 | 0.668 | +0.4 [-1.1, +1.8] | -8.7 [-11.2, -6.2] | +0.7 [-1.8, +3.2] | -8.3 [-10.6, -6.1] |

Per-domain KL vs S (mean per token, 95% CI over sequences; `results/phaseA/<arm>/kl_domains.json`):

| Domain | B | E | E-mix |
|---|---|---|---|
| all | 0.053 [0.051, 0.054] | 0.650 [0.575, 0.732] | 0.170 [0.161, 0.178] |
| code | 0.040 [0.036, 0.044] | 0.190 [0.156, 0.227] | 0.120 [0.101, 0.140] |
| math | 0.028 [0.026, 0.030] | 0.098 [0.089, 0.106] | 0.060 [0.055, 0.066] |
| tool | 0.043 [0.041, 0.045] | 0.094 [0.086, 0.104] | 0.104 [0.095, 0.114] |
| instruction | 0.088 [0.084, 0.091] | 0.194 [0.180, 0.209] | 0.224 [0.209, 0.239] |
| philosophy | 0.051 [0.050, 0.052] | 0.199 [0.191, 0.208] | 0.184 [0.176, 0.192] |
| reasoning | 0.042 [0.039, 0.044] | 0.073 [0.068, 0.078] | 0.069 [0.063, 0.075] |
| multilingual | 0.061 [0.059, 0.063] | 1.668 [1.531, 1.817] | 0.256 [0.243, 0.269] |
| Wikipedia he | 0.084 [0.081, 0.087] | 1.098 [1.027, 1.165] | 0.233 [0.221, 0.246] |
| Wikipedia ar | 0.069 [0.066, 0.072] | 1.369 [1.275, 1.458] | 0.261 [0.241, 0.280] |
| Wikipedia ja | 0.055 [0.052, 0.058] | 2.647 [2.342, 2.933] | 0.371 [0.332, 0.407] |
| Wikipedia ru | 0.048 [0.045, 0.051] | 2.920 [2.657, 3.183] | 0.238 [0.222, 0.254] |
| Wikipedia de | 0.056 [0.053, 0.058] | 1.509 [1.418, 1.593] | 0.210 [0.187, 0.237] |
| Wikipedia es | 0.052 [0.050, 0.055] | 0.466 [0.440, 0.494] | 0.223 [0.199, 0.253] |

E's KL on non-English text is 6.5x E-mix's (1.67 vs 0.26) and 27x B's; on Russian and Japanese Wikipedia it is near 3 nats per token. The mix trades a little on tool and instruction text (0.104 vs 0.094, 0.224 vs 0.194) for that.

Verdict for steps 1 and 2:
- Pruning costs multilingual knowledge far more than English. E (chat-only calibration) loses 9 to 23 points on Global-MMLU outside English, against 5.6 on English MMLU; B loses 0 to 3. The chat calibration set almost never routes non-English text, so REAP drops the experts that serve it.
- A domain-mixed calibration set at the same token count recovers about half of that loss (Hebrew +10.8, Japanese +8.8, Arabic +8.3, German +7.0, all significant) and also gains IFEval (+3.0) and GSM8K (+1.4). It does not reach B: still 8 to 12 points behind on every non-English language at 4 GB fewer.
- Code, tool use and math generation barely separate the arms. MATH-500 is the one generation task where every quantized arm is significantly below S (B -2.4, E -3.6, E-mix -3.4).
- The cost: on the old English chat/wiki KL set, E-mix is worse than E (0.226 vs 0.185), because E was calibrated on that chat distribution. English MMLU, ARC-C and ARC-E (acc_norm, as in the main tables) tie: 0.736 vs 0.734 (+0.2 [-0.3, +0.8]), 0.598 vs 0.585 (+1.3 [-0.9, +3.4]), 0.823 vs 0.822 (+0.1 [-1.1, +1.3]); `results/phaseA/Emix_vs_E_mmlu_arc.json`.

### Quantization-aware heal (HQ) and its BF16 control (HB)
- `29_heal_qad.py`: LoRA (rank 8 per expert, 16 on attention, router trained at 0.1x LR), same 6.16M-token budget, LR and schedule as EL, on a 13-gram decontaminated mixed corpus (`heal_mix.jsonl`, 3,072 x 2,048). Teacher: S, full-vocabulary forward KL.
- HB runs the LoRA in BF16 and is quantized afterwards with llm-compressor, as every other arm. HQ fake-quantizes every NVFP4 Linear in the forward pass with straight-through gradients.
- Numerics must match the deployed model. The first QAD run (HQ0) did not. It froze weight block scales at the base weight and gave gate, up, q, k and v their own global scales; it used dynamic activation global scales. The llm-compressor export and vLLM differ on all three: the export recomputes scales from the healed weight, gate+up per expert and q+k+v per layer share one global scale (vLLM packs them into one GEMM and uses one scale), and activation global scales are static calibrated values, one per layer for all experts (vLLM takes the max). On HQ0's healed weights, 3.5% of FP4 codes differ between what training saw and what the export wrote (`results/phaseA/verify/qad_scales_HQ0.json`); per-matrix global scales alone would change 32% of codes.
- Fix (`nvfp4_fq.py`, `34_act_amax.py`, `35_export_nvfp4.py`): training computes block scales from the current merged weight with the fused global scales and uses E-mix's calibrated static activation scales. The QAD arms are exported with the same functions instead of llm-compressor, so the deployed weights are the last training step's. Checks: the exporter's output equals the fake-quant on all but 0.05% of elements (floating-point ties); re-exporting E-mix with it gives the same KL on the 64 validation sequences as llm-compressor's export (0.143 [0.123, 0.165] vs 0.144 [0.124, 0.166]) and the same per-domain KL (0.169 vs 0.170 overall, every domain within 0.003; `results/phaseA/verify/`); the trainer's own step-0 validation KL with QAD on is 0.145. HB keeps the llm-compressor export, so HQ vs HB compares the training, not the exporter.
- HQ0 is kept as an ablation (mismatched numerics). Per-domain KL 0.129 overall (E-mix 0.170), multilingual 0.198 (0.256). Its suite was not run.
- Speed: about 455 to 466 tok/s with compiled fake-quant, against 520 to 626 tok/s for the BF16 LoRA heal.

HB (BF16 LoRA heal of E-mix, then llm-compressor NVFP4; 14.00 GB). Final validation KL 0.071 in BF16.
- Per-domain KL vs S: 0.137 [0.130, 0.143] (E-mix 0.170); paired vs E-mix -0.033 [-0.035, -0.031] overall and -0.051 [-0.055, -0.048] multilingual (`results/phaseA/klp_HB_vs_Emix.json`).
- HQ0 vs HB, paired: -0.0073 [-0.0077, -0.0068] overall, significant in every domain; largest on instruction (-0.013) and tool (-0.009) text (`klp_HQ0_vs_HB.json`). QAD beats heal-then-quantize even with its numerics off.
- Old set: KL 0.186 (E-mix 0.226); MMLU 0.741 vs E-mix 0.736 (+0.5 [+0.1, +0.9]), ARC-C 0.607 (+0.9 [-0.9, +2.7]), ARC-E 0.830 (+0.6 [-0.3, +1.6]).
- Suite (`suite_refEmix_HB.json`, `suite_refB_HB.json`), points with paired 95% CI:

| Benchmark | HB | HB vs E-mix | HB vs B |
|---|---|---|---|
| HumanEval | 0.915 | -2.4 [-6.7, +1.8] | -2.4 [-6.1, +0.6] |
| MBPP+ | 0.899 | -1.3 [-3.7, +1.1] | -3.2 [-5.6, -0.8] |
| MATH-500 | 0.910 | +1.0 [-1.2, +3.2] | +0.0 [-2.0, +2.0] |
| GSM8K | 0.949 | -0.5 [-1.4, +0.4] | +0.5 [-0.5, +1.6] |
| IFEval | 0.823 | -0.4 [-2.6, +2.0] | -1.5 [-3.7, +0.6] |
| BFCL simple | 0.943 | +1.0 [+0.0, +2.2] | +0.0 [-1.2, +1.2] |
| BFCL multiple | 0.905 | -1.0 [-3.5, +1.5] | +1.0 [-1.0, +3.5] |
| BFCL parallel | 0.900 | -0.5 [-2.0, +1.0] | -1.0 [-3.5, +1.5] |
| BFCL irrelevance | 0.821 | -0.4 [-3.3, +2.1] | +0.4 [-2.1, +2.9] |
| Global-MMLU he | 0.516 | +1.4 [+0.8, +2.0] | -7.2 [-7.9, -6.4] |
| Global-MMLU ar | 0.585 | +0.9 [-1.1, +2.9] | -9.4 [-12.0, -6.9] |
| Global-MMLU ja | 0.617 | +3.0 [+1.0, +5.1] | -8.7 [-11.2, -6.2] |
| Global-MMLU ru | 0.649 | +0.9 [-1.0, +2.7] | -7.2 [-9.6, -4.8] |
| Global-MMLU de | 0.659 | +1.2 [-0.6, +3.0] | -7.3 [-9.5, -5.0] |
| Global-MMLU es | 0.682 | +1.4 [-0.2, +3.0] | -6.9 [-9.2, -4.6] |

The BF16 heal cuts KL by a fifth and adds 1 to 3 points in every non-English language (significant for Hebrew and Japanese) but leaves the generation tasks within noise. It closes 10% to 25% of E-mix's gap to B per language.

HQ (QAD LoRA heal of E-mix with the deploy-exact numerics, exported by `35_export_nvfp4.py`; 14,004,266,496 bytes, as E-mix). Validation KL in the fake-quant view 0.145 at step 0, 0.100 at the end.
- Per-domain KL vs S: 0.127 [0.120, 0.133], multilingual 0.195 (E-mix 0.170 / 0.256, HB 0.137 / 0.205, HQ0 0.129 / 0.198).
- HQ vs HB, paired: -0.0097 [-0.0102, -0.0092] overall; significant in every domain, largest on instruction (-0.016), tool (-0.012) and multilingual (-0.010) text (`klp_HQ_vs_HB.json`). At the same data and budget, training through the quantizer removes 7% more KL than healing in BF16 and quantizing afterwards; vs E-mix, HQ removes 25% of the KL and HB 19%.
- HQ vs HQ0, paired: -0.0024 [-0.0028, -0.0021], significant in every domain (`klp_HQ_vs_HQ0.json`). Matching the deployed numerics adds about a third on top of mismatched QAD.
- Suite (`suite_ref{HB,Emix,B}_HQ.json`):

| Benchmark | HQ | HQ vs HB | HQ vs E-mix | HQ vs B |
|---|---|---|---|---|
| HumanEval | 0.945 | +3.0 [-0.6, +6.7] | +0.6 [-1.8, +3.0] | +0.6 [-1.2, +3.0] |
| MBPP+ | 0.918 | +1.9 [-0.5, +4.5] | +0.5 [-1.6, +2.9] | -1.3 [-3.7, +1.1] |
| MATH-500 | 0.888 | -2.2 [-4.4, +0.0] | -1.2 [-3.6, +1.0] | -2.2 [-4.4, +0.0] |
| GSM8K | 0.942 | -0.7 [-1.7, +0.2] | -1.1 [-2.2, -0.1] | -0.2 [-1.3, +1.0] |
| IFEval | 0.802 | -2.0 [-4.6, +0.6] | -2.4 [-5.2, +0.4] | -3.5 [-6.5, -0.9] |
| BFCL simple | 0.948 | +0.5 [-0.8, +1.8] | +1.5 [+0.2, +3.0] | +0.5 [-0.5, +1.5] |
| BFCL multiple | 0.900 | -0.5 [-3.0, +1.5] | -1.5 [-4.0, +1.0] | +0.5 [-2.0, +3.0] |
| BFCL parallel | 0.910 | +1.0 [+0.0, +2.5] | +0.5 [-1.0, +2.0] | +0.0 [-2.0, +2.0] |
| BFCL irrelevance | 0.804 | -1.7 [-4.6, +1.2] | -2.1 [-4.6, +0.4] | -1.2 [-4.2, +1.7] |
| Global-MMLU he | 0.520 | +0.4 [-0.1, +1.0] | +1.8 [+1.2, +2.5] | -6.8 [-7.5, -6.0] |
| Global-MMLU ar | 0.593 | +0.8 [-1.1, +2.7] | +1.7 [-0.4, +3.8] | -8.6 [-11.1, -6.1] |
| Global-MMLU ja | 0.609 | -0.8 [-2.7, +1.2] | +2.2 [+0.1, +4.4] | -9.5 [-12.1, -6.9] |
| Global-MMLU ru | 0.648 | -0.1 [-1.8, +1.7] | +0.8 [-1.2, +2.6] | -7.3 [-9.6, -5.0] |
| Global-MMLU de | 0.677 | +1.8 [+0.3, +3.5] | +3.1 [+1.3, +4.8] | -5.4 [-7.5, -3.2] |
| Global-MMLU es | 0.688 | +0.5 [-1.0, +2.2] | +1.9 [+0.3, +3.7] | -6.4 [-8.4, -4.4] |

- Old set: KL 0.173 (HB 0.186, E-mix 0.226); MMLU 0.747 vs HB 0.741 (+0.6 [+0.2, +1.0]), ARC-C 0.625 (+1.9 [+0.3, +3.5]), ARC-E 0.823 (-0.7 [-1.6, +0.3]) (`HQ_vs_HB_mmlu_arc.json`).

Verdict for step 3: QAD is the better heal on the distribution (KL, every domain, both held-out sets) and on English MMLU and ARC-C, but on the generation suite the gain over the BF16 heal is within noise except German (+1.8); MATH-500 (-2.2) and IFEval (-2.0) lean the other way, also within CI. Both heals add 1 to 3 points of non-English knowledge to E-mix; neither closes much of the 5 to 10 point gap to B. The 6M-token LoRA budget was enough to show the ordering, not to recover what pruning removed. Full-parameter QAD on both Sparks was not run: LoRA did not fall short of the full heal in the main study (EL vs EF), and the gap to B is far larger than any heal moved it.

### Self-distillation
- SDa: S generates two samples per prompt for 4,096 decontaminated prompts across the domains (`31_prompts.py`, `30_sd_gen.py`); packed into 3,008 sequences of 2,048 tokens, the student trains on them with the same KL objective, budget and QAD setting as HQ.
- SDb: on-policy. Three rounds over one 376-step schedule (125, 125, 126 steps); each round the current student (LoRA merged, BF16) answers a fresh slice of the prompts, S scores every token, and training continues with QAD on from the previous optimizer state (`32_onpolicy.sh`). Full-vocabulary KL rather than sampled-token KL, to avoid the spurious-agreement failure reported for on-policy distillation.


SDb result (on-policy, QAD, 376 steps over 3 rounds of 1,000, 1,000 and 1,008 sequences; 14,004,266,496 bytes). Heal-validation KL (fake-quant view) 0.145 at the start, 0.115 / 0.112 / 0.111 after each round; HQ reached 0.100 on the same sequences.
- Per-domain KL vs S: 0.134 [0.128, 0.141], multilingual 0.201. Paired vs HQ +0.0076 [+0.0069, +0.0082], worse in every domain, most on tool (+0.018) and philosophy (+0.011) text (`klp_SDb_vs_HQ.json`). vs HB -0.0021 [-0.0027, -0.0015]; vs E-mix -0.035 [-0.038, -0.033].
- Old set: KL 0.181 (HQ 0.173); MMLU 0.744 vs HQ 0.747 (-0.3 [-0.7, +0.1]), ARC-C 0.620 (-0.5 [-2.2, +1.2]), ARC-E 0.829 (+0.6 [-0.3, +1.6]) (`SDb_vs_HQ_mmlu_arc.json`).
- Suite (`suite_ref{HQ,Emix,B}_SDb.json`):

| Benchmark | SDb | SDb vs HQ | SDb vs E-mix | SDb vs B |
|---|---|---|---|---|
| HumanEval | 0.951 | +0.6 [-1.8, +3.7] | +1.2 [-1.8, +4.9] | +1.2 [-1.2, +3.7] |
| MBPP+ | 0.902 | -1.6 [-4.2, +0.8] | -1.1 [-3.4, +1.3] | -2.9 [-5.3, -0.5] |
| MATH-500 | 0.916 | +2.8 [+0.6, +5.0] | +1.6 [-0.8, +4.0] | +0.6 [-1.4, +2.8] |
| GSM8K | 0.938 | -0.5 [-1.6, +0.6] | -1.6 [-2.6, -0.7] | -0.6 [-1.7, +0.4] |
| IFEval | 0.802 | +0.0 [-2.8, +2.8] | -2.4 [-5.2, +0.6] | -3.5 [-6.5, -0.7] |
| BFCL simple | 0.948 | +0.0 [-1.2, +1.2] | +1.5 [+0.0, +3.0] | +0.5 [-0.8, +2.0] |
| BFCL multiple | 0.910 | +1.0 [-1.0, +3.0] | -0.5 [-2.0, +1.0] | +1.5 [+0.0, +3.5] |
| BFCL parallel | 0.905 | -0.5 [-2.5, +1.5] | +0.0 [-2.0, +2.0] | -0.5 [-3.5, +2.5] |
| BFCL irrelevance | 0.825 | +2.1 [-0.4, +4.6] | +0.0 [-2.5, +2.5] | +0.8 [-2.1, +3.8] |
| Global-MMLU he | 0.513 | -0.7 [-1.3, -0.1] | +1.1 [+0.5, +1.8] | -7.5 [-8.2, -6.7] |
| Global-MMLU ar | 0.584 | -0.9 [-2.7, +1.0] | +0.8 [-1.3, +2.9] | -9.5 [-12.1, -6.9] |
| Global-MMLU ja | 0.600 | -0.9 [-2.9, +1.1] | +1.3 [-0.9, +3.5] | -10.4 [-13.1, -7.8] |
| Global-MMLU ru | 0.651 | +0.3 [-1.4, +1.9] | +1.1 [-1.0, +3.0] | -7.0 [-9.5, -4.6] |
| Global-MMLU de | 0.668 | -1.0 [-2.5, +0.5] | +2.1 [+0.2, +3.9] | -6.4 [-8.6, -4.3] |
| Global-MMLU es | 0.685 | -0.3 [-1.8, +1.3] | +1.7 [-0.1, +3.5] | -6.7 [-8.9, -4.5] |

Verdict for SDb: at the same budget, on-policy distillation lands between the BF16 heal and the fixed-corpus QAD heal on KL. Its one benchmark gain over HQ is MATH-500 (+2.8 [+0.6, +5.0]); Hebrew is -0.7 [-1.3, -0.1]; the rest is within noise. The student's own samples cover the KL held-out text (Wikipedia, code files, tool schemas) less well than the mixed corpus does, which the per-domain KL shows most on tool text. Full-vocabulary KL on student samples did not show the spurious-agreement collapse: validation KL fell every round.

Integrity (exit 133): SDb's round-2 teacher pass and every eval hidden-state dump (S, B, E, HQ, ...) exit 133 in vLLM teardown (a fatal GIL error after the EngineCore SIGTERM), after their last write. teacher_r2 logged all 9 shards (8 x 128 + 48 = 1,072 rows, as r2.jsonl) via atomic tmp-then-rename, log mtime 17:59:17 before the 17:59:22 exit; HQ's teacher_mix exited 0 with 24 shards of 128 (3,072 rows, 1,073,743,457 bytes each); the eval dumps assert their shape before saving; the trainers' batch KL stayed in 0.055 to 0.232 with no NaN (zero-filled targets would give about 12), and every per-domain KL file holds 576 sequences. No result is affected.

SDa result (off-policy self-distillation: S's own answers, 3,008 sequences of 2,048 tokens, QAD, 376 steps, 3 h 54 min on one Spark; 14,004,266,496 bytes). Heal-validation KL (fake-quant view) 0.145 at the start, 0.116 / 0.111 / 0.110 at steps 94 / 282 / 376; HQ reached 0.100 and SDb 0.111 on the same sequences.
- Per-domain KL vs S: 0.134 [0.127, 0.140], multilingual 0.200. Paired vs HQ +0.0071 [+0.0065, +0.0077], worse in every domain except Japanese Wikipedia (+0.001, within noise), most on tool (+0.018) and philosophy (+0.010) text (`klp_SDa_vs_HQ.json`). vs SDb -0.0005 [-0.0009, -0.0001]; vs HB -0.0026 [-0.0032, -0.0020].
- Suite (`suite_ref{HQ,SDb,Emix,B}_SDa.json`):

| Benchmark | SDa | SDa vs HQ | SDa vs SDb | SDa vs E-mix | SDa vs B |
|---|---|---|---|---|---|
| HumanEval | 0.939 | -0.6 [-3.0, +1.8] | -1.2 [-3.7, +1.2] | +0.0 [-3.0, +3.0] | +0.0 [-2.4, +2.4] |
| MBPP+ | 0.905 | -1.3 [-3.4, +0.8] | +0.3 [-2.1, +2.6] | -0.8 [-2.9, +1.3] | -2.6 [-5.0, +0.0] |
| MATH-500 | 0.892 | +0.4 [-1.4, +2.4] | -2.4 [-4.8, +0.0] | -0.8 [-3.0, +1.4] | -1.8 [-4.0, +0.4] |
| GSM8K | 0.948 | +0.5 [-0.5, +1.6] | +1.0 [+0.0, +2.0] | -0.6 [-1.6, +0.4] | +0.4 [-0.6, +1.4] |
| IFEval | 0.813 | +1.1 [-1.5, +3.9] | +1.1 [-1.8, +4.1] | -1.3 [-3.7, +1.1] | -2.4 [-5.2, +0.2] |
| BFCL simple | 0.950 | +0.2 [-1.2, +1.8] | +0.2 [-1.2, +1.8] | +1.8 [+0.0, +3.5] | +0.8 [-0.8, +2.2] |
| BFCL multiple | 0.900 | +0.0 [-2.0, +2.0] | -1.0 [-3.0, +1.0] | -1.5 [-3.5, +0.5] | +0.5 [-1.5, +2.5] |
| BFCL parallel | 0.910 | +0.0 [-2.0, +2.0] | +0.5 [-1.0, +2.0] | +0.5 [-1.0, +2.5] | +0.0 [-3.0, +3.0] |
| BFCL irrelevance | 0.812 | +0.8 [-1.7, +3.3] | -1.2 [-3.3, +0.8] | -1.2 [-3.8, +1.2] | -0.4 [-2.9, +2.1] |
| Global-MMLU he | 0.515 | -0.6 [-1.1, +0.0] | +0.1 [-0.4, +0.7] | +1.2 [+0.6, +1.9] | -7.3 [-8.1, -6.6] |
| Global-MMLU ar | 0.588 | -0.5 [-2.3, +1.4] | +0.4 [-1.4, +2.1] | +1.1 [-1.0, +3.2] | -9.1 [-11.7, -6.8] |
| Global-MMLU ja | 0.618 | +0.9 [-1.1, +2.7] | +1.8 [-0.1, +3.6] | +3.1 [+0.9, +5.3] | -8.6 [-11.2, -6.0] |
| Global-MMLU ru | 0.643 | -0.5 [-2.4, +1.2] | -0.8 [-2.5, +1.0] | +0.3 [-1.8, +2.2] | -7.8 [-10.1, -5.5] |
| Global-MMLU de | 0.675 | -0.3 [-2.0, +1.4] | +0.7 [-1.0, +2.4] | +2.8 [+0.8, +4.7] | -5.7 [-7.9, -3.5] |
| Global-MMLU es | 0.677 | -1.1 [-2.7, +0.5] | -0.8 [-2.3, +0.6] | +0.9 [-0.9, +2.6] | -7.5 [-9.6, -5.3] |

Verdict for SDa: off-policy self-distillation matches on-policy (SDb) within 0.0005 KL and loses to the mixed-corpus QAD heal (HQ) by 0.007, worst on tool text. No benchmark separates SDa from HQ; it gives up SDb's MATH-500 gain (-2.4 [-4.8, +0.0]). Self-generated data, sampled once or each round, does not beat the fixed mixed corpus at this budget, so HQ stays the Phase A heal.
- Old set: KL 0.178 (HQ 0.173, SDb 0.181); MMLU 0.747 vs HQ 0.747 (-0.0 [-0.5, +0.4]), ARC-C 0.616 (-0.9 [-2.7, +0.8]), ARC-E 0.834 (+1.1 [+0.3, +2.0]) (`SDa_vs_HQ_mmlu_arc.json`).
## Operations notes

- Three out-of-memory kills on GB10 unified memory cost reruns: three parallel prunes, the full-heal save while optimizer state was resident, and the LoRA save via `save_pretrained`. Both savers now stream tensors and write adapters first. Loading the 61 GB source with `device_map="cuda"` also kept a second resident copy, so the merge capture runs on the CPU.

## Phase A (2026-11-25): calibrate REAP on a domain mix (arm E-mix)

Question: is chat data enough to decide which experts to prune? E-mix uses REAP to 96 experts/layer, then NVFP4, 14.00 GB, the same as E, but with saliency from a mixed calibration set: 817 rows, 580,325 tokens (E: 579,870 of ultrachat). Mix: chat 25%; code, math, reasoning and tool 12.5% each; instruction and philosophy 6% each; wikipedia he/ar/ru/ja/de/es 12.5% in total. 13-gram decontaminated against all evals.

Expert set (`phaseA/emix_diff.json`): 438 of 4,608 kept slots change vs E (5 to 15 per layer; most in layers 18, 19, 45 and 46). All 45 experts with zero chat tokens get traffic under the mix; 24 are kept. By each changed expert's highest-lift domain, E-mix gains 202 multilingual experts and loses 111; chat gains 4 and loses 20. Code and math swap specialists.

Per-domain KL vs S on held-out text (`phaseA/kl_domains_*.json`, 64 sequences per domain):

| Domain | B | E | E-mix |
|---|---|---|---|
| all | 0.053 | 0.650 | 0.170 |
| code | 0.040 | 0.190 | 0.120 |
| math | 0.028 | 0.098 | 0.060 |
| reasoning | 0.042 | 0.073 | 0.069 |
| tool | 0.043 | 0.094 | 0.104 |
| instruction | 0.088 | 0.194 | 0.224 |
| philosophy | 0.051 | 0.199 | 0.184 |
| wiki he | 0.084 | 1.098 | 0.233 |
| wiki ja | 0.055 | 2.647 | 0.371 |
| wiki ru | 0.048 | 2.920 | 0.238 |
| wiki ar / de / es | 0.069 / 0.056 / 0.052 | 1.369 / 1.509 / 0.466 | 0.261 / 0.210 / 0.223 |

Wide suite (`phaseA/suite_vs_{E,B}.json`; one sample per item at the model's recommended sampling, seed 0; paired bootstrap, points, 95% CI):

| Task | S | B | E | E-mix | E-mix vs E | E-mix vs B |
|---|---|---|---|---|---|---|
| Global-MMLU he | 61.2 | 58.8 | 39.4 | 50.2 | [+9.9, +11.7] | [-9.4, -7.8] |
| Global-MMLU ja | 73.3 | 70.3 | 49.9 | 58.7 | [+5.6, +11.9] | [-14.4, -8.9] |
| Global-MMLU ar | 69.1 | 67.9 | 49.3 | 57.6 | [+5.3, +11.3] | [-12.9, -7.7] |
| Global-MMLU de | 74.0 | 73.2 | 57.6 | 64.6 | [+3.9, +10.1] | [-10.8, -6.1] |
| Global-MMLU ru | 73.4 | 72.1 | 61.4 | 64.0 | [+0.2, +5.1] | [-10.4, -5.7] |
| Global-MMLU es | 74.8 | 75.2 | 66.1 | 66.8 | [-1.8, +3.2] | [-10.6, -6.1] |
| IFEval (prompt strict) | 84.1 | 83.7 | 79.7 | 82.6 | [+0.2, +5.7] | [-3.7, +1.3] |
| GSM8K | 94.2 | 94.4 | 94.0 | 95.4 | [+0.4, +2.4] | [0.0, +2.0] |
| MATH-500 | 93.4 | 91.0 | 89.8 | 90.0 | [-2.2, +2.6] | [-3.4, +1.4] |
| HumanEval | 94.5 | 93.9 | 93.3 | 93.9 | [-3.0, +4.9] | [-3.7, +3.0] |
| MBPP+ | 92.1 | 93.1 | 91.3 | 91.3 | [-2.4, +2.4] | [-4.2, +0.3] |
| BFCL simple / multiple | 94.0 / 90.0 | 94.2 / 89.5 | 94.0 / 92.0 | 93.2 / 91.5 | n.s. / n.s. | n.s. / [0.0, +4.5] |
| BFCL parallel / irrelevance | 90.5 / 82.9 | 91.0 / 81.7 | 92.5 / 81.2 | 90.5 / 82.5 | [-4.0, -0.5] / n.s. | n.s. / n.s. |

Old measurements for continuity: E-mix KL 0.226 on the 256-sequence ultrachat+wikitext set (E 0.185; ultrachat 0.194, wikitext 0.258), MMLU 0.7364, ARC-C 0.5981, ARC-E 0.8232 (E 0.7343 / 0.5853 / 0.8220).

Verdict: chat-only calibration is the wrong basis for picking experts. At the same bytes the mix cuts multilingual KL 6.5x (1.668 to 0.256) and wins back 3 to 11 points of Global-MMLU in five of six languages, plus IFEval and GSM8K. English MMLU/ARC are unchanged, and code and tools are unchanged except BFCL parallel (-2). The old chat/wiki KL set favors E, since it is E's own calibration domain: an in-domain metric. B still leads E-mix by 8 to 11 points in every language: pruning 25% costs other languages most, even calibrated.

Operations: E's per-domain hidden dump was OOM-killed once (exit 137) and passed on rerun. Exit 133 after every hidden dump is a crash at interpreter shutdown, after the file is written and checked (KL steps read it fine).

## Close

The Qwen study is complete. A port to a second family, Inkling 2 Mini (shipped in NVFP4/MXFP8, no BF16 release), was stopped after its serving and hidden-state checks. Two findings from it:
- vLLM's pooling runner could not load the model: the pooling adapter drops lm_head and the model's weight loader rejects the tensor. Skipping lm_head in the loader when the model has no lm_head parameter fixes it.
- vLLM is not stable against itself on this model. The same 4 sequences scored as one batch and one at a time differ by mean 0.04 to 0.06, p99 0.5 to 0.66 and max 1.4 to 1.7 nats per token (Qwen3-30B-A3B: max under 0.2). The hidden-state dump matches vLLM within that noise. NVFP4 activations over 48 looped layer applications likely amplify kernel-shape differences; any KL study on this model has to read arm gaps against that floor.

Arm checkpoints were not kept. Every arm rebuilds from the scripts, the REAP reports and the recorded configs.
