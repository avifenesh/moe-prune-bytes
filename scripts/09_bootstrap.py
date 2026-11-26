"""Paired bootstrap of task accuracy deltas, each arm vs the source.

Reads lm-eval --log_samples output. Items are paired by (task, doc_id).
Metrics: arc_challenge / arc_easy acc_norm, mmlu acc (micro average over all
57 subjects), gsm8k_cot_llama exact_match with the strict-match filter.

usage: 09_bootstrap.py RESULTS_ROOT OUT.json ARM [ARM...]
  RESULTS_ROOT/<arm>/lmeval/{ll,gsm}/... must exist; the first ARM is the reference.
"""

import glob
import json
import random
import sys
from collections import defaultdict

root, out, *arms = sys.argv[1:]
B = 10000


def load(arm):
    items = defaultdict(dict)  # bench -> {(task, doc_id): value}
    for f in glob.glob(f"{root}/{arm}/lmeval/**/samples_*.jsonl", recursive=True):
        for line in open(f):
            s = json.loads(line)
            task = f.split("samples_")[1].rsplit("_", 1)[0]
            if task.startswith("mmlu_"):
                bench, key = "mmlu", "acc"
            elif task in ("arc_challenge", "arc_easy"):
                bench, key = task, "acc_norm"
            elif task == "gsm8k_cot_llama":
                if s.get("filter") != "strict-match":
                    continue
                bench, key = "gsm8k_cot", "exact_match"
            else:
                continue
            items[bench][(task, s["doc_id"])] = float(s[key])
    return items


data = {a: load(a) for a in arms}
ref = arms[0]
res = {}
for a in arms:
    res[a] = {}
    for bench, ref_items in data[ref].items():
        cur = data[a].get(bench, {})
        keys = sorted(set(ref_items) & set(cur))
        if not keys:
            continue
        x = [cur[k] for k in keys]
        y = [ref_items[k] for k in keys]
        d = [xi - yi for xi, yi in zip(x, y)]
        n = len(keys)
        rng = random.Random(0)
        boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
        acc_boot = sorted(sum(x[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
        res[a][bench] = {
            "n": n,
            "acc": sum(x) / n,
            "acc_ci95": [acc_boot[int(0.025 * 2000)], acc_boot[int(0.975 * 2000)]],
            "delta_vs_ref": sum(d) / n,
            "delta_ci95": [boots[int(0.025 * B)], boots[int(0.975 * B)]],
            "n_ref_items": len(ref_items),
        }
json.dump(res, open(out, "w"), indent=1)
for a in arms:
    print(a, {b: (round(v["acc"], 4), [round(c, 4) for c in v["delta_ci95"]]) for b, v in res[a].items()})
