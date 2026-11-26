"""Phase A scoring: per-item metrics from each arm's suite and paired bootstrap vs references.

Metrics (one sample per item, recommended sampling, seed 0):
  humaneval   humaneval_instruct pass@1            mbpp        mbpp_plus_instruct_fix pass_at_1
  math500     minerva_math500 math_verify           gsm8k       gsm8k_cot_llama strict-match
  ifeval      prompt_level_strict_acc               global_mmlu_he  acc (all 57 subjects)
  global_mmlu_{ar,ja,ru,de,es} acc (20 per subject) bfcl_{simple,multiple,parallel,irrelevance}
Items are paired by (task, doc_id) or BFCL id.

usage: 33_suiteA.py RESULTS_ROOT OUT_JSON REF ARM [ARM...]
  RESULTS_ROOT/<arm>/suite/{gen,he,ml}/**/samples_*.jsonl and suite/bfcl.jsonl
"""

import glob
import json
import random
import sys
from collections import defaultdict

root, out, *arms = sys.argv[1:]
B = 4000
GEN_KEYS = {"humaneval_instruct": ("humaneval", "pass@1"), "mbpp_plus_instruct_fix": ("mbpp", "pass_at_1"),
            "minerva_math500": ("math500", "math_verify"), "gsm8k_cot_llama": ("gsm8k", "exact_match"),
            "ifeval": ("ifeval", "prompt_level_strict_acc")}


def load(arm):
    items = defaultdict(dict)
    base = f"{root}/{arm}/suite"
    for f in glob.glob(f"{base}/**/samples_*.jsonl", recursive=True):
        task = f.split("samples_")[1].rsplit("_", 1)[0]
        for line in open(f):
            s = json.loads(line)
            if task in GEN_KEYS:
                bench, key = GEN_KEYS[task]
                if task == "gsm8k_cot_llama" and s.get("filter") != "strict-match":
                    continue
                v = s.get(key)
            elif task.startswith("global_mmlu_full_"):
                lang = task.split("_")[3]
                bench, v = f"global_mmlu_{lang}", s.get("acc")
            else:
                continue
            if v is None:
                continue
            items[bench][(task, s["doc_id"])] = float(v)
    bf = f"{base}/bfcl.jsonl"
    try:
        for line in open(bf):
            r = json.loads(line)
            items[f"bfcl_{r['cat']}"][r["id"]] = float(r["correct"])
    except FileNotFoundError:
        pass
    return items


def boot(d):
    rng = random.Random(0)
    n = len(d)
    m = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [m[int(0.025 * B)], m[int(0.975 * B)]]


data = {a: load(a) for a in arms}
ref = arms[0]
res = {}
for a in arms:
    res[a] = {}
    for bench in sorted(set(data[ref]) | set(data[a])):
        cur, rf = data[a].get(bench, {}), data[ref].get(bench, {})
        keys = sorted(set(cur) & set(rf))
        if not cur:
            continue
        acc = sum(cur.values()) / len(cur)
        entry = {"n": len(cur), "acc": acc}
        if keys and a != ref:
            d = [cur[k] - rf[k] for k in keys]
            entry.update({"n_paired": len(keys), "delta_vs_ref": sum(d) / len(d), "delta_ci95": boot(d)})
        res[a][bench] = entry
json.dump(res, open(out, "w"), indent=1)
for a in arms:
    print(a, {b: (round(v["acc"], 4), [round(100 * c, 1) for c in v.get("delta_ci95", [])]) for b, v in res[a].items()})
