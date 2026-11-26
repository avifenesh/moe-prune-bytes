"""Per-category and per-subject MMLU accuracy for several arms, from lm-eval samples.

Reports, for each arm, accuracy per MMLU category (stem, humanities, social sciences,
other) with a paired-bootstrap CI of the difference vs a reference arm, and per-subject
accuracy. Subjects where an arm beats the reference by the most are listed, to see
whether a selection protects specific domains.

usage: 12_mmlu_subjects.py RESULTS_ROOT GROUPS_JSON OUT.json REF ARM [ARM...]
  GROUPS_JSON: an lm-eval results_*.json (for group_subtasks)
"""

import glob
import json
import random
import sys
from collections import defaultdict

root, groups_json, out, ref, *arms = sys.argv[1:]
groups = json.load(open(groups_json))["group_subtasks"]
cat_of = {s: c for c in ("mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other") for s in groups[c]}


def load(arm):
    items = {}
    for f in glob.glob(f"{root}/{arm}/lmeval/ll/**/samples_mmlu_*.jsonl", recursive=True):
        task = f.split("samples_")[1].rsplit("_", 1)[0]
        for line in open(f):
            s = json.loads(line)
            items[(task, s["doc_id"])] = float(s["acc"])
    return items


def boot(d, B=4000):
    rng = random.Random(0)
    n = len(d)
    m = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [m[int(0.025 * B)], m[int(0.975 * B)]]


data = {a: load(a) for a in [ref] + arms}
res = {}
for a in [ref] + arms:
    cats, subs = {}, {}
    for c in ("mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"):
        keys = [k for k in data[ref] if cat_of.get(k[0]) == c and k in data[a]]
        d = [data[a][k] - data[ref][k] for k in keys]
        cats[c] = {"n": len(keys), "acc": sum(data[a][k] for k in keys) / len(keys),
                   "delta_vs_ref": sum(d) / len(d), "delta_ci95": boot(d) if a != ref else [0, 0]}
    by = defaultdict(list)
    for k, v in data[a].items():
        by[k[0]].append(v)
    for s, v in by.items():
        subs[s] = sum(v) / len(v)
    res[a] = {"categories": cats, "subjects": subs}

ref_sub = res[ref]["subjects"]
for a in arms:
    diffs = sorted(((res[a]["subjects"][s] - ref_sub[s], s) for s in ref_sub), reverse=True)
    res[a]["top_gain_subjects_vs_ref"] = [(s, round(x, 4)) for x, s in diffs[:8]]
    res[a]["top_loss_subjects_vs_ref"] = [(s, round(x, 4)) for x, s in diffs[-8:]]
json.dump(res, open(out, "w"), indent=1)
for a in [ref] + arms:
    print(a, {c[5:]: (round(v["acc"], 4), round(100 * v["delta_vs_ref"], 1), [round(100 * x, 1) for x in v["delta_ci95"]])
              for c, v in res[a]["categories"].items()})
    if a != ref:
        print("  gains", res[a]["top_gain_subjects_vs_ref"][:5])
        print("  losses", res[a]["top_loss_subjects_vs_ref"][-5:])
