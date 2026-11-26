"""Paired bootstrap of per-domain KL differences between two arms (same 576 sequences).
Mean over sequences of per-sequence KL, as 05_kl.py reports it.

usage: 37_kl_paired.py REF_kl_domains.json ARM_kl_domains.json [OUT_JSON]
"""

import json
import random
import sys

B = 4000
ref = json.load(open(sys.argv[1]))["per_seq"]
arm = json.load(open(sys.argv[2]))["per_seq"]
assert len(ref) == len(arm) and all(a["src"] == r["src"] for a, r in zip(arm, ref))


def boot(d):
    rng = random.Random(0)
    n = len(d)
    m = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [sum(d) / n, m[int(0.025 * B)], m[int(0.975 * B)]]


groups = ["all"] + sorted({r["src"] for r in ref}) + ["multilingual"]
res = {}
for g in groups:
    idx = [i for i, r in enumerate(ref) if g == "all" or r["src"] == g or (g == "multilingual" and r["src"].startswith("wiki_"))]
    d = [arm[i]["kl"] - ref[i]["kl"] for i in idx]
    rel = [arm[i]["kl"] / ref[i]["kl"] for i in idx]
    res[g] = {"n": len(idx), "dkl": boot(d), "mean_ratio": sum(rel) / len(rel)}
if len(sys.argv) > 3:
    json.dump(res, open(sys.argv[3], "w"), indent=1)
for g in ("all", "multilingual", "code", "math", "tool", "instruction", "philosophy", "reasoning"):
    v = res[g]["dkl"]
    print(f"{g}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")
