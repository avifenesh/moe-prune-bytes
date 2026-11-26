"""Arm R: non-uniform per-layer expert budgets from softmax-normalized REAP saliency.

Per layer L: share_{L,i} = softmax(S_{L,i} / tau_L), tau_L = TAU_MULT x median nonzero
saliency of the layer. All 6,144 experts are ranked globally by share and the lowest
1,536 (the same total as E) are removed, with guardrails: every layer keeps at least
MIN_KEEP and at most 128. A layer at its floor stops losing experts; the next-lowest
share elsewhere is removed instead.

Writes a REAP-format report (score 1 = keep, 0 = remove) for ExpertPruner(uniform=False,
sparsity=0.25), and the per-layer kept counts.

usage: 19_select_R.py REAP_REPORT OUT_REPORT OUT_COUNTS_JSON [TAU_MULT] [MIN_KEEP]
"""

import json
import math
import sys

rep = json.load(open(sys.argv[1]))
tau_mult = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
min_keep = int(sys.argv[5]) if len(sys.argv) > 5 else 64
REMOVE = 1536
shares, taus = [], []
for sal in rep["saliency"]:
    nz = sorted(s for s in sal if s > 0)
    tau = tau_mult * nz[len(nz) // 2]
    m = max(sal)
    e = [math.exp((s - m) / tau) for s in sal]
    z = sum(e)
    shares.append([x / z for x in e])
    taus.append(tau)
order = sorted(((shares[L][i], L, i) for L in range(len(shares)) for i in range(128)))
kept = [128] * len(shares)
remove = set()
for share, L, i in order:
    if len(remove) == REMOVE:
        break
    if kept[L] > min_keep:
        remove.add((L, i))
        kept[L] -= 1
assert len(remove) == REMOVE
scores = [[0.0 if (L, i) in remove else 1.0 for i in range(128)] for L in range(len(shares))]
json.dump({"saliency": scores, "count": rep["count"], "selection": f"R tau_mult={tau_mult} min_keep={min_keep}"},
          open(sys.argv[2], "w"))
json.dump({"kept_per_layer": kept, "tau": taus, "tau_mult": tau_mult, "min_keep": min_keep,
           "total_kept": sum(kept)}, open(sys.argv[3], "w"), indent=1)
print("kept per layer:", kept)
print("total kept", sum(kept), "min", min(kept), "max", max(kept), "layers at floor", sum(k == min_keep for k in kept),
      "layers at 128", sum(k == 128 for k in kept))
