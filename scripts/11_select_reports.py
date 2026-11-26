"""Alternative expert selections, written as REAP-format reports so 02_prune.py applies them.

ExpertPruner(metric="saliency") drops the lowest "saliency" values per layer. Each output
report keeps the original "count" and replaces "saliency" with a selection score:

  F (keep 96): drop the 32 experts with MEDIUM routing frequency. Experts are ranked by
     count (ties broken by expert index); score = |rank - 63.5|, so ranks 48..79 go.
  G (keep 96): REAP saliency, but the bottom 10% by frequency (13 lowest-count experts per
     layer) are protected; the next-lowest-saliency experts are dropped instead.
  H (keep 112): drop the 16 medium-frequency experts (ranks 56..71).

Also writes per-layer agreement between frequency and REAP saliency: Spearman rank
correlation, and the overlap between the "medium by frequency" drop set (F) and the
REAP drop set (E), plus G vs E.

usage: 11_select_reports.py REAP_REPORT OUT_DIR
"""

import json
import sys
from pathlib import Path

rep = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
N = 128


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: (v[i], i))
    r = [0] * len(v)
    for k, i in enumerate(order):
        r[i] = k
    return r


def avg_ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = avg_ranks(a), avg_ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)


def drop_set(score, n_drop):
    return set(sorted(range(N), key=lambda i: (score[i], i))[:n_drop])


F, G, H, stats = [], [], [], []
for L, (cnt, sal) in enumerate(zip(rep["count"], rep["saliency"])):
    r = ranks(cnt)
    f = [abs(x - 63.5) for x in r]
    h = [abs(x - 63.5) for x in r]
    protected = set(sorted(range(N), key=lambda i: (cnt[i], i))[:13])
    big = max(sal) * 10 + 1
    g = [sal[i] + big if i in protected else sal[i] for i in range(N)]
    F.append(f)
    G.append(g)
    H.append(h)
    dE, dF, dG = drop_set(sal, 32), drop_set(f, 32), drop_set(g, 32)
    stats.append({
        "layer": L,
        "spearman_count_vs_saliency": spearman(cnt, sal),
        "zero_count_experts": sum(1 for c in cnt if c == 0),
        "E_drop_overlap_F_drop": len(dE & dF),
        "E_drop_overlap_G_drop": len(dE & dG),
        "E_drop_in_count_bottom13": len(dE & protected),
        "E_drop_count_rank_mean": sum(r[i] for i in dE) / 32,
    })

for name, sc in (("F", F), ("G", G), ("H", H)):
    json.dump({"count": rep["count"], "saliency": sc, "selection": name}, open(out / f"report_{name}.json", "w"))
json.dump(stats, open(out / "selection_stats.json", "w"), indent=1)

rho = [s["spearman_count_vs_saliency"] for s in stats]
print("spearman count vs saliency: mean %.3f min %.3f max %.3f" % (sum(rho) / len(rho), min(rho), max(rho)))
for key in ("E_drop_overlap_F_drop", "E_drop_overlap_G_drop", "E_drop_in_count_bottom13"):
    v = [s[key] for s in stats]
    print(key, "mean %.1f of 32 (min %d max %d)" % (sum(v) / len(v), min(v), max(v)))
v = [s["E_drop_count_rank_mean"] for s in stats]
print("mean count rank of REAP-dropped experts %.1f (0 = rarest, 127 = most used)" % (sum(v) / len(v)))
for s in stats[::8]:
    print(s)
