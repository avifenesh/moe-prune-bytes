"""E vs E-mix: which experts the domain-mixed calibration keeps differently, and why.

Inputs: the chat-only REAP report (E), the mixed REAP report (E-mix), and the source's
per-domain routing counts on the mixed calibration rows (25_route_counts.py).
For each expert, domain lift = (share of that domain's routed slots going to the expert)
/ (the expert's share of all routed slots). An expert "serves" the domain with the
highest lift.

usage: 28_emix_diff.py REPORT_E REPORT_MIX ROUTE_COUNTS OUT_JSON [KEEP]
"""

import json
import sys
from collections import Counter

rep_e, rep_m, rc_path, out = sys.argv[1:5]
keep = int(sys.argv[5]) if len(sys.argv) > 5 else 96
E, M, rc = json.load(open(rep_e)), json.load(open(rep_m)), json.load(open(rc_path))
doms = sorted(rc["counts"])
L = len(E["saliency"])


def kept(rep, layer):
    s = rep["saliency"][layer]
    return set(sorted(range(128), key=lambda i: (s[i], i))[128 - keep:])


per_layer, gained_dom, lost_dom = [], Counter(), Counter()
zero_e = [(l, i) for l in range(L) for i in range(128) if E["count"][l][i] == 0]
for layer in range(L):
    ke, km = kept(E, layer), kept(M, layer)
    gained, lost = sorted(km - ke), sorted(ke - km)
    tot = [sum(rc["counts"][d][layer][i] for d in doms) for i in range(128)]
    all_tot = sum(tot)

    def serves(i):
        if tot[i] == 0:
            return None
        lifts = {}
        for d in doms:
            dt = sum(rc["counts"][d][layer])
            lifts[d] = (rc["counts"][d][layer][i] / dt) / (tot[i] / all_tot) if dt else 0.0
        return max(lifts, key=lifts.get)

    g = [serves(i) for i in gained]
    lo = [serves(i) for i in lost]
    gained_dom.update(x for x in g if x)
    lost_dom.update(x for x in lo if x)
    per_layer.append({"layer": layer, "n_changed": len(gained), "gained": gained, "lost": lost,
                      "gained_serves": g, "lost_serves": lo})
n_changed = sum(p["n_changed"] for p in per_layer)
zero_now = [(l, i) for (l, i) in zero_e if M["count"][l][i] > 0]
zero_kept = [(l, i) for (l, i) in zero_e if i in kept(M, l)]
res = {"keep": keep, "changed_total": n_changed, "changed_per_layer": [p["n_changed"] for p in per_layer],
       "gained_by_domain": dict(gained_dom.most_common()), "lost_by_domain": dict(lost_dom.most_common()),
       "zero_token_in_E": len(zero_e), "zero_token_with_traffic_in_mix": len(zero_now),
       "zero_token_kept_in_Emix": len(zero_kept), "per_layer": per_layer,
       "route_tokens": rc["tokens"]}
json.dump(res, open(out, "w"), indent=1)
print("changed experts:", n_changed, "of", L * keep, "kept slots; per layer", res["changed_per_layer"])
print("gained experts serve:", res["gained_by_domain"])
print("lost experts serve:", res["lost_by_domain"])
print("zero-token experts in E:", len(zero_e), "with traffic in mix:", len(zero_now), "kept in E-mix:", len(zero_kept))
