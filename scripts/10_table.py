"""Collect bytes, KL, task accuracy (paired bootstrap vs S) and speed into one table.

usage: 10_table.py MPB_ROOT OUT.md OUT.json   (MPB_ROOT holds runs/ and results/)
Run 09_bootstrap.py first; it writes results/bootstrap.json.
"""

import json
import os
import statistics
import sys

root, out_md, out_json = sys.argv[1:4]
ARMS = [("S", 128, "BF16"), ("B", 128, "NVFP4"), ("D", 72, "FP8"), ("A", 64, "BF16"),
        ("E", 96, "NVFP4"), ("C", 64, "NVFP4")]
boot = json.load(open(f"{root}/results/bootstrap.json")) if os.path.exists(f"{root}/results/bootstrap.json") else {}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


rows = []
for a, k, prec in ARMS:
    r = {"arm": a, "experts": k, "precision": prec}
    b = load(f"{root}/runs/bytes_{a}.json")
    if b:
        r["weight_bytes"] = b["total_bytes"]
        r["expert_bytes"] = b["bytes"]["experts"]
    kl = load(f"{root}/results/{a}/kl.json")
    if kl:
        r["kl"] = kl["summary"]
    r["tasks"] = boot.get(a)
    sp = [load(f"{root}/results/{a}/speed_{o}.json") for o in ("fwd", "rev")]
    sp = [s for s in sp if s]
    if sp:
        pre = [x for s in sp for x in s["prefill_tok_s"]["samples"]]
        dec = [x for s in sp for x in s["decode_tok_s"]["samples"]]
        r["speed"] = {"runs": len(sp), "n": len(pre), "prefill_median": statistics.median(pre),
                      "decode_median": statistics.median(dec),
                      "prefill_run_medians": [s["prefill_tok_s"]["median"] for s in sp],
                      "decode_run_medians": [s["decode_tok_s"]["median"] for s in sp],
                      "peak_gib": max(s["torch_peak_gib_incl_kv_pool"] for s in sp)}
    rows.append(r)
json.dump(rows, open(out_json, "w"), indent=1)


def f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def task(r, t):
    v = (r.get("tasks") or {}).get(t)
    if not v:
        return "n/a"
    s = f"{v['acc']:.4f}"
    if r["arm"] != "S":
        lo, hi = v["delta_ci95"]
        s += f" ({100 * v['delta_vs_ref']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}])"
    return s


L = ["| Arm | Experts | Precision | Weight GB | KL all | KL ultrachat | KL wikitext | Top-1 agree | ARC-C | ARC-E | MMLU | GSM8K | Prefill tok/s | Decode tok/s | Peak GiB |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    kl = r.get("kl") or {}

    def klc(g):
        v = kl.get(g)
        return "0" if r["arm"] == "S" else ("n/a" if not v else f"{v['kl'][0]:.4f} [{v['kl'][1]:.4f}, {v['kl'][2]:.4f}]")

    t1 = kl.get("all", {}).get("top1")
    sp = r.get("speed") or {}
    L.append("| " + " | ".join([
        r["arm"], str(r["experts"]), r["precision"],
        f(r.get("weight_bytes") and r["weight_bytes"] / 1e9, 2),
        klc("all"), klc("ultrachat"), klc("wikitext"),
        "1" if r["arm"] == "S" else ("n/a" if not t1 else f"{t1[0]:.3f}"),
        task(r, "arc_challenge"), task(r, "arc_easy"), task(r, "mmlu"), task(r, "gsm8k_cot"),
        f(sp.get("prefill_median"), 0), f(sp.get("decode_median"), 1), f(sp.get("peak_gib"), 1),
    ]) + " |")
open(out_md, "w").write("\n".join(L) + "\n")
print("\n".join(L))
