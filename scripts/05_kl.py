"""Full-vocabulary KL, NLL and top-1 agreement of an arm against the source.

Logits = hidden @ lm_head.T with the source BF16 lm_head (fp32 math). The arm's
lm_head must be bit-identical to the source; this is checked.
Per-sequence means are saved so CIs can be bootstrapped over sequences.

usage: 05_kl.py SRC_DIR SRC_HIDDEN.pt ARM_DIR ARM_HIDDEN.pt HELDOUT_JSONL OUT.json
"""

import glob
import json
import random
import sys

import torch
from safetensors import safe_open


def lm_head(d):
    for f in glob.glob(f"{d}/*.safetensors"):
        with safe_open(f, "pt") as g:
            if "lm_head.weight" in g.keys():
                return g.get_tensor("lm_head.weight")
    raise SystemExit(f"no lm_head in {d}")


src_dir, src_h, arm_dir, arm_h, held, out = sys.argv[1:7]
W = lm_head(src_dir)
Wa = lm_head(arm_dir)
assert W.dtype == Wa.dtype == torch.bfloat16 and torch.equal(W, Wa), "lm_head differs"
W = W.cuda().float()
S = torch.load(src_h)
A = torch.load(arm_h)
rows = [json.loads(l) for l in open(held)]
assert S["src"] == A["src"]

per = []
with torch.no_grad():
    for i, r in enumerate(rows):
        tgt = torch.tensor(r["ids"][1:], device="cuda")
        ls = torch.log_softmax(S["hidden"][i, :-1].cuda().float() @ W.T, -1)
        la = torch.log_softmax(A["hidden"][i, :-1].cuda().float() @ W.T, -1)
        kl = (ls.exp() * (ls - la)).sum(-1)
        per.append({
            "src": r["src"],
            "kl": kl.mean().item(),
            "nll_src": -ls.gather(1, tgt[:, None]).mean().item(),
            "nll_arm": -la.gather(1, tgt[:, None]).mean().item(),
            "top1": (ls.argmax(-1) == la.argmax(-1)).float().mean().item(),
        })


def ci(vals, B=2000):
    rng = random.Random(0)
    n = len(vals)
    m = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [sum(vals) / n, m[int(0.025 * B)], m[int(0.975 * B)]]


summary = {}
groups = ["all"] + sorted({p["src"] for p in per})
if any(g.startswith("wiki_") for g in groups):
    groups.append("multilingual")
for grp in groups:
    sel = [p for p in per if grp == "all" or p["src"] == grp or (grp == "multilingual" and p["src"].startswith("wiki_"))]
    summary[grp] = {
        "n_seq": len(sel),
        "kl": ci([p["kl"] for p in sel]),
        "top1": ci([p["top1"] for p in sel]),
        "nll_arm": ci([p["nll_arm"] for p in sel]),
        "nll_src": sum(p["nll_src"] for p in sel) / len(sel),
        "dnll": ci([p["nll_arm"] - p["nll_src"] for p in sel]),
    }
json.dump({"summary": summary, "per_seq": per}, open(out, "w"), indent=1)
print(json.dumps(summary, indent=1))
