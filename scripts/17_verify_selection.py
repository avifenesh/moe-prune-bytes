"""Check which experts an arm actually kept: match its BF16 router rows to the source's.

For each layer, the retained expert indices are recovered by exact row matching of
mlp.gate.weight and compared with the set the selection report intends (the experts
with the highest scores under uniform pruning).

usage: 17_verify_selection.py SRC_DIR ARM_DIR REPORT_JSON KEEP
"""

import glob
import json
import sys

import torch
from safetensors import safe_open


def routers(d):
    out = {}
    for f in glob.glob(f"{d}/*.safetensors"):
        with safe_open(f, "pt") as g:
            for k in g.keys():
                if k.endswith("mlp.gate.weight"):
                    out[int(k.split(".")[2])] = g.get_tensor(k)
    return out


src, arm, rep, keep = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
S, A = routers(src), routers(arm)
scores = json.load(open(rep))["saliency"]
bad = 0
for L in sorted(S):
    rows = [int(torch.nonzero((S[L] == r).all(-1))[0]) for r in A[L]]
    want = sorted(sorted(range(128), key=lambda i: (scores[L][i], i))[128 - keep:])
    if sorted(rows) != want:
        bad += 1
        print("layer", L, "mismatch", len(set(rows) ^ set(want)))
    if rows != sorted(rows):
        print("layer", L, "order not ascending")
print("layers checked", len(S), "mismatching", bad)
