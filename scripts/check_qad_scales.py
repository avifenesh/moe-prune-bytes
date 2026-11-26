"""How much does freezing NVFP4 weight scales at the base weight (as QAD trains) differ from the
scales the real export computes from the healed weight? Compares, for a few experts and the
attention projections of some layers: fake-quant(W_healed, scales(W_base)) vs
fake-quant(W_healed, scales(W_healed)) (the export's numerics). CPU only, small reads.

usage: check_qad_scales.py BASE_DIR HEALED_DIR OUT_JSON
"""

import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_fq import fq_weight_eager as fq, weight_scales  # noqa: E402

base, healed, out = sys.argv[1:4]


def opener(d):
    wm = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    cache = {}

    def get(k):
        f = wm[k]
        if f not in cache:
            cache[f] = safe_open(f"{d}/{f}", "pt", device="cpu")
        return cache[f].get_tensor(k)
    return get


gb, gh = opener(base), opener(healed)
res = []
for layer in (0, 12, 24, 36, 47):
    names = [f"model.layers.{layer}.mlp.experts.{e}.{p}.weight" for e in (0, 31, 63, 95)
             for p in ("gate_proj", "up_proj", "down_proj")]
    names += [f"model.layers.{layer}.self_attn.{p}.weight" for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
    for n in names:
        wb, wh = gb(n), gh(n)
        q_frozen = fq(wh, weight_scales(wb)).float()
        q_export = fq(wh, weight_scales(wh)).float()
        q_base = fq(wb, weight_scales(wb)).float()
        diff = (q_frozen != q_export).float().mean().item()
        moved = (q_export != q_base).float().mean().item()
        rel = ((q_frozen - q_export).norm() / q_export.norm()).item()
        rel_delta = ((wh.float() - wb.float()).norm() / wb.float().norm()).item()
        res.append({"name": n, "frac_elems_differ": diff, "rel_err_frozen_vs_export": rel,
                    "frac_codes_moved_by_heal": moved, "rel_weight_delta": rel_delta})
summ = {k: sum(r[k] for r in res) / len(res) for k in res[0] if k != "name"}
json.dump({"summary": summ, "tensors": res}, open(out, "w"), indent=1)
print(json.dumps(summ))
