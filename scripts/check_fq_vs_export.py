"""Check nvfp4_fq.fq_weight against a real llm-compressor NVFP4 export: fake-quantize the BF16
weights with fused global scales (gate+up per expert, q+k+v per layer) and compare with the
dequantized checkpoint (FP4 codes x FP8 block scale / global scale). Also reports the static
activation amax the export stores. CPU only.

usage: check_fq_vs_export.py BF16_DIR NVFP4_DIR OUT_JSON
"""

import glob
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_fq import FP4_MAX, FP8_MAX, fq_weight_eager as fq  # noqa: E402

bf, q4, out = sys.argv[1:4]
FP4_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def index(d):
    files = sorted(glob.glob(f"{d}/*.safetensors"))
    m = {}
    for fn in files:
        with safe_open(fn, "pt", device="cpu") as f:
            for k in f.keys():
                m[k] = fn
    return m


ib, iq = index(bf), index(q4)


def get(m, k):
    with safe_open(m[k], "pt", device="cpu") as f:
        return f.get_tensor(k)


def dequant(prefix, low_first=True):
    p = get(iq, prefix + ".weight_packed")
    lo, hi = (p & 0x0F).long(), (p >> 4).long()
    codes = torch.stack((lo, hi) if low_first else (hi, lo), -1).reshape(p.shape[0], -1)
    v = FP4_VALUES[codes]
    s = get(iq, prefix + ".weight_scale").float()
    g = get(iq, prefix + ".weight_global_scale").float()
    return (v.reshape(v.shape[0], -1, 16) * s.unsqueeze(-1)).reshape(v.shape) / g


res = []
for layer in (0, 12, 40, 47):
    groups = [[f"model.layers.{layer}.mlp.experts.{e}.gate_proj", f"model.layers.{layer}.mlp.experts.{e}.up_proj"]
              for e in (0, 5, 77)]
    groups += [[f"model.layers.{layer}.mlp.experts.{e}.down_proj"] for e in (0, 5, 77)]
    groups += [[f"model.layers.{layer}.self_attn.{p}_proj" for p in "qkv"], [f"model.layers.{layer}.self_attn.o_proj"]]
    for g in groups:
        ws = [get(ib, n + ".weight") for n in g]
        gamax = torch.stack([w.float().abs().amax() for w in ws]).max().reshape(1, 1)
        for n, w in zip(g, ws):
            mine = fq(w, gamax).float()
            sep = fq(w, w.float().abs().amax().reshape(1, 1)).float()
            ref_lo, ref_hi = dequant(n, True), dequant(n, False)
            ref = ref_lo if (mine - ref_lo).abs().sum() <= (mine - ref_hi).abs().sum() else ref_hi
            res.append({"name": n, "frac_differ_fused": float((mine.bfloat16().float() != ref.bfloat16().float()).float().mean()),
                        "frac_differ_separate_global": float((sep.bfloat16().float() != ref.bfloat16().float()).float().mean()),
                        "rel_err_fused": float((mine - ref).norm() / ref.norm()),
                        "low_nibble_first": ref is ref_lo})
act = {}
for k in iq:
    if k.endswith("input_global_scale") and (".layers.12." in k):
        act[k] = FP4_MAX * FP8_MAX / float(get(iq, k))
summ = {k: sum(r[k] for r in res) / len(res) for k in ("frac_differ_fused", "frac_differ_separate_global", "rel_err_fused")}
summ["low_nibble_first_all"] = all(r["low_nibble_first"] for r in res)
json.dump({"summary": summ, "tensors": res, "layer12_input_amax": act}, open(out, "w"), indent=1)
print(json.dumps(summ))
print("worst", max(res, key=lambda r: r["frac_differ_fused"]))
