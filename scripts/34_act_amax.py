"""Static NVFP4 activation amax per GEMM input, read from a compressed-tensors NVFP4 checkpoint,
reduced the way vLLM runs it: q/k/v share one input scale (qkv_proj); o_proj its own; for the
experts one value per layer for the gate/up input and one for the down input (max amax over
experts, vLLM's amax_for_moe_activation_quant). amax = 6 * 448 / input_global_scale.

usage: 34_act_amax.py NVFP4_DIR OUT_JSON
"""

import glob
import json
import re
import sys
from collections import defaultdict

from safetensors import safe_open

d, out = sys.argv[1:3]
G = 6.0 * 448.0
amax = defaultdict(lambda: defaultdict(float))
for fn in sorted(glob.glob(f"{d}/*.safetensors")):
    with safe_open(fn, "pt", device="cpu") as f:
        for k in f.keys():
            if not k.endswith(".input_global_scale"):
                continue
            m = re.match(r"model\.layers\.(\d+)\.(.*)\.input_global_scale", k)
            layer, mod = int(m.group(1)), m.group(2)
            a = G / float(f.get_tensor(k).float().max())
            if mod.startswith("self_attn.") and mod.split(".")[1] in ("q_proj", "k_proj", "v_proj"):
                key = "qkv"
            elif mod == "self_attn.o_proj":
                key = "o"
            elif mod.endswith("gate_proj") or mod.endswith("up_proj"):
                key = "w13"
            elif mod.endswith("down_proj"):
                key = "w2"
            else:
                raise ValueError(k)
            # qkv: vLLM takes max(input_global_scale) = min amax; experts: max amax
            if key == "qkv":
                amax[layer][key] = a if amax[layer][key] == 0 else min(amax[layer][key], a)
            else:
                amax[layer][key] = max(amax[layer][key], a)
n = max(amax) + 1
res = {k: [amax[i][k] for i in range(n)] for k in ("qkv", "o", "w13", "w2")}
json.dump({"source": d, **res}, open(out, "w"), indent=1)
print({k: (round(min(v), 3), round(max(v), 3)) for k, v in res.items()})
