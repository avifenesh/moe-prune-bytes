"""Write per-layer expert counts into a pruned checkpoint's config for the patched vLLM.

Sets config["num_experts_per_layer"] from the checkpoint's own router shapes (checked
against the selection counts) and num_experts to the maximum. Drops the
moe_sparsity_config that ExpertPruner writes, which this vLLM build does not read.

usage: 21_set_per_layer.py CKPT_DIR COUNTS_JSON
"""

import glob
import json
import struct
import sys

d, counts_path = sys.argv[1:3]
want = json.load(open(counts_path))["kept_per_layer"]
rows = {}
for f in glob.glob(f"{d}/*.safetensors"):
    with open(f, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        hdr = json.loads(h.read(n))
    for k, v in hdr.items():
        if k.endswith("mlp.gate.weight"):
            rows[int(k.split(".")[2])] = v["shape"][0]
got = [rows[i] for i in range(len(rows))]
assert got == want, (got, want)
cfg = json.load(open(f"{d}/config.json"))
cfg["num_experts_per_layer"] = got
cfg["num_experts"] = max(got)
qc = cfg.get("quantization_config", {})
qc.pop("moe_sparsity_config", None)
json.dump(cfg, open(f"{d}/config.json", "w"), indent=2)
print("per-layer counts set", sum(got), min(got), max(got))
