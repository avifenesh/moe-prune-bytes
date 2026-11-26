"""Exact weight bytes of a checkpoint, by tensor group, from safetensors headers.

usage: 07_bytes.py MODEL_DIR [OUT.json]
"""

import glob
import json
import re
import struct
import sys

d = sys.argv[1]
groups = {"experts": 0, "attention": 0, "embed": 0, "lm_head": 0, "router": 0, "other": 0}
n_tensors = 0
for f in sorted(glob.glob(f"{d}/*.safetensors")):
    with open(f, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        hdr = json.loads(h.read(n))
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        n_tensors += 1
        b = meta["data_offsets"][1] - meta["data_offsets"][0]
        if ".mlp.experts." in name:
            g = "experts"
        elif ".self_attn." in name:
            g = "attention"
        elif "embed_tokens" in name:
            g = "embed"
        elif name.startswith("lm_head"):
            g = "lm_head"
        elif re.search(r"mlp\.gate\.", name):
            g = "router"
        else:
            g = "other"
        groups[g] += b
res = {"dir": d, "n_tensors": n_tensors, "bytes": groups, "total_bytes": sum(groups.values()),
       "total_gib": sum(groups.values()) / 2**30}
print(json.dumps(res, indent=1))
if len(sys.argv) > 2:
    json.dump(res, open(sys.argv[2], "w"), indent=1)
