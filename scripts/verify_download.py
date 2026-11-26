"""Verify a downloaded HF checkpoint: every shard in the index exists, header parses,
data size matches the header, total matches index total_size. Optional sha256
against the hub's LFS oids (pass a JSON file {filename: sha256}).

usage: verify_download.py MODEL_DIR [SHAS_JSON]
"""

import hashlib
import json
import os
import struct
import sys

d = sys.argv[1]
shas = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
idx = json.load(open(f"{d}/model.safetensors.index.json"))
files = sorted(set(idx["weight_map"].values()))
bad, total = [], 0
for f in files:
    p = f"{d}/{f}"
    if not os.path.exists(p):
        bad.append((f, "missing"))
        continue
    size = os.path.getsize(p)
    with open(p, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        hdr = json.loads(h.read(n))
    end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
    if size != 8 + n + end:
        bad.append((f, f"size {size} != {8 + n + end}"))
    total += end
    if f in shas:
        hsh = hashlib.sha256()
        with open(p, "rb") as h:
            for chunk in iter(lambda: h.read(1 << 24), b""):
                hsh.update(chunk)
        if hsh.hexdigest() != shas[f]:
            bad.append((f, "sha256 mismatch"))
print("files", len(files), "tensor bytes", total, "index total_size", idx["metadata"]["total_size"])
print("BAD" if bad or total != idx["metadata"]["total_size"] else "OK", bad)
