"""Export a BF16 checkpoint to compressed-tensors NVFP4 with exactly the numerics QAD trains on
(nvfp4_fq.py): block scales from the weight, one global scale per fused GEMM (gate+up per
expert, q+k+v per layer), static activation scales taken from a template NVFP4 checkpoint.
The template (an llm-compressor export of the same architecture) supplies config, tensor names
and input_global_scale; quantized weights and every unquantized tensor (router, norms,
embeddings, lm_head) come from the BF16 model.

usage: 35_export_nvfp4.py BF16_DIR TEMPLATE_NVFP4_DIR OUT_DIR
"""

import glob
import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_fq import FP4_MAX, FP8_MAX  # noqa: E402

src, tmpl, out = sys.argv[1:4]
dev = os.environ.get("EXPORT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAG = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=dev)  # FP4 E2M1 magnitudes, codes 0-7; sign is bit 3


def index(d):
    m = {}
    for fn in sorted(glob.glob(f"{d}/*.safetensors")):
        with safe_open(fn, "pt", device="cpu") as f:
            for k in f.keys():
                m[k] = fn
    return m


isrc, itm = index(src), index(tmpl)
handles = {}


def get(m, k):
    fn = m[k]
    if fn not in handles:
        handles[fn] = safe_open(fn, "pt", device="cpu")
    return handles[fn].get_tensor(k)


def quantize(w, gamax):
    """Same ops as nvfp4_fq._blocks; returns packed uint8 [out, in/2], fp8 scale [out, in/16], global."""
    wf = w.to(dev).float()
    gs = gamax / (FP4_MAX * FP8_MAX)
    b = wf.reshape(wf.shape[0], -1, 16)
    s8 = (b.abs().amax(-1, keepdim=True) / FP4_MAX / gs).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
    s = s8.float() * gs
    s = torch.where(s == 0, torch.ones_like(s), s)
    a = b.abs() / s
    q = torch.where(a < 2, torch.round(a * 2) / 2, torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2)).clamp(max=FP4_MAX)
    code = torch.where(q < 2, q * 2, torch.where(q < 4, q + 2, q / 2 + 4)).long()  # index in MAG
    code = code + 8 * (b < 0).long() * (q > 0).long()
    code = code.reshape(wf.shape[0], -1).to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.cpu(), s8.squeeze(-1).cpu(), torch.tensor([1.0 / gs], dtype=torch.float32)


qmods = sorted({k[: -len(".weight_packed")] for k in itm if k.endswith(".weight_packed")})
groups, seen = [], set()
for m in qmods:
    if m in seen:
        continue
    g = [m]
    mm = re.match(r"(.*\.)(gate_proj|up_proj)$", m)
    if mm:
        g = [mm.group(1) + "gate_proj", mm.group(1) + "up_proj"]
    mm = re.match(r"(.*\.self_attn\.)(q_proj|k_proj|v_proj)$", m)
    if mm:
        g = [mm.group(1) + p for p in ("q_proj", "k_proj", "v_proj")]
    seen.update(g)
    groups.append(g)

os.makedirs(out, exist_ok=True)
for f in os.listdir(tmpl):
    if not f.endswith(".safetensors") and not f.endswith(".safetensors.index.json"):
        shutil.copy(os.path.join(tmpl, f), os.path.join(out, f))

shard, shard_bytes, n_shard, weight_map = {}, 0, 0, {}


def flush():
    global shard, shard_bytes, n_shard
    if not shard:
        return
    name = f"model-{n_shard:05d}.safetensors"
    save_file(shard, os.path.join(out, name), metadata={"format": "pt"})
    for k in shard:
        weight_map[k] = name
    n_shard += 1
    shard, shard_bytes = {}, 0


def add(k, t):
    global shard_bytes
    shard[k] = t.contiguous()
    shard_bytes += t.numel() * t.element_size()
    if shard_bytes > 4e9:
        flush()


done = set()
with torch.no_grad():
    for g in groups:
        ws = [get(isrc, m + ".weight") for m in g]
        gamax = max(float(w.float().abs().amax()) for w in ws)
        for m, w in zip(g, ws):
            p, s8, gsc = quantize(w, gamax)
            add(m + ".weight_packed", p)
            add(m + ".weight_scale", s8)
            add(m + ".weight_global_scale", gsc)
            add(m + ".input_global_scale", get(itm, m + ".input_global_scale"))
            done.update(m + x for x in (".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale"))
    for k in itm:
        if k in done:
            continue
        t = get(isrc, k)
        ref = get(itm, k)
        assert t.shape == ref.shape and t.dtype == ref.dtype, (k, t.shape, ref.shape, t.dtype, ref.dtype)
        add(k, t)
        done.add(k)
flush()
missing = set(itm) - done
assert not missing, list(missing)[:5]
json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
print("exported", len(done), "tensors,", n_shard, "shards, groups", len(groups))
