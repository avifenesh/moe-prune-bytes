"""Arm M: 96 experts per layer, keep REAP's top-96 as centroids, drop only experts with
zero calibration tokens, and merge every other low-saliency expert into a centroid.

Per layer (statistics on calibration tokens from calib.jsonl, the same set as every arm):
  1. Centroids = the 96 highest-REAP-saliency experts (E's kept set). Router rows of the
     centroids are kept unchanged, as in REAP and REAM. A top-k linear router cannot send a
     removed expert's tokens to its centroid without duplicate expert slots.
  2. Experts with zero calibration tokens are dropped.
  3. Each other non-centroid j goes to its most similar centroid c, similarity =
     cos(gate logits over tokens) + mean over tokens of cos(sigma_c E_c(x), sigma_j E_j(x))
     (REAM, Eq. 7-8), at most CAP members per centroid (REAM's C = 16 for 96 experts).
  4. Neuron alignment: Hungarian matching of j's intermediate neurons to c's on
     cost = C_act + C_wt, each normalized by its mean (activation vectors over tokens, and
     concatenated gate/up rows + down column).
  5. Merge: W_c <- sum_i w_i W_i over the group, w = softmax(S_i / tau), S = REAP saliency,
     tau = median nonzero saliency of the layer (scale-free; reported per layer).

Writes a copy of the E pruned BF16 checkpoint with merged centroid experts, plus stats.

usage: 18_merge.py SRC_DIR E_BF16_DIR REAP_REPORT CALIB_JSONL OUT_DIR STATS_JSON
  env: N_SEQ (16) SEQ (512) CAP (16) WEIGHTS (softmax | saliency)
"""

import glob
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForCausalLM, AutoTokenizer

src, e_dir, rep_path, calib, out, stats_path = sys.argv[1:7]
N_SEQ = int(os.environ.get("N_SEQ", "16"))
SEQ = int(os.environ.get("SEQ", "512"))
CAP = int(os.environ.get("CAP", "16"))
WEIGHTS = os.environ.get("WEIGHTS", "softmax")
KEEP = 96
rep = json.load(open(rep_path))
tok = AutoTokenizer.from_pretrained(src)
rows = [json.loads(l)["text"] for l in open(calib)][:N_SEQ]
ids = [tok(t, add_special_tokens=False)["input_ids"][:SEQ] for t in rows]

# Capture forward on the CPU: one 61 GB copy of the source. Loading it with
# device_map="cuda" on GB10 unified memory kept a second resident copy (about 120 GB).
CAP_DEV = os.environ.get("CAPTURE_DEVICE", "cpu")
torch.set_num_threads(os.cpu_count())
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map=CAP_DEV)
model.eval()
caps = {}


def hook(L):
    def f(mod, args):
        caps.setdefault(L, []).append(args[0].reshape(-1, args[0].shape[-1]).detach())
    return f


hs = [layer.mlp.register_forward_pre_hook(hook(L)) for L, layer in enumerate(model.model.layers)]
with torch.no_grad():
    for t in ids:
        model.model(input_ids=torch.tensor([t], device=CAP_DEV))
for h in hs:
    h.remove()
# free the model: per-layer weights are read from the source safetensors below
caps = {L: torch.cat(v).cpu() for L, v in caps.items()}
n_layers = len(model.model.layers)
del model
import gc

gc.collect()
torch.cuda.empty_cache()
src_files = sorted(glob.glob(f"{src}/*.safetensors"))
if os.path.exists(f"{src}/model.safetensors.index.json"):
    src_map = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
else:  # single-file checkpoint
    src_map = {}
    for f in src_files:
        with safe_open(f, "pt") as g:
            src_map.update({k: os.path.basename(f) for k in g.keys()})


def load_src(names):
    out_ = {}
    by_file = {}
    for n in names:
        by_file.setdefault(src_map[n], []).append(n)
    for f, ns in by_file.items():
        with safe_open(f"{src}/{f}", "pt", device="cuda") as g:
            for n in ns:
                out_[n] = g.get_tensor(n)
    return out_

# map E checkpoint's expert slots to original indices (kept set, ascending)
def kept_indices(L):
    sal = rep["saliency"][L]
    return sorted(sorted(range(128), key=lambda i: (sal[i], i))[128 - KEEP:])


shards = sorted(glob.glob(f"{e_dir}/*.safetensors"))
os.makedirs(out, exist_ok=True)
for f in os.listdir(e_dir):
    if not f.endswith(".safetensors"):
        shutil.copy(f"{e_dir}/{f}", out)
new_tensors = {}
stats = []
with torch.no_grad():
    for L in range(n_layers):
        x = caps[L].cuda().float()
        pre = f"model.layers.{L}.mlp."
        names = [pre + "gate.weight"] + [f"{pre}experts.{e}.{w}_proj.weight" for e in range(128) for w in ("gate", "up", "down")]
        t = load_src(names)
        Wg = t[pre + "gate.weight"].float()
        gu = torch.stack([torch.cat([t[f"{pre}experts.{e}.gate_proj.weight"], t[f"{pre}experts.{e}.up_proj.weight"]], 0)
                          for e in range(128)]).float()  # (128, 2I, H)
        dn = torch.stack([t[f"{pre}experts.{e}.down_proj.weight"] for e in range(128)]).float()  # (128, H, I)
        del t
        inter = dn.shape[-1]
        logits = x @ Wg.T
        sig = torch.softmax(logits, -1)
        sal = torch.tensor(rep["saliency"][L])
        cnt = torch.tensor(rep["count"][L])
        cents = kept_indices(L)
        cset = set(cents)
        zero = [j for j in range(128) if j not in cset and cnt[j] == 0]
        mergers = [j for j in range(128) if j not in cset and cnt[j] > 0]

        def act(e):
            g_, u_ = (x @ gu[e].T).split(inter, -1)
            return torch.nn.functional.silu(g_) * u_  # (T, I)

        acts = {e: act(e) for e in cents + mergers}
        onorm = {e: torch.nn.functional.normalize(sig[:, e:e + 1] * (acts[e] @ dn[e].T), dim=-1).to(torch.bfloat16)
                 for e in cents + mergers}
        lg = torch.nn.functional.normalize(logits - logits.mean(0), dim=0)
        C = torch.stack([onorm[c] for c in cents], 0)  # (96, T, H)
        sim = torch.zeros(len(mergers), len(cents))
        for a, j in enumerate(mergers):
            out_sim = torch.einsum("cth,th->ct", C, onorm[j]).float().mean(-1)  # (96,)
            sim[a] = (lg[:, cents].T @ lg[:, j]).cpu() + out_sim.cpu()
        del C
        groups = {c: [] for c in cents}
        for a in sorted(range(len(mergers)), key=lambda a: -sim[a].max().item()):
            for b in torch.argsort(sim[a], descending=True).tolist():
                if len(groups[cents[b]]) < CAP:
                    groups[cents[b]].append(mergers[a])
                    break
        nz = sal[sal > 0]
        tau = float(nz.median())
        sizes, cw = [], []
        for c, mem in groups.items():
            if not mem:
                continue
            members = [c] + mem
            Wsum_gu = torch.zeros_like(gu[c])
            Wsum_dn = torch.zeros_like(dn[c])
            s = torch.tensor([float(sal[i]) for i in members])
            w = torch.softmax(s / tau, 0) if WEIGHTS == "softmax" else s / s.sum()
            cw.append(float(w[0]))
            ha = torch.nn.functional.normalize(acts[c], dim=0)  # columns = neurons
            wc = torch.cat([gu[c][:inter], gu[c][inter:], dn[c].T], 1)
            for i, wi in zip(members, w.tolist()):
                if i == c:
                    perm = torch.arange(inter)
                else:
                    hj = torch.nn.functional.normalize(acts[i], dim=0)
                    wj = torch.cat([gu[i][:inter], gu[i][inter:], dn[i].T], 1)
                    c_act = torch.cdist(ha.T, hj.T)
                    c_wt = torch.cdist(wc, wj)
                    cost = c_act / c_act.mean() + c_wt / c_wt.mean()
                    _, col = linear_sum_assignment(cost.cpu().numpy())
                    perm = torch.as_tensor(col)
                g_i, u_i = gu[i][:inter][perm], gu[i][inter:][perm]
                Wsum_gu += wi * torch.cat([g_i, u_i], 0)
                Wsum_dn += wi * dn[i][:, perm]
            slot = cents.index(c)
            base = f"model.layers.{L}.mlp.experts.{slot}"
            new_tensors[f"{base}.gate_proj.weight"] = Wsum_gu[:inter].to(torch.bfloat16).cpu()
            new_tensors[f"{base}.up_proj.weight"] = Wsum_gu[inter:].to(torch.bfloat16).cpu()
            new_tensors[f"{base}.down_proj.weight"] = Wsum_dn.to(torch.bfloat16).cpu()
            sizes.append(len(members))
        stats.append({"layer": L, "dropped_zero": len(zero), "merged": len(mergers), "groups": len(sizes),
                      "group_sizes": sorted(sizes, reverse=True), "tau": tau,
                      "centroid_weight_mean": sum(cw) / max(1, len(cw)),
                      "best_sim_mean": float(sim.max(1).values.mean()) if len(mergers) else None})
        print(json.dumps(stats[-1]), flush=True)
        del acts, onorm, gu, dn, x
        caps[L] = None
        torch.cuda.empty_cache()

# rewrite shards: E checkpoint tensors, with merged centroids replaced
for f in shards:
    with safe_open(f, "pt") as g:
        d = {k: (new_tensors[k] if k in new_tensors else g.get_tensor(k)) for k in g.keys()}
    save_file(d, f"{out}/{os.path.basename(f)}", metadata={"format": "pt"})
json.dump({"weights": WEIGHTS, "cap": CAP, "n_seq": N_SEQ, "seq": SEQ, "layers": stats,
           "replaced_tensors": len(new_tensors)}, open(stats_path, "w"), indent=1)
print("done", len(new_tensors), "tensors replaced")
