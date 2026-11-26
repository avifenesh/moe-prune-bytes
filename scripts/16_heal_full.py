"""Full-parameter heal by distillation, FSDP2 across nodes (torchrun), then save BF16.

Same objective, data, validation and token budget as 15_heal.py (LoRA heal):
forward KL(teacher || student), full vocabulary, T=1; teacher logits from the source's
final hidden states @ the shared, frozen lm_head.

Trainable: every decoder-layer parameter (attention, all experts, routers, norms).
Frozen: embed_tokens and lm_head (lm_head must stay identical to the source).
Memory per node for a 96-expert model (23.3B params, world 2): bf16 param shard 23 GB,
bf16 grad shard 23 GB, optimizer about 23 GB, activations with checkpointing a few GB:
about 75 GB of 119 GB.

Optimizer (FactoredSR below): Adafactor-style factored second moment (row and column
EMAs per matrix, per expert for 3D tensors), update RMS clipping at 1.0, bf16 first
moment (beta1 0.9), and bf16 stochastic rounding of the parameter update, so there is
no fp32 master copy and small updates are not lost to bf16 rounding. Chosen because
full AdamW (16 B/param with fp32 master) does not fit 2 x 119 GB, and torchao's 8-bit
AdamW fails to compile with torch 2.13 on this platform (aten.view.dtype on its state
subclass). Factoring is done on each rank's local shard.

Batch: MICRO sequences of 2048 tokens per rank per step, no accumulation, so
tokens/step = world * MICRO * 2048. Schedule: linear warmup WARMUP of steps, linear
decay to 0, grad clip 1.0, beta1 0.9, beta2 0.999 (bias-corrected), wd 0.

usage (each node): torchrun --nnodes 2 --nproc-per-node 1 --node-rank R --master-addr A
  --master-port P 16_heal_full.py STUDENT_DIR TEACHER_DIR HEAL_JSONL OUT_DIR LOG_JSON
  env: LR (5e-5) MICRO (4) N_VAL (64) TRAIN_SEQ WARMUP (0.03) MAX_STEPS SAVE (1)
"""

import glob
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM

student, tdir, heal, out, log_path = sys.argv[1:6]
LR = float(os.environ.get("LR", "5e-5"))
MICRO = int(os.environ.get("MICRO", "4"))
N_VAL = int(os.environ.get("N_VAL", "64"))
WARM = float(os.environ.get("WARMUP", "0.03"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
SAVE = os.environ.get("SAVE", "1") == "1"
VAL_EVERY = int(os.environ.get("VAL_EVERY", "25"))  # quick check on the first 16 val seqs
VAL_SUB = 16
CH = 512

class FactoredSR(torch.optim.Optimizer):
    """Factored second moment + bf16 momentum + bf16 stochastic rounding (see module doc)."""

    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-30, clip=1.0):
        super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, clip=clip))

    @staticmethod
    def _sround(x):
        bits = x.view(torch.int32)
        noise = torch.randint_like(bits, 0, 1 << 16)
        return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                P = p.to_local() if hasattr(p, "to_local") else p
                G = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                if P.numel() == 0:
                    continue
                st = self.state[p]
                grad = G.float()
                sq = grad * grad + g["eps"]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(P, dtype=torch.bfloat16)
                    if P.dim() >= 2:
                        st["r"] = torch.zeros(P.shape[:-1], device=P.device)
                        st["c"] = torch.zeros(P.shape[:-2] + P.shape[-1:], device=P.device)
                    else:
                        st["v"] = torch.zeros_like(P, dtype=torch.float32)
                st["t"] += 1
                b2 = g["beta2"]
                corr = 1 - b2 ** st["t"]
                if P.dim() >= 2:
                    st["r"].mul_(b2).add_(sq.mean(-1), alpha=1 - b2)
                    st["c"].mul_(b2).add_(sq.mean(-2), alpha=1 - b2)
                    r = st["r"] / corr
                    c = st["c"] / corr
                    v = (r / r.mean(-1, keepdim=True)).unsqueeze(-1) * c.unsqueeze(-2)
                else:
                    st["v"].mul_(b2).add_(sq, alpha=1 - b2)
                    v = st["v"] / corr
                u = grad / v.sqrt()
                u.div_((u.pow(2).mean().sqrt() / g["clip"]).clamp(min=1.0))
                m = st["m"].float().mul_(g["beta1"]).add_(u, alpha=1 - g["beta1"])
                st["m"].copy_(self._sround(m))
                P.copy_(self._sround(P.float() - g["lr"] * m))


dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)

rows = [json.loads(l) for l in open(heal)]
tfiles = sorted(glob.glob(f"{tdir}/teacher_*.pt"))
per_file = torch.load(tfiles[0], mmap=True).shape[0]
n_train_all = len(rows) - N_VAL
TRAIN_SEQ = int(os.environ.get("TRAIN_SEQ", str(n_train_all)))
per_step = world * MICRO
steps = MAX_STEPS or TRAIN_SEQ // per_step
val_all = list(range(len(rows) - N_VAL, len(rows)))
val_idx = val_all[rank::world]
val_sub = val_all[:VAL_SUB][rank::world]
_cache = {}


def teacher(i):
    f = i // per_file
    if f not in _cache:
        _cache.clear()
        _cache[f] = torch.load(tfiles[f], mmap=True)
    return _cache[f][i % per_file]


model = AutoModelForCausalLM.from_pretrained(student, dtype=torch.bfloat16, device_map="cpu",
                                             experts_implementation="grouped_mm")
model.config.use_cache = False
H_DIM = model.config.hidden_size
model.model.embed_tokens.weight.requires_grad_(False)
model.lm_head.weight.requires_grad_(False)
W = model.lm_head.weight.detach().float().cuda()
# bf16 reduce halves the reduce-scatter bytes over the QSFP link (2 ranks: one add)
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
for layer in model.model.layers:
    fully_shard(layer, mp_policy=mp)
# the decoder (embeddings, layers, final norm) is the FSDP root: training calls it
# directly and computes logits with W, so the frozen lm_head stays unsharded
fully_shard(model.model, mp_policy=mp)
model.to("cuda")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
params = [p for p in model.parameters() if p.requires_grad]
n_train = sum(p.numel() for p in params) * world  # sharded: local numel x world (approx)
opt = FactoredSR(params, lr=LR)
inner = model.model


def kl_batch(idx, train):
    """Full-vocab KL over 512-position chunks; chunks backprop into a detached copy of the
    final hidden states and the network backward runs once."""
    ids = torch.tensor([rows[i]["ids"] for i in idx], device="cuda")
    th = torch.stack([teacher(i) for i in idx]).cuda()
    h = inner(input_ids=ids).last_hidden_state.reshape(-1, H_DIM)
    hd = h.detach().requires_grad_(train)
    tht = th.reshape(-1, H_DIM)
    n = h.shape[0]
    total = 0.0
    for c in range(0, n, CH):
        with torch.no_grad():
            lt = torch.log_softmax(tht[c:c + CH].float() @ W.T, -1)
        with torch.set_grad_enabled(train):
            ls = torch.log_softmax(hd[c:c + CH].float() @ W.T, -1)
            kl = (lt.exp() * (lt - ls)).sum(-1).sum() / n
        if train:
            kl.backward()
        total += kl.item()
    if train:
        h.backward(hd.grad)
    return total


def validate(sub=False):
    idx_all = val_sub if sub else val_idx
    model.eval()
    s, k = torch.zeros(2, device="cuda"), 0
    with torch.no_grad():
        for j in range(0, len(idx_all), MICRO):
            part = idx_all[j:j + MICRO]
            s[0] += kl_batch(part, False) * len(part)
            s[1] += len(part)
    dist.all_reduce(s)
    model.train()
    return (s[0] / s[1]).item()


log = {"config": {"student": student, "lr": LR, "micro_per_rank": MICRO, "world": world, "seq_len": 2048,
                  "steps": steps, "train_seq": steps * per_step, "train_tokens": steps * per_step * 2048,
                  "warmup": WARM, "optimizer": "FactoredSR: factored 2nd moment, bf16 momentum, bf16 stochastic rounding",
                  "trainable_params_approx": n_train}, "steps": [], "val": []}
if rank == 0:
    print(json.dumps(log["config"]), flush=True)
log["val"].append({"step": 0, "kl": validate()})
log["val_sub"] = [{"step": 0, "kl": validate(sub=True)}]
if rank == 0:
    print("val_sub step 0", log["val_sub"][0], flush=True)
    print("val step 0", log["val"][-1], flush=True)
model.train()
t0 = time.time()
warm = max(1, int(WARM * steps))
for s in range(steps):
    frac = (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm))
    for g in opt.param_groups:
        g["lr"] = LR * frac
    idx = list(range(s * per_step + rank * MICRO, s * per_step + (rank + 1) * MICRO))
    loss = torch.tensor(kl_batch(idx, True), device="cuda")
    dist.all_reduce(loss)
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    gn = gn.full_tensor().item() if hasattr(gn, "full_tensor") else float(gn)
    opt.step()
    opt.zero_grad(set_to_none=True)
    el = time.time() - t0
    rec = {"step": s + 1, "kl": loss.item() / world, "grad_norm": gn, "lr": float(opt.param_groups[0]["lr"]),
           "tok_s": (s + 1) * per_step * 2048 / el}
    log["steps"].append(rec)
    if rank == 0:
        print(json.dumps(rec), flush=True)
    if (s + 1) % VAL_EVERY == 0:
        log.setdefault("val_sub", []).append({"step": s + 1, "kl": validate(sub=True)})
        if rank == 0:
            print("val_sub", log["val_sub"][-1], flush=True)
    if not math.isfinite(rec["kl"]):
        raise SystemExit("non-finite loss")
    if (s + 1) % max(1, steps // 4) == 0 and s + 1 < steps:
        log["val"].append({"step": s + 1, "kl": validate()})
        if rank == 0:
            print("val", log["val"][-1], flush=True)
            json.dump(log, open(log_path, "w"), indent=1)
log["val"].append({"step": steps, "kl": validate()})
log["train_seconds"] = time.time() - t0
if rank == 0:
    print("val final", log["val"][-1], flush=True)
    json.dump(log, open(log_path, "w"), indent=1)
def hf_tensors(name, t):
    """Map a training parameter to checkpoint tensors in the source's per-expert layout."""
    if name.endswith("mlp.experts.gate_up_proj"):
        base = name[: -len("gate_up_proj")]
        inter = t.shape[1] // 2  # concatenated layout: [gate; up]
        for e in range(t.shape[0]):
            yield f"{base}{e}.gate_proj.weight", t[e, :inter].contiguous()
            yield f"{base}{e}.up_proj.weight", t[e, inter:].contiguous()
    elif name.endswith("mlp.experts.down_proj"):
        base = name[: -len("down_proj")]
        for e in range(t.shape[0]):
            yield f"{base}{e}.down_proj.weight", t[e].contiguous()
    else:
        yield name, t.contiguous()


if SAVE:
    # Stream the weights out: free the optimizer, then gather one parameter at a time
    # (every rank joins each all-gather in the same order) and let rank 0 write
    # per-expert safetensors shards of about 4 GB. No full state dict is ever held.
    import gc
    import shutil

    from safetensors.torch import save_file

    del opt
    params = None
    gc.collect()
    torch.cuda.empty_cache()
    if rank == 0:
        os.makedirs(out, exist_ok=True)
    buf, buf_bytes, shard, wmap = {}, 0, 0, {}

    def flush():
        global buf, buf_bytes, shard
        if rank == 0 and buf:
            fname = f"model-{shard:05d}.safetensors"
            save_file(buf, f"{out}/{fname}", metadata={"format": "pt"})
            for k in buf:
                wmap[k] = fname
            shard += 1
        buf, buf_bytes = {}, 0

    names = [n for n, _ in model.named_parameters()]
    for n, p in zip(names, [p for _, p in model.named_parameters()]):
        full = p.full_tensor() if hasattr(p, "full_tensor") else p.data
        if rank == 0:
            for k, t in hf_tensors(n, full.detach().to(torch.bfloat16).cpu()):
                buf[k] = t
                buf_bytes += t.numel() * t.element_size()
            if buf_bytes > 4e9:
                flush()
        del full
    flush()
    if rank == 0:
        total = sum(os.path.getsize(f"{out}/{f}") for f in set(wmap.values()))
        json.dump({"metadata": {"total_size": total}, "weight_map": wmap},
                  open(f"{out}/model.safetensors.index.json", "w"), indent=1)
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                  "generation_config.json"):
            if os.path.exists(f"{student}/{f}"):
                shutil.copy(f"{student}/{f}", out)
        print("saved", out, len(wmap), "tensors", total, "bytes", flush=True)
    dist.barrier()
if rank == 0:
    print("done", flush=True)
dist.destroy_process_group()
