"""Heal a pruned MoE by distillation, LoRA, with optional NVFP4 quantization-aware forward.

Same objective and data plumbing as 15_heal.py: forward KL(teacher || student) over the
full vocabulary, teacher logits = teacher hidden @ the shared frozen lm_head.

Trainable: router weights (full), LoRA rank R_ATTN on attention q/k/v/o, rank R_EXP per
expert on gate_up and down. Adapters are fp32; AdamW (betas 0.9/0.999, wd 0).

QAD=1: every quantized Linear of the deployed NVFP4 model (experts and attention, not the
router or lm_head) runs with fake-quant NVFP4 in the forward pass and straight-through
gradients, with the numerics vLLM runs the export with (nvfp4_fq.py, 35_export_nvfp4.py):
  weights: block scales from the current merged weight W + scale * B @ A; one global scale
           amax/(6*448) per fused GEMM (gate+up per expert, q+k+v per layer; down and o alone);
           per 16-element block an FP8-E4M3 scale, values on the FP4 E2M1 grid;
  activations: static global scale per GEMM input (ACT_AMAX json from 34_act_amax.py; experts
           share one per layer), dynamic per-16-element block scales.
35_export_nvfp4.py exports the healed model with the same functions, so the deployed weights
are the ones the last training step saw. QAD=0 runs the same LoRA without fake-quant.

Batching: MICRO sequences of 2,048 tokens per forward, ACCUM micro-batches per step.

usage: 29_heal_qad.py STUDENT_DIR TEACHER_DIR HEAL_JSONL OUT_DIR LOG_JSON
  env: QAD (1) LR (1e-4) R_EXP (8) R_ATTN (16) MICRO (8) ACCUM (1) N_VAL (64)
       ROUTER_LR_MULT (0.1) WARMUP (0.03) MAX_STEPS (0) VAL_EVERY (25) SAVE (1)
       ACT_AMAX (required with QAD=1)
"""

import gc
import glob
import json
import math
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers.integrations import moe as tmoe

student, tdir, heal, out, log_path = sys.argv[1:6]
QAD = os.environ.get("QAD", "1") == "1"
LR = float(os.environ.get("LR", "1e-4"))
R_EXP = int(os.environ.get("R_EXP", "8"))
R_ATTN = int(os.environ.get("R_ATTN", "16"))
MICRO = int(os.environ.get("MICRO", "8"))
ACCUM = int(os.environ.get("ACCUM", "1"))
N_VAL = int(os.environ.get("N_VAL", "64"))
RMULT = float(os.environ.get("ROUTER_LR_MULT", "0.1"))
WARM = float(os.environ.get("WARMUP", "0.03"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
VAL_EVERY = int(os.environ.get("VAL_EVERY", "25"))
SAVE = os.environ.get("SAVE", "1") == "1"
# on-policy rounds: continue adapters + optimizer from STATE_IN, run steps [STEP_OFFSET,
# STEP_OFFSET + MAX_STEPS) of a TOTAL_STEPS schedule, write STATE_OUT
STATE_IN = os.environ.get("STATE_IN", "")
STATE_OUT = os.environ.get("STATE_OUT", "")
STEP_OFFSET = int(os.environ.get("STEP_OFFSET", "0"))
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "0"))
CH = 512
VAL_SUB = 16
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)

rows = [json.loads(l) for l in open(heal)]
tfiles = sorted(glob.glob(f"{tdir}/teacher_*.pt"))
per_file = torch.load(tfiles[0], mmap=True).shape[0]
TRAIN_SEQ = len(rows) - N_VAL
val_idx = list(range(len(rows) - N_VAL, len(rows)))
val_sub = val_idx[:VAL_SUB]
per_step = MICRO * ACCUM
steps = MAX_STEPS or TRAIN_SEQ // per_step
_cache = {}


def teacher(i):
    f = i // per_file
    if f not in _cache:
        _cache.clear()
        _cache[f] = torch.load(tfiles[f], mmap=True)
    return _cache[f][i % per_file]


# ---------------- NVFP4 fake quant ----------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_fq import fq_act as _fq_act_c, fq_weight as _fq_weight_c, matrix_amax  # noqa: E402

ACT = json.load(open(os.environ["ACT_AMAX"])) if QAD else None


def ste_w(w, gamax):
    with torch.no_grad():
        q = _fq_weight_c(w.detach(), gamax)
    return w + (q - w).detach()


def fq_a(x, amax):
    with torch.no_grad():
        q = _fq_act_c(x.detach(), amax)
    return x + (q - x).detach()


# ---------------- model ----------------
def grouped_mm_qlora_forward(self, hidden_states, top_k_index, top_k_weights):
    num_top_k = top_k_index.size(-1)
    num_tokens, hidden_dim = hidden_states.shape
    sample_weights = top_k_weights.reshape(-1)
    expert_ids_g, perm = torch.sort(top_k_index.reshape(-1))
    x = hidden_states[perm // num_top_k]
    counts = torch.histc(expert_ids_g.int(), bins=self.num_experts, min=0, max=self.num_experts - 1)
    offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
    bf = x.dtype
    inter = self.down_proj.shape[-1]
    w_gu = self.gate_up_proj + (self.lora_scale * torch.bmm(self.lora_B_gu, self.lora_A_gu)).to(bf)
    w_d = self.down_proj + (self.lora_scale * torch.bmm(self.lora_B_d, self.lora_A_d)).to(bf)
    if QAD:
        w_gu = ste_w(w_gu, matrix_amax(w_gu))  # [E, 2i, h]: gate and up share one global per expert
        w_d = ste_w(w_d, matrix_amax(w_d))
        x = fq_a(x, self.act_w13)
    h = self._apply_gate(tmoe._grouped_linear(x, w_gu, offs))
    if QAD:
        h = fq_a(h, self.act_w2)
    d = tmoe._grouped_linear(h, w_d, offs)
    w = d * sample_weights[perm].unsqueeze(-1)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return w[inv].view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


tmoe.ALL_EXPERTS_FUNCTIONS.register("grouped_mm_qlora", grouped_mm_qlora_forward)


class QLoRALinear(torch.nn.Module):
    def __init__(self, base, r, gen, act_amax=None):
        super().__init__()
        self.base = base
        base.weight.requires_grad_(False)
        out_f, in_f = base.weight.shape
        bound = 1.0 / math.sqrt(in_f)
        self.lora_A = torch.nn.Parameter((torch.rand(r, in_f, device="cuda", generator=gen) * 2 - 1) * bound)
        self.lora_B = torch.nn.Parameter(torch.zeros(out_f, r, device="cuda"))
        self.scale = 2.0  # alpha = 2r
        self.act_amax = act_amax
        self.siblings = [self]  # modules sharing one weight global scale (q, k, v)

    def merged_weight(self):
        return self.base.weight + (self.scale * self.lora_B @ self.lora_A).to(self.base.weight.dtype)

    def forward(self, x):
        w = self.merged_weight()
        if QAD:
            with torch.no_grad():
                gamax = torch.stack([matrix_amax(w) if m is self else matrix_amax(m.merged_weight())
                                     for m in self.siblings]).amax(0)
            w = ste_w(w, gamax)
            shp = x.shape
            x = fq_a(x.reshape(-1, shp[-1]), self.act_amax).reshape(shp)
        return F.linear(x, w, self.base.bias)


model = AutoModelForCausalLM.from_pretrained(student, dtype=torch.bfloat16, device_map="cuda",
                                             experts_implementation="grouped_mm")
model.config.use_cache = False
for p in model.parameters():
    p.requires_grad_(False)
W = model.lm_head.weight.detach().float()
gen = torch.Generator(device="cuda").manual_seed(0)
experts, attn = [], []
for li, layer in enumerate(model.model.layers):
    m = layer.mlp.experts
    E, two_i, hdim = m.gate_up_proj.shape
    inter = m.down_proj.shape[-1]
    for k, shape, fan in (("lora_A_gu", (E, R_EXP, hdim), hdim), ("lora_A_d", (E, R_EXP, inter), inter)):
        setattr(m, k, torch.nn.Parameter((torch.rand(*shape, device="cuda", generator=gen) * 2 - 1) / math.sqrt(fan)))
    m.lora_B_gu = torch.nn.Parameter(torch.zeros(E, two_i, R_EXP, device="cuda"))
    m.lora_B_d = torch.nn.Parameter(torch.zeros(E, hdim, R_EXP, device="cuda"))
    m.lora_scale = 2.0
    if QAD:
        m.act_w13 = torch.tensor(ACT["w13"][li], device="cuda")
        m.act_w2 = torch.tensor(ACT["w2"][li], device="cuda")
    experts.append(m)
    sa = layer.self_attn
    qkv = []
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        a = torch.tensor(ACT["o" if name == "o_proj" else "qkv"][li], device="cuda") if QAD else None
        q = QLoRALinear(getattr(sa, name), R_ATTN, gen, a)
        setattr(sa, name, q)
        attn.append(q)
        if name != "o_proj":
            qkv.append(q)
    for q in qkv:
        q.siblings = qkv
    layer.mlp.gate.weight.requires_grad_(True)
model.config._experts_implementation = "grouped_mm_qlora"
router = [l.mlp.gate.weight for l in model.model.layers]
lora = [p for m in experts for p in (m.lora_A_gu, m.lora_B_gu, m.lora_A_d, m.lora_B_d)] + \
       [p for q in attn for p in (q.lora_A, q.lora_B)]
opt = torch.optim.AdamW([{"params": lora, "lr": LR}, {"params": router, "lr": LR * RMULT}],
                        betas=(0.9, 0.999), weight_decay=0.0)
base_lrs = [g["lr"] for g in opt.param_groups]
named_train = {n: p for n, p in model.named_parameters() if p.requires_grad}
if STATE_IN:
    st = torch.load(STATE_IN, map_location="cuda")
    with torch.no_grad():
        for n, t in st["params"].items():
            named_train[n].copy_(t)
    opt.load_state_dict(st["opt"])
    for g, b in zip(opt.param_groups, base_lrs):
        g["lr"] = b
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
inner = model.model


def kl_batch(idx, train):
    ids = torch.tensor([rows[i]["ids"] for i in idx], device="cuda")
    th = torch.stack([teacher(i) for i in idx]).cuda().reshape(-1, W.shape[1])
    h = inner(input_ids=ids).last_hidden_state.reshape(-1, W.shape[1])
    hd = h.detach().requires_grad_(train)
    n, total = h.shape[0], 0.0
    for c in range(0, n, CH):
        with torch.no_grad():
            lt = torch.log_softmax(th[c:c + CH].float() @ W.T, -1)
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
    model.eval()
    idx = val_sub if sub else val_idx
    vals = []
    with torch.no_grad():
        for j in range(0, len(idx), MICRO):
            part = idx[j:j + MICRO]
            vals.append(kl_batch(part, False) * len(part))
    model.train()
    return sum(vals) / len(idx)


log = {"config": {"student": student, "qad": QAD, "lr": LR, "router_lr": LR * RMULT, "r_exp": R_EXP,
                  "r_attn": R_ATTN, "micro": MICRO, "accum": ACCUM, "steps": steps,
                  "train_tokens": steps * per_step * 2048, "warmup": WARM,
                  "trainable_lora": sum(p.numel() for p in lora), "trainable_router": sum(p.numel() for p in router)},
       "steps": [], "val": [], "val_sub": []}
print(json.dumps(log["config"]), flush=True)
log["val"].append({"step": 0, "kl": validate()})
log["val_sub"].append({"step": 0, "kl": validate(sub=True)})
print("val step 0", log["val"][-1], log["val_sub"][-1], flush=True)
t0 = time.time()
sched_total = TOTAL_STEPS or steps
warm = max(1, int(WARM * sched_total))
for s in range(steps):
    gs_ = STEP_OFFSET + s
    frac = (gs_ + 1) / warm if gs_ < warm else max(0.0, (sched_total - gs_) / (sched_total - warm))
    for g, b in zip(opt.param_groups, base_lrs):
        g["lr"] = b * frac
    tot = 0.0
    for k in range(ACCUM):
        base = (s * ACCUM + k) * MICRO
        tot += kl_batch(list(range(base, base + MICRO)), True) / ACCUM
    for p in lora + router:
        if p.grad is not None and ACCUM > 1:
            p.grad.div_(ACCUM)
    gn = torch.nn.utils.clip_grad_norm_(lora + router, 1.0).item()
    opt.step()
    opt.zero_grad(set_to_none=True)
    rec = {"step": s + 1, "kl": tot, "grad_norm": gn, "lr": opt.param_groups[0]["lr"],
           "tok_s": (s + 1) * per_step * 2048 / (time.time() - t0)}
    log["steps"].append(rec)
    print(json.dumps(rec), flush=True)
    if not math.isfinite(tot):
        raise SystemExit("non-finite loss")
    if (s + 1) % VAL_EVERY == 0:
        log["val_sub"].append({"step": s + 1, "kl": validate(sub=True)})
        print("val_sub", log["val_sub"][-1], flush=True)
    if (s + 1) % max(1, steps // 4) == 0 and s + 1 < steps:
        log["val"].append({"step": s + 1, "kl": validate()})
        print("val", log["val"][-1], flush=True)
        json.dump(log, open(log_path, "w"), indent=1)
log["val"].append({"step": steps, "kl": validate()})
log["train_seconds"] = time.time() - t0
if STATE_OUT:
    torch.save({"params": {n: p.detach() for n, p in named_train.items()}, "opt": opt.state_dict()}, STATE_OUT)
print("val final", log["val"][-1], flush=True)
json.dump(log, open(log_path, "w"), indent=1)

if SAVE:
    os.makedirs(out, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, f"{out}.adapters.pt")
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():  # merge adapters into the BF16 weights; quantization happens afterwards
        for m in experts:
            m.gate_up_proj.data += (m.lora_scale * torch.bmm(m.lora_B_gu, m.lora_A_gu)).to(torch.bfloat16)
            m.down_proj.data += (m.lora_scale * torch.bmm(m.lora_B_d, m.lora_A_d)).to(torch.bfloat16)
            for k in ("lora_A_gu", "lora_B_gu", "lora_A_d", "lora_B_d"):
                delattr(m, k)
        for layer in model.model.layers:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                q = getattr(layer.self_attn, name)
                q.base.weight.data = q.merged_weight().detach()
                setattr(layer.self_attn, name, q.base)
    model.config._experts_implementation = "grouped_mm"

    def hf_tensors(name, t):
        if name.endswith("mlp.experts.gate_up_proj"):
            b, inter = name[: -len("gate_up_proj")], t.shape[1] // 2
            for e in range(t.shape[0]):
                yield f"{b}{e}.gate_proj.weight", t[e, :inter].contiguous()
                yield f"{b}{e}.up_proj.weight", t[e, inter:].contiguous()
        elif name.endswith("mlp.experts.down_proj"):
            b = name[: -len("down_proj")]
            for e in range(t.shape[0]):
                yield f"{b}{e}.down_proj.weight", t[e].contiguous()
        else:
            yield name, t.contiguous()

    buf, nbytes, shard, wmap = {}, 0, 0, {}
    for n, p in model.named_parameters():
        for k, t in hf_tensors(n, p.detach().to(torch.bfloat16).cpu()):
            buf[k] = t
            nbytes += t.numel() * t.element_size()
        if nbytes > 4e9:
            fname = f"model-{shard:05d}.safetensors"
            save_file(buf, f"{out}/{fname}", metadata={"format": "pt"})
            wmap.update({k: fname for k in buf})
            buf, nbytes, shard = {}, 0, shard + 1
    if buf:
        fname = f"model-{shard:05d}.safetensors"
        save_file(buf, f"{out}/{fname}", metadata={"format": "pt"})
        wmap.update({k: fname for k in buf})
    total = sum(os.path.getsize(f"{out}/{f}") for f in set(wmap.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": wmap},
              open(f"{out}/model.safetensors.index.json", "w"), indent=1)
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
        if os.path.exists(f"{student}/{f}"):
            shutil.copy(f"{student}/{f}", out)
    print("saved", out, len(wmap), "tensors", total, flush=True)
print("done", flush=True)
