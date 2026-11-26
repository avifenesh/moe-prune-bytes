"""Heal a pruned BF16 MoE by distillation from the source, then merge and save BF16.

Objective: forward KL(teacher || student) over the full vocabulary, temperature 1,
mean over tokens. Teacher logits = teacher_hidden @ lm_head.T (source lm_head, which
the student shares and keeps frozen), from 14_teacher_hidden.py.

Trainable (fits one GB10, 119 GB unified memory):
  - LoRA on every expert, per-expert rank R_EXP, alpha 2*R_EXP, on gate_up_proj and
    down_proj. Implemented here as an experts forward ("grouped_mm_lora") that adds the
    low-rank path with two extra grouped GEMMs per projection over the same routed tokens.
    PEFT's target_parameters wrapper materializes W + BA for every expert on every call;
    on this model that cost about 17 s per 2048-token sequence against 2.4 s without it.
  - LoRA on attention q/k/v/o, rank 16, alpha 32.
  - Router (mlp.gate) weights, full, at LR * ROUTER_LR_MULT.
Frozen: embeddings, norms, lm_head, base expert and attention weights.

Data: heal.jsonl; the last N_VAL sequences are held out for validation KL.
Schedule: AdamW (betas 0.9/0.999, wd 0), linear warmup WARMUP of steps, linear decay
to 0, grad clip 1.0, MICRO=1 sequence of 2048 tokens, ACCUM micro-batches per step.

usage: 15_heal.py STUDENT_DIR TEACHER_DIR HEAL_JSONL OUT_DIR LOG_JSON
  env: LR (2e-4) R_EXP (8) ACCUM (8) TRAIN_SEQ (all but N_VAL) N_VAL (64)
       ROUTER_LR_MULT (0.1) WARMUP (0.03) MAX_STEPS (0 = from TRAIN_SEQ) SAVE (1)
"""

import glob
import json
import math
import os
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.integrations import moe as tmoe

student, tdir, heal, out, log_path = sys.argv[1:6]
LR = float(os.environ.get("LR", "2e-4"))
R_EXP = int(os.environ.get("R_EXP", "8"))
ACCUM = int(os.environ.get("ACCUM", "8"))
N_VAL = int(os.environ.get("N_VAL", "64"))
RMULT = float(os.environ.get("ROUTER_LR_MULT", "0.1"))
WARM = float(os.environ.get("WARMUP", "0.03"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
SAVE = os.environ.get("SAVE", "1") == "1"
VAL_EVERY = int(os.environ.get("VAL_EVERY", "25"))  # quick check on the first 16 val seqs
VAL_SUB = 16
CH = 512
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)

rows = [json.loads(l) for l in open(heal)]
tfiles = sorted(glob.glob(f"{tdir}/teacher_*.pt"))
per_file = torch.load(tfiles[0], mmap=True).shape[0]
n_train_all = len(rows) - N_VAL
TRAIN_SEQ = int(os.environ.get("TRAIN_SEQ", str(n_train_all)))
assert TRAIN_SEQ <= n_train_all
val_idx = list(range(len(rows) - N_VAL, len(rows)))
val_sub = val_idx[:VAL_SUB]
steps = MAX_STEPS or TRAIN_SEQ // ACCUM
_cache = {}


def teacher(i):
    f = i // per_file
    if f not in _cache:
        _cache.clear()
        _cache[f] = torch.load(tfiles[f], mmap=True)
    return _cache[f][i % per_file]




def grouped_mm_lora_forward(self, hidden_states, top_k_index, top_k_weights):
    """grouped_mm_experts_forward for gated, non-transposed, bias-free experts, plus LoRA."""
    num_top_k = top_k_index.size(-1)
    num_tokens, hidden_dim = hidden_states.shape
    sample_weights = top_k_weights.reshape(-1)
    expert_ids_g, perm = torch.sort(top_k_index.reshape(-1))
    x = hidden_states[perm // num_top_k]
    counts = torch.histc(expert_ids_g.int(), bins=self.num_experts, min=0, max=self.num_experts - 1)
    offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
    gu = tmoe._grouped_linear(x, self.gate_up_proj, offs)
    bf = x.dtype  # adapters are fp32 parameters, cast for the bf16 grouped GEMM
    gu = gu + self.lora_scale * tmoe._grouped_linear(tmoe._grouped_linear(x, self.lora_A_gu.to(bf), offs), self.lora_B_gu.to(bf), offs)
    h = self._apply_gate(gu)
    d = tmoe._grouped_linear(h, self.down_proj, offs)
    d = d + self.lora_scale * tmoe._grouped_linear(tmoe._grouped_linear(h, self.lora_A_d.to(bf), offs), self.lora_B_d.to(bf), offs)
    w = d * sample_weights[perm].unsqueeze(-1)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return w[inv].view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


tmoe.ALL_EXPERTS_FUNCTIONS.register("grouped_mm_lora", grouped_mm_lora_forward)

model = AutoModelForCausalLM.from_pretrained(student, dtype=torch.bfloat16, device_map="cuda",
                                             experts_implementation="grouped_mm")
model.config.use_cache = False
W = model.lm_head.weight.detach().float()  # frozen, shared with the teacher
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
pm = get_peft_model(model, cfg)
experts = [m for n, m in pm.named_modules() if n.endswith("mlp.experts")]
g = torch.Generator(device="cuda").manual_seed(0)
for m in experts:
    E, two_i, hdim = m.gate_up_proj.shape
    inter = m.down_proj.shape[-1]

    def a_init(e, r, fan_in):  # kaiming-uniform like PEFT's lora_A, B starts at zero
        bound = 1.0 / math.sqrt(fan_in)
        return torch.nn.Parameter((torch.rand(e, r, fan_in, device="cuda", generator=g) * 2 - 1) * bound)

    m.lora_A_gu = a_init(E, R_EXP, hdim)
    m.lora_B_gu = torch.nn.Parameter(torch.zeros(E, two_i, R_EXP, device="cuda"))
    m.lora_A_d = a_init(E, R_EXP, inter)
    m.lora_B_d = torch.nn.Parameter(torch.zeros(E, hdim, R_EXP, device="cuda"))
    m.lora_scale = 2.0  # alpha 2*R_EXP over rank R_EXP
model.config._experts_implementation = "grouped_mm_lora"
router = []
for n, p in pm.named_parameters():
    if n.endswith("mlp.gate.weight"):
        p.requires_grad_(True)
        router.append(p)
lora = [p for n, p in pm.named_parameters() if p.requires_grad and "lora_" in n]
n_lora = sum(p.numel() for p in lora)
n_router = sum(p.numel() for p in router)
opt = torch.optim.AdamW([{"params": lora, "lr": LR}, {"params": router, "lr": LR * RMULT}],
                        betas=(0.9, 0.999), weight_decay=0.0)
base_lrs = [g_["lr"] for g_ in opt.param_groups]
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
inner = pm.base_model.model.model  # Qwen3MoeModel: returns final-norm hidden states


def merge_expert_lora():
    with torch.no_grad():
        for m in experts:
            m.gate_up_proj.data = (m.gate_up_proj.float() + m.lora_scale * torch.bmm(m.lora_B_gu.float(), m.lora_A_gu.float())).to(torch.bfloat16)
            m.down_proj.data = (m.down_proj.float() + m.lora_scale * torch.bmm(m.lora_B_d.float(), m.lora_A_d.float())).to(torch.bfloat16)
            for k in ("lora_A_gu", "lora_B_gu", "lora_A_d", "lora_B_d"):
                delattr(m, k)
    model.config._experts_implementation = "grouped_mm"


def kl_loss(ids, th, train):
    """Full-vocab KL over 512-position chunks. Each chunk backprops only into a detached
    copy of the final hidden states; the network backward then runs once."""
    h = inner(input_ids=ids[None]).last_hidden_state[0]
    hd = h.detach().requires_grad_(train)
    total = 0.0
    for c in range(0, h.shape[0], CH):
        with torch.no_grad():
            lt = torch.log_softmax(th[c:c + CH].float() @ W.T, -1)
        with torch.set_grad_enabled(train):
            ls = torch.log_softmax(hd[c:c + CH].float() @ W.T, -1)
            kl = (lt.exp() * (lt - ls)).sum(-1).sum() / h.shape[0]
        if train:
            kl.backward()
        total += kl.item()
    if train:
        h.backward(hd.grad)
    return total


def validate(sub=False):
    pm.eval()
    vals = []
    with torch.no_grad():
        for i in (val_sub if sub else val_idx):
            vals.append(kl_loss(torch.tensor(rows[i]["ids"], device="cuda"), teacher(i).cuda(), False))
    pm.train()
    return sum(vals) / len(vals)


log = {"config": {"student": student, "lr": LR, "router_lr": LR * RMULT, "r_exp": R_EXP, "r_attn": 16,
                  "accum": ACCUM, "seq_len": 2048, "steps": steps, "train_seq": TRAIN_SEQ, "n_val": N_VAL,
                  "train_tokens": steps * ACCUM * 2048, "warmup": WARM, "trainable_lora": n_lora,
                  "trainable_router": n_router}, "steps": [], "val": []}
print(json.dumps(log["config"]), flush=True)
log["val"].append({"step": 0, "kl": validate()})
log["val_sub"] = [{"step": 0, "kl": validate(sub=True)}]
print("val step 0", log["val"][-1], "sub", log["val_sub"][0], flush=True)
pm.train()
t0 = time.time()
warm = max(1, int(WARM * steps))
for s in range(steps):
    frac = (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm))
    for g_, b in zip(opt.param_groups, base_lrs):
        g_["lr"] = b * frac
    tot = 0.0
    for k in range(ACCUM):
        i = s * ACCUM + k
        th = teacher(i).cuda(non_blocking=True)
        tot += kl_loss(torch.tensor(rows[i]["ids"], device="cuda"), th, True) / ACCUM
    # gradients were accumulated as a sum over micro-batches; average them
    for p in lora + router:
        if p.grad is not None:
            p.grad.div_(ACCUM)
    gn = torch.nn.utils.clip_grad_norm_(lora + router, 1.0).item()
    opt.step()
    opt.zero_grad(set_to_none=True)
    el = time.time() - t0
    log["steps"].append({"step": s + 1, "kl": tot, "grad_norm": gn, "lr": opt.param_groups[0]["lr"],
                         "tok_s": (s + 1) * ACCUM * 2048 / el})
    print(json.dumps(log["steps"][-1]), flush=True)
    if (s + 1) % VAL_EVERY == 0:
        log.setdefault("val_sub", []).append({"step": s + 1, "kl": validate(sub=True)})
        print("val_sub", log["val_sub"][-1], flush=True)
    if not math.isfinite(tot):
        raise SystemExit("non-finite loss")
    if (s + 1) % max(1, steps // 4) == 0 and s + 1 < steps:
        log["val"].append({"step": s + 1, "kl": validate()})
        print("val", log["val"][-1], flush=True)
        json.dump(log, open(log_path, "w"), indent=1)
log["val"].append({"step": steps, "kl": validate()})
print("val final", log["val"][-1], flush=True)
log["train_seconds"] = time.time() - t0
json.dump(log, open(log_path, "w"), indent=1)
def hf_tensors(name, t):
    """Map a parameter to checkpoint tensors in the source's per-expert layout."""
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
    # 1) adapters and router first (small), so a failure later cannot lose the run
    import gc
    import shutil

    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in pm.named_parameters() if p.requires_grad},
               f"{out}.adapters.pt")
    # 2) free the optimizer, merge in place, check, then stream tensors to shards
    del opt
    lora = router = None
    gc.collect()
    torch.cuda.empty_cache()
    merged = pm.merge_and_unload()
    merge_expert_lora()
    inner = merged.model
    log["val_after_bf16_merge"] = validate()
    print("val after merge", log["val_after_bf16_merge"], flush=True)
    buf, nbytes, shard, wmap = {}, 0, 0, {}
    for n, p in merged.named_parameters():
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
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "generation_config.json"):
        if os.path.exists(f"{student}/{f}"):
            shutil.copy(f"{student}/{f}", out)
    print("saved", out, len(wmap), "tensors", total, "bytes", flush=True)
    json.dump(log, open(log_path, "w"), indent=1)
print("done", flush=True)
