"""Teacher targets for healing: source (S) final hidden states for the heal set, via vLLM.

Same method as 04_hidden.py, written in chunks so memory stays bounded. The trainer
rebuilds full-vocab teacher logits as hidden @ lm_head.T (lm_head is shared and frozen).

usage: 14_teacher_hidden.py MODEL_DIR HEAL_JSONL OUT_DIR [CHUNK]
"""

import json
import os
import sys

import torch
from vllm import LLM
from vllm.config import PoolerConfig
from vllm.inputs import TokensPrompt


def main():
    model_dir, path, out = sys.argv[1:4]
    chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(path)]
    llm = LLM(model=model_dir, runner="pooling", convert="embed",
              pooler_config=PoolerConfig(task="token_embed", tok_pooling_type="ALL", use_activation=False),
              max_model_len=2560, gpu_memory_utilization=0.80, enable_prefix_caching=False, seed=0)
    for c in range(0, len(rows), chunk):
        f = f"{out}/teacher_{c // chunk:04d}.pt"
        if os.path.exists(f):
            continue
        part = rows[c:c + chunk]
        outs = llm.encode([TokensPrompt(prompt_token_ids=r["ids"]) for r in part], pooling_task="token_embed", use_tqdm=False)
        hs = torch.stack([o.outputs.data.to(torch.bfloat16).cpu() for o in outs])
        torch.save(hs, f + ".tmp")
        os.replace(f + ".tmp", f)
        print("saved", f, tuple(hs.shape), flush=True)


if __name__ == "__main__":
    main()
