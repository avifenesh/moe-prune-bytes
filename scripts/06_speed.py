"""Batch-1 prefill and decode speed with vLLM, plus memory.

prefill: 2048 random-token prompt, 1 output token, prefix caching off.
decode:  128-token prompt, 513 output tokens, ignore_eos; decode tok/s is
         512 / (t(513 tokens) - t(1 token)) on the same prompt.
N repetitions after 3 warmups; median and all samples are recorded.
Memory: vLLM's own "model loading took X GiB" line and torch peak after a
profile run; the engine runs in-process so torch sees the same allocator.

usage: 06_speed.py MODEL_DIR ARM ORDER_TAG OUT.json [N]
"""

import json
import os
import random
import statistics
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

model_dir, arm, tag, out = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 5

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
llm = LLM(model=model_dir, max_model_len=4096, gpu_memory_utilization=0.80,
          enable_prefix_caching=False, max_num_seqs=1, seed=0)
load_s = time.time() - t0
weights_gib = torch.cuda.memory_allocated() / 2**30  # includes KV pool
rng = random.Random(0)


def prompt(n):
    return TokensPrompt(prompt_token_ids=[rng.randrange(1000, 150000) for _ in range(n)])


def timed(p, n_out):
    sp = SamplingParams(max_tokens=n_out, ignore_eos=True, temperature=0.0)
    torch.cuda.synchronize()
    t = time.perf_counter()
    o = llm.generate([p], sp, use_tqdm=False)
    torch.cuda.synchronize()
    assert len(o[0].outputs[0].token_ids) == n_out
    return time.perf_counter() - t


for _ in range(3):
    timed(prompt(2048), 1)
    timed(prompt(128), 64)

pre, dec = [], []
for _ in range(N):
    pre.append(2048 / timed(prompt(2048), 1))
    p = prompt(128)
    t1 = timed(p, 1)
    t513 = timed(p, 513)
    dec.append(512 / (t513 - t1))

res = {
    "arm": arm, "order": tag, "N": N, "load_s": load_s,
    "prefill_tok_s": {"median": statistics.median(pre), "samples": pre},
    "decode_tok_s": {"median": statistics.median(dec), "samples": dec},
    "torch_peak_gib_incl_kv_pool": torch.cuda.max_memory_allocated() / 2**30,
    "allocated_after_load_gib_incl_kv_pool": weights_gib,
}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
