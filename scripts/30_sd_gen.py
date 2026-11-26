"""Generate self-distillation sequences: a model answers mixed-domain prompts, and the
conversations are packed into 2,048-token sequences for KL distillation.

Teacher-generated arm: MODEL = the BF16 source. On-policy arm: MODEL = the current student.
Sampling: the source's recommended settings (temperature 0.7, top_p 0.8, top_k 20),
max 2,048 new tokens, N_SAMPLES answers per prompt, seed SEED.

usage: 30_sd_gen.py MODEL_DIR PROMPTS_JSONL OUT_JSONL N_SEQ [N_SAMPLES] [SEED] [FIRST] [COUNT]
  FIRST/COUNT pick a slice of the prompt file (on-policy rounds use disjoint slices).
"""

import json
import random
import sys

from vllm import LLM, SamplingParams


def main():
    model, prompts_path, out, n_seq = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    n_samples = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    first = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    count = int(sys.argv[8]) if len(sys.argv) > 8 else None
    rows = [json.loads(l) for l in open(prompts_path)]
    rows = rows[first:first + count] if count else rows[first:]
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.80, seed=seed)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False) for r in rows]
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=2048, n=n_samples, seed=seed)
    res = llm.generate(prompts, sp)
    convs = []
    for r, o in zip(rows, res):
        for c in o.outputs:
            ids = tok.apply_chat_template(r["messages"] + [{"role": "assistant", "content": c.text}], tokenize=False)
            convs.append((r["src"], tok(ids, add_special_tokens=False)["input_ids"]))
    random.Random(seed).shuffle(convs)
    seqs, buf, srcs = [], [], []
    for src, ids in convs:
        buf.extend(ids)
        srcs.append(src)
        while len(buf) >= 2048 and len(seqs) < n_seq:
            seqs.append({"src": max(set(srcs), key=srcs.count), "ids": buf[:2048]})
            buf, srcs = buf[2048:], [src]
        if len(seqs) >= n_seq:
            break
    with open(out, "w") as f:
        for s in seqs:
            f.write(json.dumps(s) + "\n")
    gen_tokens = sum(len(c.token_ids) for o in res for c in o.outputs)
    print("prompts", len(rows), "answers", len(convs), "generated tokens", gen_tokens, "sequences", len(seqs),
          "of", n_seq, flush=True)


if __name__ == "__main__":
    main()
