"""Per-domain routing counts of a model on a JSONL of texts, using vLLM's routed-experts output.

For each row (field "text" or "ids", plus "src"), runs the prompt with max_tokens=1 and
counts, per source domain, how often each (layer, expert) is in the top-k of a token.

usage: 25_route_counts.py MODEL_DIR JSONL OUT_JSON
"""

import json
import sys
from collections import defaultdict

import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    model, path, out = sys.argv[1:4]
    rows = [json.loads(l) for l in open(path)]
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.80, enable_prefix_caching=True,
              enable_return_routed_experts=True)
    tok = llm.get_tokenizer()
    prompts = [TokensPrompt(prompt_token_ids=r["ids"] if "ids" in r else tok(r["text"], add_special_tokens=False)["input_ids"])
               for r in rows]
    res = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0))
    counts = defaultdict(lambda: None)
    tokens = defaultdict(int)
    for r, o in zip(rows, res):
        re = o.outputs[0].routed_experts
        if re is None:
            raise SystemExit("no routed_experts returned")
        re = np.asarray(re)  # [seq_len, layers, topk]
        L = re.shape[1]
        c = counts[r["src"]]
        if c is None:
            c = np.zeros((L, 128), dtype=np.int64)
            counts[r["src"]] = c
        for layer in range(L):
            np.add.at(c[layer], re[:, layer, :].reshape(-1), 1)
        tokens[r["src"]] += re.shape[0]
    json.dump({"tokens": tokens, "counts": {k: v.tolist() for k, v in counts.items()},
               "seq_len_sample": int(np.asarray(res[0].outputs[0].routed_experts).shape[0]),
               "prompt_len_sample": len(prompts[0]["prompt_token_ids"])}, open(out, "w"))
    print("done", dict(tokens))


if __name__ == "__main__":
    main()
