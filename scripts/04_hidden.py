"""Dump final hidden states (after the last norm) for the held-out set with vLLM.

Every arm keeps lm_head in BF16 and unchanged, so full-vocabulary logits for
any arm are hidden @ lm_head.T with the source lm_head. That gives exact
full-vocab KL against the source while each arm runs on its own real kernels.
05_kl.py checks that each arm's lm_head is bit-identical to the source.

usage: 04_hidden.py MODEL_DIR HELDOUT_JSONL OUT.pt [check]
  check: do not dump; load OUT.pt, run the generate runner on 4 sequences and
         compare vLLM's own prompt logprobs to the ones recomputed from OUT.pt.
"""

import json
import sys

import torch
from vllm import LLM, SamplingParams
from vllm.config import PoolerConfig
from vllm.inputs import TokensPrompt



def main():
    model_dir, held_path, out_path = sys.argv[1:4]
    check = len(sys.argv) > 4 and sys.argv[4] == "check"
    rows = [json.loads(l) for l in open(held_path)]
    prompts = [TokensPrompt(prompt_token_ids=r["ids"]) for r in rows]
    if not check:
        llm = LLM(
            model=model_dir,
            runner="pooling",
            convert="embed",
            pooler_config=PoolerConfig(task="token_embed", tok_pooling_type="ALL", use_activation=False),
            max_model_len=2560,
            gpu_memory_utilization=0.80,
            enable_prefix_caching=False,
            seed=0,
        )
        hs = []
        for c in range(0, len(prompts), 64):  # chunks: fp32 outputs of all rows at once do not fit
            outs = llm.encode(prompts[c:c + 64], pooling_task="token_embed", use_tqdm=False)
            hs.extend(o.outputs.data.to(torch.bfloat16).cpu() for o in outs)
            del outs
        hs = torch.stack(hs)
        assert hs.shape[:2] == (len(rows), 2048), hs.shape
        torch.save({"hidden": hs, "src": [r["src"] for r in rows]}, out_path)
        print("saved", out_path, tuple(hs.shape))
        return

    import glob

    from safetensors import safe_open

    hs = torch.load(out_path)["hidden"]
    W = None
    for f in glob.glob(f"{model_dir}/*.safetensors"):
        with safe_open(f, "pt") as g:
            if "lm_head.weight" in g.keys():
                W = g.get_tensor("lm_head.weight").cuda().float()
    gen = LLM(model=model_dir, max_model_len=2560, gpu_memory_utilization=0.80,
              enable_prefix_caching=False, max_logprobs=1)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)
    idx = [0, 1, 128, 129]
    res = gen.generate([prompts[i] for i in idx], sp)
    for i, r in zip(idx, res):
        ids = rows[i]["ids"]
        ref = torch.tensor([r.prompt_logprobs[t][ids[t]].logprob for t in range(1, 2048)])
        lp = torch.log_softmax(hs[i, :-1].cuda().float() @ W.T, -1)
        mine = lp.gather(1, torch.tensor(ids[1:], device="cuda")[:, None])[:, 0].cpu()
        print(f"check seq {i}: vllm nll {-ref.mean():.5f} hidden nll {-mine.mean():.5f} "
              f"max |diff| {(ref - mine).abs().max():.4f}")


if __name__ == "__main__":
    main()
