"""Tool use: BFCL v3 single-turn categories (simple, multiple, parallel, irrelevance).

Prompts go through the model's own chat template with `tools=` (Qwen3 emits
<tool_call>{"name": ..., "arguments": {...}}</tool_call>). Scoring follows BFCL's AST
checker: function name must match; every argument given must be one of the allowed
values in possible_answer (light normalization: strings case- and space-insensitive,
int/float equal by value, lists element-wise); an argument may be omitted only when ""
is among its allowed values; no extra arguments. Parallel calls match in any order.
Irrelevance passes only when the model makes no call.

Sampling: the model's recommended settings (temperature 0.7, top_p 0.8, top_k 20), seed 0.

usage: 26_bfcl.py MODEL_DIR BFCL_DIR OUT_JSONL
  BFCL_DIR holds BFCL_v3_{cat}.json and possible_answer/BFCL_v3_{cat}.json
"""

import json
import os
import re
import sys

from vllm import LLM, SamplingParams

CATS = ["simple", "multiple", "parallel", "irrelevance"]
TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string", "integer": "integer"}


def fix_schema(x):
    if isinstance(x, dict):
        y = {k: fix_schema(v) for k, v in x.items()}
        if isinstance(y.get("type"), str):
            y["type"] = TYPE_MAP.get(y["type"], y["type"])
        return y
    if isinstance(x, list):
        return [fix_schema(v) for v in x]
    return x


def tools_of(funcs):
    return [{"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                              "parameters": fix_schema(f["parameters"])}} for f in funcs]


def parse_calls(text):
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try:
            c = json.loads(m.group(1))
            args = c.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            calls.append((c.get("name"), args))
        except Exception:
            calls.append(("__unparseable__", {}))
    return calls


def norm(v):
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v.strip().lower())
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    return v


def match_call(call, gt):
    """call = (name, args); gt = {name: {param: [allowed...]}}"""
    (gname, gparams), = gt.items()
    name, args = call
    if name != gname or not isinstance(args, dict):
        return False
    for k in args:
        if k not in gparams:
            return False
    for p, allowed in gparams.items():
        if p in args:
            if norm(args[p]) not in [norm(a) for a in allowed if a != ""]:
                return False
        elif "" not in allowed:
            return False
    return True


def score(cat, calls, gts):
    if cat == "irrelevance":
        return len(calls) == 0
    if cat in ("simple", "multiple"):
        return len(calls) == 1 and match_call(calls[0], gts[0])
    if len(calls) != len(gts):
        return False
    used = set()
    for g in gts:
        hit = next((i for i, c in enumerate(calls) if i not in used and match_call(c, g)), None)
        if hit is None:
            return False
        used.add(hit)
    return True


def main():
    model, bdir, out = sys.argv[1:4]
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.80, seed=0)
    tok = llm.get_tokenizer()
    items = []
    for cat in CATS:
        qs = [json.loads(l) for l in open(f"{bdir}/BFCL_v3_{cat}.json")]
        if os.environ.get("BFCL_LIMIT"):
            qs = qs[: int(os.environ["BFCL_LIMIT"])]
        ans = {}
        pa = f"{bdir}/possible_answer/BFCL_v3_{cat}.json"
        if os.path.exists(pa):
            ans = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(pa)}
        for q in qs:
            msgs = q["question"][0]
            prompt = tok.apply_chat_template(msgs, tools=tools_of(q["function"]), add_generation_prompt=True, tokenize=False)
            items.append({"cat": cat, "id": q["id"], "prompt": prompt, "gt": ans.get(q["id"], [])})
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=1024, seed=0)
    res = llm.generate([it["prompt"] for it in items], sp)
    with open(out, "w") as f:
        for it, r in zip(items, res):
            text = r.outputs[0].text
            calls = parse_calls(text)
            ok = score(it["cat"], calls, it["gt"])
            f.write(json.dumps({"cat": it["cat"], "id": it["id"], "correct": float(ok), "text": text}) + "\n")
    rows = [json.loads(l) for l in open(out)]
    for cat in CATS:
        sel = [r["correct"] for r in rows if r["cat"] == cat]
        print(cat, len(sel), round(sum(sel) / len(sel), 4))


if __name__ == "__main__":
    main()
