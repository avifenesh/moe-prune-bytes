"""13-gram set of every eval item, for decontamination (same sources as 22_phaseA_data.py)."""

import json

from datasets import load_dataset
from huggingface_hub import hf_hub_download

NG = 13


def build_bad(tok, data_dir):
    def grams(ids):
        return {hash(tuple(ids[i:i + NG])) for i in range(len(ids) - NG + 1)}

    bad, n = set(), 0

    def add(text):
        nonlocal n
        n += 1
        bad.update(grams(tok(text, add_special_tokens=False)["input_ids"]))

    for ex in load_dataset("cais/mmlu", "all", split="test"):
        add(ex["question"] + " " + " ".join(ex["choices"]))
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for ex in load_dataset("allenai/ai2_arc", cfg, split="test"):
            add(ex["question"] + " " + " ".join(ex["choices"]["text"]))
    for ex in load_dataset("openai/gsm8k", "main", split="test"):
        add(ex["question"] + " " + ex["answer"])
    for ex in load_dataset("HuggingFaceH4/MATH-500", split="test"):
        add(ex["problem"] + " " + ex["solution"])
    for ex in load_dataset("openai/openai_humaneval", split="test"):
        add(ex["prompt"] + ex["canonical_solution"])
    for ex in load_dataset("evalplus/mbppplus", split="test"):
        add(ex["prompt"] + " " + ex["code"])
    for ex in load_dataset("google/IFEval", split="train"):
        add(ex["prompt"])
    for ex in load_dataset("CohereLabs/Global-MMLU", "he", split="test"):
        add(ex["question"] + " " + " ".join(ex[k] for k in ("option_a", "option_b", "option_c", "option_d")))
    for cat in ("simple", "multiple", "parallel", "irrelevance"):
        p = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f"BFCL_v3_{cat}.json", repo_type="dataset")
        for line in open(p):
            add(json.dumps(json.loads(line)))
    for fname in ("heldout.jsonl", "heldout_domains.jsonl"):
        for line in open(f"{data_dir}/{fname}"):
            n += 1
            bad.update(grams(json.loads(line)["ids"]))
    print("eval items", n, "13-gram hashes", len(bad), flush=True)
    return bad
