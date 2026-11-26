"""Heal (distillation) training set, disjoint from every eval set.

Mix: 50% HuggingFaceFW/fineweb-edu (sample-10BT, first shard), 50% ultrachat_200k
train_sft (chat template; the 512 calibration rows are excluded). Packed into
sequences of exactly 2048 tokens.

Decontamination: any document sharing a 13-token n-gram with an eval item is dropped.
Eval items: MMLU test, ARC-Challenge and ARC-Easy test (question + choices), the 256
held-out KL sequences (ultrachat test_sft + wikitext-103 test), GSM8K test.

usage: 13_heal_data.py OUT_DIR N_SEQ_PER_SOURCE
writes OUT_DIR/heal.jsonl ({"src", "ids"}) and OUT_DIR/heal_meta.json
"""

import hashlib
import json
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

out = Path(sys.argv[1])
n_per = int(sys.argv[2])
SEQ = 2048
NG = 13
tok = AutoTokenizer.from_pretrained("~/models/Qwen3-30B-A3B-Instruct-2507")


def grams(ids):
    return {hash(tuple(ids[i:i + NG])) for i in range(len(ids) - NG + 1)}


bad = set()
n_eval = 0


def add_eval(text):
    global n_eval
    n_eval += 1
    bad.update(grams(tok(text, add_special_tokens=False)["input_ids"]))


for sub in ("all",):
    for ex in load_dataset("cais/mmlu", sub, split="test"):
        add_eval(ex["question"] + " " + " ".join(ex["choices"]))
for cfg in ("ARC-Challenge", "ARC-Easy"):
    for ex in load_dataset("allenai/ai2_arc", cfg, split="test"):
        add_eval(ex["question"] + " " + " ".join(ex["choices"]["text"]))
for ex in load_dataset("openai/gsm8k", "main", split="test"):
    add_eval(ex["question"])
for line in open("~/mpb/data/heldout.jsonl"):
    ids = json.loads(line)["ids"]
    n_eval += 1
    bad.update(grams(ids))
print("eval items", n_eval, "13-gram hashes", len(bad), flush=True)

calib_ids = {json.loads(l)["prompt_id"] for l in open("~/mpb/data/calib.jsonl")}


def pack(docs, src):
    rows, buf, dropped, used = [], [], 0, 0
    for text in docs:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if grams(ids) & bad:
            dropped += 1
            continue
        used += 1
        buf.extend(ids)
        while len(buf) >= SEQ and len(rows) < n_per:
            rows.append({"src": src, "ids": buf[:SEQ]})
            buf = buf[SEQ:]
        if len(rows) >= n_per:
            break
    assert len(rows) == n_per, (src, len(rows))
    return rows, {"docs_used": used, "docs_dropped_contamination": dropped}


fw_path = hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet", repo_type="dataset")
fw = load_dataset("parquet", data_files=fw_path, split="train")
fw_rows, fw_meta = pack((ex["text"] for ex in fw), "fineweb-edu")
print("fineweb-edu", fw_meta, flush=True)

uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=7)
uc_rows, uc_meta = pack((tok.apply_chat_template(ex["messages"], tokenize=False) for ex in uc
                         if ex["prompt_id"] not in calib_ids), "ultrachat-train")
print("ultrachat", uc_meta, flush=True)

rows = [r for pair in zip(fw_rows, uc_rows) for r in pair]  # interleave sources
with open(out / "heal.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
meta = {"n_seq": len(rows), "seq_len": SEQ, "tokens": len(rows) * SEQ, "fineweb-edu": fw_meta,
        "ultrachat-train": uc_meta, "ngram": NG, "eval_items": n_eval,
        "sha256": hashlib.sha256((out / "heal.jsonl").read_bytes()).hexdigest()}
json.dump(meta, open(out / "heal_meta.json", "w"), indent=1)
print(meta)
