"""Build the fixed calibration set and the held-out KL set.

Calibration: 512 conversations from ultrachat_200k train_sft (seed 42),
chat template applied, truncated to 2048 tokens. Used for REAP saliency and
for NVFP4 calibration.

Held-out KL set: 256 sequences of exactly 2048 tokens, disjoint from calibration.
  - 128 packed from ultrachat_200k test_sft (same distribution as calibration)
  - 128 packed from wikitext-103 test (different distribution)

Outputs (JSONL, token ids, so every arm sees identical inputs):
  calib.jsonl   {"text": str}  (chat-templated text; tokenized by the consumer)
  heldout.jsonl {"src": "ultrachat"|"wikitext", "ids": [2048 ints]}
"""

import hashlib
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SEQ = 2048
N_CAL = 512
N_HELD = 128

tok = AutoTokenizer.from_pretrained(MODEL)

# calibration
cal = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=42)
cal = cal.select(range(N_CAL))
with open(OUT / "calib.jsonl", "w") as f:
    for ex in cal:
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        f.write(json.dumps({"prompt_id": ex["prompt_id"], "text": text}) + "\n")
cal_ids = set(cal["prompt_id"])


def pack(texts, n, src):
    out, buf = [], []
    for t in texts:
        buf.extend(tok(t, add_special_tokens=False)["input_ids"])
        while len(buf) >= SEQ and len(out) < n:
            out.append({"src": src, "ids": buf[:SEQ]})
            buf = buf[SEQ:]
        if len(out) >= n:
            break
    assert len(out) == n, (src, len(out))
    return out


test = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft").shuffle(seed=1234)
test = test.filter(lambda ex: ex["prompt_id"] not in cal_ids)
uc = pack((tok.apply_chat_template(ex["messages"], tokenize=False) for ex in test), N_HELD, "ultrachat")

wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
docs, cur = [], []
for line in wt["text"]:
    if line.startswith(" = ") and not line.startswith(" = = ") and cur:
        docs.append("".join(cur))
        cur = []
    cur.append(line)
docs.append("".join(cur))
random.Random(1234).shuffle(docs)
wk = pack(docs, N_HELD, "wikitext")

with open(OUT / "heldout.jsonl", "w") as f:
    for r in uc + wk:
        f.write(json.dumps(r) + "\n")

for name in ("calib.jsonl", "heldout.jsonl"):
    h = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    print(name, h)
