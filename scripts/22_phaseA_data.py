"""Phase A data: per-domain held-out KL sets, a domain-mixed calibration set (E-mix) and a
domain-mixed heal corpus. Every document is 13-gram decontaminated against all evals.

Domains and sources (open datasets, streamed):
  code        codeparrot/codeparrot-clean-valid (Python)
  math        open-r1/OpenR1-Math-220k (problem + solution)
  multilingual wikimedia/wikipedia 20231101.{he,ar,ru,ja,de,es}
  tool        glaiveai/glaive-function-calling-v2 (system + chat transcript)
  instruction HuggingFaceH4/no_robots (chat template)
  philosophy  AiresPucrs/stanford-encyclopedia-philosophy
  reasoning   open-thoughts/OpenThoughts-114k (chat template, long reasoning)
  chat        HuggingFaceH4/ultrachat_200k (calibration and heal only; the old held-out set covers it)

Document ranges are disjoint between the three outputs: per source, documents are taken in
a fixed shuffled order and assigned held-out first, then calibration, then heal.

Eval items for decontamination: MMLU, ARC-C/E, GSM8K, MATH-500, HumanEval, MBPP(+),
IFEval, BFCL v3 (simple, multiple, parallel, irrelevance), Global-MMLU (he), and the old
256 held-out sequences.

usage: 22_phaseA_data.py TOKENIZER_DIR OUT_DIR
writes heldout_domains.jsonl {"src","lang","ids"[2048]}, calib_mix.jsonl {"src","text"},
       heal_mix.jsonl {"src","ids"[2048]}, phaseA_meta.json
"""

import hashlib
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

tok_dir, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(tok_dir)
SEQ, NG = 2048, 13
rng = random.Random(20261124)


def enc(t):
    return tok(t, add_special_tokens=False)["input_ids"]


def grams(ids):
    return {hash(tuple(ids[i:i + NG])) for i in range(len(ids) - NG + 1)}


# ---- decontamination set ----
bad, n_eval = set(), 0


def add_eval(text):
    global n_eval
    n_eval += 1
    bad.update(grams(enc(text)))


for ex in load_dataset("cais/mmlu", "all", split="test"):
    add_eval(ex["question"] + " " + " ".join(ex["choices"]))
for cfg in ("ARC-Challenge", "ARC-Easy"):
    for ex in load_dataset("allenai/ai2_arc", cfg, split="test"):
        add_eval(ex["question"] + " " + " ".join(ex["choices"]["text"]))
for ex in load_dataset("openai/gsm8k", "main", split="test"):
    add_eval(ex["question"] + " " + ex["answer"])
for ex in load_dataset("HuggingFaceH4/MATH-500", split="test"):
    add_eval(ex["problem"] + " " + ex["solution"])
for ex in load_dataset("openai/openai_humaneval", split="test"):
    add_eval(ex["prompt"] + ex["canonical_solution"])
for ex in load_dataset("evalplus/mbppplus", split="test"):
    add_eval(ex["prompt"] + " " + ex["code"])
for ex in load_dataset("google/IFEval", split="train"):
    add_eval(ex["prompt"])
for ex in load_dataset("CohereLabs/Global-MMLU", "he", split="test"):
    add_eval(ex["question"] + " " + " ".join(ex[k] for k in ("option_a", "option_b", "option_c", "option_d")))
for cat in ("simple", "multiple", "parallel", "irrelevance"):
    p = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f"BFCL_v3_{cat}.json", repo_type="dataset")
    for line in open(p):
        add_eval(json.dumps(json.loads(line)))
for line in open(out / "heldout.jsonl"):
    n_eval += 1
    bad.update(grams(json.loads(line)["ids"]))
print("eval items", n_eval, "13-gram hashes", len(bad), flush=True)


# ---- sources ----
def chat(msgs):
    return tok.apply_chat_template(msgs, tokenize=False)


def src_code():
    for ex in load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True).shuffle(seed=1, buffer_size=10000):
        yield ex["content"]


def src_math():
    for ex in load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True).shuffle(seed=1, buffer_size=10000):
        yield ex["problem"] + "\n\n" + ex["solution"]


def src_wiki(lang):
    def g():
        for ex in load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True).shuffle(seed=1, buffer_size=10000):
            yield ex["title"] + "\n\n" + ex["text"]
    return g


def src_tool():
    for ex in load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True).shuffle(seed=1, buffer_size=10000):
        yield ex["system"] + "\n\n" + ex["chat"]


def src_instr():
    for split in ("test", "train"):
        for ex in load_dataset("HuggingFaceH4/no_robots", split=split).shuffle(seed=1):
            yield chat(ex["messages"])


def src_phil():
    for ex in load_dataset("AiresPucrs/stanford-encyclopedia-philosophy", split="train").shuffle(seed=1):
        yield ex["text"]


def src_reason():
    for ex in load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True).shuffle(seed=1, buffer_size=10000):
        msgs = [{"role": "system", "content": ex["system"]}] + [
            {"role": "user" if c["from"] == "user" else "assistant", "content": c["value"]} for c in ex["conversations"]]
        yield chat(msgs)


def src_chat():
    calib_ids = {json.loads(l)["prompt_id"] for l in open(out / "calib.jsonl")}
    for ex in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=11):
        if ex["prompt_id"] not in calib_ids:
            yield chat(ex["messages"])


LANGS = ["he", "ar", "ru", "ja", "de", "es"]
SOURCES = {"code": src_code, "math": src_math, "tool": src_tool, "instruction": src_instr,
           "philosophy": src_phil, "reasoning": src_reason, "chat": src_chat}
SOURCES.update({f"wiki_{l}": src_wiki(l) for l in LANGS})

# held-out sequences per source, calibration rows per source, heal sequences per source
HELD = {"code": 64, "math": 64, "tool": 64, "instruction": 64, "philosophy": 64, "reasoning": 64, "chat": 0}
HELD.update({f"wiki_{l}": 32 for l in LANGS})
# 819 rows x up to 1152 tokens: about the old calibration's 579,870 tokens (many tool,
# instruction and math documents are shorter than the cap, so 512 rows gave only 363k)
CAL = {"chat": 205, "code": 102, "math": 102, "reasoning": 102, "tool": 102, "instruction": 51, "philosophy": 51}
CAL.update({f"wiki_{l}": 17 for l in LANGS})
CAL_MAX_TOK = 1152
HEAL = {"chat": 768, "code": 384, "math": 384, "reasoning": 384, "tool": 384, "instruction": 192, "philosophy": 192}
HEAL.update({f"wiki_{l}": 64 for l in LANGS})

held, cal, heal, meta = [], [], [], {}
for name, gen in SOURCES.items():
    stats = {"docs": 0, "dropped": 0}
    it = gen()
    buf = []

    def clean_docs():
        for text in it:
            ids = enc(text)
            if len(ids) < 64:
                continue
            stats["docs"] += 1
            if grams(ids) & bad:
                stats["dropped"] += 1
                continue
            yield text, ids

    docs = clean_docs()
    # held-out: packed 2048-token sequences
    while sum(1 for r in held if r["src"] == name) < HELD[name]:
        _, ids = next(docs)
        buf.extend(ids)
        while len(buf) >= SEQ and sum(1 for r in held if r["src"] == name) < HELD[name]:
            held.append({"src": name, "ids": buf[:SEQ]})
            buf = buf[SEQ:]
    buf = []
    # calibration: whole documents truncated to CAL_MAX_TOK tokens
    for _ in range(CAL[name]):
        text, ids = next(docs)
        cal.append({"src": name, "text": tok.decode(ids[:CAL_MAX_TOK])})
    # heal: packed 2048-token sequences
    n = 0
    while n < HEAL[name]:
        _, ids = next(docs)
        buf.extend(ids)
        while len(buf) >= SEQ and n < HEAL[name]:
            heal.append({"src": name, "ids": buf[:SEQ]})
            buf = buf[SEQ:]
            n += 1
    meta[name] = stats
    print(name, stats, flush=True)

rng.shuffle(heal)
import os
for fname, rows in (("heldout_domains.jsonl", held), ("calib_mix.jsonl", cal), ("heal_mix.jsonl", heal)):
    with open(out / fname, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
cal_tokens = sum(len(enc(r["text"])) for r in cal)
meta = {"sources": meta, "n_eval_items": n_eval, "ngram": NG, "held_seqs": len(held), "cal_rows": len(cal),
        "cal_tokens": cal_tokens, "heal_seqs": len(heal), "heal_tokens": len(heal) * SEQ,
        "sha256": {f: hashlib.sha256((out / f).read_bytes()).hexdigest()
                   for f in ("heldout_domains.jsonl", "calib_mix.jsonl", "heal_mix.jsonl")}}
json.dump(meta, open(out / "phaseA_meta.json", "w"), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != "sources"}), flush=True)
os._exit(0)  # skip interpreter teardown: streaming threads crash at finalize
