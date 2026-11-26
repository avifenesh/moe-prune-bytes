"""Mixed-domain prompts for self-distillation (teacher-generated and on-policy arms).

Domains match the heal mix: chat (ultrachat first turns), math (OpenR1-Math problems),
code (Magicoder OSS-Instruct problems), reasoning (OpenThoughts problems), tool use
(glaive function-calling: system with function specs + first user turn), instruction
(no_robots train prompts), philosophy (explain an SEP excerpt), multilingual (wikipedia
excerpt in he/ar/ru/ja/de/es with a task in the same language). 13-gram decontaminated
against every eval, and documents are taken past the ranges used by 22_phaseA_data.py.

usage: 31_prompts.py TOKENIZER_DIR DATA_DIR OUT_JSONL N_TOTAL
"""

import json
import random
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

tok_dir, data_dir, out_path, n_total = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(tok_dir)
NG = 13
rng = random.Random(7)


def enc(t):
    return tok(t, add_special_tokens=False)["input_ids"]


def grams(ids):
    return {hash(tuple(ids[i:i + NG])) for i in range(len(ids) - NG + 1)}


sys.path.insert(0, __file__.rsplit("/", 1)[0])
from evalset import build_bad  # noqa: E402

bad = build_bad(tok, data_dir)

SHARE = {"chat": 0.25, "math": 0.125, "code": 0.125, "reasoning": 0.125, "tool": 0.125,
         "instruction": 0.0625, "philosophy": 0.0625, "multilingual": 0.125}
TASK = {"he": "סכם את הקטע הבא ולאחר מכן הסבר את הרעיון המרכזי שלו:",
        "ar": "لخص النص التالي ثم اشرح فكرته الرئيسية:",
        "ru": "Кратко изложите следующий текст, а затем объясните его основную мысль:",
        "ja": "次の文章を要約し、その主旨を説明してください：",
        "de": "Fasse den folgenden Text zusammen und erkläre dann seinen Kerngedanken:",
        "es": "Resume el siguiente texto y luego explica su idea principal:"}


def gen_chat():
    for ex in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=23):
        yield [{"role": "user", "content": ex["messages"][0]["content"]}]


def gen_math():
    for ex in load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True).shuffle(seed=23, buffer_size=20000):
        yield [{"role": "user", "content": ex["problem"]}]


def gen_code():
    for ex in load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train").shuffle(seed=23):
        yield [{"role": "user", "content": ex["problem"]}]


def gen_reason():
    for ex in load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True).shuffle(seed=23, buffer_size=20000):
        yield [{"role": "user", "content": ex["conversations"][0]["value"]}]


def gen_tool():
    for ex in load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True).shuffle(seed=23, buffer_size=20000):
        user = ex["chat"].split("USER:", 1)[-1].split("ASSISTANT:", 1)[0].strip()
        sysmsg = ex["system"].replace("SYSTEM:", "").strip()
        if user:
            yield [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]


def gen_instr():
    for ex in load_dataset("HuggingFaceH4/no_robots", split="train").shuffle(seed=23):
        msgs = [m for m in ex["messages"] if m["role"] != "assistant"]
        if msgs and msgs[-1]["role"] == "user":
            yield msgs


def gen_phil():
    for ex in load_dataset("AiresPucrs/stanford-encyclopedia-philosophy", split="train").shuffle(seed=23):
        ex_t = ex["text"][:2500]
        yield [{"role": "user", "content": "Explain the argument in this passage, then give one strong objection to it:\n\n" + ex_t}]


def gen_multi():
    its = {l: iter(load_dataset("wikimedia/wikipedia", f"20231101.{l}", split="train", streaming=True)
                   .shuffle(seed=23, buffer_size=20000)) for l in TASK}
    while True:
        for l, it in its.items():
            ex = next(it)
            yield [{"role": "user", "content": TASK[l] + "\n\n" + ex["text"][:2000]}], l


GENS = {"chat": gen_chat, "math": gen_math, "code": gen_code, "reasoning": gen_reason, "tool": gen_tool,
        "instruction": gen_instr, "philosophy": gen_phil, "multilingual": gen_multi}
rows = []
for dom, share in SHARE.items():
    need, got, dropped = int(round(share * n_total)), 0, 0
    for item in GENS[dom]():
        msgs, lang = (item if isinstance(item, tuple) else (item, None))
        text = " ".join(m["content"] for m in msgs)
        if grams(enc(text)) & bad:
            dropped += 1
            continue
        rows.append({"src": dom, "lang": lang, "messages": msgs})
        got += 1
        if got == need:
            break
    print(dom, "prompts", got, "dropped", dropped, flush=True)
rng.shuffle(rows)
with open(out_path, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("total", len(rows), flush=True)
import os

os._exit(0)
