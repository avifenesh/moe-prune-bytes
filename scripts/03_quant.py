"""Quantize a (pruned or full) BF16 checkpoint with llm-compressor.

scheme NVFP4: FP4 E2M1 weights, group 16 with FP8 E4M3 scales plus a per-tensor
  global scale; activations FP4 on the fly with a calibrated global scale.
  Calibrated on the fixed calibration set, every expert sees calibration data.
scheme FP8_DYNAMIC: FP8 E4M3 weights with per-channel scales, per-token dynamic
  FP8 activations. No calibration data needed.
lm_head and the router gates stay BF16 in every scheme. Embeddings stay BF16.

usage: 03_quant.py IN_DIR CALIB_JSONL SCHEME OUT_DIR [N_CAL]
"""

import json
import sys

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

in_dir, calib_path, scheme, out_dir = sys.argv[1:5]
n_cal = int(sys.argv[5]) if len(sys.argv) > 5 else 512
SEQ = 2048

tok = AutoTokenizer.from_pretrained(in_dir)
with load_context():
    model = AutoModelForCausalLM.from_pretrained(in_dir, dtype="bfloat16")

recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head", "re:.*mlp.gate$"])

if scheme == "FP8_DYNAMIC":
    oneshot(model=model, recipe=recipe)
else:
    rows = [json.loads(l) for l in open(calib_path)][:n_cal]
    ds = Dataset.from_list([{"text": r["text"]} for r in rows])
    ds = ds.map(
        lambda ex: tok(ex["text"], max_length=SEQ, truncation=True, add_special_tokens=False),
        remove_columns=["text"],
    )
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=SEQ,
        num_calibration_samples=len(rows),
        moe_calibrate_all_experts=True,
    )

model.save_pretrained(out_dir, save_compressed=True)
tok.save_pretrained(out_dir)
print("saved", out_dir)
