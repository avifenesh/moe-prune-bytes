"""One REAP calibration pass that writes per-expert saliency and prunes nothing.

The saliency report is then used by 02_prune.py to cut any number of experts
from the BF16 checkpoint. One calibration pass serves every pruning level.

usage: 01_reap_report.py MODEL_DIR CALIB_JSONL REPORT_JSON
"""

import json
import sys

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.utils import load_context

model_dir, calib_path, report_path = sys.argv[1:4]
SEQ = 2048

tok = AutoTokenizer.from_pretrained(model_dir)
rows = [json.loads(l) for l in open(calib_path)]
ds = Dataset.from_list([{"text": r["text"]} for r in rows])
ds = ds.map(
    lambda ex: tok(ex["text"], max_length=SEQ, truncation=True, add_special_tokens=False),
    remove_columns=["text"],
)

with load_context():
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="bfloat16")

oneshot(
    model=model,
    dataset=ds,
    recipe=REAPPruningModifier(sparsity=0.0, report_path=report_path),
    max_seq_length=SEQ,
    num_calibration_samples=len(rows),
    moe_calibrate_all_experts=False,
)
print("report written", report_path)
