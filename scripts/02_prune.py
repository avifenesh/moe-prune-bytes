"""Remove experts from the BF16 checkpoint using the REAP saliency report.

Real removal: expert tensors are dropped from the safetensors files, router
rows are sliced, and config.num_experts is rewritten. Uniform per layer, so
vLLM loads the result as a plain Qwen3-MoE with fewer experts.

usage: 02_prune.py MODEL_DIR REPORT_JSON KEEP OUT_DIR
"""

import sys

from compressed_tensors.entrypoints.convert import convert_checkpoint

from llmcompressor.entrypoints.converter import ExpertPruner

model_dir, report, keep, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
N = 128
sparsity = (N - keep) / N

pruner = ExpertPruner.from_pretrained(
    model_dir,
    sparsity=sparsity,
    metric="saliency",
    uniform=True,
    saliency_report_path=report,
    expert_pattern=r"mlp\.experts\.\d+\.(gate|up|down)_proj",
)
convert_checkpoint(model_stub=model_dir, save_directory=out, converter=pruner, max_workers=8)
print("pruned", keep, "experts ->", out)
