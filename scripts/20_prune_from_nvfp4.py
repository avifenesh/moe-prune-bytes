"""Prune experts from an NVFP4 (compressed-tensors) checkpoint with ExpertPruner.

usage: 20_prune_from_nvfp4.py IN_DIR REPORT SPARSITY uniform|nonuniform OUT_DIR
"""

import sys

from compressed_tensors.entrypoints.convert import convert_checkpoint

from llmcompressor.entrypoints.converter import ExpertPruner

src, report, sparsity, mode, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
pruner = ExpertPruner.from_pretrained(src, sparsity=sparsity, metric="saliency", uniform=(mode == "uniform"),
                                      saliency_report_path=report,
                                      expert_pattern=r"mlp\.experts\.\d+\.(gate|up|down)_proj")
convert_checkpoint(model_stub=src, save_directory=out, converter=pruner, max_workers=8)
print("pruned", mode, "->", out)
