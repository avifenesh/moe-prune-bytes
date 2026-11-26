"""Patch vLLM's Qwen3-MoE so each layer can have its own number of routed experts.

Reads an optional `num_experts_per_layer` list from the HF config (indexed by the decoder
layer number parsed from the module prefix). Without the list the model is unchanged.
Expert-weight loading needs no change: checkpoint expert ids per layer run 0..n_L-1.

usage: patch_vllm_qwen3_per_layer.py IN_QWEN3_MOE_PY OUT_PY
"""

import sys

src = open(sys.argv[1]).read()
old = "        self.n_routed_experts = config.num_experts\n"
new = ("        _per_layer = getattr(config, \"num_experts_per_layer\", None)\n"
       "        _n_exp = _per_layer[extract_layer_index(prefix)] if _per_layer else config.num_experts\n"
       "        self.n_routed_experts = _n_exp\n")
assert src.count(old) == 1, "anchor 1"
src = src.replace(old, new)
old2 = ("        self.gate = GateLinear(\n"
        "            config.hidden_size,\n"
        "            config.num_experts,\n")
new2 = ("        self.gate = GateLinear(\n"
        "            config.hidden_size,\n"
        "            _n_exp,\n")
assert src.count(old2) == 1, "anchor 2"
src = src.replace(old2, new2)
assert "extract_layer_index" in src.split("class ")[0] or "import" in src
open(sys.argv[2], "w").write(src)
print("patched")
