#!/usr/bin/env bash
# Bytes for every arm, paired bootstrap vs S, final table.
set -euo pipefail
cd ~/mpb
S=~/models/Qwen3-30B-A3B-Instruct-2507
M=~/models/mpb
for a in S:$S A:$M/r64 B:$M/full-nvfp4 C:$M/r64-nvfp4 D:$M/r72-fp8 E:$M/r96-nvfp4; do
  python3 scripts/07_bytes.py ${a#*:} runs/bytes_${a%%:*}.json > /dev/null
done
python3 scripts/09_bootstrap.py results results/bootstrap.json S A B C D E > results/bootstrap.log
python3 scripts/10_table.py . results/table.md results/table.json
echo FINAL_OK
