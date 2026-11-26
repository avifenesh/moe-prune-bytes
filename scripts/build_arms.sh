#!/usr/bin/env bash
# Build all arms on the Spark: REAP report, pruning, quantization, bytes.
# Each step is skipped when its output exists, so the script can be rerun.
set -euo pipefail
cd ~/mpb
P=".venv-lc/bin/python"
S=~/models/Qwen3-30B-A3B-Instruct-2507
M=~/models/mpb
mkdir -p runs $M
run() { systemd-run --user --scope -q -p MemoryMax=108G -- "$@"; }
[ -f runs/reap_report.json ] || run $P scripts/01_reap_report.py $S data/calib.jsonl runs/reap_report.json > runs/reap.log 2>&1
echo "reap done $(date +%T)"
for k in 64 72 96; do
  [ -f $M/r$k/config.json ] || run $P scripts/02_prune.py $S runs/reap_report.json $k $M/r$k > runs/prune_r$k.log 2>&1
  echo "prune r$k done $(date +%T)"
done
[ -f $M/r72-fp8/config.json ] || run $P scripts/03_quant.py $M/r72 data/calib.jsonl FP8_DYNAMIC $M/r72-fp8 > runs/quant_D.log 2>&1
echo "D done $(date +%T)"
[ -f $M/r64-nvfp4/config.json ] || run $P scripts/03_quant.py $M/r64 data/calib.jsonl NVFP4 $M/r64-nvfp4 > runs/quant_C.log 2>&1
echo "C done $(date +%T)"
[ -f $M/full-nvfp4/config.json ] || run $P scripts/03_quant.py $S data/calib.jsonl NVFP4 $M/full-nvfp4 > runs/quant_B.log 2>&1
echo "B done $(date +%T)"
[ -f $M/r96-nvfp4/config.json ] || run $P scripts/03_quant.py $M/r96 data/calib.jsonl NVFP4 $M/r96-nvfp4 > runs/quant_E.log 2>&1
echo "E done $(date +%T)"
for a in S:$S A:$M/r64 B:$M/full-nvfp4 C:$M/r64-nvfp4 D:$M/r72-fp8 E:$M/r96-nvfp4; do
  python3 scripts/07_bytes.py ${a#*:} runs/bytes_${a%%:*}.json > /dev/null
done
echo BUILD_OK
