#!/usr/bin/env bash
# Run on a Spark: one Phase A heal arm on E-mix's BF16 student (r96mix), then NVFP4 (same
# calibration as E-mix) and the full Phase A evaluation.
# usage: heal_arm_A.sh NAME QAD TRAIN_JSONL TEACHER_DIR
set -uo pipefail
cd ~/mpb
NAME=$1; QADV=$2; DATA=$3; TDIR=$4
M=~/models/mpb
STU=$M/r96mix
OUT=$M/$NAME
mkdir -p runs
[ -f $OUT-nvfp4/config.json ] || [ -f $OUT/model.safetensors.index.json ] || QAD=$QADV systemd-run --user --scope -q -p MemoryMax=110G -- \
  .venv-lc/bin/python scripts/29_heal_qad.py $STU $TDIR $DATA $OUT runs/heal_$NAME.json > runs/heal_$NAME.log 2>&1
echo "$NAME train $? $(date +%T)"
[ -f $OUT-nvfp4/config.json ] || grep -q "^saved" runs/heal_$NAME.log || exit 1
[ -f $OUT-nvfp4/config.json ] || systemd-run --user --scope -q -p MemoryMax=108G -- .venv-lc/bin/python scripts/03_quant.py $OUT phaseA/calib_mix.jsonl NVFP4 $OUT-nvfp4 > runs/quant_$NAME.log 2>&1
echo "$NAME quant $? $(date +%T)"
# disk: keep the adapters ($OUT.adapters.pt) and the NVFP4 model; the BF16 merge is rebuildable
[ -f $OUT-nvfp4/model.safetensors.index.json ] || [ -n "$(ls $OUT-nvfp4/*.safetensors 2>/dev/null)" ] && rm -rf $OUT
python3 scripts/07_bytes.py $OUT-nvfp4 runs/bytes_$NAME.json > /dev/null
NO_SUITE="S" bash scripts/phaseA_baselines.sh S=~/models/Qwen3-30B-A3B-Instruct-2507 $NAME=$OUT-nvfp4
bash scripts/extras_A.sh $NAME $OUT-nvfp4
echo "HEAL_ARM_DONE $NAME"
