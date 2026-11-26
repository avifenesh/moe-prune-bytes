#!/usr/bin/env bash
# Run on a Spark: one Phase A QAD heal arm on E-mix's BF16 student (r96mix), exported to NVFP4
# with the training numerics (35_export_nvfp4.py; E-mix's checkpoint as the template for
# names and static activation scales), then the Phase A evaluation.
# usage: heal_arm_A2.sh NAME TRAIN_JSONL TEACHER_DIR
set -uo pipefail
cd ~/mpb
NAME=$1; DATA=$2; TDIR=$3
M=~/models/mpb
STU=$M/r96mix
OUT=$M/$NAME
export ACT_AMAX=phaseA/act_amax_emix.json
mkdir -p runs
[ -f $OUT-nvfp4/config.json ] || [ -f $OUT/model.safetensors.index.json ] || QAD=1 systemd-run --user --scope -q -p MemoryMax=110G -- \
  .venv-lc/bin/python scripts/29_heal_qad.py $STU $TDIR $DATA $OUT runs/heal_$NAME.json > runs/heal_$NAME.log 2>&1
echo "$NAME train $? $(date +%T)"
[ -f $OUT-nvfp4/config.json ] || grep -q "^saved" runs/heal_$NAME.log || exit 1
bash scripts/export_A.sh $OUT $OUT-nvfp4 runs/export_$NAME.log
echo "$NAME export $? $(date +%T)"
python3 scripts/07_bytes.py $OUT-nvfp4 runs/bytes_$NAME.json > /dev/null
NO_SUITE="S" bash scripts/phaseA_baselines.sh S=~/models/Qwen3-30B-A3B-Instruct-2507 $NAME=$OUT-nvfp4
bash scripts/extras_A.sh $NAME $OUT-nvfp4
echo "HEAL_ARM_DONE $NAME"
