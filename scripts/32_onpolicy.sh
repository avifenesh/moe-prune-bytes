#!/usr/bin/env bash
# On-policy self-distillation (arm SDb) on one Spark, 3 rounds over one 376-step schedule:
#   round r: the current student (LoRA merged, exported to NVFP4 with the training numerics)
#   answers its own slice of prompts ->
#   packed sequences + the shared 64 heal validation sequences -> source hidden states ->
#   continue LoRA (QAD as set) training for this round's steps from the previous state.
# usage: 32_onpolicy.sh STUDENT_DIR OUT_PREFIX QAD
set -uo pipefail
cd ~/mpb
S=~/models/Qwen3-30B-A3B-Instruct-2507
STU0=$1; OUT=$2; QADV=$3
D=scripts/dock.sh
export ACT_AMAX=phaseA/act_amax_emix.json
TOTAL=376; ROUNDS=3; PER=(125 125 126); OFF=(0 125 250)
mkdir -p phaseA/onpol runs
stu=$STU0
tail -n 64 phaseA/heal_mix.jsonl > phaseA/onpol/val64.jsonl
for r in 0 1 2; do
  n_seq=$(( ${PER[$r]} * 8 ))
  f=phaseA/onpol/r$r.jsonl
  # sample from the deployed form of the current student: its NVFP4 export (training numerics)
  q=${stu}-nvfp4g
  if [ ! -f $f.gen ]; then
    [ -f $q/model.safetensors.index.json ] || systemd-run --user --scope -q -p MemoryMax=60G -- \
      .venv-lc/bin/python scripts/35_export_nvfp4.py $stu ~/models/mpb/r96mix-nvfp4 $q > runs/onpol_export_r$r.log 2>&1
    bash $D mpbA-gen-r$r python3 scripts/30_sd_gen.py $q phaseA/sd_prompts.jsonl $f.gen $n_seq 2 $r $(( r * 1365 )) 1365 > runs/onpol_gen_r$r.log 2>&1
  fi
  rm -rf $q
  echo "round $r gen $? $(date +%T) $(wc -l < $f.gen) seqs"
  cat $f.gen phaseA/onpol/val64.jsonl > $f
  mkdir -p phaseA/onpol/teacher_r$r
  [ "$(ls phaseA/onpol/teacher_r$r | wc -l)" -ge 1 ] || bash $D mpbA-teach-r$r python3 scripts/14_teacher_hidden.py $S $f phaseA/onpol/teacher_r$r 128 > runs/onpol_teacher_r$r.log 2>&1
  echo "round $r teacher $? $(date +%T)"
  st_in=""; [ $r -gt 0 ] && st_in=phaseA/onpol/state_r$((r-1)).pt
  QAD=$QADV STATE_IN=$st_in STATE_OUT=phaseA/onpol/state_r$r.pt STEP_OFFSET=${OFF[$r]} MAX_STEPS=${PER[$r]} TOTAL_STEPS=$TOTAL \
    systemd-run --user --scope -q -p MemoryMax=110G -- .venv-lc/bin/python scripts/29_heal_qad.py $STU0 phaseA/onpol/teacher_r$r $f ${OUT}_r$r runs/onpol_r$r.json > runs/onpol_train_r$r.log 2>&1
  echo "round $r train $? $(date +%T)"
  grep -q "^saved" runs/onpol_train_r$r.log || exit 1
  [ $r -gt 0 ] && rm -rf ${OUT}_r$((r-1))
  rm -rf phaseA/onpol/teacher_r$r
  stu=${OUT}_r$r
done
echo ONPOL_DONE ${OUT}_r2
