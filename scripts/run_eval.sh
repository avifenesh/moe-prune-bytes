#!/usr/bin/env bash
# Evaluate arms on the Spark: hidden states, KL vs S, lm-eval, speed (forward pass of ABBA).
# usage: run_eval.sh ARM [ARM...]   (S must run first: KL needs S hidden states)
# A step is skipped when its output exists.
set -uo pipefail
cd ~/mpb
S=~/models/Qwen3-30B-A3B-Instruct-2507
M=~/models/mpb
D=scripts/dock.sh
declare -A DIR=([S]=$S [A]=$M/r64 [B]=$M/full-nvfp4 [C]=$M/r64-nvfp4 [D]=$M/r72-fp8 [E]=$M/r96-nvfp4)
H=~/mpb/hidden
mkdir -p $H results
for a in "$@"; do
  m=${DIR[$a]}
  R=results/$a
  mkdir -p $R
  [ -f $H/$a.pt ] || bash $D mpb-hidden-$a python3 scripts/04_hidden.py $m data/heldout.jsonl $H/$a.pt > $R/hidden.log 2>&1
  echo "$a hidden $? $(date +%T)"
  if [ $a = S ] && ! [ -f $R/hidden_check.log ]; then
    bash $D mpb-check python3 scripts/04_hidden.py $m data/heldout.jsonl $H/S.pt check > $R/hidden_check.log 2>&1
    echo "S check $? $(date +%T)"
  fi
  if [ $a != S ] && ! [ -f $R/kl.json ]; then
    bash $D mpb-kl-$a python3 scripts/05_kl.py $S $H/S.pt $m $H/$a.pt data/heldout.jsonl $R/kl.json > $R/kl.log 2>&1
    echo "$a kl $? $(date +%T)"
  fi
  [ -d $R/lmeval/gsm ] || bash $D mpb-lmeval-$a bash scripts/08_lmeval.sh $m $R/lmeval > $R/lmeval.log 2>&1
  echo "$a lmeval $? $(date +%T)"
  [ -f $R/speed_fwd.json ] || bash $D mpb-speed-$a python3 scripts/06_speed.py $m $a fwd $R/speed_fwd.json 5 > $R/speed_fwd.log 2>&1
  echo "$a speed_fwd $? $(date +%T)"
done
echo EVAL_DONE "$@"
