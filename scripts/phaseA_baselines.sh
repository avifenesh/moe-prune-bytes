#!/usr/bin/env bash
# Run on a Spark: Phase A baselines for S, B, E. Per-domain KL on heldout_domains.jsonl
# (S hidden states first), then the generation + multilingual + tool suite per arm.
# usage: [NO_SUITE="S ..."] phaseA_baselines.sh ARM=DIR [ARM=DIR ...]   (S must be first)
set -uo pipefail
cd ~/mpb
S=~/models/Qwen3-30B-A3B-Instruct-2507
H=~/mpb/hiddenA
D=scripts/dock.sh
export IMAGE=mpb-eval:v0.34.0-a
mkdir -p $H resultsA
for spec in "$@"; do
  a=${spec%%=*}; m=${spec#*=}
  R=resultsA/$a
  mkdir -p $R
  [ -f $H/$a.pt ] || bash $D mpbA-hidden-$a python3 scripts/04_hidden.py $m phaseA/heldout_domains.jsonl $H/$a.pt > $R/hidden.log 2>&1
  echo "$a hidden $? $(date +%T)"
  if [ $a != S ]; then
    [ -f $R/kl_domains.json ] || bash $D mpbA-kl-$a python3 scripts/05_kl.py $S $H/S.pt $m $H/$a.pt phaseA/heldout_domains.jsonl $R/kl_domains.json > $R/kl.log 2>&1
    echo "$a kl $? $(date +%T)"
    rm -f $H/$a.pt
  fi
  case " ${NO_SUITE:-} " in *" $a "*) echo "$a suite skipped"; continue ;; esac
  bash $D mpbA-suite-$a bash scripts/27_gen_suite.sh $m $R/suite > $R/suite.log 2>&1
  echo "$a suite $? $(date +%T)"
done
echo BASELINES_DONE
