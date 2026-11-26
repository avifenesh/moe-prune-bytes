#!/usr/bin/env bash
# Run on a Spark: LoRA heal of E (teacher targets from 14_teacher_hidden.py), then NVFP4 + eval.
# LR 1e-4: about 10x the full-parameter LR that healed (1e-5), per LoRA-without-regret
# guidance; progress is watched through the 16-sequence validation every 25 steps.
set -uo pipefail
cd ~/mpb
S=~/models/Qwen3-30B-A3B-Instruct-2507
STU=~/models/mpb/r96
P=.venv-heal/bin/python
mkdir -p heal/teacher heal/runs
[ "$(ls heal/teacher | wc -l)" -ge 24 ] || bash scripts/dock.sh mpb-teacher python3 scripts/14_teacher_hidden.py $S heal/heal.jsonl heal/teacher 128 > heal/runs/teacher.log 2>&1
echo "teacher $? $(date +%T)"
[ -f ~/models/mpb/E-heal-lora/config.json ] || LR=1e-4 systemd-run --user --scope -q -p MemoryMax=110G -- $P scripts/15_heal.py $STU heal/teacher heal/heal.jsonl ~/models/mpb/E-heal-lora heal/runs/E_lora.json > heal/runs/E_lora.log 2>&1
echo "E lora $? $(date +%T)"
[ -f ~/models/mpb/E-heal-lora/config.json ] && bash scripts/heal_eval.sh EL ~/models/mpb/E-heal-lora
echo LORA_DONE
