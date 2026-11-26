#!/usr/bin/env bash
# Run on a Spark: arm SDa, teacher-generated self-distillation data. The BF16 source answers
# the 4,096 mixed-domain prompts (2 samples each); 3,008 packed sequences + the shared 64 heal
# validation sequences; source hidden states; then the QAD heal arm (heal_arm_A2.sh).
# usage: sd_a.sh
set -uo pipefail
cd ~/mpb
S=~/models/Qwen3-30B-A3B-Instruct-2507
D=scripts/dock.sh
mkdir -p phaseA/sda
[ -f phaseA/sda/gen.jsonl ] || bash $D mpbA-gen-sda python3 scripts/30_sd_gen.py $S phaseA/sd_prompts.jsonl phaseA/sda/gen.jsonl 3008 2 0 > runs/sda_gen.log 2>&1
echo "sda gen $? $(date +%T) $(wc -l < phaseA/sda/gen.jsonl)"
tail -n 64 phaseA/heal_mix.jsonl > phaseA/sda/val64.jsonl
cat phaseA/sda/gen.jsonl phaseA/sda/val64.jsonl > phaseA/sda/train.jsonl
mkdir -p phaseA/sda/teacher
[ "$(ls phaseA/sda/teacher | wc -l)" -ge 24 ] || bash $D mpbA-teach-sda python3 scripts/14_teacher_hidden.py $S phaseA/sda/train.jsonl phaseA/sda/teacher 128 > runs/sda_teacher.log 2>&1
echo "sda teacher $? $(date +%T)"
bash scripts/heal_arm_A2.sh SDa phaseA/sda/train.jsonl phaseA/sda/teacher
rm -rf phaseA/sda/teacher
echo SDA_DONE
