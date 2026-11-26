#!/usr/bin/env bash
# Run on node 0: full-parameter heal of E across both Sparks (FSDP2, NCCL over RoCE on the
# QSFP link). Rank 0 = this node (NODE0_ADDR), rank 1 = the second Spark (NODE1_ADDR).
# Needs the same code, venv and teacher targets on both nodes.
# usage: heal_full_launch.sh LR MAX_STEPS SAVE OUT_DIR LOG_JSON
set -uo pipefail
LR=$1; MAXS=$2; SAVE=$3; OUT=$4; LOG=$5
STU=~/models/mpb/r96
NCCL="NCCL_NET=IB NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 NCCL_IB_GID_INDEX=3 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_SOCKET_IFNAME=enp1s0f0np0 GLOO_SOCKET_IFNAME=enp1s0f0np0 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0"
ARGS="--nnodes 2 --nproc-per-node 1 --master-addr ${NODE0_ADDR:?set NODE0_ADDR} --master-port 29655"
CMD="cd ~/mpb && env $NCCL LR=$LR MAX_STEPS=$MAXS SAVE=$SAVE MICRO=4 systemd-run --user --scope -q -p MemoryMax=112G -- .venv-heal/bin/torchrun $ARGS"
TAIL="scripts/16_heal_full.py $STU heal/teacher heal/heal.jsonl $OUT $LOG"
ssh -n ${NODE1_ADDR:?set NODE1_ADDR} "$CMD --node-rank 1 $TAIL > ~/mpb/heal/runs/full_rank1.log 2>&1" &
r1=$!
bash -c "$CMD --node-rank 0 $TAIL"
rc=$?
wait $r1
echo "full heal rank0 exit $rc rank1 exit $?"
