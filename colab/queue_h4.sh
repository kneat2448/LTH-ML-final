#!/bin/bash
# Run H4 fixed-mask jobs in one GPU slot once a seed-1 chain has exited, under the same MPS setup.
# Usage: nohup bash colab/queue_h4.sh <cond-to-wait-for> "<args>" ["<args>" ...] >> /content/h4_queue.out 2>&1 &
#   each "<args>" is passed to `python -m src.fixed_mask`, run in order (finished trainings are skipped).
# Example: bash colab/queue_h4.sh low "configs/h4_high_anchor.yaml --select-lam" configs/h4_high_anchor.yaml
cd /content/drive/MyDrive/final_project
export LTH_RAW_DATA=/content/cifar_raw CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
WAIT=$1; shift
while pgrep -f "src.imp configs/resnet20_$WAIT.yaml --seeds 1" >/dev/null; do sleep 60; done
echo "$(date -u +%T) $WAIT seed 1 finished"
for job in "$@"; do
  echo "$(date -u +%T) start fixed_mask $job"
  python -m src.fixed_mask $job || echo "$(date -u +%T) FAILED: $job"
done
echo "$(date -u +%T) queue done"
