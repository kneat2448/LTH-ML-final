#!/bin/bash
# Start or resume the Phase B seed-1 chains under MPS (same setup as resume_phaseA.sh), plus the watcher.
# Finished trainings are skipped (src.imp resumes). Usage: bash colab/resume_phaseB.sh [conditions...]
# Default conditions: low high warm03. Only launches a chain whose seed-1 process is not already running.
# Optional: WAIT_FOR=<cond> waits until that condition's Phase A (seed-0) chain has exited before starting.
cd /content/drive/MyDrive/final_project
CONDS=${@:-low high warm03}
mkdir -p /content/cifar_raw
[ -f /content/cifar_raw/cifar-10-python.tar.gz ] || cp data/cifar-10-python.tar.gz /content/cifar_raw/
# extract once here: chains started together would each extract it and race
[ -d /content/cifar_raw/cifar-10-batches-py ] || tar -xzf /content/cifar_raw/cifar-10-python.tar.gz -C /content/cifar_raw
export LTH_RAW_DATA=/content/cifar_raw CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
if [ -n "$WAIT_FOR" ]; then
  while pgrep -f "src.imp configs/resnet20_$WAIT_FOR.yaml$" >/dev/null; do sleep 60; done
  echo "$(date -u +%T) $WAIT_FOR (seed 0) finished"
fi
for c in $CONDS; do
  if pgrep -f "src.imp configs/resnet20_$c.yaml --seeds 1" >/dev/null; then echo "$c seed 1 already running"; continue; fi
  nohup python -m src.imp configs/resnet20_$c.yaml --seeds 1 >> /content/phaseB_${c}_s1.out 2>&1 &
  echo "$(date -u +%T) started $c seed 1"
done
pgrep -f "colab/watcher.sh" >/dev/null || { nohup bash colab/watcher.sh >> /content/watcher.out 2>&1 & echo "started watcher"; }
