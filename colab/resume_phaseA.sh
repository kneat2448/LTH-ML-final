#!/bin/bash
# Resume Phase A after a Colab VM restart: CIFAR copy, MPS daemon, the three parallel chains, the watcher.
# Finished trainings are skipped (src.imp resumes). Usage: bash colab/resume_phaseA.sh [conditions...]
# Default conditions: low high warm03. Only relaunches a chain that is not already running.
cd /content/drive/MyDrive/final_project
CONDS=${@:-low high warm03}
mkdir -p /content/cifar_raw
[ -f /content/cifar_raw/cifar-10-python.tar.gz ] || cp data/cifar-10-python.tar.gz /content/cifar_raw/
export LTH_RAW_DATA=/content/cifar_raw CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
for c in $CONDS; do
  if pgrep -f "src.imp configs/resnet20_$c.yaml" >/dev/null; then echo "$c already running"; continue; fi
  nohup python -m src.imp configs/resnet20_$c.yaml >> /content/phaseA_$c.out 2>&1 &
  echo "started $c"
done
pgrep -f "colab/watcher.sh" >/dev/null || { nohup bash colab/watcher.sh >> /content/watcher.out 2>&1 & echo "started watcher"; }
