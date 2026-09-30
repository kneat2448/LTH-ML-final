#!/bin/bash
# Start chain $2 once chain $1 has finished (its src.imp process exited), under the same MPS setup.
# Usage: nohup bash colab/start_after.sh low high_warm > /content/start_after.out 2>&1 &
cd /content/drive/MyDrive/final_project
while pgrep -f "src.imp configs/resnet20_$1.yaml$" >/dev/null; do sleep 60; done
echo "$(date -u +%T) $1 finished; starting $2"
bash colab/resume_phaseA.sh "$2"
