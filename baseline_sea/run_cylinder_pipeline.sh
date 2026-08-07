#!/bin/bash
set -e
cd ./SEA-baseline
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6

echo "=== [$(date)] Waiting for spatial encoder training to finish ==="
while pgrep -f "main.py cylinder_flow_clean encoder train" > /dev/null 2>&1; do
    sleep 30
done

echo "=== [$(date)] Spatial training complete. Checking checkpoint ==="
ls -la checkpoints-cylinder/encoder_decoder_cylinder_flow_run1.pt

echo "=== [$(date)] Starting temporal model training ==="
python3 -u main.py cylinder_flow_clean temporal train 2>&1 | tee checkpoints-cylinder/temporal_train.log

echo "=== [$(date)] Temporal training complete ==="
ls -la checkpoints-cylinder/temporal_*

echo "=== [$(date)] All done! ==="
