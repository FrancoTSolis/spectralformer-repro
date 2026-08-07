#!/bin/bash
set -e
cd ./SEA-baseline
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

echo "=== [$(date)] Waiting for spatial encoder training to finish ==="
# Wait for the spatial training process to complete
while pgrep -f "main.py sonic_flow encoder train" > /dev/null 2>&1; do
    sleep 30
done

echo "=== [$(date)] Spatial training complete. Checking checkpoint ==="
ls -la checkpoints-sonic/encoder_decoder_sonic_flow_run1.pt

echo "=== [$(date)] Starting temporal model training ==="
python3 -u main.py sonic_flow temporal train 2>&1 | tee checkpoints-sonic/temporal_train.log

echo "=== [$(date)] Temporal training complete ==="
ls -la checkpoints-sonic/temporal_*

echo "=== [$(date)] Running test evaluation ==="
python3 -u eval_sonic_test.py 2>&1 | tee checkpoints-sonic/eval_test.log

echo "=== [$(date)] All done! ==="
