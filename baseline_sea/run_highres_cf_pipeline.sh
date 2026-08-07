#!/bin/bash
set -e
cd ./SEA-baseline
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "=== [$(date)] Waiting for spatial encoder training to finish ==="
while pgrep -f "main.py cylinder_flow_highres encoder train" > /dev/null 2>&1; do
    sleep 60
done

echo "=== [$(date)] Spatial training complete. Checking checkpoint ==="
ls -la checkpoints-cylinder-highres/encoder_decoder_cylinder_flow_highres_run1.pt

echo "=== [$(date)] Starting temporal model training ==="
python3 -u main.py cylinder_flow_highres temporal train 2>&1 | tee checkpoints-cylinder-highres/temporal_train.log

echo "=== [$(date)] Temporal training complete ==="
echo "=== [$(date)] All done! ==="
