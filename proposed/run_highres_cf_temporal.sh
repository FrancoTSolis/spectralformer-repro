#!/bin/bash
set -e
cd ./SEA
export PYTHONUNBUFFERED=1

echo "=== [$(date)] Waiting for highres CF spatial encoder training to finish ==="
while pgrep -f "main.py cylinder_flow_highres encoder train" > /dev/null 2>&1; do
    sleep 30
done

echo "=== [$(date)] Spatial training complete. Checking checkpoint ==="
ls -la checkpoints-cylinder-flow-highres/encoder_decoder_cylinder_flow_highres_run1.pt

echo "=== [$(date)] Starting temporal model training ==="
python3 -u main.py cylinder_flow_highres temporal train 2>&1 | tee checkpoints-cylinder-flow-highres/temporal_train.log

echo "=== [$(date)] Temporal training complete ==="
echo "=== [$(date)] All done! ==="
