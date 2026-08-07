#!/bin/bash
set -e
cd ./SEA
export PYTHONUNBUFFERED=1

echo "=== [$(date)] Waiting for WE1 spatial encoder training to finish ==="
while pgrep -f "main.py wave_equation encoder train" > /dev/null 2>&1; do
    sleep 30
done

echo "=== [$(date)] Spatial training complete. Checking checkpoint ==="
ls -la checkpoints-wave-equation/encoder_decoder_wave_equation_run1.pt

echo "=== [$(date)] Starting temporal model training ==="
python3 -u main.py wave_equation temporal train 2>&1 | tee checkpoints-wave-equation/temporal_train.log

echo "=== [$(date)] Temporal training complete ==="
ls -la checkpoints-wave-equation/temporal_*

echo "=== [$(date)] All done! ==="
