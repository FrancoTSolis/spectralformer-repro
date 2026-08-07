#!/bin/bash
set -e
cd ./SEA-baseline
export PYTHONUNBUFFERED=1

echo "=== [$(date)] Waiting for E1 spatial encoder to finish ==="
while pgrep -f "main.py advection_equation encoder train" > /dev/null 2>&1; do
    sleep 30
done

echo "=== [$(date)] Spatial done. Starting temporal ==="
python3 -u main.py advection_equation temporal train 2>&1 | tee checkpoints-advection-equation/temporal_train.log
echo "=== [$(date)] Done! ==="
