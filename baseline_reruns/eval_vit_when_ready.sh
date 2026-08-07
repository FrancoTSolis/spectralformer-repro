#!/usr/bin/env bash
set -u

ROOT="./"
HERE="$ROOT/baseline-reruns"

declare -A GPU=( [cylinder]=0 [multiphase]=2 [we1]=4 [e1]=6 )
declare -A TRAIN_LOG=(
  [cylinder]="$ROOT/train_baseline_cylinder_temporal.log"
  [multiphase]="$ROOT/train_baseline_multiphase_temporal.log"
  [we1]="$ROOT/train_baseline_we1_temporal.log"
  [e1]="$ROOT/train_baseline_e1_temporal.log"
)

while true; do
  pending=0
  for dataset in cylinder multiphase we1 e1; do
    result="$HERE/results/vit-sea_${dataset}.json"
    eval_log="$HERE/logs/vit_sea_${dataset}_eval.log"
    if [ -f "$result" ]; then
      continue
    fi
    pending=1
    if pgrep -f "eval_vit_sea.py --dataset ${dataset}" >/dev/null 2>&1; then
      continue
    fi
    if [ -f "${TRAIN_LOG[$dataset]}" ] && \
       rg -q "Total training time:" "${TRAIN_LOG[$dataset]}"; then
      echo "$(date): launching ViT-SEA ${dataset} evaluation on GPU ${GPU[$dataset]}"
      cd "$HERE"
      nohup python3 -u eval_vit_sea.py \
        --dataset "$dataset" --gpu "${GPU[$dataset]}" > "$eval_log" 2>&1 &
    fi
  done

  if [ "$pending" -eq 0 ]; then
    echo "$(date): all ViT-SEA evaluations complete"
    exit 0
  fi
  sleep 30
done
