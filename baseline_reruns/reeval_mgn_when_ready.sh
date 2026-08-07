#!/usr/bin/env bash
set -u

HERE="./baseline-reruns"

run_pair() {
  local gpu="$1"
  local variant="$2"
  local first="$3"
  local second="$4"
  local marker="$HERE/results/.${variant}_${first}_${second}_canonical_eval_done"

  while [ ! -f "$marker" ]; do
    if [ -f "$HERE/results/${variant}_${first}.json" ] && \
       [ -f "$HERE/results/${variant}_${second}.json" ] && \
       ! pgrep -f "train_mgn.py.*--gpu ${gpu}" >/dev/null 2>&1; then
      cd "$HERE"
      python3 -u train_mgn.py --dataset "$first" --variant "$variant" \
        --gpu "$gpu" --eval-only > "logs/${variant}_${first}_canonical_eval.log" 2>&1
      python3 -u train_mgn.py --dataset "$second" --variant "$variant" \
        --gpu "$gpu" --eval-only > "logs/${variant}_${second}_canonical_eval.log" 2>&1
      touch "$marker"
      echo "$(date): canonical re-evaluation complete for $variant $first $second"
      return
    fi
    sleep 30
  done
}

run_pair 1 mgn cylinder multiphase &
run_pair 3 mgn-ni cylinder multiphase &
run_pair 5 mgn we1 e1 &
run_pair 7 mgn-ni we1 e1 &
wait
