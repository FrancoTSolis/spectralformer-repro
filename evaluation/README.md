# Checkpoint-only evidence pipeline

This directory implements phases 1–2 of the accepted submission plan. It is an
isolated paper artifact: it reads the existing `SEA`/`SEA-baseline` repositories,
datasets, W&B logs, and checkpoints, but never trains, calls `backward`, steps an
optimizer, writes a checkpoint, initializes W&B, or writes into a historical
model/data directory.

## Files

- `run_manifest.json` — canonical evidence registry. It pins full revisions,
  local W&B provenance, data shapes/hashes, exact seed-42 trajectory indices,
  horizons, checkpoint hashes, historical metrics, limitations, and paper use.
- `evaluate.py` — orchestration entry point. It starts a fresh worker process
  and, by default, exports the exact recorded Git revision to a temporary source
  snapshot. The existing repository checkout is not changed.
- `evaluator_worker.py` — inference-only worker. It imports exactly one source
  tree in its process, applies mandatory safety overrides, evaluates every test
  trajectory and forecast step, and writes JSON/CSV.
- `metrics.py` — metric and trajectory-bootstrap implementation.
- `aggregate_historical.py` — fallback exporter. Its output is explicitly
  labeled historical log aggregation, never a new evaluation.
- `tests/` — deterministic metric, manifest, and no-training smoke tests.
- `results/` — append-only evaluation/fallback outputs.

## Safety and reproducibility contract

1. There is no training mode.
2. W&B is disabled and authentication-bearing config keys are removed in memory.
3. The worker forces seed 42, trajectory splits, full rollout lengths, no time
   shifting, no plotting, no checkpoint saves, and an output-local `save_dir`.
4. The orchestrator fingerprints every protected input before and after a run.
5. An output directory must not already exist. Re-running with the same explicit
   path fails rather than overwriting it.
6. Exact code revisions are materialized with `git archive`. Historical
   `configs/` files are deliberately omitted because they contain authentication
   data; their safe runtime values are pinned in `run_manifest.json`. Use
   `--source-mode working-tree` only when tracked files are clean and `HEAD`
   equals the manifest revision.
7. Training results are single-seed. Bootstrap intervals quantify test
   trajectory uncertainty; they are not training-seed uncertainty.

## Environment and preflight

Use a Python environment compatible with the original repositories (PyTorch,
NumPy, SciPy, scikit-learn, PyTorch Geometric, and any partition dependency
required by that revision). The orchestrator itself has only standard-library
dependencies; select a repository-compatible worker interpreter with
`--python`.

Before a GPU evaluation:

```bash
nvidia-smi
ps -eo pid,user,cmd | rg 'python|torchrun|accelerate|deepspeed'
```

List and validate manifest entries without loading a model:

```bash
cd ./log-specatraformer-submission/experiments
python evaluate.py --list-runs
python evaluate.py --run-id multiphase_graphspectralformer_canonical --dry-run
```

Add `--verify-hashes` to recompute all pinned hashes before inference. This is
slow for the multi-gigabyte field arrays and checkpoints.

## Canonical evaluation commands

The surviving repository-compatible interpreter is
`python` (Python 3.12.8,
PyTorch 2.4.0+cu121, NumPy 1.26.4). Each command creates a timestamped
append-only directory under `results/`. Set `CUDA_VISIBLE_DEVICES` to an
actually free physical GPU; the worker then addresses it as `cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=3 python evaluate.py \
  --run-id cylinder_sea_baseline_canonical \
  --python python \
  --device cuda:0 --batch-size 1 --bootstrap-samples 2000

CUDA_VISIBLE_DEVICES=3 python evaluate.py \
  --run-id multiphase_sea_baseline_canonical \
  --python python \
  --device cuda:0 --batch-size 1 --bootstrap-samples 2000

CUDA_VISIBLE_DEVICES=3 python evaluate.py \
  --run-id multiphase_graphspectralformer_canonical \
  --python python \
  --device cuda:0 --batch-size 1 --bootstrap-samples 2000
```

Cylinder conditioning is reconstructed in memory from the exact historical
source recipe in `train/train_temporal.py:39–47` at revisions `00bb57f...` and
`9633fc6...`: 101 Reynolds numbers linearly spaced from 300 to 1000, their
inverses, powers 1–3, and per-column maximum normalization in alternating
`Re^k, nu^k` order. The archival `101×1` file remains fingerprinted as a
provenance mismatch but is not read for runtime values and is never modified.
The canonical baseline is ready. The proposed run is intentionally blocked by
the manifest because its exact patch membership/order and `U_patch` were not
saved; see the final blocker under `results/`.

## Output contract

A successful run writes:

- `results.json` — provenance, split, aggregate/per-field metrics, final-step
  metrics, every horizon, every trajectory, trajectory-bootstrap 95% intervals,
  latent diagnostics, parameters, inference/decode/preprocessing wall time, and
  peak GPU memory.
- `summary.csv` — aggregate and final-step per-field/macro metrics.
- `horizon_metrics.csv` — per-field/macro error curves with intervals.
- `trajectory_metrics.csv` — every held-out trajectory.
- `model_profile.csv` — spatial/temporal/combined parameters, timing, and memory.
- `execution_plan.json`, `execution.json`, `evaluation.log`, `run_spec.json`.

If evaluation fails, the directory contains `blocker.json` and
`evaluation.log`; no `results.json` is fabricated.

Metric aggregation is fixed:

- relative squared L2: spatial squared-error sum divided by spatial truth-energy
  sum, computed per trajectory/step/field;
- relative L2: square root before aggregation;
- raw RMSE: spatial node RMSE in original field units;
- aggregate: mean over every test trajectory and every predicted step;
- final-step: same metrics at the last predicted step;
- macro: arithmetic mean over fields;
- uncertainty: percentile 95% interval from whole-trajectory bootstrap samples.

## Historical fallback

When a dependency, source/config, or data mismatch prevents a new evaluation:

```bash
python aggregate_historical.py
```

To export only selected runs:

```bash
python aggregate_historical.py \
  --run-id cylinder_sea_baseline_canonical \
  --run-id cylinder_graphspectralformer_canonical \
  --run-id multiphase_fallback_metis_legacy \
  --run-id multiphase_fallback_spectral_legacy
```

The resulting JSON/CSV says
`historical_log_aggregation_not_new_evaluation`. Preserve each row’s original
validation/test, minibatch, horizon, and single-seed limitations.

## Evidence policy

- Use the source-conditioned standardized cylinder baseline for its paper row.
  The proposed method remains historical-only until its exact spatial basis is
  recovered; diagnostic reruns and fabricated confidence intervals are barred.
  The older slide per-field rows are minibatches, not aggregate test metrics.
- Canonical newer multiphase validation values are not test results. Use the
  successful standardized outputs in the two `sanitized_v2` result directories,
  not the historical validation scalars.
- The two older multiphase test runs are fallback evidence at **199 predicted
  steps**, never 400.
- Cylinder ablations with validation-only/incomplete provenance are omitted
  unless standardized checkpoint evaluation succeeds.
- Sonic is a negative result only: 39 predicted steps, validation, final step of
  one retained minibatch, and eight fields. It is not positive or test evidence.
- Never relabel relative squared L2 as RMSE. Never turn trajectory-bootstrap
  intervals into multi-seed error bars.

## Smoke tests

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
python evaluate.py --list-runs
python evaluate.py --run-id multiphase_sea_baseline_canonical --dry-run
```

