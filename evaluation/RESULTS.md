# Phase 1–2 execution record

## Completed standardized evaluation

Both canonical multiphase checkpoints were evaluated at their recorded Git
revisions on the fixed seed-42 held-out trajectory indices
`[10, 22, 18, 20, 7, 14, 28, 38]`. Every trajectory and all 199 predicted steps
were included. These are single-training-seed models; intervals resample test
trajectories.

Preferred result directories:

- Baseline:
  `results/multiphase_sea_baseline_canonical__standard_eval_sanitized_v2/`
- Proposed:
  `results/multiphase_graphspectralformer_canonical__standard_eval_sanitized_v2/`

Macro aggregate results:

- SEA baseline:
  - relative squared L2 `1.2308506457` (95% CI `0.7399453052–2.0178626423`)
  - relative L2 `0.8268581849` (95% CI `0.6698105380–1.0539135617`)
  - raw RMSE `0.1610314865` (95% CI `0.1449088726–0.1799982972`)
  - final-step relative squared L2 `2.3377258004`
- GraphSpectralFormer:
  - relative squared L2 `0.8175139794` (95% CI `0.6541920469–1.0445962344`)
  - relative L2 `0.7884077655` (95% CI `0.7039454286–0.8883872674`)
  - raw RMSE `0.1938166942` (95% CI `0.1724667617–0.2163688930`)
  - final-step relative squared L2 `0.9584998644`

Interpretation must be metric-specific. The proposed checkpoint is lower on
macro relative squared L2 and final-step relative squared L2, but higher on raw
RMSE. Its pressure relative squared L2 (`0.2497386660`) is also worse than the
baseline (`0.1146842650`). The trajectory-bootstrap intervals overlap.

Resource measurements:

- Baseline: `202,774,068` combined parameters, `22.98 s` autoregressive model
  wall time, `44.13 s` preprocessing, `3,609,381,888` bytes peak GPU memory.
- Proposed: `221,991,939` combined parameters, `26.37 s` autoregressive model
  wall time, `194.25 s` preprocessing, `3,795,555,328` bytes peak GPU memory.

The exact per-field, final-step, per-horizon, per-trajectory, latent, timing,
parameter, and memory records are in each directory’s JSON/CSV files.

## Cylinder follow-up

The historical six-feature conditioning is now reconstructed exactly in worker
memory from `train/train_temporal.py:39–47` at the recorded revisions. The
resulting float32 tensor is `101×6` with SHA-256
`36a1c08c42b899f47112f7fd1c68bfcd7272779e1f2c5e19a5992cccdec84556`.
The archival `101×1` file is fingerprinted but not used for runtime values.

The canonical SEA baseline completed over all 20 held-out trajectories and 400
steps:

- Output:
  `results/cylinder_sea_baseline_canonical__standard_eval_source_conditioning_v3/`
- Aggregate relative squared L2:
  - `u`: `0.0022925666` (95% CI `0.0007044787–0.0045825389`)
  - `v`: `0.0298694156` (95% CI `0.0090150148–0.0598558148`)
  - `p`: `0.0060593515` (95% CI `0.0016586587–0.0124657861`)
  - macro: `0.0127404446` (95% CI `0.0038222351–0.0256073058`)
- Aggregate relative L2:
  - `u`: `0.0326761439` (95% CI `0.0214212809–0.0469940744`)
  - `v`: `0.1157857617` (95% CI `0.0738832547–0.1687421228`)
  - `p`: `0.0487188941` (95% CI `0.0294272832–0.0729072152`)
  - macro: `0.0657269332` (95% CI `0.0414609155–0.0960416333`)
- Aggregate raw RMSE:
  - `u`: `0.0327095247` (95% CI `0.0212661947–0.0473689775`)
  - `v`: `0.0436849838` (95% CI `0.0267443736–0.0649730538`)
  - `p`: `0.0187559562` (95% CI `0.0109994693–0.0285658019`)
  - macro: `0.0317168216` (95% CI `0.0196455396–0.0469517593`)
- Final-step macro relative squared L2: `0.0126584257`
  (95% CI `0.0037611209–0.0248124377`).
- Resources: `87,850,025` combined parameters; `20.93 s` preprocessing;
  `87.26 s` autoregressive model time; `96.16 s` total evaluation wall time;
  `2,748,507,648` bytes peak allocated GPU memory.

The proposed conditioning also reconstructs exactly, but its spatial checkpoint
does not persist patch membership/order or `U_patch`. The recorded commit lacks
the uncommitted processor state used by the W&B run. Multiple surviving
processor reconstructions disagree at horizon 1 by two to three orders of
magnitude, so none is paper-usable. The final blocker is:

- `results/cylinder_graphspectralformer_canonical__standard_eval_final_blocker/blocker.json`

The only defensible proposed scalar remains the historical macro relative
squared L2 `0.0048621331`, with legacy minibatch aggregation and no
trajectory-bootstrap CI. No CI was fabricated.

Curve assessment:

- The standardized baseline export has 4,800 finite rows, all horizons 1–400,
  all three metrics, per-field/macro rows, and 95% trajectory-bootstrap bands;
  it is suitable as plot input.
- The historical proposed CSV has 400 finite steps but represents only the last
  test minibatch and has no confidence bands. It is not suitable for a
  main-paper comparative curve.
- Diagnostic proposed rerun curves are invalid because of spatial-basis
  incompatibility. See `results/cylinder_curve_export_audit.json`.

The historical fallback export remains:

- `results/historical_evidence__phase2_fallback/historical_evidence.json`
- `results/historical_evidence__phase2_fallback/historical_evidence.csv`

## Other recorded blockers

- `results/multiphase_sea_baseline_canonical__standard_eval/` is a completed
  pre-policy run whose metrics match `sanitized_v2`; only `sanitized_v2` is
  paper-usable.
- `results/multiphase_graphspectralformer_canonical__standard_eval/` completed
  after its parent process was interrupted, but it is superseded because its
  temporary source snapshot predated archive-time config omission. Its metrics
  match `sanitized_v2`; only `sanitized_v2` is paper-usable.
- `results/multiphase_graphspectralformer_canonical__standard_eval_sanitized/`
  records a missing safe-profile `batch_size`; the profile was fixed and the
  successful `sanitized_v2` result supersedes it.

No metric from either blocked directory is treated as a result.

## Commands executed

Environment/process checks:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
pgrep -af 'python|torchrun|accelerate|deepspeed'
conda env list
python -c 'import torch,numpy,scipy,sklearn,torch_geometric'
```

Validation and fallback:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
python evaluate.py --list-runs
python evaluate.py --run-id multiphase_sea_baseline_canonical --dry-run
python aggregate_historical.py \
  --output-dir results/historical_evidence__phase2_fallback
```

Final test-only evaluations:

```bash
CUDA_VISIBLE_DEVICES=3 python evaluate.py \
  --run-id multiphase_sea_baseline_canonical \
  --python python \
  --device cuda:0 --batch-size 1 --bootstrap-samples 2000 \
  --output-dir results/multiphase_sea_baseline_canonical__standard_eval_sanitized_v2

CUDA_VISIBLE_DEVICES=6 python evaluate.py \
  --run-id multiphase_graphspectralformer_canonical \
  --python python \
  --device cuda:0 --batch-size 1 --bootstrap-samples 2000 \
  --output-dir results/multiphase_graphspectralformer_canonical__standard_eval_sanitized_v2

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
  --run-id cylinder_sea_baseline_canonical \
  --python python \
  --device cuda:0 --batch-size 1 --bootstrap-samples 2000 \
  --output-dir results/cylinder_sea_baseline_canonical__standard_eval_source_conditioning_v3
```

The proposed cylinder command was run with append-only `v3`, `v5`, `v6`, and
`v7` compatibility profiles while auditing the missing spatial basis. All are
recorded as diagnostics in the final blocker and excluded from paper evidence.

Artifact hashes were computed with `sha256sum` and pinned in
`run_manifest.json`. No existing SEA, SEA-baseline, spectraformer, checkpoint,
or data file was modified.

