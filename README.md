# SpectralFormer: Geometry-Adaptive Spectral Mixing for Long-Horizon Mesh Graph Dynamics

> Submitted to LoG 2026 (anonymous review)

This repository contains the code to reproduce all experiments in the paper.

## Repository Structure

```
spectralformer-repro/
├── proposed/              # SpectralFormer (proposed method)
│   ├── main.py            # Entry point for spatial encoder-decoder and temporal model
│   ├── models/            # Model definitions (base blocks, encoder-decoder, temporal, transforms)
│   ├── configs/            # Per-dataset training configurations
│   ├── train/             # Training loops (encoder, temporal)
│   └── utils/             # Data loading & training utilities
├── baseline_sea/          # ViT-SEA baseline (ablation without spectral mixing)
│   ├── main.py
│   ├── models/
│   ├── configs/
│   ├── train/
│   └── utils/
├── baseline_reruns/       # MGN, GMR, PbGMR baselines
│   ├── train_mgn.py       # MeshGraphNet training
│   ├── train_mesh_reduced.py  # GMR / PbGMR training
│   ├── mesh_reduced_models.py
│   ├── eval_vit_sea.py    # Evaluate ViT-SEA checkpoints
│   ├── common.py          # Shared helpers
│   ├── prepare.py         # Data preparation
│   ├── aggregate.py       # Result aggregation
│   ├── cache/             # Trajectory split definitions (JSON)
│   └── upstream/          # PbGMR RealNVP upstream code
├── spectral_search/       # Experiment E1: FFT/FNO spectral specialization
│   └── train_spectral_recurrence.py
├── evaluation/            # Manifest-driven evaluation pipeline
│   ├── evaluate.py        # Main evaluation driver
│   ├── evaluator_worker.py
│   ├── metrics.py         # RMSE, correlation, rollout metrics
│   ├── manifest_lib.py    # Manifest parsing
│   └── run_manifest.json  # Evaluation manifest template
├── data_conversion/       # Data preprocessing scripts
│   ├── convert_we1_to_sea.py   # Convert WE-1 data to SEA format
│   └── convert_e1_to_sea.py    # Convert E-1 data to SEA format
├── eval_per_trajectory.py # Per-trajectory evaluation script
├── eval_all_datasets.py   # Evaluate across all datasets
├── requirements.txt
└── README.md
```

## Environment Setup

**Python 3.10+** is required. We recommend creating a conda environment:

```bash
conda create -n spectralformer python=3.10 -y
conda activate spectralformer
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| PyTorch | 2.x | CUDA 11.8+ recommended |
| PyG (torch-geometric) | 2.5+ | With torch-scatter, torch-sparse |
| PhysicsNeMo | latest | NVIDIA physics-informed ML toolkit |
| einops | 0.7+ | Tensor rearrangement |
| wandb | latest | Experiment tracking (optional) |

Install PyTorch Geometric after PyTorch:

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.x.x+cuXXX.html
```

## Data Preparation

### MP-PDE Benchmark Data

Download the MP-PDE benchmark datasets:

1. **Cylinder Flow (CF)** — 2D incompressible flow past a cylinder
2. **Wave Equation (WE-1)** — 1D wave equation on irregular mesh
3. **Advection Equation (E-1)** — 1D advection equation

Place raw data under `./data/` and convert to the SEA mesh format:

```bash
python data_conversion/convert_we1_to_sea.py --input data/WE-1/ --output data/we1_sea/
python data_conversion/convert_e1_to_sea.py --input data/E-1/ --output data/e1_sea/
```

## Training

### SpectralFormer (Proposed)

Training proceeds in two stages: (1) spatial encoder-decoder, then (2) temporal model.

**Cylinder Flow:**
```bash
cd proposed
python main.py --config configs/cylinder_flow.py --stage encoder
python main.py --config configs/cylinder_flow.py --stage temporal
```

**Wave Equation:**
```bash
cd proposed
python main.py --config configs/wave_equation.py --stage encoder
python main.py --config configs/wave_equation.py --stage temporal
```

**Advection Equation:**
```bash
cd proposed
python main.py --config configs/advection_equation.py --stage encoder
python main.py --config configs/advection_equation.py --stage temporal
```

### ViT-SEA Baseline

```bash
cd baseline_sea
python main.py --config configs/cylinder_flow.py --stage encoder
python main.py --config configs/cylinder_flow.py --stage temporal
```

### MGN / GMR / PbGMR Baselines

```bash
cd baseline_reruns

# MeshGraphNet
python train_mgn.py --dataset cylinder

# GMR (Graph Mesh Reduced)
python train_mesh_reduced.py --dataset cylinder --model gmr

# PbGMR (Physics-based GMR)
python train_mesh_reduced.py --dataset cylinder --model pbgmr
```

### Spectral Recurrence Ablation (E1)

```bash
cd spectral_search
python train_spectral_recurrence.py
```

## Evaluation

### Per-Dataset Evaluation

```bash
python eval_all_datasets.py
```

### Per-Trajectory Evaluation

```bash
python eval_per_trajectory.py --dataset cylinder --checkpoint_dir <path_to_checkpoints>
```

### Manifest-Driven Evaluation

The evaluation pipeline supports batch evaluation via manifests:

```bash
cd evaluation
python evaluate.py --manifest run_manifest.json
```

## Notes

- **Checkpoints** are not included in this repository due to size. Train from scratch or contact the authors after the review period.
- **Data** must be obtained separately from the MP-PDE benchmark sources.
- W&B logging is optional. Set `WANDB_API_KEY` in config files or disable wandb in the config to train offline.
- All configs use relative paths by default. Adjust `data_dir` and `checkpoint_dir` in the config files to match your setup.

## License

See individual `LICENSE` files in subdirectories for upstream code licenses.
