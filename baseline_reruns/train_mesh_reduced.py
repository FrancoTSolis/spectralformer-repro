"""Train and evaluate directly comparable mesh-reduced baselines.

This harness shares ``common.py``'s seed-42 trajectory splits, horizons,
conditioning, graph construction, and train-only field statistics.  It trains
the spatial autoencoder and temporal model in separate stages, selects both
checkpoints on validation data, and evaluates decoded autoregressive rollouts
with the common per-field RelMSE aggregation.

The two variants are:

* ``gmr-gmus``: Han et al. (ICLR 2022) GMR/GMUS with a one-layer,
  four-head residual temporal Transformer.
* ``pbgmr-gmus``: Sun et al. (NeurIPS 2023) position-based,
  residual/layer-normalized PbGMR/PbGMUS with a conditional RealNVP head.

NVIDIA PhysicsNeMo supplies the public PbGMR/GMUS spatial component.  The
authors' pinned temporal/flow release supplies the conditional RealNVP; this
harness preserves its coupling and invertible-batch-normalization semantics
while removing hard-coded devices and unrelated package dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from common import (
    DATASETS,
    DatasetSpec,
    edge_features as make_edge_features,
    field_and_delta_stats,
    load_conditioning,
    load_coordinates,
    load_edges,
    trajectory_split,
)
from mesh_reduced_models import (
    GMRResidualTemporalTransformer,
    MeshReducedAutoencoder,
    PbGMRRealNVPTemporal,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHYSICSNEMO_ROOT = ROOT / "physicsnemo"
PHYSICSNEMO_COMMIT = "77b3c68001159b948a16804fa76eb127735fb6d1"
PHYSICSNEMO_SPATIAL_SOURCE = (
    PHYSICSNEMO_ROOT / "physicsnemo/models/mesh_reduced/mesh_reduced.py"
)
PHYSICSNEMO_TEMPORAL_SOURCE = (
    PHYSICSNEMO_ROOT / "physicsnemo/models/mesh_reduced/temporal_model.py"
)
OFFICIAL_CYLINDER_PIVOTS = (
    PHYSICSNEMO_ROOT
    / "examples/cfd/vortex_shedding_mesh_reduced/dataset/meshPosition_pivotal.txt"
)
SUN_AUTHOR_ROOT = (
    HERE / "upstream/Unified_Sequential_Flow_Generative_Model"
)
SUN_AUTHOR_COMMIT = "383882bb56b33c48127a1874384e9897b5cde6e4"
SUN_AUTHOR_FLOW_SOURCE = SUN_AUTHOR_ROOT / "flow.py"
SUN_AUTHOR_TEMPORAL_SOURCE = SUN_AUTHOR_ROOT / "sequential_flow.py"
SUN_AUTHOR_README = SUN_AUTHOR_ROOT / "README.md"

VARIANTS = ("gmr-gmus", "pbgmr-gmus", "pbgmr-gmus-realnvp")
VARIANT_ALIASES = {"pbgmr-gmus-realnvp": "pbgmr-gmus"}
FIELD_NAMES = {
    "cylinder": ("u", "v", "p"),
    "multiphase": ("field_0", "field_1", "field_2"),
    "we1": ("field_0",),
    "e1": ("field_0",),
}


@dataclass
class Contract:
    spec: DatasetSpec
    fields: np.ndarray
    conditioning: np.ndarray
    coordinates: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    stats: dict[str, np.ndarray]
    condition_mean: np.ndarray
    condition_std: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    split_path: Path
    cache_usage: dict[str, str]

    @property
    def field_count(self) -> int:
        return int(self.fields.shape[-1])

    @property
    def condition_dim(self) -> int:
        return int(self.conditioning.shape[-1])

    @property
    def temporal_conditioning(self) -> bool:
        return self.conditioning.ndim == 3


@dataclass
class ReductionGeometry:
    pivotal_indices: np.ndarray
    pivotal_positions: np.ndarray
    reduction_indices: np.ndarray
    reduction_weights: np.ndarray
    expansion_indices: np.ndarray
    expansion_weights: np.ndarray
    metadata: dict[str, Any]

    @property
    def pivotal_count(self) -> int:
        return int(self.pivotal_positions.shape[0])


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    manifest: Path
    autoencoder_best: Path
    autoencoder_last: Path
    temporal_best: Path
    temporal_last: Path
    latent_cache: Path
    latent_cache_metadata: Path
    latent_stats: Path
    latent_stats_metadata: Path
    results_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate GMR-GMUS and PbGMR-GMUS-RealNVP"
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--gpu",
        default="cpu",
        help="'cpu', an integer CUDA index, or 'cuda:<index>'",
    )
    parser.add_argument(
        "--stage",
        choices=("autoencoder", "temporal", "evaluate"),
        default="autoencoder",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--restart-temporal",
        action="store_true",
        help="Discard only temporal checkpoints and retrain from the selected autoencoder",
    )
    parser.add_argument(
        "--finetune-from-best",
        action="store_true",
        help="Start a fresh optimizer from the validation-best temporal checkpoint",
    )
    parser.add_argument(
        "--eval-only",
        "--eval",
        dest="eval_only",
        action="store_true",
        help="Alias for --stage evaluate",
    )
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument(
        "--shape-smoke",
        action="store_true",
        help="CPU-safe synthetic forward/backward test; writes no files",
    )

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--val-pairs", type=int, default=128)
    parser.add_argument(
        "--val-trajectories",
        type=int,
        default=0,
        help="Validation trajectories for temporal selection; 0 means all",
    )
    parser.add_argument(
        "--temporal-horizon",
        type=int,
        default=0,
        help="Training/validation horizon; 0 uses the exact dataset horizon",
    )
    parser.add_argument("--encode-chunk-size", type=int, default=8)
    parser.add_argument("--decode-chunk-size", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument(
        "--pivotal-count",
        type=int,
        default=256,
        help="Resolved to min(value, node_count) on small meshes",
    )
    parser.add_argument(
        "--gmr-interpolation-k",
        type=int,
        default=3,
        help="Han GMUS interpolation; PhysicsNeMo public default is 3",
    )
    parser.add_argument(
        "--flow-eval-mode",
        choices=("center", "sample-mean"),
        default="center",
        help="Deterministic zero-base flow center or Monte Carlo sample mean",
    )
    parser.add_argument("--flow-samples", type=int, default=8)
    parser.add_argument(
        "--validation-flow-samples",
        type=int,
        default=8,
        help="Fixed-seed ensemble size for PbGMR decoded validation selection",
    )
    parser.add_argument("--flow-seed", type=int, default=42)
    parser.add_argument(
        "--result-tag",
        help="Optional suffix for alternate evaluation outputs",
    )

    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument(
        "--run-name",
        help="Checkpoint namespace; defaults to <variant>_<dataset>",
    )
    args = parser.parse_args()
    args.variant = VARIANT_ALIASES.get(args.variant, args.variant)
    if args.eval_only:
        args.stage = "evaluate"
    if args.steps < 1 and args.stage != "evaluate" and not args.shape_smoke:
        parser.error("--steps must be positive")
    if args.val_interval < 1:
        parser.error("--val-interval must be positive")
    if args.val_pairs < 1:
        parser.error("--val-pairs must be positive")
    if args.val_trajectories < 0:
        parser.error("--val-trajectories cannot be negative")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.encode_chunk_size < 1 or args.decode_chunk_size < 1:
        parser.error("encode/decode chunk sizes must be positive")
    if args.gmr_interpolation_k < 1:
        parser.error("--gmr-interpolation-k must be positive")
    if args.pivotal_count < 1:
        parser.error("--pivotal-count must be positive")
    if args.flow_samples < 1:
        parser.error("--flow-samples must be positive")
    if args.validation_flow_samples < 1:
        parser.error("--validation-flow-samples must be positive")
    if sum(
        int(flag)
        for flag in (
            args.resume,
            args.restart_temporal,
            args.finetune_from_best,
        )
    ) > 1:
        parser.error(
            "--resume, --restart-temporal, and --finetune-from-best are mutually exclusive"
        )
    return args


def resolve_device(gpu: str) -> torch.device:
    normalized = str(gpu).strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda:"):
        index = int(normalized.split(":", 1)[1])
    else:
        index = int(normalized)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA index {index} outside available range 0..{torch.cuda.device_count() - 1}"
        )
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def seed_everything(device: torch.device) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _condition_stats(
    conditioning: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(conditioning[train_indices], dtype=np.float64)
    flattened = selected.reshape(-1, selected.shape[-1])
    mean = flattened.mean(axis=0)
    variance = np.maximum(
        np.square(flattened).mean(axis=0) - np.square(mean), 1e-12
    )
    return mean.astype(np.float32), np.maximum(
        np.sqrt(variance), 1e-6
    ).astype(np.float32)


def _verify_split_cache(
    path: Path,
    spec: DatasetSpec,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"required split audit is missing: {path}; run prepare.py first"
        )
    payload = json.loads(path.read_text())
    expected = {
        "dataset": spec.name,
        "seed": 42,
        "forecast_horizon": spec.horizon,
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "test_indices": test.tolist(),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: cached {key} does not match common.py")


def load_contract(dataset_name: str) -> Contract:
    spec = DATASETS[dataset_name]
    fields = np.load(spec.fields_path, mmap_mode="r")
    if fields.ndim != 4:
        raise ValueError(f"{dataset_name}: expected [trajectory,time,node,field]")
    actual_horizon = int(fields.shape[1] - 1)
    if actual_horizon != spec.horizon:
        raise ValueError(
            f"{dataset_name}: common.py horizon={spec.horizon}, data={actual_horizon}"
        )

    train, validation, test = trajectory_split(
        fields.shape[0], spec.train_fraction, spec.val_fraction
    )
    split_path = HERE / "cache" / f"{dataset_name}_split.json"
    _verify_split_cache(split_path, spec, train, validation, test)

    graph_path = HERE / "cache" / f"{dataset_name}_graph.npz"
    if graph_path.exists():
        with np.load(graph_path) as graph:
            coordinates = np.asarray(graph["coordinates"], dtype=np.float32)
            edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
            graph_edge_features = np.asarray(
                graph["edge_features"], dtype=np.float32
            )
        graph_source = str(graph_path)
    else:
        coordinates = load_coordinates(spec)
        edge_index = load_edges(spec, coordinates)
        graph_edge_features = make_edge_features(coordinates, edge_index)
        graph_source = "computed in memory with common.py"

    conditioning_path = HERE / "cache" / f"{dataset_name}_conditioning.npy"
    if conditioning_path.exists():
        conditioning = np.load(conditioning_path, mmap_mode="r")
        conditioning_source = str(conditioning_path)
    else:
        conditioning = load_conditioning(spec, fields.shape[1])
        conditioning_source = "computed in memory with common.py"

    stats_path = HERE / "cache" / f"{dataset_name}_stats.npz"
    if stats_path.exists():
        with np.load(stats_path) as archive:
            stats = {key: archive[key] for key in archive.files}
        stats_source = str(stats_path)
    else:
        stats = field_and_delta_stats(fields, train)
        stats_source = "computed in memory with common.py"

    if coordinates.shape[0] != fields.shape[2]:
        raise ValueError("coordinate and field node counts differ")
    if conditioning.shape[0] != fields.shape[0]:
        raise ValueError("conditioning and field trajectory counts differ")
    required_stats = {"field_mean", "field_std", "delta_mean", "delta_std"}
    if required_stats.difference(stats):
        raise ValueError(f"normalization cache lacks {required_stats.difference(stats)}")
    condition_mean, condition_std = _condition_stats(conditioning, train)
    return Contract(
        spec=spec,
        fields=fields,
        conditioning=conditioning,
        coordinates=np.ascontiguousarray(coordinates),
        edge_index=np.ascontiguousarray(edge_index),
        edge_features=np.ascontiguousarray(graph_edge_features),
        stats=stats,
        condition_mean=condition_mean,
        condition_std=condition_std,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        split_path=split_path,
        cache_usage={
            "split": str(split_path),
            "graph": graph_source,
            "conditioning": conditioning_source,
            "normalization": stats_source,
        },
    )


def _inverse_square_knn(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(int(k), int(source_positions.shape[0]))
    if k < 1:
        raise ValueError("interpolation requires at least one source")
    distances, indices = cKDTree(source_positions).query(
        target_positions, k=k
    )
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = np.empty_like(distances, dtype=np.float64)
    for row in range(distances.shape[0]):
        exact = np.flatnonzero(distances[row] <= 1e-12)
        if len(exact):
            weights[row] = 0.0
            weights[row, exact[0]] = 1.0
        else:
            inverse = 1.0 / np.maximum(np.square(distances[row]), 1e-16)
            weights[row] = inverse / inverse.sum()
    return (
        np.ascontiguousarray(indices, dtype=np.int64),
        np.ascontiguousarray(weights, dtype=np.float32),
    )


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def build_geometry(
    contract: Contract,
    variant: str,
    requested_pivotal_count: int,
    gmr_interpolation_k: int,
) -> ReductionGeometry:
    node_count = int(contract.coordinates.shape[0])
    pivotal_count = min(requested_pivotal_count, node_count)
    source_description: str

    if (
        contract.spec.name == "cylinder"
        and pivotal_count == 256
        and OFFICIAL_CYLINDER_PIVOTS.exists()
    ):
        pivotal_positions = np.asarray(
            np.loadtxt(OFFICIAL_CYLINDER_PIVOTS), dtype=np.float32
        )
        distances, pivotal_indices = cKDTree(contract.coordinates).query(
            pivotal_positions, k=1
        )
        if np.max(distances) > 1e-7 or len(np.unique(pivotal_indices)) != 256:
            raise ValueError("official PhysicsNeMo pivots do not match cylinder mesh")
        pivotal_indices = np.asarray(pivotal_indices, dtype=np.int64)
        source_description = str(OFFICIAL_CYLINDER_PIVOTS)
    else:
        rng = np.random.RandomState(42)
        pivotal_indices = rng.choice(
            node_count, size=pivotal_count, replace=False
        ).astype(np.int64)
        pivotal_positions = contract.coordinates[pivotal_indices].copy()
        source_description = (
            "seed-42 uniform mesh-coordinate selection; no authoritative "
            "pivotal file exists for this SEA dataset"
        )

    if variant == "gmr-gmus":
        reduction_indices = pivotal_indices[:, None]
        reduction_weights = np.ones((pivotal_count, 1), dtype=np.float32)
        expansion_indices, expansion_weights = _inverse_square_knn(
            pivotal_positions,
            contract.coordinates,
            gmr_interpolation_k,
        )
        interpolation_k = min(gmr_interpolation_k, pivotal_count)
        reduction_mode = "pivotal graph nodes"
        residual_layernorm = False
    else:
        reduction_indices, reduction_weights = _inverse_square_knn(
            contract.coordinates, pivotal_positions, 10
        )
        expansion_indices, expansion_weights = _inverse_square_knn(
            pivotal_positions, contract.coordinates, 10
        )
        interpolation_k = min(10, pivotal_count, node_count)
        reduction_mode = "position centers independent of graph adjacency"
        residual_layernorm = True

    metadata = {
        "requested_pivotal_count": requested_pivotal_count,
        "resolved_pivotal_count": pivotal_count,
        "pivotal_source": source_description,
        "pivotal_indices": pivotal_indices.tolist(),
        "pivotal_positions_sha256": _sha256_array(pivotal_positions),
        "reduction_mode": reduction_mode,
        "interpolation": "normalized inverse squared distance",
        "interpolation_k": interpolation_k,
        "residual_layernorm": residual_layernorm,
    }
    return ReductionGeometry(
        pivotal_indices=np.ascontiguousarray(pivotal_indices),
        pivotal_positions=np.ascontiguousarray(pivotal_positions),
        reduction_indices=reduction_indices,
        reduction_weights=reduction_weights,
        expansion_indices=expansion_indices,
        expansion_weights=expansion_weights,
        metadata=metadata,
    )


def model_configuration(
    contract: Contract, geometry: ReductionGeometry, variant: str
) -> dict[str, Any]:
    return {
        "variant": variant,
        "dataset": contract.spec.name,
        "node_input_dim": contract.field_count,
        "edge_input_dim": int(contract.edge_features.shape[-1]),
        "decoded_field_count": contract.field_count,
        "node_count": int(contract.fields.shape[2]),
        "processor_blocks": 3,
        "processor_width": 128,
        "processor_mlp_hidden_layers": 2,
        "latent_per_pivot": 4,
        "pivotal_count": geometry.pivotal_count,
        "latent_dim": geometry.pivotal_count * 4,
        "residual_layernorm_graphnet": variant != "gmr-gmus",
        "geometry": geometry.metadata,
        "temporal_conditioning": contract.temporal_conditioning,
        "temporal_architecture": (
            {
                "layers": 1,
                "heads": 4,
                "residual_latent_prediction": True,
            }
            if variant == "gmr-gmus"
            else {
                "transformer_encoder_layers": 2,
                "transformer_decoder_layers": 1,
                "heads": 4,
                "coupling_layers": 2,
                "coupling_hidden_size": geometry.pivotal_count * 4,
                "coupling_hidden_layers_argument": 2,
                "invertible_batch_norm": True,
                "author_flow_commit": SUN_AUTHOR_COMMIT,
            }
        ),
    }


def build_autoencoder(
    contract: Contract,
    geometry: ReductionGeometry,
    variant: str,
    device: torch.device,
) -> MeshReducedAutoencoder:
    return MeshReducedAutoencoder(
        node_input_dim=contract.field_count,
        output_dim=contract.field_count,
        edge_features=torch.from_numpy(contract.edge_features).float(),
        edge_index=torch.from_numpy(contract.edge_index).long(),
        reduction_indices=torch.from_numpy(geometry.reduction_indices).long(),
        reduction_weights=torch.from_numpy(geometry.reduction_weights).float(),
        expansion_indices=torch.from_numpy(geometry.expansion_indices).long(),
        expansion_weights=torch.from_numpy(geometry.expansion_weights).float(),
        pivotal_count=geometry.pivotal_count,
        width=128,
        latent_per_pivot=4,
        processor_blocks=3,
        residual_layernorm=variant != "gmr-gmus",
    ).to(device)


def build_temporal(
    contract: Contract,
    latent_dim: int,
    variant: str,
    device: torch.device,
) -> torch.nn.Module:
    if variant == "gmr-gmus":
        model: torch.nn.Module = GMRResidualTemporalTransformer(
            latent_dim, contract.condition_dim, heads=4
        )
    else:
        model = PbGMRRealNVPTemporal(
            latent_dim,
            contract.condition_dim,
            heads=4,
            coupling_layers=2,
        )
    return model.to(device)


def _normalizer_tensors(
    contract: Contract, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(contract.stats["field_mean"], device=device),
        torch.as_tensor(contract.stats["field_std"], device=device),
        torch.as_tensor(contract.condition_mean, device=device),
        torch.as_tensor(contract.condition_std, device=device),
    )


def snapshot_batch(
    contract: Contract,
    trajectories: np.ndarray,
    times: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    fields = np.asarray(
        contract.fields[trajectories, times], dtype=np.float32
    )
    field_mean, field_std, condition_mean, condition_std = _normalizer_tensors(
        contract, device
    )
    target = (
        torch.from_numpy(fields).to(device) - field_mean
    ) / field_std
    del condition_mean, condition_std
    return target, target


def trajectory_conditions(
    contract: Contract,
    trajectories: np.ndarray,
    start: int,
    stop: int,
    device: torch.device,
) -> torch.Tensor:
    if contract.conditioning.ndim == 2:
        raw = np.broadcast_to(
            np.asarray(
                contract.conditioning[trajectories, None], dtype=np.float32
            ),
            (len(trajectories), stop - start, contract.condition_dim),
        ).copy()
    else:
        raw = np.asarray(
            contract.conditioning[trajectories, start:stop],
            dtype=np.float32,
        )
    condition_mean = torch.as_tensor(contract.condition_mean, device=device)
    condition_std = torch.as_tensor(contract.condition_std, device=device)
    return (
        torch.from_numpy(raw).to(device) - condition_mean
    ) / condition_std


@torch.no_grad()
def encode_trajectories(
    autoencoder: MeshReducedAutoencoder,
    contract: Contract,
    trajectories: np.ndarray,
    horizon: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    autoencoder.eval()
    field_mean = torch.as_tensor(contract.stats["field_mean"], device=device)
    field_std = torch.as_tensor(contract.stats["field_std"], device=device)
    encoded_chunks = []
    for start in range(0, horizon + 1, chunk_size):
        stop = min(start + chunk_size, horizon + 1)
        raw_fields = np.asarray(
            contract.fields[trajectories, start:stop], dtype=np.float32
        )
        batch, steps, nodes, fields = raw_fields.shape
        state = (
            torch.from_numpy(raw_fields).to(device) - field_mean
        ) / field_std
        latent = autoencoder.encode(
            state.reshape(batch * steps, nodes, fields)
        )
        encoded_chunks.append(latent.reshape(batch, steps, -1))
    return torch.cat(encoded_chunks, dim=1)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_paths(args: argparse.Namespace) -> RunPaths:
    name = args.run_name or f"{args.variant}_{args.dataset}"
    run_dir = args.output_root / "checkpoints" / "mesh_reduced" / name
    return RunPaths(
        run_dir=run_dir,
        manifest=run_dir / "provenance.json",
        autoencoder_best=run_dir / "autoencoder_best.pt",
        autoencoder_last=run_dir / "autoencoder_last.pt",
        temporal_best=run_dir / "temporal_best.pt",
        temporal_last=run_dir / "temporal_last.pt",
        latent_cache=run_dir / "latents.npy",
        latent_cache_metadata=run_dir / "latents.json",
        latent_stats=run_dir / "latent_stats.npz",
        latent_stats_metadata=run_dir / "latent_stats.json",
        results_dir=args.output_root / "results",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, hash_content: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        record.update({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if hash_content:
            record["sha256"] = _sha256_file(path)
    return record


def _physicsnemo_head() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(PHYSICSNEMO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to verify local PhysicsNeMo commit") from error
    head = result.stdout.strip()
    if head != PHYSICSNEMO_COMMIT:
        raise RuntimeError(
            f"PhysicsNeMo HEAD {head} != required {PHYSICSNEMO_COMMIT}"
        )
    return head


def provenance(
    args: argparse.Namespace,
    contract: Contract,
    geometry: ReductionGeometry,
    configuration: dict[str, Any],
    autoencoder: MeshReducedAutoencoder,
    temporal: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    pb_variant = args.variant == "pbgmr-gmus"
    limitations = [
        "SEA inputs do not expose the cell-volume features used by the flow "
        "papers; this harness uses every common.py field and conditioning channel.",
        "The authoritative 256 cylinder pivotal positions are reused. Other SEA "
        "meshes use fixed seed-42 mesh-coordinate centers because no matching "
        "published pivotal files exist.",
        "Small meshes resolve the requested 256 pivots to the node count.",
        "Han et al. grow the free-running loss horizon after a convergence "
        "threshold. This harness exposes the same curriculum explicitly through "
        "--temporal-horizon rather than guessing a dataset-specific threshold.",
    ]
    if contract.temporal_conditioning:
        limitations.append(
            "The paper assumes a static physical parameter. For temporal "
            "conditioning, the known condition at each prediction step is added "
            "to the temporal token; static-conditioning behavior is unchanged."
        )
    if pb_variant:
        limitations.extend(
            [
                "The pinned author flow is available locally, but cannot be used "
                "as-is: it includes hard-coded cuda:0 operations, debugger traps, "
                "and unrelated GluonTS imports. The integration mirrors its "
                "alternating masks, scale/shift networks, tanh log-scales, and "
                "invertible batch normalization in dependency-free PyTorch.",
                "The author source's variance comment specifies a biased batch "
                "variance, while its torch.var call inherits version-dependent "
                "defaults. This harness explicitly uses unbiased=False.",
                "Default deterministic evaluation maps zero Gaussian base noise "
                "through the flow; --flow-eval-mode sample-mean is also available.",
                "Sun et al.'s inference appendix seeds z0 and z1, while the common "
                "one-initial-frame protocol requires predicting z1 from z0. The "
                "implemented factorization includes p(z1 | condition, z0), as in "
                "their equation (11).",
            ]
        )

    return {
        "schema_version": 1,
        "method": args.variant,
        "implementation_status": (
            "paper equations reimplemented over the common fixed graph"
            if not pb_variant
            else "local spatial implementation cross-checked against PhysicsNeMo "
            "plus source-aligned integration of the pinned author RealNVP"
        ),
        "seed": 42,
        "architecture": {
            **configuration,
            "autoencoder_parameters": parameter_count(autoencoder),
            "temporal_parameters": parameter_count(temporal),
            "temporal": (
                {
                    "type": "residual causal attention",
                    "layers": 1,
                    "heads": 4,
                    "residual_prediction": True,
                }
                if not pb_variant
                else {
                    "type": "encoder-decoder Transformer + conditional RealNVP",
                    "encoder_layers": 2,
                    "masked_decoder_layers": 1,
                    "heads": 4,
                    "activation": "gelu",
                    "coupling_layers": 2,
                    "flow_hidden_size": configuration["latent_dim"],
                    "flow_hidden_layers_argument": 2,
                    "flow_mask": "alternating even/odd latent dimensions",
                    "flow_log_scale": "tanh(s)",
                    "flow_batch_norm": "after every coupling layer",
                }
            ),
        },
        "training_protocol": {
            "stages": [
                "train spatial autoencoder reconstruction",
                "freeze validation-selected autoencoder",
                "train temporal model",
                "evaluate validation-selected temporal checkpoint",
            ],
            "requested_stage": args.stage,
            "requested_steps": args.steps,
            "temporal_training_horizon": (
                args.temporal_horizon or contract.spec.horizon
            ),
            "autoencoder_validation": {
                "metric": "normalized decoded reconstruction MSE",
                "fixed_pairs": args.val_pairs,
                "seed": 420042,
            },
            "temporal_validation": {
                "trajectories": (
                    "all"
                    if args.val_trajectories == 0
                    else args.val_trajectories
                ),
                "metric": (
                    "decoded free-running validation RelMSE, reducing spatial "
                    "nodes first and averaging trajectories, steps, and fields"
                ),
                "pbgmr_validation_rollout": (
                    f"fixed-seed mean of {args.validation_flow_samples} flow samples"
                ),
            },
        },
        "data_contract": {
            "dataset": contract.spec.name,
            "trajectory_split_seed": 42,
            "train_indices": contract.train_indices.tolist(),
            "validation_indices": contract.validation_indices.tolist(),
            "test_indices": contract.test_indices.tolist(),
            "forecast_horizon": contract.spec.horizon,
            "decoded_field_count": contract.field_count,
            "decoded_field_names": list(FIELD_NAMES[contract.spec.name]),
            "normalization_scope": "training trajectories only",
            "latent_normalization_scope": (
                "training trajectories and all frames after selecting the "
                "validation-best spatial autoencoder"
            ),
            "field_mean": contract.stats["field_mean"].tolist(),
            "field_std": contract.stats["field_std"].tolist(),
            "condition_mean": contract.condition_mean.tolist(),
            "condition_std": contract.condition_std.tolist(),
            "cache_usage": contract.cache_usage,
            "artifacts": {
                "fields": _file_record(
                    contract.spec.fields_path, hash_content=False
                ),
                "coordinates": _file_record(
                    contract.spec.coordinates_path, hash_content=False
                ),
                "conditioning": _file_record(
                    contract.spec.input_path, hash_content=False
                ),
                "edges": _file_record(
                    contract.spec.edge_path, hash_content=False
                ),
            },
        },
        "authoritative_sources": {
            "physicsnemo_commit": _physicsnemo_head(),
            "physicsnemo_spatial": _file_record(
                PHYSICSNEMO_SPATIAL_SOURCE, hash_content=True
            ),
            "physicsnemo_temporal": _file_record(
                PHYSICSNEMO_TEMPORAL_SOURCE, hash_content=True
            ),
            "sun_author_repository": {
                "url": "https://github.com/luningsun/"
                "Unified_Sequential_Flow_Generative_Model",
                "commit": SUN_AUTHOR_COMMIT,
                "readme": _file_record(
                    SUN_AUTHOR_README, hash_content=True
                ),
                "flow": _file_record(
                    SUN_AUTHOR_FLOW_SOURCE, hash_content=True
                ),
                "temporal": _file_record(
                    SUN_AUTHOR_TEMPORAL_SOURCE, hash_content=True
                ),
            },
            "common_py": _file_record(HERE / "common.py", hash_content=True),
            "paper_faithful_models": _file_record(
                HERE / "mesh_reduced_models.py", hash_content=True
            ),
            "split_json": _file_record(
                contract.split_path, hash_content=True
            ),
            "han_2022": {
                "title": "Predicting Physics in Mesh-Reduced Space with Temporal Attention",
                "venue": "ICLR 2022",
                "url": "https://openreview.net/forum?id=XctLdNfCmP",
                "implemented_equations": ["2", "3", "4", "5", "6", "7-12"],
            },
            "sun_2023": {
                "title": "Unifying Predictions of Deterministic and Stochastic "
                "Physics in Mesh-reduced Space with Sequential Flow Generative Model",
                "venue": "NeurIPS 2023",
                "url": "https://openreview.net/forum?id=2JtwuJtoa0",
                "implemented_equations": [
                    "3",
                    "5-10",
                    "11-13",
                    "16-18",
                ],
            },
        },
        "evaluation_contract": {
            "initial_frames": 1,
            "autoregressive_steps": contract.spec.horizon,
            "all_decoded_fields": True,
            "elementary_value": (
                "sum_nodes((prediction-target)^2) / sum_nodes(target^2)"
            ),
            "aggregation": (
                "arithmetic mean over trajectories and forecast steps, then "
                "arithmetic mean over fields"
            ),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "argv": sys.argv,
        },
        "limitations": limitations,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def load_or_create_latent_cache(
    autoencoder: MeshReducedAutoencoder,
    contract: Contract,
    paths: RunPaths,
    encode_chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    checkpoint_sha256 = _sha256_file(paths.autoencoder_best)
    expected_shape = (
        int(contract.fields.shape[0]),
        int(contract.fields.shape[1]),
        int(autoencoder.latent_dim),
    )
    expected_metadata = {
        "autoencoder_checkpoint": str(paths.autoencoder_best),
        "autoencoder_sha256": checkpoint_sha256,
        "shape": list(expected_shape),
        "dtype": "float32",
        "trajectory_order": "original dataset order",
    }
    if paths.latent_cache.exists() and paths.latent_cache_metadata.exists():
        metadata = json.loads(paths.latent_cache_metadata.read_text())
        if metadata == expected_metadata:
            cached = np.load(paths.latent_cache, mmap_mode="r")
            if tuple(cached.shape) != expected_shape:
                raise ValueError("latent cache shape disagrees with metadata")
            print(f"using latent cache {paths.latent_cache}", flush=True)
            return cached

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    temporary = paths.latent_cache.with_name(
        f"{paths.latent_cache.name}.tmp-{os.getpid()}"
    )
    cached = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=expected_shape,
    )
    trajectory_batch_size = contract.spec.rollout_batch_size
    autoencoder.eval()
    for start in range(0, contract.fields.shape[0], trajectory_batch_size):
        stop = min(start + trajectory_batch_size, contract.fields.shape[0])
        trajectory_ids = np.arange(start, stop, dtype=np.int64)
        latent = encode_trajectories(
            autoencoder,
            contract,
            trajectory_ids,
            contract.spec.horizon,
            encode_chunk_size,
            device,
        )
        cached[start:stop] = latent.float().cpu().numpy()
        cached.flush()
        print(
            f"latent cache {stop}/{contract.fields.shape[0]} trajectories",
            flush=True,
        )
    del cached
    os.replace(temporary, paths.latent_cache)
    _atomic_json(paths.latent_cache_metadata, expected_metadata)
    return np.load(paths.latent_cache, mmap_mode="r")


def load_or_create_latent_stats(
    latent_cache: np.ndarray,
    contract: Contract,
    paths: RunPaths,
) -> dict[str, np.ndarray]:
    cache_metadata = json.loads(paths.latent_cache_metadata.read_text())
    expected_metadata = {
        "latent_cache_autoencoder_sha256": cache_metadata["autoencoder_sha256"],
        "scope": "training trajectories and all available frames",
        "dimensions": int(latent_cache.shape[-1]),
    }
    if paths.latent_stats.exists() and paths.latent_stats_metadata.exists():
        metadata = json.loads(paths.latent_stats_metadata.read_text())
        if metadata == expected_metadata:
            with np.load(paths.latent_stats) as archive:
                return {key: archive[key] for key in archive.files}

    total = np.zeros(latent_cache.shape[-1], dtype=np.float64)
    squared_total = np.zeros_like(total)
    count = 0
    for start in range(0, len(contract.train_indices), 16):
        trajectory_ids = contract.train_indices[start : start + 16]
        values = np.asarray(latent_cache[trajectory_ids], dtype=np.float32)
        flattened = values.reshape(-1, values.shape[-1]).astype(np.float64)
        total += flattened.sum(axis=0)
        squared_total += np.square(flattened).sum(axis=0)
        count += flattened.shape[0]
    mean = total / count
    variance = np.maximum(squared_total / count - np.square(mean), 1e-12)
    stats = {
        "latent_mean": mean.astype(np.float32),
        "latent_std": np.maximum(np.sqrt(variance), 1e-6).astype(np.float32),
    }
    temporary = paths.latent_stats.with_name(
        f"{paths.latent_stats.stem}.tmp-{os.getpid()}.npz"
    )
    np.savez(temporary, **stats)
    os.replace(temporary, paths.latent_stats)
    _atomic_json(paths.latent_stats_metadata, expected_metadata)
    return stats


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    validation_metric: float,
    best_validation_metric: float,
    metric_name: str,
    configuration: dict[str, Any],
    manifest: dict[str, Any],
    rng: np.random.RandomState,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "validation_metric": validation_metric,
        "best_validation_metric": best_validation_metric,
        "validation_metric_name": metric_name,
        "model_configuration": configuration,
        "provenance": manifest,
        "numpy_rng_state": rng.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    model_device = next(model.parameters()).device
    if model_device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(model_device)
        payload["cuda_rng_device"] = str(model_device)
    return payload


def _load_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    configuration: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_configuration") != configuration:
        raise ValueError(f"{path}: model/data configuration mismatch")
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def _restore_training_state(
    checkpoint: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    rng: np.random.RandomState,
) -> tuple[int, float]:
    optimizer.load_state_dict(checkpoint["optimizer"])
    rng.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    parameter_device = optimizer.param_groups[0]["params"][0].device
    if parameter_device.type == "cuda" and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state(
            checkpoint["cuda_rng_state"].cpu(), device=parameter_device
        )
    return (
        int(checkpoint["step"]),
        float(checkpoint["best_validation_metric"]),
    )


@torch.no_grad()
def validate_autoencoder(
    model: MeshReducedAutoencoder,
    contract: Contract,
    pair_count: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    rng = np.random.RandomState(420_042)
    squared_error = 0.0
    value_count = 0
    remaining = pair_count
    while remaining:
        current_batch = min(batch_size, remaining)
        trajectories = rng.choice(
            contract.validation_indices, size=current_batch, replace=True
        )
        times = rng.randint(
            0, contract.fields.shape[1], size=current_batch
        )
        node_features, target = snapshot_batch(
            contract, trajectories, times, device
        )
        prediction = model(node_features)
        squared_error += torch.square(prediction - target).sum().item()
        value_count += target.numel()
        remaining -= current_batch
    model.train()
    return squared_error / value_count


def train_autoencoder(
    args: argparse.Namespace,
    model: MeshReducedAutoencoder,
    contract: Contract,
    configuration: dict[str, Any],
    manifest: dict[str, Any],
    paths: RunPaths,
    device: torch.device,
) -> None:
    if not args.resume and (
        paths.autoencoder_best.exists() or paths.autoencoder_last.exists()
    ):
        raise FileExistsError(
            f"{paths.run_dir} already has autoencoder checkpoints; use "
            "--resume or a different --run-name"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.RandomState(42)
    start_step = 0
    best_validation = float("inf")
    if args.resume:
        checkpoint = _load_model_checkpoint(
            paths.autoencoder_last, model, configuration, device
        )
        start_step, best_validation = _restore_training_state(
            checkpoint, optimizer, rng
        )
        print(f"resumed autoencoder at step {start_step}", flush=True)

    batch_size = args.batch_size or contract.spec.batch_size
    wall_start = time.time()
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        trajectories = rng.choice(
            contract.train_indices, size=batch_size, replace=True
        )
        times = rng.randint(0, contract.fields.shape[1], size=batch_size)
        node_features, target = snapshot_batch(
            contract, trajectories, times, device
        )
        prediction = model(node_features)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % 50 == 0:
            elapsed = max(time.time() - wall_start, 1e-9)
            print(
                f"autoencoder step={step}/{args.steps} "
                f"loss={loss.item():.6e} steps_per_second="
                f"{(step - start_step) / elapsed:.3f}",
                flush=True,
            )

        if step % args.val_interval == 0 or step == args.steps:
            validation = validate_autoencoder(
                model,
                contract,
                args.val_pairs,
                batch_size,
                device,
            )
            improved = validation < best_validation
            best_validation = min(best_validation, validation)
            payload = _checkpoint_payload(
                model,
                optimizer,
                step,
                validation,
                best_validation,
                "normalized_decoded_reconstruction_mse",
                configuration,
                manifest,
                rng,
            )
            _atomic_torch_save(paths.autoencoder_last, payload)
            if improved:
                _atomic_torch_save(paths.autoencoder_best, payload)
            print(
                f"autoencoder validation step={step} mse={validation:.6e} "
                f"best={best_validation:.6e}",
                flush=True,
            )


@torch.no_grad()
def validate_temporal(
    temporal: torch.nn.Module,
    autoencoder: MeshReducedAutoencoder,
    latent_cache: np.ndarray,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    contract: Contract,
    variant: str,
    horizon: int,
    validation_count: int,
    decode_chunk_size: int,
    validation_flow_samples: int,
    device: torch.device,
) -> float:
    temporal.eval()
    autoencoder.eval()
    indices = contract.validation_indices
    if validation_count:
        indices = indices[:validation_count]
    field_mean = torch.as_tensor(contract.stats["field_mean"], device=device)
    field_std = torch.as_tensor(contract.stats["field_std"], device=device)
    ratio_sum = 0.0
    value_count = 0
    batch_size = contract.spec.rollout_batch_size
    flow_generator = _flow_generator(device, 420_042)
    for start in range(0, len(indices), batch_size):
        trajectory = indices[start : start + batch_size]
        initial = torch.from_numpy(
            np.asarray(latent_cache[trajectory, 0], dtype=np.float32)
        ).to(device)
        initial = (initial - latent_mean) / latent_std
        conditions = trajectory_conditions(
            contract, trajectory, 0, horizon, device
        )
        if variant == "gmr-gmus":
            sample_count = 1
            prediction = temporal(initial, conditions)
        else:
            sample_count = validation_flow_samples
            initial = initial.repeat_interleave(sample_count, dim=0)
            conditions = conditions.repeat_interleave(sample_count, dim=0)
            prediction = temporal.rollout(
                initial,
                conditions,
                sample=True,
                generator=flow_generator,
                temporal_conditioning=contract.temporal_conditioning,
            )
        prediction = prediction * latent_std + latent_mean
        for time_start in range(0, horizon, decode_chunk_size):
            time_stop = min(time_start + decode_chunk_size, horizon)
            chunk = prediction[:, time_start:time_stop]
            decoded_normalized = autoencoder.decode(
                chunk.reshape(-1, chunk.shape[-1])
            )
            if sample_count == 1:
                decoded_normalized = decoded_normalized.reshape(
                    len(trajectory),
                    time_stop - time_start,
                    contract.fields.shape[2],
                    contract.field_count,
                )
            else:
                decoded_normalized = decoded_normalized.reshape(
                    len(trajectory),
                    sample_count,
                    time_stop - time_start,
                    contract.fields.shape[2],
                    contract.field_count,
                ).mean(dim=1)
            decoded = decoded_normalized
            decoded = decoded * field_std + field_mean
            target = torch.from_numpy(
                np.asarray(
                    contract.fields[
                        trajectory,
                        time_start + 1 : time_stop + 1,
                    ],
                    dtype=np.float32,
                )
            ).to(device)
            numerator = torch.square(decoded - target).sum(dim=2)
            denominator = torch.square(target).sum(dim=2)
            ratios = numerator / torch.clamp(denominator, min=1e-12)
            ratio_sum += ratios.sum().item()
            value_count += ratios.numel()
    temporal.train()
    return ratio_sum / value_count


def train_temporal(
    args: argparse.Namespace,
    temporal: torch.nn.Module,
    autoencoder: MeshReducedAutoencoder,
    contract: Contract,
    configuration: dict[str, Any],
    manifest: dict[str, Any],
    paths: RunPaths,
    device: torch.device,
) -> None:
    _load_model_checkpoint(
        paths.autoencoder_best, autoencoder, configuration, device
    )
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    latent_cache = load_or_create_latent_cache(
        autoencoder,
        contract,
        paths,
        args.encode_chunk_size,
        device,
    )
    latent_stats = load_or_create_latent_stats(latent_cache, contract, paths)
    latent_mean = torch.from_numpy(latent_stats["latent_mean"]).to(device)
    latent_std = torch.from_numpy(latent_stats["latent_std"]).to(device)
    if args.restart_temporal:
        paths.temporal_best.unlink(missing_ok=True)
        paths.temporal_last.unlink(missing_ok=True)
    if not (args.resume or args.finetune_from_best) and (
        paths.temporal_best.exists() or paths.temporal_last.exists()
    ):
        raise FileExistsError(
            f"{paths.run_dir} already has temporal checkpoints; use "
            "--resume or a different --run-name"
        )

    optimizer = torch.optim.Adam(
        temporal.parameters(), lr=args.learning_rate
    )
    rng = np.random.RandomState(42)
    start_step = 0
    best_validation = float("inf")
    if args.resume:
        checkpoint = _load_model_checkpoint(
            paths.temporal_last, temporal, configuration, device
        )
        start_step, best_validation = _restore_training_state(
            checkpoint, optimizer, rng
        )
        print(f"resumed temporal model at step {start_step}", flush=True)
    elif args.finetune_from_best:
        checkpoint = _load_model_checkpoint(
            paths.temporal_best, temporal, configuration, device
        )
        best_validation = float(checkpoint["best_validation_metric"])
        print(
            "fine-tuning from validation-best temporal checkpoint "
            f"(value={best_validation:.6e})",
            flush=True,
        )

    horizon = args.temporal_horizon or contract.spec.horizon
    if not 1 <= horizon <= contract.spec.horizon:
        raise ValueError(
            f"--temporal-horizon must be in 1..{contract.spec.horizon}"
        )
    batch_size = args.batch_size or 1
    wall_start = time.time()
    temporal.train()
    for step in range(start_step + 1, args.steps + 1):
        trajectories = rng.choice(
            contract.train_indices, size=batch_size, replace=True
        )
        latent = torch.from_numpy(
            np.asarray(
                latent_cache[trajectories, : horizon + 1],
                dtype=np.float32,
            )
        ).to(device)
        latent = (latent - latent_mean) / latent_std
        conditions = trajectory_conditions(
            contract, trajectories, 0, horizon, device
        )
        if args.variant == "gmr-gmus":
            prediction = temporal(latent[:, 0], conditions)
            loss = F.mse_loss(prediction, latent[:, 1:])
        else:
            loss = temporal(
                latent,
                conditions,
                temporal_conditioning=contract.temporal_conditioning,
            )
        metric_name = "decoded_validation_rollout_relmse"
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(temporal.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % 50 == 0:
            elapsed = max(time.time() - wall_start, 1e-9)
            print(
                f"temporal step={step}/{args.steps} loss={loss.item():.6e} "
                f"steps_per_second={(step - start_step) / elapsed:.3f}",
                flush=True,
            )

        if step % args.val_interval == 0 or step == args.steps:
            validation = validate_temporal(
                temporal,
                autoencoder,
                latent_cache,
                latent_mean,
                latent_std,
                contract,
                args.variant,
                horizon,
                args.val_trajectories,
                args.decode_chunk_size,
                args.validation_flow_samples,
                device,
            )
            improved = validation < best_validation
            best_validation = min(best_validation, validation)
            payload = _checkpoint_payload(
                temporal,
                optimizer,
                step,
                validation,
                best_validation,
                metric_name,
                configuration,
                manifest,
                rng,
            )
            _atomic_torch_save(paths.temporal_last, payload)
            if improved:
                _atomic_torch_save(paths.temporal_best, payload)
            print(
                f"temporal validation step={step} value={validation:.6e} "
                f"best={best_validation:.6e}",
                flush=True,
            )


def _flow_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


@torch.no_grad()
def evaluate_rollout(
    args: argparse.Namespace,
    autoencoder: MeshReducedAutoencoder,
    temporal: torch.nn.Module,
    contract: Contract,
    paths: RunPaths,
    device: torch.device,
) -> dict[str, Any]:
    autoencoder.eval()
    temporal.eval()
    field_mean = torch.as_tensor(contract.stats["field_mean"], device=device)
    field_std = torch.as_tensor(contract.stats["field_std"], device=device)
    field_count = contract.field_count
    horizon = contract.spec.horizon
    latent_cache = load_or_create_latent_cache(
        autoencoder,
        contract,
        paths,
        args.encode_chunk_size,
        device,
    )
    latent_stats = load_or_create_latent_stats(latent_cache, contract, paths)
    latent_mean = torch.from_numpy(latent_stats["latent_mean"]).to(device)
    latent_std = torch.from_numpy(latent_stats["latent_std"]).to(device)
    global_ratio_sum = np.zeros(field_count, dtype=np.float64)
    trajectory_records: list[dict[str, Any]] = []
    flow_generator = _flow_generator(device, args.flow_seed)

    rollout_batch_size = contract.spec.rollout_batch_size
    for start in range(0, len(contract.test_indices), rollout_batch_size):
        trajectory_ids = contract.test_indices[
            start : start + rollout_batch_size
        ]
        initial_features, _ = snapshot_batch(
            contract,
            trajectory_ids,
            np.zeros(len(trajectory_ids), dtype=np.int64),
            device,
        )
        initial_latent = autoencoder.encode(initial_features)
        initial_latent = (initial_latent - latent_mean) / latent_std
        conditions = trajectory_conditions(
            contract, trajectory_ids, 0, horizon, device
        )

        sample_count = 1
        if args.variant == "gmr-gmus":
            latent_prediction = temporal(initial_latent, conditions)
        elif args.flow_eval_mode == "center":
            latent_prediction = temporal.rollout(
                initial_latent,
                conditions,
                sample=False,
                generator=None,
                temporal_conditioning=contract.temporal_conditioning,
            )
        else:
            sample_count = args.flow_samples
            repeated_latent = initial_latent.repeat_interleave(
                sample_count, dim=0
            )
            repeated_conditions = conditions.repeat_interleave(
                sample_count, dim=0
            )
            latent_prediction = temporal.rollout(
                repeated_latent,
                repeated_conditions,
                sample=True,
                generator=flow_generator,
                temporal_conditioning=contract.temporal_conditioning,
            )
        latent_prediction = latent_prediction * latent_std + latent_mean

        batch_ratio_sum = torch.zeros(
            len(trajectory_ids),
            field_count,
            dtype=torch.float64,
            device=device,
        )
        for time_start in range(0, horizon, args.decode_chunk_size):
            time_stop = min(time_start + args.decode_chunk_size, horizon)
            chunk = latent_prediction[:, time_start:time_stop]
            decoded_normalized = autoencoder.decode(
                chunk.reshape(-1, chunk.shape[-1])
            )
            nodes = decoded_normalized.shape[1]
            if sample_count == 1:
                decoded_normalized = decoded_normalized.reshape(
                    len(trajectory_ids),
                    time_stop - time_start,
                    nodes,
                    field_count,
                )
                decoded = decoded_normalized * field_std + field_mean
            else:
                decoded_normalized = decoded_normalized.reshape(
                    len(trajectory_ids),
                    sample_count,
                    time_stop - time_start,
                    nodes,
                    field_count,
                )
                decoded = (
                    decoded_normalized * field_std + field_mean
                ).mean(dim=1)

            target = torch.from_numpy(
                np.asarray(
                    contract.fields[
                        trajectory_ids,
                        time_start + 1 : time_stop + 1,
                    ],
                    dtype=np.float32,
                )
            ).to(device)
            numerator = torch.square(decoded - target).sum(dim=2).double()
            denominator = torch.square(target).sum(dim=2).double()
            batch_ratio_sum += (
                numerator / torch.clamp(denominator, min=1e-12)
            ).sum(dim=1)

        ratio_sum_np = batch_ratio_sum.cpu().numpy()
        global_ratio_sum += ratio_sum_np.sum(axis=0)
        per_trajectory = ratio_sum_np / horizon
        for row, trajectory_id in enumerate(trajectory_ids):
            trajectory_records.append(
                {
                    "trajectory_index": int(trajectory_id),
                    "per_field_relmse": per_trajectory[row].tolist(),
                    "macro_relmse": float(per_trajectory[row].mean()),
                }
            )
        print(
            f"rollout {start + len(trajectory_ids)}/"
            f"{len(contract.test_indices)} complete",
            flush=True,
        )

    per_field = global_ratio_sum / (len(contract.test_indices) * horizon)
    names = FIELD_NAMES[contract.spec.name]
    return {
        "method": args.variant,
        "dataset": contract.spec.name,
        "test_trajectory_count": int(len(contract.test_indices)),
        "test_trajectory_indices": contract.test_indices.tolist(),
        "forecast_horizon": horizon,
        "decoded_field_count": field_count,
        "field_names": list(names),
        "per_field_elementary_relmse_sum": global_ratio_sum.tolist(),
        "elementary_values_per_field": int(
            len(contract.test_indices) * horizon
        ),
        "per_field_relmse": per_field.tolist(),
        "per_field_relmse_by_name": {
            name: float(per_field[index])
            for index, name in enumerate(names)
        },
        "macro_relmse": float(per_field.mean()),
        "aggregation": (
            "reduce spatial nodes first; arithmetic mean over every held-out "
            "trajectory and forecast step; macro is the arithmetic mean over fields"
        ),
        "initial_frames_provided": 1,
        "flow_evaluation": (
            None
            if args.variant == "gmr-gmus"
            else {
                "mode": args.flow_eval_mode,
                "samples": sample_count,
                "seed": args.flow_seed,
            }
        ),
        "per_trajectory": trajectory_records,
    }


def save_results(
    args: argparse.Namespace,
    results: dict[str, Any],
    manifest: dict[str, Any],
    paths: RunPaths,
    autoencoder_checkpoint: dict[str, Any],
    temporal_checkpoint: dict[str, Any],
) -> Path:
    results.update(
        {
            "checkpoints": {
                "autoencoder": str(paths.autoencoder_best),
                "autoencoder_step": int(autoencoder_checkpoint["step"]),
                "autoencoder_validation_metric": float(
                    autoencoder_checkpoint["validation_metric"]
                ),
                "autoencoder_validation_metric_name": autoencoder_checkpoint[
                    "validation_metric_name"
                ],
                "temporal": str(paths.temporal_best),
                "temporal_step": int(temporal_checkpoint["step"]),
                "temporal_validation_metric": float(
                    temporal_checkpoint["validation_metric"]
                ),
                "temporal_validation_metric_name": temporal_checkpoint[
                    "validation_metric_name"
                ],
            },
            "split_file": str(Path(manifest["data_contract"]["cache_usage"]["split"])),
            "provenance": manifest,
        }
    )
    results["created_utc"] = datetime.now(timezone.utc).isoformat()
    suffix = f"_{args.result_tag}" if args.result_tag else ""
    path = paths.results_dir / f"{args.variant}_{args.dataset}{suffix}.json"
    if path.exists():
        raise FileExistsError(path)
    _atomic_json(path, results)
    return path


def shape_smoke(variant: str, device: torch.device) -> None:
    torch.manual_seed(42)
    node_count = 12
    pivotal_count = 4
    field_count = 2
    condition_dim = 3
    sender = torch.arange(node_count, dtype=torch.long)
    receiver = (sender + 1) % node_count
    edge_index = torch.stack(
        (
            torch.cat((sender, receiver)),
            torch.cat((receiver, sender)),
        )
    )
    edge_features = torch.randn(edge_index.shape[1], 3)
    if variant == "gmr-gmus":
        reduction_indices = torch.arange(pivotal_count)[:, None]
        reduction_weights = torch.ones(pivotal_count, 1)
        residual = False
    else:
        reduction_indices = torch.stack(
            (
                torch.arange(pivotal_count),
                (torch.arange(pivotal_count) + 1) % node_count,
                (torch.arange(pivotal_count) + 2) % node_count,
            ),
            dim=1,
        )
        reduction_weights = torch.full((pivotal_count, 3), 1.0 / 3.0)
        residual = True
    expansion_indices = (
        torch.arange(node_count)[:, None] % pivotal_count
    )
    expansion_weights = torch.ones(node_count, 1)
    autoencoder = MeshReducedAutoencoder(
        field_count,
        field_count,
        edge_features,
        edge_index,
        reduction_indices,
        reduction_weights,
        expansion_indices,
        expansion_weights,
        pivotal_count=pivotal_count,
        width=128,
        latent_per_pivot=4,
        processor_blocks=3,
        residual_layernorm=residual,
    ).to(device)
    node_features = torch.randn(
        2, node_count, field_count, device=device
    )
    target = torch.randn(2, node_count, field_count, device=device)
    reconstruction = autoencoder(node_features)
    reconstruction_loss = F.mse_loss(reconstruction, target)
    reconstruction_loss.backward()
    if reconstruction.shape != target.shape or not torch.isfinite(
        reconstruction_loss
    ):
        raise AssertionError("autoencoder smoke failed")

    latent_dim = pivotal_count * 4
    horizon = 3
    conditions = torch.randn(2, horizon, condition_dim, device=device)
    latent_sequence = torch.randn(
        2, horizon + 1, latent_dim, device=device
    )
    if variant == "gmr-gmus":
        temporal = GMRResidualTemporalTransformer(
            latent_dim, condition_dim, heads=4
        ).to(device)
        prediction = temporal(latent_sequence[:, 0], conditions)
        temporal_loss = F.mse_loss(prediction, latent_sequence[:, 1:])
        temporal_loss.backward()
        generated_shape = tuple(prediction.shape)
    else:
        temporal = PbGMRRealNVPTemporal(
            latent_dim, condition_dim, heads=4, coupling_layers=2
        ).to(device)
        temporal_loss = temporal(
            latent_sequence,
            conditions,
            temporal_conditioning=True,
        )
        temporal_loss.backward()
        temporal.eval()
        with torch.no_grad():
            generated = temporal.rollout(
                latent_sequence[:, 0],
                conditions,
                sample=False,
                temporal_conditioning=True,
            )
        generated_shape = tuple(generated.shape)
    if not torch.isfinite(temporal_loss):
        raise AssertionError("temporal smoke failed")
    print(
        json.dumps(
            {
                "shape_smoke": "passed",
                "variant": variant,
                "device": str(device),
                "reconstruction_shape": list(reconstruction.shape),
                "latent_rollout_shape": list(generated_shape),
                "autoencoder_parameters": parameter_count(autoencoder),
                "temporal_parameters": parameter_count(temporal),
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.gpu)
    seed_everything(device)
    if args.shape_smoke:
        shape_smoke(args.variant, device)
        return

    contract = load_contract(args.dataset)
    geometry = build_geometry(
        contract,
        args.variant,
        args.pivotal_count,
        args.gmr_interpolation_k,
    )
    configuration = model_configuration(contract, geometry, args.variant)
    autoencoder = build_autoencoder(
        contract, geometry, args.variant, device
    )
    temporal_build_device = (
        torch.device("cpu") if args.stage == "autoencoder" else device
    )
    temporal = build_temporal(
        contract,
        autoencoder.latent_dim,
        args.variant,
        temporal_build_device,
    )
    paths = run_paths(args)
    manifest = provenance(
        args,
        contract,
        geometry,
        configuration,
        autoencoder,
        temporal,
        device,
    )

    if paths.manifest.exists():
        existing = json.loads(paths.manifest.read_text())
        identity_keys = (
            "dataset",
            "trajectory_split_seed",
            "train_indices",
            "validation_indices",
            "test_indices",
            "forecast_horizon",
            "decoded_field_count",
            "field_mean",
            "field_std",
            "condition_mean",
            "condition_std",
        )
        existing_data = existing.get("data_contract", {})
        current_data = manifest.get("data_contract", {})
        data_mismatch = any(
            existing_data.get(key) != current_data.get(key)
            for key in identity_keys
        )
        if existing.get("architecture") != manifest.get("architecture") or data_mismatch:
            raise ValueError(
                f"{paths.manifest}: existing run is incompatible; choose "
                "a different --run-name"
            )
        _atomic_json(paths.manifest, manifest)
    else:
        _atomic_json(paths.manifest, manifest)

    if args.stage == "autoencoder":
        # The full Pb temporal model is large and is needed here only to record
        # its exact parameter count in provenance.
        del temporal
        train_autoencoder(
            args,
            autoencoder,
            contract,
            configuration,
            manifest,
            paths,
            device,
        )
        print(
            f"autoencoder stage complete: {paths.autoencoder_best}", flush=True
        )
        return

    if args.stage == "temporal":
        train_temporal(
            args,
            temporal,
            autoencoder,
            contract,
            configuration,
            manifest,
            paths,
            device,
        )
        if args.skip_rollout:
            print(
                f"temporal stage complete: {paths.temporal_best}", flush=True
            )
            return

    autoencoder_checkpoint = _load_model_checkpoint(
        paths.autoencoder_best, autoencoder, configuration, device
    )
    temporal_checkpoint = _load_model_checkpoint(
        paths.temporal_best, temporal, configuration, device
    )
    results = evaluate_rollout(
        args, autoencoder, temporal, contract, paths, device
    )
    result_path = save_results(
        args,
        results,
        manifest,
        paths,
        autoencoder_checkpoint,
        temporal_checkpoint,
    )
    print(
        json.dumps(
            {
                "method": results["method"],
                "dataset": results["dataset"],
                "macro_relmse": results["macro_relmse"],
                "result": str(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
