"""Shared data contract for directly comparable baseline reruns.

All methods use the same trajectory-level seed-42 splits, forecast horizons,
fields, conditioning variables, and decoded-space RelMSE aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from scipy.spatial import Delaunay


ROOT = Path("./")
SEA_ROOT = ROOT / "SEA"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_dir: Path
    train_fraction: float
    val_fraction: float
    horizon: int
    batch_size: int
    rollout_batch_size: int

    @property
    def fields_path(self) -> Path:
        return self.data_dir / "field_data.npy"

    @property
    def coordinates_path(self) -> Path:
        return self.data_dir / "coordinates.npy"

    @property
    def input_path(self) -> Path:
        return self.data_dir / "input_data.npy"

    @property
    def edge_path(self) -> Path:
        return self.data_dir / "edge_data.npz"


DATASETS = {
    "cylinder": DatasetSpec(
        "cylinder",
        SEA_ROOT / "data_old/CF/all_data",
        0.6,
        0.2,
        400,
        1,
        2,
    ),
    "multiphase": DatasetSpec(
        "multiphase",
        SEA_ROOT / "data/MP/all_data",
        0.6,
        0.2,
        199,
        1,
        1,
    ),
    "we1": DatasetSpec(
        "we1",
        SEA_ROOT / "data_computed/WE1",
        0.889,
        0.056,
        248,
        16,
        16,
    ),
    "e1": DatasetSpec(
        "e1",
        SEA_ROOT / "data_computed/E1",
        0.889,
        0.056,
        248,
        8,
        16,
    ),
}


def trajectory_split(
    trajectory_count: int,
    train_fraction: float,
    val_fraction: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the repository's trajectory split exactly."""
    indices = np.arange(trajectory_count)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    train_length = int(np.round(trajectory_count * train_fraction))
    val_length = int(np.round(trajectory_count * val_fraction))
    return (
        indices[:train_length],
        indices[train_length : train_length + val_length],
        indices[train_length + val_length :],
    )


def load_coordinates(spec: DatasetSpec) -> np.ndarray:
    coordinates = np.asarray(np.load(spec.coordinates_path), dtype=np.float32)
    if coordinates.ndim != 2:
        raise ValueError(f"{spec.name}: expected 2-D coordinates, got {coordinates.shape}")
    if coordinates.shape[0] in (2, 3):
        coordinates = coordinates.T
    if coordinates.shape[1] not in (2, 3):
        raise ValueError(f"{spec.name}: invalid coordinate shape {coordinates.shape}")
    return np.ascontiguousarray(coordinates)


def _bidirectional_unique_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.shape[0] != 2:
        edges = edges.T
    directed = np.concatenate([edges, edges[::-1]], axis=1)
    directed = np.unique(directed.T, axis=0).T
    keep = directed[0] != directed[1]
    return np.ascontiguousarray(directed[:, keep])


def load_edges(spec: DatasetSpec, coordinates: np.ndarray) -> np.ndarray:
    if spec.edge_path.exists():
        with np.load(spec.edge_path) as archive:
            return _bidirectional_unique_edges(archive["edge_index"])

    # The multiphase SEA pipeline constructs Delaunay edges from the fixed mesh.
    triangles = Delaunay(coordinates[:, :2]).simplices
    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        axis=0,
    )
    return _bidirectional_unique_edges(edges.T)


def edge_features(coordinates: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    sender, receiver = edge_index
    relative = coordinates[sender] - coordinates[receiver]
    length = np.linalg.norm(relative, axis=1, keepdims=True)
    features = np.concatenate([relative, length], axis=1).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return np.ascontiguousarray((features - mean) / np.maximum(std, 1e-6))


def cylinder_conditioning() -> np.ndarray:
    """Exact six-feature recipe recorded in the paper evaluation manifest."""
    reynolds = torch.from_numpy(np.linspace(300, 1000, 101)).float().reshape(-1, 1)
    inverse_reynolds = 1.0 / reynolds
    columns = []
    for power in (1, 2, 3):
        re_power = reynolds**power
        inverse_power = inverse_reynolds**power
        columns.extend(
            [re_power / re_power.max(), inverse_power / inverse_power.max()]
        )
    return torch.cat(columns, dim=1).contiguous().numpy()


def load_conditioning(spec: DatasetSpec, frame_count: int) -> np.ndarray:
    if spec.name == "cylinder":
        return cylinder_conditioning()
    conditioning = np.asarray(np.load(spec.input_path), dtype=np.float32)
    if conditioning.ndim == 1:
        conditioning = conditioning[:, None]
    if conditioning.ndim not in (2, 3):
        raise ValueError(
            f"{spec.name}: expected static or temporal conditioning, got {conditioning.shape}"
        )
    if conditioning.ndim == 3 and conditioning.shape[1] != frame_count:
        raise ValueError(
            f"{spec.name}: conditioning length {conditioning.shape[1]} != {frame_count}"
        )
    return np.ascontiguousarray(conditioning)


def condition_at(
    conditioning: np.ndarray, trajectory_indices: np.ndarray, time_indices: np.ndarray
) -> np.ndarray:
    if conditioning.ndim == 2:
        return conditioning[trajectory_indices]
    return conditioning[trajectory_indices, time_indices]


def iter_trajectory_chunks(
    fields: np.ndarray, indices: np.ndarray, chunk_size: int = 16
) -> Iterator[np.ndarray]:
    for start in range(0, len(indices), chunk_size):
        yield np.asarray(fields[indices[start : start + chunk_size]], dtype=np.float32)


def field_and_delta_stats(
    fields: np.ndarray, train_indices: np.ndarray
) -> dict[str, np.ndarray]:
    field_sum = np.zeros(fields.shape[-1], dtype=np.float64)
    field_sq_sum = np.zeros(fields.shape[-1], dtype=np.float64)
    field_count = 0
    delta_sum = np.zeros(fields.shape[-1], dtype=np.float64)
    delta_sq_sum = np.zeros(fields.shape[-1], dtype=np.float64)
    delta_count = 0

    for chunk in iter_trajectory_chunks(fields, train_indices):
        flat = chunk.reshape(-1, chunk.shape[-1]).astype(np.float64)
        field_sum += flat.sum(axis=0)
        field_sq_sum += np.square(flat).sum(axis=0)
        field_count += flat.shape[0]

        delta = (chunk[:, 1:] - chunk[:, :-1]).reshape(-1, chunk.shape[-1])
        delta = delta.astype(np.float64)
        delta_sum += delta.sum(axis=0)
        delta_sq_sum += np.square(delta).sum(axis=0)
        delta_count += delta.shape[0]

    field_mean = field_sum / field_count
    field_var = np.maximum(field_sq_sum / field_count - np.square(field_mean), 1e-12)
    field_std = np.sqrt(field_var)
    delta_mean_raw = delta_sum / delta_count
    delta_var_raw = np.maximum(
        delta_sq_sum / delta_count - np.square(delta_mean_raw), 1e-12
    )

    # Delta statistics in normalized-state coordinates.
    delta_mean = delta_mean_raw / field_std
    delta_std = np.sqrt(delta_var_raw) / field_std
    return {
        "field_mean": field_mean.astype(np.float32),
        "field_std": field_std.astype(np.float32),
        "delta_mean": delta_mean.astype(np.float32),
        "delta_std": np.maximum(delta_std, 1e-6).astype(np.float32),
    }


def sample_pairs(
    rng: np.random.RandomState,
    trajectory_indices: np.ndarray,
    frame_count: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    trajectories = rng.choice(trajectory_indices, size=batch_size, replace=True)
    times = rng.randint(0, frame_count - 1, size=batch_size)
    return trajectories, times


def write_split_audit(
    path: Path,
    spec: DatasetSpec,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": spec.name,
        "seed": 42,
        "unit": "trajectory",
        "train_fraction": spec.train_fraction,
        "validation_fraction": spec.val_fraction,
        "forecast_horizon": spec.horizon,
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "test_indices": test.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
