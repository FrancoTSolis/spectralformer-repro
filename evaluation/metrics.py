"""Metric aggregation for checkpoint-only rollout evaluation.

All public metrics are computed in physical field space.  The elementary
sample is one (trajectory, forecast step, field) tuple; spatial nodes are
reduced first.  Aggregate values are arithmetic means over trajectories and
forecast steps, followed by an arithmetic mean over fields for the macro row.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


METRIC_NAMES = ("relative_squared_l2", "relative_l2", "raw_rmse")


def metric_arrays(
    prediction: np.ndarray, truth: np.ndarray, epsilon: float = 1.0e-12
) -> Dict[str, np.ndarray]:
    """Return [trajectory, horizon, field] metric arrays.

    ``prediction`` and ``truth`` must be shaped
    [trajectory, horizon, spatial..., field].
    """

    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape:
        raise ValueError(
            f"prediction/truth shape mismatch: {prediction.shape} != {truth.shape}"
        )
    if prediction.ndim < 4:
        raise ValueError(
            "expected [trajectory, horizon, spatial..., field], "
            f"received {prediction.shape}"
        )
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("prediction and truth must contain only finite values")

    spatial_axes = tuple(range(2, prediction.ndim - 1))
    spatial_count = int(np.prod(prediction.shape[2:-1]))
    squared_error = np.sum((prediction - truth) ** 2, axis=spatial_axes)
    squared_truth = np.sum(truth**2, axis=spatial_axes)
    denominator = np.maximum(squared_truth, epsilon)

    relative_squared_l2 = squared_error / denominator
    return {
        "relative_squared_l2": relative_squared_l2,
        "relative_l2": np.sqrt(relative_squared_l2),
        "raw_rmse": np.sqrt(squared_error / spatial_count),
    }


def _ci(values: np.ndarray) -> List[float]:
    low, high = np.percentile(values, [2.5, 97.5], axis=0)
    return [float(low), float(high)]


def _value_ci(value: float, ci95: Sequence[float]) -> Dict[str, object]:
    return {"value": float(value), "ci95": [float(ci95[0]), float(ci95[1])]}


def _bootstrap_indices(
    trajectory_count: int, samples: int, seed: int
) -> Iterable[np.ndarray]:
    if trajectory_count <= 0:
        raise ValueError("at least one trajectory is required")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        yield rng.integers(0, trajectory_count, size=trajectory_count)


def summarize_metrics(
    arrays: Mapping[str, np.ndarray],
    field_names: Sequence[str],
    trajectory_ids: Sequence[int],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
    trajectory_wall_seconds: Sequence[float] | None = None,
) -> Dict[str, object]:
    """Summarize elementary metrics and trajectory-bootstrap confidence intervals."""

    if set(arrays) != set(METRIC_NAMES):
        raise ValueError(f"expected metrics {METRIC_NAMES}, received {tuple(arrays)}")

    first = np.asarray(arrays[METRIC_NAMES[0]], dtype=np.float64)
    if first.ndim != 3:
        raise ValueError(f"metric arrays must be [trajectory,horizon,field], got {first.shape}")
    trajectory_count, horizon_count, field_count = first.shape
    if field_count != len(field_names):
        raise ValueError("field-name count does not match metric field dimension")
    if trajectory_count != len(trajectory_ids):
        raise ValueError("trajectory-id count does not match metric trajectory dimension")

    normalized: Dict[str, np.ndarray] = {}
    for metric in METRIC_NAMES:
        value = np.asarray(arrays[metric], dtype=np.float64)
        if value.shape != first.shape:
            raise ValueError(f"{metric} has inconsistent shape {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{metric} contains non-finite values")
        normalized[metric] = value

    aggregate_boot = {
        metric: np.empty((bootstrap_samples, field_count), dtype=np.float64)
        for metric in METRIC_NAMES
    }
    final_boot = {
        metric: np.empty((bootstrap_samples, field_count), dtype=np.float64)
        for metric in METRIC_NAMES
    }
    horizon_boot = {
        metric: np.empty(
            (bootstrap_samples, horizon_count, field_count), dtype=np.float64
        )
        for metric in METRIC_NAMES
    }

    for sample_index, indices in enumerate(
        _bootstrap_indices(trajectory_count, bootstrap_samples, bootstrap_seed)
    ):
        for metric, values in normalized.items():
            sampled = values[indices]
            aggregate_boot[metric][sample_index] = sampled.mean(axis=(0, 1))
            final_boot[metric][sample_index] = sampled[:, -1, :].mean(axis=0)
            horizon_boot[metric][sample_index] = sampled.mean(axis=0)

    aggregate: Dict[str, object] = {"per_field": {}, "macro": {}}
    final_step: Dict[str, object] = {"per_field": {}, "macro": {}}
    for field_index, field_name in enumerate(field_names):
        aggregate["per_field"][field_name] = {}
        final_step["per_field"][field_name] = {}
        for metric, values in normalized.items():
            aggregate["per_field"][field_name][metric] = _value_ci(
                values[:, :, field_index].mean(),
                _ci(aggregate_boot[metric][:, field_index]),
            )
            final_step["per_field"][field_name][metric] = _value_ci(
                values[:, -1, field_index].mean(),
                _ci(final_boot[metric][:, field_index]),
            )

    for metric, values in normalized.items():
        aggregate_macro_boot = aggregate_boot[metric].mean(axis=1)
        final_macro_boot = final_boot[metric].mean(axis=1)
        aggregate["macro"][metric] = _value_ci(
            values.mean(), _ci(aggregate_macro_boot)
        )
        final_step["macro"][metric] = _value_ci(
            values[:, -1, :].mean(), _ci(final_macro_boot)
        )

    per_horizon: List[Dict[str, object]] = []
    for horizon_index in range(horizon_count):
        row: Dict[str, object] = {
            "horizon": horizon_index + 1,
            "per_field": {},
            "macro": {},
        }
        for field_index, field_name in enumerate(field_names):
            row["per_field"][field_name] = {}
            for metric, values in normalized.items():
                row["per_field"][field_name][metric] = _value_ci(
                    values[:, horizon_index, field_index].mean(),
                    _ci(horizon_boot[metric][:, horizon_index, field_index]),
                )
        for metric, values in normalized.items():
            row["macro"][metric] = _value_ci(
                values[:, horizon_index, :].mean(),
                _ci(horizon_boot[metric][:, horizon_index, :].mean(axis=1)),
            )
        per_horizon.append(row)

    per_trajectory: List[Dict[str, object]] = []
    if trajectory_wall_seconds is None:
        trajectory_wall_seconds = [float("nan")] * trajectory_count
    if len(trajectory_wall_seconds) != trajectory_count:
        raise ValueError("trajectory timing count does not match trajectory count")

    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        row = {
            "trajectory_id": int(trajectory_id),
            "inference_wall_seconds": float(trajectory_wall_seconds[trajectory_index]),
            "aggregate": {"per_field": {}, "macro": {}},
            "final_step": {"per_field": {}, "macro": {}},
        }
        for field_index, field_name in enumerate(field_names):
            row["aggregate"]["per_field"][field_name] = {
                metric: float(values[trajectory_index, :, field_index].mean())
                for metric, values in normalized.items()
            }
            row["final_step"]["per_field"][field_name] = {
                metric: float(values[trajectory_index, -1, field_index])
                for metric, values in normalized.items()
            }
        row["aggregate"]["macro"] = {
            metric: float(values[trajectory_index].mean())
            for metric, values in normalized.items()
        }
        row["final_step"]["macro"] = {
            metric: float(values[trajectory_index, -1].mean())
            for metric, values in normalized.items()
        }
        per_trajectory.append(row)

    return {
        "aggregation_rule": (
            "reduce spatial nodes first; arithmetic mean over every held-out "
            "trajectory and forecast step; macro is the arithmetic mean over fields"
        ),
        "bootstrap": {
            "unit": "trajectory",
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "interval": "percentile_95",
        },
        "trajectory_count": trajectory_count,
        "horizon_count": horizon_count,
        "aggregate": aggregate,
        "final_step": final_step,
        "per_horizon": per_horizon,
        "per_trajectory": per_trajectory,
    }

