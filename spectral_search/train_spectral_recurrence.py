"""Validation-only search for full-channel spectral GraphSpectralFormer variants.

E1 uses every FFT channel on its uniform one-dimensional grid. WE1 uses every
eigenvector of the full graph Laplacian and a condition-dependent second-order
modal recurrence. Both models predict complete autoregressive rollouts from
one initial frame and select checkpoints only on validation trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "baseline-reruns"))

from common import DATASETS, trajectory_split  # noqa: E402


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @property
    def output(self) -> nn.Linear:
        return self.net[-1]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class FFTAdvectionRecurrence(nn.Module):
    """Conditioned stable complex multiplier over all rFFT channels."""

    def __init__(
        self,
        node_count: int,
        condition_dim: int,
        hidden_dim: int,
        dense_correction: bool,
        signed_gain: bool,
    ) -> None:
        super().__init__()
        self.node_count = node_count
        self.mode_count = node_count // 2 + 1
        self.conditioner = MLP(
            condition_dim, hidden_dim, 4 * self.mode_count
        )
        nn.init.zeros_(self.conditioner.output.weight)
        nn.init.zeros_(self.conditioner.output.bias)
        if not signed_gain:
            with torch.no_grad():
                self.conditioner.output.bias[
                    self.mode_count : 2 * self.mode_count
                ].fill_(-5.0)
        self.dense_correction = dense_correction
        self.signed_gain = signed_gain
        if dense_correction:
            self.correction = MLP(
                2 * self.mode_count + condition_dim,
                hidden_dim,
                2 * self.mode_count,
            )
            nn.init.zeros_(self.correction.output.weight)
            nn.init.zeros_(self.correction.output.bias)
            self.correction_gate = nn.Parameter(torch.tensor(-5.0))

    def multiplier_and_bias(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.conditioner(condition).reshape(
            condition.shape[0], 4, self.mode_count
        )
        phase = math.pi * torch.tanh(raw[:, 0])
        if self.signed_gain:
            gain = torch.exp(0.01 * torch.tanh(raw[:, 1]))
        else:
            gain = torch.exp(
                -0.001 * torch.nn.functional.softplus(raw[:, 1])
            )
        multiplier = torch.polar(gain, phase)
        bias = 1e-3 * torch.complex(
            torch.tanh(raw[:, 2]), torch.tanh(raw[:, 3])
        )
        return multiplier, bias

    def step(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        coefficients = torch.fft.rfft(state, dim=-1, norm="ortho")
        if self.dense_correction:
            real_view = torch.cat(
                (coefficients.real, coefficients.imag, condition), dim=-1
            )
            correction = self.correction(real_view)
            correction = torch.complex(
                correction[:, : self.mode_count],
                correction[:, self.mode_count :],
            )
            coefficients = coefficients + torch.sigmoid(
                self.correction_gate
            ) * correction
        multiplier, bias = self.multiplier_and_bias(condition)
        next_coefficients = coefficients * multiplier + bias
        return torch.fft.irfft(
            next_coefficients,
            n=self.node_count,
            dim=-1,
            norm="ortho",
        )

    def rollout(
        self, initial: torch.Tensor, condition: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        state = initial
        predictions = []
        for _ in range(horizon):
            state = self.step(state, condition)
            predictions.append(state)
        return torch.stack(predictions, dim=1)


class GraphWaveRecurrence(nn.Module):
    """Full-Laplacian second-order recurrence over all graph modes."""

    def __init__(
        self,
        basis: torch.Tensor,
        condition_dim: int,
        hidden_dim: int,
        stable: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("basis", basis.float())
        self.mode_count = basis.shape[1]
        self.stable = stable
        self.conditioner = MLP(
            condition_dim, hidden_dim, 4 * self.mode_count
        )
        nn.init.zeros_(self.conditioner.output.weight)
        nn.init.zeros_(self.conditioner.output.bias)
        alpha_bias = -4.0 if stable else math.atanh(0.99)
        beta_bias = -5.0 if stable else math.log(0.99 / (1.05 - 0.99))
        with torch.no_grad():
            self.conditioner.output.bias[: self.mode_count].fill_(alpha_bias)
            self.conditioner.output.bias[
                self.mode_count : 2 * self.mode_count
            ].fill_(beta_bias)

    def parameters_for(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.conditioner(condition).reshape(
            condition.shape[0], 4, self.mode_count
        )
        if self.stable:
            decay = torch.exp(
                -0.01 * torch.nn.functional.softplus(raw[:, 1])
            )
            omega = math.pi * torch.sigmoid(raw[:, 0])
            alpha = 2.0 * decay * torch.cos(omega)
            beta = -torch.square(decay)
        else:
            alpha = 2.0 * torch.tanh(raw[:, 0])
            beta = -1.05 * torch.sigmoid(raw[:, 1])
        first_gain = 1.0 + 0.1 * torch.tanh(raw[:, 2])
        bias = 1e-3 * torch.tanh(raw[:, 3])
        return alpha, beta, first_gain, bias

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self.basis

    def decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        return coefficients @ self.basis.T

    def rollout(
        self, initial: torch.Tensor, condition: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        alpha, beta, first_gain, bias = self.parameters_for(condition)
        previous = self.encode(initial)
        current = first_gain * previous + bias
        predictions = [self.decode(current)]
        for _ in range(1, horizon):
            following = alpha * current + beta * previous + bias
            previous, current = current, following
            predictions.append(self.decode(current))
        return torch.stack(predictions, dim=1)


class SpectralConv1d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        scale = 1.0 / width
        self.weight_real = nn.Parameter(
            scale * torch.randn(width, width, modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(width, width, modes)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        coefficients = torch.fft.rfft(value, dim=-1, norm="ortho")
        used_modes = min(self.modes, coefficients.shape[-1])
        output = torch.zeros(
            value.shape[0],
            self.width,
            coefficients.shape[-1],
            dtype=coefficients.dtype,
            device=value.device,
        )
        weight = torch.complex(
            self.weight_real[..., :used_modes],
            self.weight_imag[..., :used_modes],
        )
        output[..., :used_modes] = torch.einsum(
            "bim,iom->bom", coefficients[..., :used_modes], weight
        )
        return torch.fft.irfft(
            output, n=value.shape[-1], dim=-1, norm="ortho"
        )


class FNO1DRecurrence(nn.Module):
    """Nonlinear full-channel Fourier neural recurrence for E1."""

    def __init__(
        self,
        coordinates: torch.Tensor,
        condition_dim: int,
        width: int,
        layers: int,
        modes: int,
    ) -> None:
        super().__init__()
        coordinate = coordinates.float()
        coordinate = 2.0 * (
            (coordinate - coordinate.min())
            / torch.clamp(coordinate.max() - coordinate.min(), min=1e-8)
        ) - 1.0
        self.register_buffer("coordinate", coordinate[None, None])
        self.lift = nn.Conv1d(2 + condition_dim, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, modes) for _ in range(layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(layers)]
        )
        self.norms = nn.ModuleList(
            [nn.GroupNorm(1, width) for _ in range(layers)]
        )
        self.project = nn.Sequential(
            nn.Conv1d(width, 2 * width, 1),
            nn.GELU(),
            nn.Conv1d(2 * width, 1, 1),
        )
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def step(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        coordinate = self.coordinate.expand(state.shape[0], -1, -1)
        condition_nodes = condition[:, :, None].expand(
            -1, -1, state.shape[-1]
        )
        hidden = self.lift(
            torch.cat((state[:, None], coordinate, condition_nodes), dim=1)
        )
        for spectral, pointwise, norm in zip(
            self.spectral, self.pointwise, self.norms
        ):
            update = spectral(hidden) + pointwise(hidden)
            hidden = hidden + torch.nn.functional.gelu(norm(update))
        delta = self.project(hidden)[:, 0]
        return state + delta

    def rollout(
        self, initial: torch.Tensor, condition: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        state = initial
        predictions = []
        for _ in range(horizon):
            state = self.step(state, condition)
            predictions.append(state)
        return torch.stack(predictions, dim=1)


class BurgersSpectralRecurrence(nn.Module):
    """Learned de-aliased pseudo-spectral integrator for E1 Burgers dynamics."""

    def __init__(
        self,
        coordinates: torch.Tensor,
        dt_init: float,
        nu_init: float,
        filter_init: float,
    ) -> None:
        super().__init__()
        node_count = coordinates.numel()
        dx = float((coordinates[1] - coordinates[0]).abs())
        wave_number = 2.0 * math.pi * torch.fft.fftfreq(
            node_count, d=dx
        )
        self.register_buffer("wave_number", wave_number.float())
        cutoff = (wave_number.abs() <= (2.0 / 3.0) * wave_number.abs().max())
        self.register_buffer("dealias_mask", cutoff)
        normalized_frequency = wave_number.abs() / torch.clamp(
            wave_number.abs().max(), min=1e-8
        )
        self.register_buffer("normalized_frequency", normalized_frequency.float())
        self.raw_dt = nn.Parameter(
            torch.tensor(math.log(dt_init / (0.03 - dt_init)))
        )
        self.raw_nu = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(nu_init, 1e-8) / 0.01)))
        )
        self.raw_filter = nn.Parameter(
            torch.tensor(
                math.log(math.expm1(max(filter_init, 1e-8) / 0.1))
            )
        )

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = 0.03 * torch.sigmoid(self.raw_dt)
        viscosity = 0.01 * torch.nn.functional.softplus(self.raw_nu)
        filtering = 0.1 * torch.nn.functional.softplus(self.raw_filter)
        return dt, viscosity, filtering

    def rhs(self, state: torch.Tensor, viscosity: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft(state, dim=-1, norm="ortho")
        derivative = torch.fft.ifft(
            1j * self.wave_number * spectrum, dim=-1, norm="ortho"
        ).real
        diffusion = torch.fft.ifft(
            -torch.square(self.wave_number) * spectrum,
            dim=-1,
            norm="ortho",
        ).real
        nonlinear = -state * derivative
        nonlinear_spectrum = torch.fft.fft(
            nonlinear, dim=-1, norm="ortho"
        )
        nonlinear = torch.fft.ifft(
            nonlinear_spectrum * self.dealias_mask,
            dim=-1,
            norm="ortho",
        ).real
        return nonlinear + viscosity * diffusion

    def step(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        del condition
        dt, viscosity, filtering = self.coefficients()
        k1 = self.rhs(state, viscosity)
        k2 = self.rhs(state + 0.5 * dt * k1, viscosity)
        k3 = self.rhs(state + 0.5 * dt * k2, viscosity)
        k4 = self.rhs(state + dt * k3, viscosity)
        following = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        spectrum = torch.fft.fft(following, dim=-1, norm="ortho")
        spectral_filter = torch.exp(
            -filtering * torch.pow(self.normalized_frequency, 8)
        )
        return torch.fft.ifft(
            spectrum * spectral_filter, dim=-1, norm="ortho"
        ).real

    def rollout(
        self, initial: torch.Tensor, condition: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        state = initial
        predictions = []
        for _ in range(horizon):
            state = self.step(state, condition)
            predictions.append(state)
        return torch.stack(predictions, dim=1)


def graph_laplacian_basis(
    node_count: int, edge_index: np.ndarray, edge_weight: np.ndarray
) -> torch.Tensor:
    adjacency = torch.zeros(node_count, node_count, dtype=torch.float64)
    sender = torch.from_numpy(edge_index[0]).long()
    receiver = torch.from_numpy(edge_index[1]).long()
    weights = torch.from_numpy(edge_weight).double()
    adjacency.index_put_((sender, receiver), weights, accumulate=True)
    adjacency = 0.5 * (adjacency + adjacency.T)
    degree = adjacency.sum(dim=1)
    inverse_sqrt = torch.where(
        degree > 0, degree.rsqrt(), torch.zeros_like(degree)
    )
    laplacian = torch.eye(node_count, dtype=torch.float64) - (
        inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    )
    _, basis = torch.linalg.eigh(laplacian)
    return basis.float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("e1", "we1"), required=True)
    parser.add_argument(
        "--model",
        choices=(
            "fft-ar1",
            "fft-dense",
            "fft-gain",
            "fft-gain-dense",
            "graph-wave",
            "graph-wave-stored",
            "graph-wave-aux",
            "graph-wave-stable",
            "graph-wave-aux-stable",
            "fno1d",
            "burgers-spectral",
        ),
        required=True,
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-horizon", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--dt-init", type=float, default=0.016)
    parser.add_argument("--nu-init", type=float, default=1e-4)
    parser.add_argument("--filter-init", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("mse", "relmse"), default="mse")
    parser.add_argument("--val-interval", type=int, default=250)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--finetune-from-best",
        action="store_true",
        help="Initialize model weights from this run's validation-best checkpoint",
    )
    return parser.parse_args()


def load_data(dataset: str, model_name: str):
    spec = DATASETS[dataset]
    fields = np.load(spec.fields_path, mmap_mode="r")
    conditions = np.asarray(np.load(spec.input_path), dtype=np.float32)
    if conditions.ndim == 3:
        conditions = conditions[:, 0]
    if dataset == "we1" and model_name in (
        "graph-wave-aux",
        "graph-wave-aux-stable",
    ):
        profiles = np.load(HERE / "we1_initial_profiles.npy").astype(np.float32)
        if profiles.shape[0] != conditions.shape[0]:
            raise ValueError("WE1 auxiliary profile count mismatch")
        conditions = np.concatenate((conditions, profiles), axis=1)
    train, validation, test = trajectory_split(
        fields.shape[0], spec.train_fraction, spec.val_fraction
    )
    with np.load(ROOT / "baseline-reruns/cache" / f"{dataset}_stats.npz") as ar:
        field_mean = ar["field_mean"].astype(np.float32)
        field_std = ar["field_std"].astype(np.float32)
    train_conditions = conditions[train].astype(np.float64)
    condition_mean = train_conditions.mean(axis=0).astype(np.float32)
    condition_std = np.maximum(
        train_conditions.std(axis=0), 1e-6
    ).astype(np.float32)
    return (
        spec,
        fields,
        conditions,
        train,
        validation,
        test,
        field_mean,
        field_std,
        condition_mean,
        condition_std,
    )


def build_model(
    args: argparse.Namespace,
    node_count: int,
    condition_dim: int,
) -> nn.Module:
    if args.model.startswith("fft"):
        return FFTAdvectionRecurrence(
            node_count,
            condition_dim,
            args.hidden_dim,
            dense_correction=args.model in ("fft-dense", "fft-gain-dense"),
            signed_gain=args.model in ("fft-gain", "fft-gain-dense"),
        )
    if args.model == "fno1d":
        coordinates = np.load(DATASETS["e1"].coordinates_path)
        if coordinates.shape[0] in (2, 3):
            coordinates = coordinates[0]
        else:
            coordinates = coordinates[:, 0]
        return FNO1DRecurrence(
            torch.from_numpy(np.asarray(coordinates, dtype=np.float32)),
            condition_dim,
            args.width,
            args.layers,
            args.modes,
        )
    if args.model == "burgers-spectral":
        coordinates = np.load(DATASETS["e1"].coordinates_path)
        if coordinates.shape[0] in (2, 3):
            coordinates = coordinates[0]
        else:
            coordinates = coordinates[:, 0]
        return BurgersSpectralRecurrence(
            torch.from_numpy(np.asarray(coordinates, dtype=np.float32)),
            args.dt_init,
            args.nu_init,
            args.filter_init,
        )
    if args.model in (
        "graph-wave-stored",
        "graph-wave-aux",
        "graph-wave-stable",
        "graph-wave-aux-stable",
    ):
        source = (
            ROOT
            / "mesh_operator/data/mp_pde/WE1/train/processed/"
            "pde_250-100/data_1.pt"
        )
        sample = torch.load(source, map_location="cpu", weights_only=False)
        return GraphWaveRecurrence(
            sample.U.float(),
            condition_dim,
            args.hidden_dim,
            stable=args.model in (
                "graph-wave-stable",
                "graph-wave-aux-stable",
            ),
        )
    graph_path = ROOT / "baseline-reruns/cache/we1_graph.npz"
    with np.load(graph_path) as graph:
        edge_index = graph["edge_index"].astype(np.int64)
        edge_features = graph["edge_features"].astype(np.float32)
    edge_length = np.maximum(edge_features[:, -1], -8.0)
    edge_weight = np.exp(-np.square(edge_length)).astype(np.float32)
    basis = graph_laplacian_basis(node_count, edge_index, edge_weight)
    return GraphWaveRecurrence(basis, condition_dim, args.hidden_dim)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    fields: np.ndarray,
    conditions: np.ndarray,
    indices: np.ndarray,
    field_mean: torch.Tensor,
    field_std: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    horizon: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, list[dict[str, float]]]:
    model.eval()
    ratio_sum = 0.0
    value_count = 0
    records = []
    for start in range(0, len(indices), batch_size):
        trajectory_ids = indices[start : start + batch_size]
        initial_raw = torch.from_numpy(
            np.asarray(fields[trajectory_ids, 0, :, 0], dtype=np.float32)
        ).to(device)
        condition_raw = torch.from_numpy(conditions[trajectory_ids]).to(device)
        initial = (initial_raw - field_mean) / field_std
        condition = (condition_raw - condition_mean) / condition_std
        prediction = model.rollout(initial, condition, horizon)
        prediction = prediction * field_std + field_mean
        target = torch.from_numpy(
            np.asarray(
                fields[trajectory_ids, 1 : horizon + 1, :, 0],
                dtype=np.float32,
            )
        ).to(device)
        numerator = torch.square(prediction - target).sum(dim=-1)
        denominator = torch.square(target).sum(dim=-1)
        ratios = numerator / torch.clamp(denominator, min=1e-12)
        ratio_sum += ratios.sum().item()
        value_count += ratios.numel()
        trajectory_values = ratios.mean(dim=1).cpu().numpy()
        for trajectory_id, value in zip(trajectory_ids, trajectory_values):
            records.append(
                {"trajectory_index": int(trajectory_id), "relmse": float(value)}
            )
    return ratio_sum / value_count, records


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    run_dir = HERE / "checkpoints" / args.run_name
    checkpoint_path = run_dir / "best.pt"
    result_path = HERE / "results" / f"{args.run_name}.json"
    if args.test:
        checkpoint_metadata = torch.load(checkpoint_path, map_location="cpu")
        stored_args = checkpoint_metadata["args"]
        for key in (
            "model",
            "hidden_dim",
            "width",
            "layers",
            "modes",
            "dt_init",
            "nu_init",
            "filter_init",
        ):
            if key in stored_args:
                setattr(args, key, stored_args[key])
    (
        spec,
        fields,
        conditions,
        train_indices,
        validation_indices,
        test_indices,
        field_mean_np,
        field_std_np,
        condition_mean_np,
        condition_std_np,
    ) = load_data(args.dataset, args.model)
    model = build_model(args, fields.shape[2], conditions.shape[-1]).to(device)
    field_mean = torch.as_tensor(field_mean_np[0], device=device)
    field_std = torch.as_tensor(field_std_np[0], device=device)
    condition_mean = torch.from_numpy(condition_mean_np).to(device)
    condition_std = torch.from_numpy(condition_std_np).to(device)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.test:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        metric, records = evaluate(
            model,
            fields,
            conditions,
            test_indices,
            field_mean,
            field_std,
            condition_mean,
            condition_std,
            spec.horizon,
            args.batch_size,
            device,
        )
        payload = {
            "dataset": args.dataset,
            "model": args.model,
            "run_name": args.run_name,
            "test_relmse": metric,
            "test_trajectories": len(test_indices),
            "horizon": spec.horizon,
            "selected_step": checkpoint["step"],
            "validation_relmse": checkpoint["validation_relmse"],
            "per_trajectory": records,
        }
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload | {"per_trajectory": "..."}), flush=True)
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.02
    )
    rng = np.random.RandomState(42)
    best_validation = float("inf")
    if args.finetune_from_best:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_validation = float(checkpoint["validation_relmse"])
        print(
            f"fine-tuning validation-best checkpoint ({best_validation:.6e})",
            flush=True,
        )
    wall_start = time.time()
    for step in range(1, args.steps + 1):
        trajectory_ids = rng.choice(
            train_indices, size=args.batch_size, replace=True
        )
        initial_raw = torch.from_numpy(
            np.asarray(fields[trajectory_ids, 0, :, 0], dtype=np.float32)
        ).to(device)
        target_raw = torch.from_numpy(
            np.asarray(
                fields[
                    trajectory_ids,
                    1 : args.train_horizon + 1,
                    :,
                    0,
                ],
                dtype=np.float32,
            )
        ).to(device)
        condition_raw = torch.from_numpy(conditions[trajectory_ids]).to(device)
        initial = (initial_raw - field_mean) / field_std
        target = (target_raw - field_mean) / field_std
        condition = (condition_raw - condition_mean) / condition_std
        prediction = model.rollout(initial, condition, args.train_horizon)
        if args.loss == "relmse":
            prediction_raw = prediction * field_std + field_mean
            numerator = torch.square(prediction_raw - target_raw).sum(dim=-1)
            denominator = torch.square(target_raw).sum(dim=-1)
            loss = torch.mean(
                numerator / torch.clamp(denominator, min=1e-12)
            )
        else:
            loss = torch.mean(torch.square(prediction - target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0:
            elapsed = max(time.time() - wall_start, 1e-9)
            print(
                f"step={step}/{args.steps} loss={loss.item():.6e} "
                f"steps_per_second={step / elapsed:.2f}",
                flush=True,
            )
        if step % args.val_interval == 0 or step == args.steps:
            validation, _ = evaluate(
                model,
                fields,
                conditions,
                validation_indices,
                field_mean,
                field_std,
                condition_mean,
                condition_std,
                spec.horizon,
                args.batch_size,
                device,
            )
            if validation < best_validation:
                best_validation = validation
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "validation_relmse": validation,
                        "args": vars(args),
                        "field_mean": field_mean_np,
                        "field_std": field_std_np,
                        "condition_mean": condition_mean_np,
                        "condition_std": condition_std_np,
                        "train_indices": train_indices.tolist(),
                        "validation_indices": validation_indices.tolist(),
                        "test_indices": test_indices.tolist(),
                    },
                    checkpoint_path,
                )
            print(
                f"validation step={step} relmse={validation:.6e} "
                f"best={best_validation:.6e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
