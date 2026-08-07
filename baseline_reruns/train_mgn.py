"""Train/evaluate MGN and MGN-NI on the paper's exact data contract.

Architecture source:
  NVIDIA PhysicsNeMo MeshGraphNet (PyG), an Apache-2.0 maintained
  implementation of Pfaff et al. (ICLR 2021), configured with the original
  15 message-passing blocks, width 128, and two-layer MLPs.

MGN-NI differs only by the original normalized-state Gaussian noise injection.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from common import DATASETS, condition_at, sample_pairs, trajectory_split


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHYSICSNEMO_ROOT = ROOT / "physicsnemo"
sys.path.insert(0, str(PHYSICSNEMO_ROOT))
os.environ.setdefault("PHYSICSNEMO_FORCE_TE", "False")

# PhysicsNeMo main targets newer PyTorch; preserve behavior on the repository's
# existing torch 2.2 runtime without modifying the upstream checkout.
if not hasattr(torch.nn.Module, "register_load_state_dict_pre_hook"):
    def _register_public_pre_hook(self, hook):
        return self._register_load_state_dict_pre_hook(hook, with_module=True)

    torch.nn.Module.register_load_state_dict_pre_hook = (  # type: ignore[attr-defined]
        _register_public_pre_hook
    )

from physicsnemo.models.meshgraphnet.meshgraphnet import MeshGraphNet  # noqa: E402


PHYSICSNEMO_COMMIT = "77b3c68001159b948a16804fa76eb127735fb6d1"
DEEPMIND_REFERENCE_COMMIT = "f5de0ede8430809180254ee957abf36ed62579ef"


class BatchedGraphCache:
    def __init__(
        self,
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        node_count: int,
        device: torch.device,
    ) -> None:
        self.base_edge_index = torch.from_numpy(edge_index).long().to(device)
        self.base_edge_features = torch.from_numpy(edge_features).float().to(device)
        self.node_count = node_count
        self.device = device
        self._cache: dict[int, tuple[Data, torch.Tensor]] = {}

    def get(self, batch_size: int) -> tuple[Data, torch.Tensor]:
        if batch_size not in self._cache:
            offsets = (
                torch.arange(batch_size, device=self.device, dtype=torch.long)
                * self.node_count
            )
            edge_index = (
                self.base_edge_index.unsqueeze(0) + offsets[:, None, None]
            ).permute(1, 0, 2).reshape(2, -1)
            edge_features = self.base_edge_features.repeat(batch_size, 1)
            graph = Data(edge_index=edge_index, num_nodes=batch_size * self.node_count)
            graph = graph.to(self.device)
            self._cache[batch_size] = graph, edge_features
        return self._cache[batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--variant", choices=("mgn", "mgn-ni"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--val-pairs", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    return parser.parse_args()


def load_contract(dataset_name: str):
    spec = DATASETS[dataset_name]
    fields = np.load(spec.fields_path, mmap_mode="r")
    train, validation, test = trajectory_split(
        fields.shape[0], spec.train_fraction, spec.val_fraction
    )
    with np.load(HERE / "cache" / f"{dataset_name}_graph.npz") as graph:
        edge_index = graph["edge_index"]
        edge_features = graph["edge_features"]
    with np.load(HERE / "cache" / f"{dataset_name}_stats.npz") as archive:
        stats = {key: archive[key] for key in archive.files}
    conditioning = np.load(
        HERE / "cache" / f"{dataset_name}_conditioning.npy", mmap_mode="r"
    )
    return spec, fields, conditioning, train, validation, test, edge_index, edge_features, stats


def tensors_for_pairs(
    fields: np.ndarray,
    conditioning: np.ndarray,
    trajectories: np.ndarray,
    times: np.ndarray,
    stats: dict[str, np.ndarray],
    noise_std: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = np.asarray(fields[trajectories, times], dtype=np.float32)
    following = np.asarray(fields[trajectories, times + 1], dtype=np.float32)
    conditions = condition_at(conditioning, trajectories, times).astype(np.float32)

    field_mean = torch.as_tensor(stats["field_mean"], device=device)
    field_std = torch.as_tensor(stats["field_std"], device=device)
    delta_mean = torch.as_tensor(stats["delta_mean"], device=device)
    delta_std = torch.as_tensor(stats["delta_std"], device=device)

    current_t = torch.from_numpy(current).to(device)
    following_t = torch.from_numpy(following).to(device)
    current_normalized = (current_t - field_mean) / field_std
    following_normalized = (following_t - field_mean) / field_std
    if noise_std > 0:
        current_normalized = current_normalized + torch.randn_like(
            current_normalized
        ) * noise_std

    condition_t = torch.from_numpy(conditions).to(device)
    condition_nodes = condition_t[:, None, :].expand(
        -1, current_normalized.shape[1], -1
    )
    node_features = torch.cat([current_normalized, condition_nodes], dim=-1)
    target = (
        following_normalized - current_normalized - delta_mean
    ) / delta_std
    return node_features.reshape(-1, node_features.shape[-1]), target.reshape(
        -1, target.shape[-1]
    )


@torch.no_grad()
def validate(
    model: MeshGraphNet,
    graph_cache: BatchedGraphCache,
    fields: np.ndarray,
    conditioning: np.ndarray,
    validation_indices: np.ndarray,
    stats: dict[str, np.ndarray],
    pair_count: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    rng = np.random.RandomState(420_042)
    total_squared_error = 0.0
    total_values = 0
    remaining = pair_count
    while remaining:
        current_batch = min(batch_size, remaining)
        trajectories, times = sample_pairs(
            rng, validation_indices, fields.shape[1], current_batch
        )
        node_features, target = tensors_for_pairs(
            fields,
            conditioning,
            trajectories,
            times,
            stats,
            0.0,
            device,
        )
        graph, edge_features = graph_cache.get(current_batch)
        prediction = model(node_features, edge_features, graph)
        total_squared_error += torch.square(prediction - target).sum().item()
        total_values += target.numel()
        remaining -= current_batch
    model.train()
    return total_squared_error / total_values


@torch.no_grad()
def evaluate_rollout(
    model: MeshGraphNet,
    graph_cache: BatchedGraphCache,
    dataset_name: str,
    fields: np.ndarray,
    conditioning: np.ndarray,
    test_indices: np.ndarray,
    stats: dict[str, np.ndarray],
    rollout_batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    field_mean = torch.as_tensor(stats["field_mean"], device=device)
    field_std = torch.as_tensor(stats["field_std"], device=device)
    delta_mean = torch.as_tensor(stats["delta_mean"], device=device)
    delta_std = torch.as_tensor(stats["delta_std"], device=device)
    field_count = fields.shape[-1]

    global_ratio_sum = np.zeros(field_count, dtype=np.float64)
    trajectory_records = []

    for start in range(0, len(test_indices), rollout_batch_size):
        trajectory_ids = test_indices[start : start + rollout_batch_size]
        batch_size = len(trajectory_ids)
        graph, batched_edge_features = graph_cache.get(batch_size)
        initial = np.asarray(fields[trajectory_ids, 0], dtype=np.float32)
        state = (torch.from_numpy(initial).to(device) - field_mean) / field_std
        ratio_sum = torch.zeros(
            batch_size, field_count, dtype=torch.float64, device=device
        )

        for time_index in range(fields.shape[1] - 1):
            times = np.full(batch_size, time_index, dtype=np.int64)
            conditions = condition_at(
                conditioning, trajectory_ids, times
            ).astype(np.float32)
            condition_t = torch.from_numpy(conditions).to(device)
            condition_nodes = condition_t[:, None, :].expand(
                -1, state.shape[1], -1
            )
            node_features = torch.cat([state, condition_nodes], dim=-1).reshape(
                -1, state.shape[-1] + condition_t.shape[-1]
            )
            delta_prediction = model(
                node_features, batched_edge_features, graph
            ).reshape(batch_size, state.shape[1], field_count)
            state = state + delta_prediction * delta_std + delta_mean
            decoded = state * field_std + field_mean
            target = torch.from_numpy(
                np.asarray(fields[trajectory_ids, time_index + 1], dtype=np.float32)
            ).to(device)
            numerator = torch.square(decoded - target).sum(dim=1).double()
            denominator = torch.square(target).sum(dim=1).double()
            ratio_sum += numerator / torch.clamp(denominator, min=1e-12)

        ratio_sum_np = ratio_sum.cpu().numpy()
        global_ratio_sum += ratio_sum_np.sum(axis=0)
        per_field = ratio_sum_np / (fields.shape[1] - 1)
        for row, trajectory_id in enumerate(trajectory_ids):
            trajectory_records.append(
                {
                    "trajectory_index": int(trajectory_id),
                    "per_field_relmse": per_field[row].tolist(),
                    "macro_relmse": float(per_field[row].mean()),
                }
            )
        print(
            f"rollout {start + batch_size}/{len(test_indices)} complete",
            flush=True,
        )

    per_field_relmse = global_ratio_sum / (
        len(test_indices) * (fields.shape[1] - 1)
    )
    return {
        "dataset": dataset_name,
        "test_trajectory_count": int(len(test_indices)),
        "forecast_horizon": int(fields.shape[1] - 1),
        "per_field_relmse": per_field_relmse.tolist(),
        "macro_relmse": float(per_field_relmse.mean()),
        "aggregation": (
            "reduce spatial nodes first; arithmetic mean over every held-out "
            "trajectory and forecast step; macro is the arithmetic mean over fields"
        ),
        "per_trajectory": trajectory_records,
    }


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    (
        spec,
        fields,
        conditioning,
        train_indices,
        validation_indices,
        test_indices,
        edge_index,
        edge_features,
        stats,
    ) = load_contract(args.dataset)
    batch_size = args.batch_size or spec.batch_size
    condition_dim = conditioning.shape[-1]
    field_count = fields.shape[-1]
    model = MeshGraphNet(
        input_dim_nodes=field_count + condition_dim,
        input_dim_edges=edge_features.shape[-1],
        output_dim=field_count,
        processor_size=15,
        num_layers_node_processor=2,
        num_layers_edge_processor=2,
        hidden_dim_processor=128,
        hidden_dim_node_encoder=128,
        num_layers_node_encoder=2,
        hidden_dim_edge_encoder=128,
        num_layers_edge_encoder=2,
        hidden_dim_node_decoder=128,
        num_layers_node_decoder=2,
        aggregation="sum",
        norm_type="LayerNorm",
    ).to(device)
    graph_cache = BatchedGraphCache(
        edge_index, edge_features, fields.shape[2], device
    )

    run_name = f"{args.variant}_{args.dataset}"
    checkpoint_dir = HERE / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    results_path = HERE / "results" / f"{run_name}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    start_step = 0
    best_validation = float("inf")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if (args.resume or args.eval_only) and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if args.resume and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint["step"])
            best_validation = float(checkpoint["validation_mse"])
        print(f"loaded {checkpoint_path} at step {start_step}", flush=True)
    elif args.eval_only:
        raise FileNotFoundError(checkpoint_path)

    provenance = {
        "architecture": "MeshGraphNet, 15 message-passing blocks, width 128",
        "implementation": "NVIDIA PhysicsNeMo PyG MeshGraphNet",
        "physicsnemo_commit": PHYSICSNEMO_COMMIT,
        "original_deepmind_reference_commit": DEEPMIND_REFERENCE_COMMIT,
        "variant": args.variant,
        "noise_injection_std_normalized": (
            args.noise_std if args.variant == "mgn-ni" else 0.0
        ),
        "seed": 42,
    }

    if not args.eval_only:
        model.train()
        rng = np.random.RandomState(42)
        noise_std = args.noise_std if args.variant == "mgn-ni" else 0.0
        wall_start = time.time()
        for step in range(start_step + 1, args.steps + 1):
            trajectories, times = sample_pairs(
                rng, train_indices, fields.shape[1], batch_size
            )
            node_features, target = tensors_for_pairs(
                fields,
                conditioning,
                trajectories,
                times,
                stats,
                noise_std,
                device,
            )
            graph, batched_edge_features = graph_cache.get(batch_size)
            prediction = model(node_features, batched_edge_features, graph)
            loss = torch.mean(torch.square(prediction - target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % 50 == 0:
                elapsed = time.time() - wall_start
                print(
                    f"step={step}/{args.steps} loss={loss.item():.6e} "
                    f"steps_per_second={step / max(elapsed, 1e-6):.3f}",
                    flush=True,
                )

            if step % args.val_interval == 0 or step == args.steps:
                validation_mse = validate(
                    model,
                    graph_cache,
                    fields,
                    conditioning,
                    validation_indices,
                    stats,
                    args.val_pairs,
                    batch_size,
                    device,
                )
                print(
                    f"validation step={step} mse={validation_mse:.6e}",
                    flush=True,
                )
                if validation_mse < best_validation:
                    best_validation = validation_mse
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step,
                            "validation_mse": validation_mse,
                            "provenance": provenance,
                            "train_indices": train_indices.tolist(),
                            "validation_indices": validation_indices.tolist(),
                            "test_indices": test_indices.tolist(),
                        },
                        checkpoint_path,
                    )
                    print(f"saved best checkpoint: {checkpoint_path}", flush=True)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

    if args.skip_rollout:
        print("training/checkpoint smoke test complete; rollout skipped", flush=True)
        return

    results = evaluate_rollout(
        model,
        graph_cache,
        args.dataset,
        fields,
        conditioning,
        test_indices,
        stats,
        spec.rollout_batch_size,
        device,
    )
    results.update(
        {
            "method": args.variant,
            "checkpoint": str(checkpoint_path),
            "validation_mse": float(
                torch.load(checkpoint_path, map_location="cpu")["validation_mse"]
            ),
            "provenance": provenance,
            "split_file": str(HERE / "cache" / f"{args.dataset}_split.json"),
        }
    )
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: results[k] for k in ("method", "dataset", "macro_relmse")}))
    print(f"wrote {results_path}", flush=True)


if __name__ == "__main__":
    main()
