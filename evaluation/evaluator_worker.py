#!/usr/bin/env python3
"""Isolated checkpoint-only SEA evaluator.

This file is invoked only by ``evaluate.py`` in a fresh Python process.  It
contains no optimizer, backward, scheduler, checkpoint-save, or W&B path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import METRIC_NAMES, metric_arrays, summarize_metrics


SECRET_KEY = re.compile(
    r"(api[_-]?key|password|passwd|secret|access[_-]?token|auth[_-]?token|credential)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal checkpoint-only worker")
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def scrub_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in config.items() if not SECRET_KEY.search(key)}


def build_config(run: Mapping[str, Any], output_dir: Path, device: torch.device) -> Dict[str, Any]:
    config = scrub_config(run["evaluation"].get("config_overrides", {}))
    if not config:
        raise RuntimeError("manifest does not contain a pinned safe config profile")

    config.update(
        {
            "device": str(device),
            "field_data_path": run["data"]["field_data"]["path"],
            "input_path": run["data"]["input_data"]["path"],
            "coordinates_path": run["data"]["coordinates"]["path"],
            "encoder_decoder_path": run["artifacts"]["spatial_checkpoint"]["path"],
            "pretrained_model_path": run["artifacts"]["temporal_checkpoint"]["path"],
            "save_dir": str(output_dir / "preprocessing"),
            "random_seed": 42,
            "train_fraction": run["split"]["train_fraction"],
            "val_fraction": run["split"]["validation_fraction"],
            "dataset_src_len": run["horizon"]["predicted_steps"],
            "dataset_overlap": 0,
            "dataset_time_shifting_flag": False,
            "use_wandb": False,
            "final_save": False,
            "perform_initial_test": False,
            "test_mesh_structure": False,
            "load_pretrained": False,
            "epoch_num": 0,
            "validation_interval": 0,
            "full_eval_interval": 0,
            "print_split_sizes": True,
            "case_name": run["case"],
            "run_name": run["id"],
        }
    )
    edge_policy = run["evaluation"].get("runtime_edge_source", {})
    edge = run["data"].get("edges")
    if edge_policy.get("kind") in {"reconstruct_delaunay", "reconstruct_knn"}:
        config["edge_path"] = None
    elif edge:
        config["edge_path"] = edge["path"]
    elif "edge_path" in config:
        config["edge_path"] = None

    if config["random_seed"] != 42:
        raise RuntimeError("safety override failed: random seed is not 42")
    if config["use_wandb"] or config["final_save"] or config["epoch_num"] != 0:
        raise RuntimeError("unsafe configuration survived checkpoint-only overrides")
    return scrub_config(config)


def load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint is not a state dictionary: {path}")
    return {str(key).replace("module.", ""): value for key, value in payload.items()}


def set_positional_default(function: Any, parameter_name: str, value: Any) -> Any:
    signature = inspect.signature(function)
    before = signature.parameters[parameter_name].default
    defaults = list(function.__defaults__ or ())
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    first_default = len(positional) - len(defaults)
    position = next(
        index
        for index, parameter in enumerate(positional)
        if parameter.name == parameter_name
    )
    default_index = position - first_default
    if default_index < 0:
        raise RuntimeError(f"{parameter_name} is not a defaulted parameter")
    defaults[default_index] = value
    function.__defaults__ = tuple(defaults)
    return before


def instantiate_temporal_model(
    config: Mapping[str, Any], device: torch.device
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    temporal_module = importlib.import_module("models.temporal")
    block_default_before = inspect.signature(
        temporal_module.BaseBlockTemporal.__init__
    ).parameters["ib_num"].default
    if block_default_before != config["ib_num"]:
        set_positional_default(
            temporal_module.BaseBlockTemporal.__init__,
            "ib_num",
            config["ib_num"],
        )
    model_class = temporal_module.TemporalModel
    signature = inspect.signature(model_class.__init__)
    candidates = {
        "num_layers": config["num_layers"],
        "embed_dim": config["embed_dim"],
        "n_heads": config["n_heads"],
        "max_len": config["block_size"],
        "block_size": config["block_size"],
        "scale_ratio": config["scale_ratio"],
        "src_len": config["src_len"],
        "num_variables": config["num_fields"],
        "num_fields": config["num_fields"],
        "down_proj": config["down_proj"],
        "dropout": config["dropout"],
        "exchange_mode": config["exchange_mode"],
        "pos_encoding_mode": config["pos_encoding_mode"],
        "ib_scale_mode": config["ib_scale_mode"],
        "ib_addition_mode": config["ib_addition_mode"],
        "ib_mlp_layers": config["ib_mlp_layers"],
        "ib_num": config.get("ib_num", 1),
        "add_info_after_cross": config["add_info_after_cross"],
        "LN_type": config["LN_type"],
    }
    kwargs = {
        name: candidates[name]
        for name in signature.parameters
        if name != "self" and name in candidates
    }
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.default is inspect.Parameter.empty
        and name not in kwargs
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if missing:
        raise RuntimeError(f"unsupported TemporalModel constructor; missing {missing}")
    model = model_class(**kwargs).to(device)
    return model, {
        "base_block_ib_num_default_before": block_default_before,
        "base_block_ib_num_default_used": config["ib_num"],
        "in_memory_default_restoration": block_default_before != config["ib_num"],
    }


def normalize_boundary_input(
    boundary: torch.Tensor, horizon: int, field_count: int
) -> torch.Tensor:
    if boundary.ndim == 1:
        boundary = boundary.unsqueeze(0)
    if boundary.ndim == 2:
        boundary = boundary.unsqueeze(1).expand(-1, horizon, -1)
    elif boundary.ndim == 3 and boundary.shape[1] == 1:
        boundary = boundary.expand(-1, horizon, -1)
    elif boundary.ndim == 3 and boundary.shape[1] == field_count:
        boundary = boundary.mean(dim=1, keepdim=True).expand(-1, horizon, -1)
    elif boundary.ndim != 3 or boundary.shape[1] != horizon:
        raise ValueError(
            f"cannot normalize boundary input {tuple(boundary.shape)} to horizon {horizon}"
        )
    return boundary


def decode_rollout(
    encoded: torch.Tensor,
    processor: Any,
    mesh_processor: Any,
    config: Mapping[str, Any],
) -> torch.Tensor:
    train_utils = importlib.import_module("utils.train_utils")
    batch, horizon, _, _ = encoded.shape
    patch_count = int(
        getattr(
            processor,
            "P",
            (config["m"] - 1)
            * (config["n"] - 1)
            * ((config["k"] - 1) if config["dimension"] == "3D" else 1),
        )
    )
    decoded = train_utils.inverse_transform_processed_data(
        encoded, batch, horizon, patch_count, len(config["field_groups"])
    )
    decoded = processor.decode_data(decoded)
    if config["SEA_mixed"]:
        flat_batch, patches, fields, cells = decoded.shape
        decoded = decoded.reshape(flat_batch, patches, cells, fields)
    elif config["SEA_isolate"]:
        decoded = decoded.permute(0, 1, 3, 2)
    else:
        raise RuntimeError("exactly one of SEA_isolate/SEA_mixed must be enabled")
    decoded = mesh_processor.inverse_scale_and_unpatch(decoded.cpu())
    _, node_count, field_count = decoded.shape
    return decoded.reshape(batch, horizon, node_count, field_count)


def parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_metric_csvs(output_dir: Path, summary: Mapping[str, Any]) -> None:
    common_header = [
        "scope",
        "trajectory_id",
        "horizon",
        "field",
        "metric",
        "value",
        "ci95_low",
        "ci95_high",
    ]
    with (output_dir / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(common_header)
        for scope in ("aggregate", "final_step"):
            section = summary[scope]
            for field, metrics in section["per_field"].items():
                for metric, record in metrics.items():
                    writer.writerow(
                        [scope, "", "", field, metric, record["value"], *record["ci95"]]
                    )
            for metric, record in section["macro"].items():
                writer.writerow(
                    [scope, "", "", "macro", metric, record["value"], *record["ci95"]]
                )

    with (output_dir / "horizon_metrics.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(common_header)
        for row in summary["per_horizon"]:
            for field, metrics in row["per_field"].items():
                for metric, record in metrics.items():
                    writer.writerow(
                        [
                            "horizon",
                            "",
                            row["horizon"],
                            field,
                            metric,
                            record["value"],
                            *record["ci95"],
                        ]
                    )
            for metric, record in row["macro"].items():
                writer.writerow(
                    [
                        "horizon",
                        "",
                        row["horizon"],
                        "macro",
                        metric,
                        record["value"],
                        *record["ci95"],
                    ]
                )

    with (output_dir / "trajectory_metrics.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(common_header + ["inference_wall_seconds"])
        for row in summary["per_trajectory"]:
            for scope in ("aggregate", "final_step"):
                section = row[scope]
                for field, metrics in section["per_field"].items():
                    for metric, value in metrics.items():
                        writer.writerow(
                            [
                                scope,
                                row["trajectory_id"],
                                "",
                                field,
                                metric,
                                value,
                                "",
                                "",
                                row["inference_wall_seconds"],
                            ]
                        )
                for metric, value in section["macro"].items():
                    writer.writerow(
                        [
                            scope,
                            row["trajectory_id"],
                            "",
                            "macro",
                            metric,
                            value,
                            "",
                            "",
                            row["inference_wall_seconds"],
                        ]
                    )


def write_profile_csv(output_dir: Path, profile: Mapping[str, Any]) -> None:
    with (output_dir / "model_profile.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        for key, value in profile.items():
            if isinstance(value, Mapping):
                for child_key, child_value in value.items():
                    writer.writerow([f"{key}.{child_key}", child_value])
            else:
                writer.writerow([key, value])


def generate_source_conditioning(
    run: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor | None, Dict[str, Any]]:
    recipe = run["evaluation"].get("conditioning_recipe")
    if recipe is None:
        return None, {"kind": "archival_input_file"}
    if recipe.get("id") != "cylinder_re_nu_polynomial_v1":
        raise RuntimeError(f"unsupported conditioning recipe: {recipe.get('id')}")
    if run["id"] not in {
        "cylinder_sea_baseline_canonical",
        "cylinder_graphspectralformer_canonical",
    }:
        raise RuntimeError("source-derived cylinder conditioning is restricted to canonical runs")

    source_revisions = {
        source["revision"] for source in recipe.get("sources", [])
    }
    if run["code"]["revision"] not in source_revisions:
        raise RuntimeError(
            "conditioning recipe does not cite the evaluated source revision"
        )

    reynolds = torch.from_numpy(
        np.linspace(
            recipe["reynolds"]["start"],
            recipe["reynolds"]["stop"],
            recipe["trajectory_count"],
        )
    ).float().reshape([-1, 1])
    inverse_reynolds = 1 / reynolds
    columns = []
    for power in recipe["powers"]:
        re_power = reynolds**power
        inverse_power = inverse_reynolds**power
        columns.append(re_power / re_power.max())
        columns.append(inverse_power / inverse_power.max())
    conditioning = torch.cat(columns, dim=1).contiguous()

    if tuple(conditioning.shape) != (
        recipe["trajectory_count"],
        recipe["output_features"],
    ):
        raise RuntimeError(
            f"conditioning recipe produced unexpected shape {tuple(conditioning.shape)}"
        )
    raw = conditioning.numpy().tobytes(order="C")
    observed_hash = hashlib.sha256(raw).hexdigest()
    expected_hash = recipe.get("sha256_float32_c_order")
    if expected_hash is not None and observed_hash != expected_hash:
        raise RuntimeError(
            f"conditioning recipe hash {observed_hash} != {expected_hash}"
        )
    metadata = {
        "kind": "source_derived_in_memory",
        "recipe_id": recipe["id"],
        "shape": list(conditioning.shape),
        "dtype": str(conditioning.numpy().dtype),
        "sha256_float32_c_order": observed_hash,
        "source_path": recipe["source_path"],
        "sources": recipe["sources"],
        "archival_input_file_read_for_values": False,
    }
    return conditioning.to(device), metadata


def load_numpy_tensor(path: Path, device: torch.device) -> torch.Tensor:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            key = "x" if "x" in loaded.files else loaded.files[0]
            array = loaded[key]
        finally:
            loaded.close()
    elif isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
        item = loaded.item()
        array = item["x"] if isinstance(item, Mapping) and "x" in item else item
    else:
        array = loaded
    if not isinstance(array, np.ndarray):
        raise TypeError(f"unsupported NumPy payload in {path}: {type(array)}")
    return torch.from_numpy(array).to(device)


def install_conditioning_loader(
    train_temporal: Any,
    run: Mapping[str, Any],
    config: Mapping[str, Any],
    conditioning: torch.Tensor | None,
    device: torch.device,
) -> None:
    if conditioning is None:
        return

    def load_and_convert_override(_config: Mapping[str, Any]):
        field_data = load_numpy_tensor(Path(config["field_data_path"]), device)
        coordinates = load_numpy_tensor(Path(config["coordinates_path"]), device)
        runtime_repository = Path(
            run["evaluation"]["runtime_source"]["repository"]
        ).name
        if runtime_repository == "SEA-baseline":
            return field_data, coordinates, conditioning
        if runtime_repository == "SEA":
            if config.get("static_mesh", True):
                coordinates = coordinates.T
            edge_path = config.get("edge_path")
            if edge_path is None:
                edge_policy = run["evaluation"]["runtime_edge_source"]
                if edge_policy["kind"] == "reconstruct_knn":
                    edge_index, distances = train_temporal.build_edges(
                        coordinates,
                        method="knn",
                        k=edge_policy["neighbors"],
                    )
                else:
                    edge_index, distances = train_temporal.build_edges(
                        coordinates,
                        method="delaunay",
                        max_edge_len=edge_policy["max_edge_length"],
                    )
            else:
                with np.load(edge_path) as edge_data:
                    edge_index = torch.from_numpy(edge_data["edge_index"]).to(device)
                    edge_attr = torch.from_numpy(edge_data["edge_attr"]).to(device)
                if edge_attr.dim() == 2:
                    edge_attr = edge_attr.unsqueeze(0)
                distances = edge_attr[0, :, 2]
            sigma = torch.mean(distances)
            edge_weight = torch.exp(
                -(distances * distances) / (2.0 * (sigma**2) + 1.0e-12)
            )
            return (
                field_data,
                coordinates,
                conditioning,
                edge_index,
                edge_weight,
            )
        raise RuntimeError(
            f"unsupported cylinder runtime repository: {runtime_repository}"
        )

    train_temporal.load_and_convert = load_and_convert_override


def install_spatial_basis_compatibility(
    run: Mapping[str, Any], device: torch.device
) -> Dict[str, Any]:
    specification = run["evaluation"].get("spatial_basis")
    if specification is None:
        return {"kind": "runtime_native"}
    if specification.get("id") != "cylinder_global_u_patch_basis_v1":
        raise RuntimeError(f"unsupported spatial basis: {specification.get('id')}")
    if run["id"] != "cylinder_graphspectralformer_canonical":
        raise RuntimeError("pinned global-U patch basis is restricted to canonical proposed")

    basis_path = Path(specification["artifact"]["path"])
    try:
        global_basis = torch.load(
            basis_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        global_basis = torch.load(basis_path, map_location="cpu")
    if not isinstance(global_basis, torch.Tensor):
        raise TypeError(f"global basis is not a tensor: {basis_path}")
    expected_shape = tuple(specification["artifact"]["shape"])
    if tuple(global_basis.shape) != expected_shape:
        raise RuntimeError(
            f"global basis shape {tuple(global_basis.shape)} != {expected_shape}"
        )
    global_basis = global_basis.float().contiguous()
    embedding_modes = int(specification["partition_embedding_modes"])
    local_modes = int(specification["local_modes"])

    data_processors = importlib.import_module("utils.data_processors")
    partitioner_class = data_processors.SpectralPartitioner

    def spectral_embedding_override(_laplacian, k, **_kwargs):
        if k != embedding_modes:
            raise RuntimeError(
                f"partition requested {k} modes; expected {embedding_modes}"
            )
        embedding = global_basis[:, :k].numpy()
        row_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / (row_norm + 1.0e-12)
        return torch.from_numpy(embedding.astype(np.float32))

    def patch_basis_override(self, global_indices, _coordinates, _k_eigs):
        basis = global_basis.to(self.device)
        indices = global_indices.to(self.device)
        patch_basis = basis[indices, :local_modes].contiguous()
        eigenvalues = torch.zeros(local_modes, device=self.device)
        return patch_basis, eigenvalues

    partitioner_class._spectral_embedding = staticmethod(
        spectral_embedding_override
    )
    partitioner_class._compute_patch_gft = patch_basis_override
    return {
        "kind": "pinned_global_basis_patch_selection",
        "id": specification["id"],
        "artifact": specification["artifact"],
        "partition_embedding_modes": embedding_modes,
        "local_modes": local_modes,
        "partition_method": specification["partition_method"],
        "patch_basis_method": specification["patch_basis_method"],
    }


def preflight_data_contract(
    run: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor | None, Dict[str, Any]]:
    input_path = Path(run["data"]["input_data"]["path"])
    if input_path.suffix != ".npy":
        raise RuntimeError(f"unsupported input-data format: {input_path}")
    input_data = np.load(input_path, mmap_mode="r")
    observed_features = 1 if input_data.ndim == 1 else int(input_data.shape[-1])
    required_features = int(run["evaluation"]["required_input_features"])
    conditioning, conditioning_metadata = generate_source_conditioning(run, device)
    if conditioning is None and observed_features != required_features:
        raise RuntimeError(
            "input feature mismatch: temporal checkpoint requires "
            f"{required_features}, but {input_path} provides {observed_features} "
            f"with shape {list(input_data.shape)}"
        )
    if conditioning is not None and conditioning.shape[-1] != required_features:
        raise RuntimeError(
            "source-derived conditioning feature count does not match the checkpoint"
        )
    conditioning_metadata["archival_input_file"] = {
        "path": str(input_path),
        "shape": list(input_data.shape),
        "observed_features": observed_features,
        "used_for_runtime_values": conditioning is None,
    }

    field_path = Path(run["data"]["field_data"]["path"])
    if field_path.suffix != ".npy":
        raise RuntimeError(f"unsupported field-data format: {field_path}")
    field_data = np.load(field_path, mmap_mode="r")
    expected_frames = int(run["horizon"]["available_frames"])
    if field_data.ndim != 4 or field_data.shape[1] != expected_frames:
        raise RuntimeError(
            f"field-data/horizon mismatch: {field_path} has shape "
            f"{list(field_data.shape)}, expected {expected_frames} frames"
        )
    return conditioning, conditioning_metadata


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    if not source_root.is_dir() or source_root.name not in {"source", "SEA", "SEA-baseline"}:
        raise RuntimeError(f"invalid isolated source root: {source_root}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if (output_dir / "results.json").exists():
        raise FileExistsError(f"refusing to overwrite {output_dir / 'results.json'}")

    run = json.loads(args.run_spec.read_text(encoding="utf-8"))
    if not run.get("evaluation", {}).get("supported"):
        raise RuntimeError(f"run is not enabled for checkpoint evaluation: {run.get('id')}")
    if run["split"]["seed"] != 42:
        raise RuntimeError("only the fixed seed-42 split is permitted")

    sys.path.insert(0, str(source_root))
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.pop("WANDB_API_KEY", None)

    device = select_device(args.device)
    set_determinism(42)
    conditioning, conditioning_metadata = preflight_data_contract(run, device)
    config = build_config(run, output_dir, device)
    (output_dir / "preprocessing").mkdir(exist_ok=False)

    temporal_model, temporal_compatibility = instantiate_temporal_model(
        config, device
    )
    state = load_state_dict(Path(run["artifacts"]["temporal_checkpoint"]["path"]), device)
    try:
        temporal_model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "temporal checkpoint/config incompatibility at revision "
            f"{run['code']['revision']}: {exc}"
        ) from exc
    temporal_model.eval()
    temporal_parameters = parameter_count(temporal_model)

    train_temporal = importlib.import_module("train.train_temporal")
    spatial_basis_metadata = install_spatial_basis_compatibility(run, device)
    install_conditioning_loader(
        train_temporal, run, config, conditioning, device
    )
    preprocess_start = time.perf_counter()
    loaders = train_temporal.get_datasets(config)
    if len(loaders) != 5:
        raise RuntimeError(f"unexpected get_datasets return length: {len(loaders)}")
    _, _, repository_test_loader, mesh_processor, processor = loaders
    preprocess_seconds = time.perf_counter() - preprocess_start

    expected_test_ids = run["split"]["test_indices"]
    dataset = repository_test_loader.dataset
    if len(dataset) != len(expected_test_ids):
        raise RuntimeError(
            "test dataset does not contain exactly one full rollout per held-out "
            f"trajectory: got {len(dataset)}, expected {len(expected_test_ids)}"
        )
    test_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    spatial_model = getattr(processor, "model_spatial", None)
    spatial_parameters = parameter_count(spatial_model) if spatial_model is not None else {
        "total": 0,
        "trainable": 0,
    }

    if device.type == "cuda":
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_gpu_memory = int(torch.cuda.memory_allocated(device))
    else:
        baseline_gpu_memory = 0

    physical_batches = {metric: [] for metric in METRIC_NAMES}
    encoded_batches = {metric: [] for metric in METRIC_NAMES}
    trajectory_seconds: list[float] = []
    trajectory_cursor = 0
    model_inference_seconds = 0.0
    decode_seconds = 0.0
    evaluation_start = time.perf_counter()

    with torch.inference_mode():
        for source, target, original_target, boundary in test_loader:
            source = source.to(device)
            target = target.to(device)
            boundary = normalize_boundary_input(
                boundary.to(device), target.shape[1], config["num_fields"]
            )
            if target.shape[1] != run["horizon"]["predicted_steps"]:
                raise RuntimeError(
                    f"observed horizon {target.shape[1]} does not match manifest "
                    f"{run['horizon']['predicted_steps']}"
                )

            sync(device)
            batch_start = time.perf_counter()
            autoregressive = source[:, 0:1]
            for step in range(target.shape[1]):
                output = temporal_model(autoregressive, boundary[:, : step + 1])
                autoregressive = torch.cat(
                    (autoregressive, output[:, -1:, :, :]), dim=1
                )
            encoded_prediction = autoregressive[:, 1:]
            sync(device)
            batch_model_seconds = time.perf_counter() - batch_start
            model_inference_seconds += batch_model_seconds
            trajectory_seconds.extend(
                [batch_model_seconds / source.shape[0]] * source.shape[0]
            )

            decode_start = time.perf_counter()
            physical_prediction = decode_rollout(
                encoded_prediction, processor, mesh_processor, config
            )
            sync(device)
            decode_seconds += time.perf_counter() - decode_start

            physical = metric_arrays(
                physical_prediction.detach().cpu().numpy(),
                original_target.detach().cpu().numpy(),
            )
            encoded = metric_arrays(
                encoded_prediction.detach().cpu().numpy().transpose(0, 1, 3, 2),
                target.detach().cpu().numpy().transpose(0, 1, 3, 2),
            )
            for metric in METRIC_NAMES:
                physical_batches[metric].append(physical[metric])
                encoded_batches[metric].append(encoded[metric])
            trajectory_cursor += source.shape[0]

            del output, autoregressive, encoded_prediction, physical_prediction
            if device.type == "cuda":
                torch.cuda.empty_cache()

    sync(device)
    evaluation_wall_seconds = time.perf_counter() - evaluation_start
    if trajectory_cursor != len(expected_test_ids):
        raise RuntimeError("not every held-out trajectory was evaluated")
    peak_gpu_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )

    physical_arrays = {
        metric: np.concatenate(values, axis=0)
        for metric, values in physical_batches.items()
    }
    encoded_arrays = {
        metric: np.concatenate(values, axis=0)
        for metric, values in encoded_batches.items()
    }
    physical_summary = summarize_metrics(
        physical_arrays,
        run["evaluation"]["field_names"],
        expected_test_ids,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=42,
        trajectory_wall_seconds=trajectory_seconds,
    )
    encoded_names = [
        f"latent_group_{index}" for index in range(encoded_arrays[METRIC_NAMES[0]].shape[2])
    ]
    encoded_summary = summarize_metrics(
        encoded_arrays,
        encoded_names,
        expected_test_ids,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=42,
        trajectory_wall_seconds=trajectory_seconds,
    )

    profile = {
        "temporal_parameters": temporal_parameters,
        "spatial_parameters": spatial_parameters,
        "combined_parameters": {
            "total": temporal_parameters["total"] + spatial_parameters["total"],
            "trainable": temporal_parameters["trainable"]
            + spatial_parameters["trainable"],
        },
        "preprocessing_wall_seconds": preprocess_seconds,
        "autoregressive_model_wall_seconds": model_inference_seconds,
        "decode_wall_seconds": decode_seconds,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "wall_seconds_per_trajectory": model_inference_seconds
        / len(expected_test_ids),
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "gpu_memory_at_evaluation_start_bytes": baseline_gpu_memory,
        "device": str(device),
        "evaluation_batch_size": args.batch_size,
    }
    runtime_source = run["evaluation"].get("runtime_source", run["code"])
    result = {
        "schema_version": 1,
        "provenance": {
            "kind": "new_checkpoint_only_evaluation",
            "run_id": run["id"],
            "checkpoint_recorded_revision": run["code"]["revision"],
            "runtime_source_revision": runtime_source["revision"],
            "runtime_source_repository": runtime_source["repository"],
            "training_seed_count": 1,
            "test_split_seed": 42,
            "test_trajectory_indices": expected_test_ids,
            "historical_artifacts_overwritten": False,
            "temporal_compatibility": temporal_compatibility,
        },
        "dataset": {
            "case": run["case"],
            "predicted_steps": run["horizon"]["predicted_steps"],
            "field_names": run["evaluation"]["field_names"],
            "test_trajectory_count": len(expected_test_ids),
            "conditioning": conditioning_metadata,
            "runtime_edge_source": run["evaluation"].get(
                "runtime_edge_source", {"kind": "not_applicable"}
            ),
            "spatial_basis": spatial_basis_metadata,
        },
        "metrics": physical_summary,
        "latent_metrics": encoded_summary,
        "profile": profile,
    }

    write_metric_csvs(output_dir, physical_summary)
    write_profile_csv(output_dir, profile)
    with (output_dir / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

