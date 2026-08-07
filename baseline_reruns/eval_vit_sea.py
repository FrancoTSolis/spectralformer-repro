"""Common decoded-space evaluator for matched-split ViT-SEA reruns."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from common import DATASETS, trajectory_split


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEA_BASELINE = ROOT / "SEA-baseline"
SEA_COMMIT = "59dffe0c03510e695be9cc26bd637d0a741f6735"

CONFIGS = {
    "cylinder": (
        "cylinder_flow_retrain",
        "checkpoints-cylinder-retrain",
        "cylinder_flow",
        "retrain1",
        "temporal",
    ),
    "multiphase": (
        "multiphase_flow_retrain",
        "checkpoints-multiphase-retrain",
        "multiphase_flow",
        "retrain1",
        "temporal_Checkpoint",
    ),
    "we1": (
        "wave_equation_retrain",
        "checkpoints-wave-equation-matched",
        "wave_equation",
        "matched1",
        "temporal",
    ),
    "e1": (
        "advection_equation_retrain",
        "checkpoints-advection-equation-matched",
        "advection_equation",
        "matched1",
        "temporal",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=CONFIGS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (
        module_name,
        checkpoint_dir_name,
        case_name,
        run_name,
        checkpoint_prefix,
    ) = CONFIGS[args.dataset]
    spec = DATASETS[args.dataset]
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    sys.path.insert(0, str(SEA_BASELINE))
    os.chdir(SEA_BASELINE)
    config_module = importlib.import_module(f"configs.{module_name}")
    config = config_module.get_config_temporal()
    config["device"] = str(device)
    config["use_wandb"] = False
    config["perform_initial_test"] = False
    config["test_mesh_structure"] = False

    checkpoint_dir = SEA_BASELINE / checkpoint_dir_name
    model_path = Path(args.model_path) if args.model_path else (
        checkpoint_dir / f"{checkpoint_prefix}_{case_name}_{run_name}.pt"
    )
    encoder_path = checkpoint_dir / f"encoder_decoder_{case_name}_{run_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not encoder_path.exists():
        raise FileNotFoundError(encoder_path)
    config["pretrained_model_path"] = str(model_path)
    config["encoder_decoder_path"] = str(encoder_path)
    config["load_pretrained"] = True

    from train.train_temporal import get_datasets, get_model
    from utils.train_utils import inverse_transform_processed_data

    model, _, _ = get_model(config, device)
    _, _, test_loader, mesh_processor, processor = get_datasets(config)
    model.eval()

    raw_fields = np.load(spec.fields_path, mmap_mode="r")
    _, _, test_indices = trajectory_split(
        raw_fields.shape[0], spec.train_fraction, spec.val_fraction
    )
    field_count = raw_fields.shape[-1]
    global_ratio_sum = np.zeros(field_count, dtype=np.float64)
    per_trajectory = []
    trajectory_offset = 0

    with torch.no_grad():
        for batch_index, (data, target, original_data, conditioning) in enumerate(
            test_loader
        ):
            data = data.to(device)
            target = target.to(device)
            original_data = original_data.to(device)
            conditioning = conditioning.to(device)
            batch_size, sequence_length, variable_count, _ = data.shape
            if conditioning.dim() == 2:
                conditioning = conditioning.unsqueeze(1).expand(
                    batch_size, sequence_length, -1
                )
            elif conditioning.shape[1] == 1:
                conditioning = conditioning.expand(
                    batch_size, sequence_length, -1
                )
            elif conditioning.shape[1] != sequence_length:
                raise ValueError(
                    f"invalid conditioning shape {conditioning.shape}; "
                    f"expected sequence length {sequence_length}"
                )

            autoregressive = data[:, :1]
            for step in range(target.shape[1]):
                output = model(autoregressive, conditioning[:, : step + 1])
                autoregressive = torch.cat(
                    [autoregressive, output[:, -1:]], dim=1
                )
            encoded_prediction = autoregressive[:, 1:]

            if config["dimension"] == "3D":
                patch_count = (
                    (config["m"] - 1)
                    * (config["n"] - 1)
                    * (config["k"] - 1)
                )
            else:
                patch_count = (config["m"] - 1) * (config["n"] - 1)
            decoded = inverse_transform_processed_data(
                encoded_prediction,
                batch_size,
                target.shape[1],
                patch_count,
                len(config["field_groups"]),
            )
            decoded = processor.decode_data(decoded)
            if config["SEA_mixed"]:
                batch, patches, variables, cells = decoded.shape
                decoded = decoded.reshape(batch, patches, cells, variables)
            elif config["SEA_isolate"]:
                decoded = decoded.permute(0, 1, 3, 2)
            else:
                raise ValueError("invalid SEA field layout")

            decoded = (
                mesh_processor.inverse_scale_and_unpatch(decoded.cpu())
                .to(device)
                .reshape(
                    batch_size,
                    target.shape[1],
                    original_data.shape[2],
                    field_count,
                )
            )
            numerator = torch.square(decoded - original_data).sum(dim=2).double()
            denominator = torch.square(original_data).sum(dim=2).double()
            ratios = (
                numerator / torch.clamp(denominator, min=1e-12)
            ).cpu().numpy()
            global_ratio_sum += ratios.sum(axis=(0, 1))
            batch_trajectory_ids = test_indices[
                trajectory_offset : trajectory_offset + batch_size
            ]
            for row, trajectory_id in enumerate(batch_trajectory_ids):
                per_trajectory.append(
                    {
                        "trajectory_index": int(trajectory_id),
                        "per_field_relmse": ratios[row].mean(axis=0).tolist(),
                        "macro_relmse": float(ratios[row].mean()),
                    }
                )
            trajectory_offset += batch_size
            print(
                f"batch={batch_index + 1}/{len(test_loader)} "
                f"trajectories={trajectory_offset}/{len(test_indices)}",
                flush=True,
            )

    if trajectory_offset != len(test_indices):
        raise RuntimeError(
            f"evaluated {trajectory_offset} samples, expected {len(test_indices)}"
        )
    per_field_relmse = global_ratio_sum / (
        len(test_indices) * (raw_fields.shape[1] - 1)
    )
    results = {
        "method": "vit-sea",
        "dataset": args.dataset,
        "model_path": str(model_path),
        "encoder_path": str(encoder_path),
        "official_repository": "https://github.com/anonymous/SEA",
        "source_commit": SEA_COMMIT,
        "test_trajectory_count": int(len(test_indices)),
        "forecast_horizon": int(raw_fields.shape[1] - 1),
        "per_field_relmse": per_field_relmse.tolist(),
        "macro_relmse": float(per_field_relmse.mean()),
        "aggregation": (
            "reduce spatial nodes first; arithmetic mean over every held-out "
            "trajectory and forecast step; macro is the arithmetic mean over fields"
        ),
        "split_file": str(HERE / "cache" / f"{args.dataset}_split.json"),
        "per_trajectory": per_trajectory,
    }
    output_path = (
        Path(args.output)
        if args.output
        else HERE / "results" / f"vit-sea_{args.dataset}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "method": "vit-sea",
                "dataset": args.dataset,
                "per_field_relmse": results["per_field_relmse"],
                "macro_relmse": results["macro_relmse"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
