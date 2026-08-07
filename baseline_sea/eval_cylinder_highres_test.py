#!/usr/bin/env python3
"""Evaluate high-res cylinder flow baseline on the test set."""

import sys, os, json, csv
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
os.environ["WANDB_MODE"] = "disabled"

from configs.cylinder_flow_highres import get_config_temporal
from train.train_temporal import get_datasets, get_model
from utils.train_utils import full_autoregressive_evaluation

def main():
    config = get_config_temporal()
    device = torch.device(config['device'])

    temporal_ckpt = config['save_dir'] + '/temporal_Checkpoint_cylinder_flow_highres_run1.pt'
    if not os.path.exists(temporal_ckpt):
        temporal_ckpt = config['save_dir'] + '/temporal_cylinder_flow_highres_run1.pt'

    print(f"Using temporal checkpoint: {temporal_ckpt}")
    config['pretrained_model_path'] = temporal_ckpt
    config['load_pretrained'] = True
    config['use_wandb'] = False
    config['perform_initial_test'] = False
    config['test_mesh_structure'] = False

    model, loss_fn, _ = get_model(config, device)
    _, _, testLoader, mesh_processor, processor = get_datasets(config)

    print("Running full autoregressive evaluation on test set...")
    test_results = full_autoregressive_evaluation(
        model, testLoader, loss_fn, device,
        processor, mesh_processor, config,
        epoch=0, plot_traj=True
    )

    print("\n=== Test Results ===")
    for key, value in test_results.items():
        print(f"{key}: {value}")

    rollout_csv = config['save_dir'] + '/rollout_error_cylinder_flow_highres_run1.csv'
    if os.path.exists(rollout_csv):
        with open(rollout_csv) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = np.array([[float(x) for x in row] for row in reader])

        results = {
            'model': 'SEA_baseline',
            'dataset': 'cylinder_flow_highres',
            'split': 'test',
            'mesh_nodes': 7697,
            'horizon': int(data.shape[0]),
            'fields': header[1:],
            'macro_relmse': float(data.mean(axis=0)[1:].mean()),
            'per_field_relmse': {header[i+1]: float(data.mean(axis=0)[i+1]) for i in range(len(header)-1)},
            'final_step_relmse': {header[i+1]: float(data[-1][i+1]) for i in range(len(header)-1)},
            'final_step_macro': float(data[-1][1:].mean()),
        }

        outpath = './log-specatraformer-submission/experiments/results/cylinder_highres_baseline_test_results.json'
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {outpath}")
        print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
