#!/usr/bin/env python3
"""Evaluate retrained sonic flow proposed models and compare with baseline."""

import sys
import os
import json
import csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
os.environ["WANDB_MODE"] = "disabled"

def eval_checkpoint(config_name, run_name, label):
    """Evaluate a single checkpoint and return results."""
    if config_name == 'sonic_flow_v3':
        from configs.sonic_flow_v3 import get_config_temporal
    elif config_name == 'sonic_flow_v4':
        from configs.sonic_flow_v4 import get_config_temporal
    else:
        from configs.sonic_flow import get_config_temporal

    from train.train_temporal import get_datasets, get_model
    from utils.train_utils import full_autoregressive_evaluation

    config = get_config_temporal()
    config['run_name'] = run_name
    device = torch.device(config['device'])

    ckpt_dir = config['save_dir']
    temporal_ckpt = f"{ckpt_dir}/temporal_Checkpoint_sonic_flow_{run_name}.pt"
    if not os.path.exists(temporal_ckpt):
        temporal_ckpt = f"{ckpt_dir}/temporal_sonic_flow_{run_name}.pt"
    if not os.path.exists(temporal_ckpt):
        print(f"[{label}] No checkpoint found at {temporal_ckpt}")
        return None

    print(f"\n{'='*60}")
    print(f"[{label}] Evaluating: {temporal_ckpt}")
    print(f"{'='*60}")

    config['pretrained_model_path'] = temporal_ckpt
    config['load_pretrained'] = True
    config['use_wandb'] = False
    config['perform_initial_test'] = False
    config['test_mesh_structure'] = False

    model, loss_fn, _ = get_model(config, device)
    _, _, testLoader, mesh_processor, processor = get_datasets(config)

    test_results = full_autoregressive_evaluation(
        model, testLoader, loss_fn, device,
        processor, mesh_processor, config,
        epoch=0, plot_traj=False
    )

    rollout_csv = f"{ckpt_dir}/rollout_error_sonic_flow_{run_name}.csv"
    if os.path.exists(rollout_csv):
        with open(rollout_csv) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = np.array([[float(x) for x in row] for row in reader])

        results = {
            'label': label,
            'config': config_name,
            'run_name': run_name,
            'macro_relmse': float(data.mean(axis=0)[1:].mean()),
            'final_step_macro': float(data[-1][1:].mean()),
        }
        print(f"[{label}] Macro RelMSE: {results['macro_relmse']:.6f}")
        print(f"[{label}] Final step macro: {results['final_step_macro']:.6f}")
        return results
    return None

def main():
    print("Evaluating all sonic flow retrained models...")
    print("Baseline reference: macro_relmse = 0.006")
    print("Previous proposed: macro_relmse = 0.344")
    print()

    results = []

    for config_name, run_name, label in [
        ('sonic_flow', 'run2', 'Original (run2)'),
        ('sonic_flow_v3', 'run3', 'V3: lower LR, batch=4'),
        ('sonic_flow_v4', 'run4', 'V4: 2 layers'),
        ('sonic_flow', 'run1', 'Retrained (run1)'),
    ]:
        try:
            r = eval_checkpoint(config_name, run_name, label)
            if r:
                results.append(r)
        except Exception as e:
            print(f"[{label}] ERROR: {e}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Label':<30} {'Macro RelMSE':>15} {'Final Step':>15}")
    print("-"*60)
    print(f"{'Baseline (target)':<30} {'0.006000':>15} {'N/A':>15}")
    for r in results:
        print(f"{r['label']:<30} {r['macro_relmse']:>15.6f} {r['final_step_macro']:>15.6f}")

    outpath = './log-specatraformer-submission/experiments/results/sonic_retrained_comparison.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {outpath}")

if __name__ == '__main__':
    main()
