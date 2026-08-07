#!/usr/bin/env python3
"""Evaluate all trained models and generate structured comparison results."""

import sys
import os
import json
import csv
import numpy as np
import torch

def eval_temporal_model(model_dir, config_name, model_type='proposed'):
    """Evaluate a trained temporal model and return structured results."""
    sys.path.insert(0, model_dir)
    os.environ["WANDB_MODE"] = "disabled"
    
    config_module = __import__(f'configs.{config_name}', fromlist=['get_config_temporal'])
    config = config_module.get_config_temporal()
    device = torch.device(config['device'])
    
    temporal_ckpt = f"{config['save_dir']}/temporal_{config['case_name']}_{config['run_name']}.pt"
    if not os.path.exists(temporal_ckpt):
        temporal_ckpt = f"{config['save_dir']}/temporal_Checkpoint_{config['case_name']}_{config['run_name']}.pt"
    
    if not os.path.exists(temporal_ckpt):
        return None, f"No checkpoint found at {temporal_ckpt}"
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type}: {config_name} from {model_dir}")
    print(f"Checkpoint: {temporal_ckpt}")
    print(f"{'='*60}")
    
    config['pretrained_model_path'] = temporal_ckpt
    config['load_pretrained'] = True
    config['use_wandb'] = False
    config['perform_initial_test'] = False
    config['test_mesh_structure'] = False
    
    from train.train_temporal import get_datasets, get_model
    from utils.train_utils import full_autoregressive_evaluation
    
    model, loss_fn, _ = get_model(config, device)
    _, _, testLoader, mesh_processor, processor = get_datasets(config)
    
    test_results = full_autoregressive_evaluation(
        model, testLoader, loss_fn, device,
        processor, mesh_processor, config,
        epoch=0, plot_traj=True
    )
    
    rollout_csv = f"{config['save_dir']}/rollout_error_{config['case_name']}_{config['run_name']}.csv"
    results = {
        'model': model_type,
        'dataset': config['case_name'],
        'decoded_rel_mse': test_results.get('decoded_rel_mse', None),
        'encoded_rel_mse': test_results.get('encoded_rel_mse', None),
    }
    
    if os.path.exists(rollout_csv):
        with open(rollout_csv) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = np.array([[float(x) for x in row] for row in reader])
        
        results['horizon'] = int(data.shape[0])
        results['fields'] = header[1:]
        results['macro_relmse'] = float(data.mean(axis=0)[1:].mean())
        results['per_field_relmse'] = {header[i+1]: float(data.mean(axis=0)[i+1]) for i in range(len(header)-1)}
        results['final_step_relmse'] = {header[i+1]: float(data[-1][i+1]) for i in range(len(header)-1)}
        results['final_step_macro'] = float(data[-1][1:].mean())
    
    return results, None


def main():
    output_dir = './log-specatraformer-submission/experiments/results'
    os.makedirs(output_dir, exist_ok=True)
    
    evaluations = [
        ('./SEA', 'wave_equation', 'GraphSpectralFormer'),
        ('./SEA-baseline', 'wave_equation', 'SEA_baseline'),
        ('./SEA', 'advection_equation', 'GraphSpectralFormer'),
        ('./SEA-baseline', 'advection_equation', 'SEA_baseline'),
        ('./SEA', 'cylinder_flow_highres', 'GraphSpectralFormer'),
        ('./SEA-baseline', 'cylinder_flow_highres', 'SEA_baseline'),
    ]
    
    all_results = []
    for model_dir, config_name, model_type in evaluations:
        try:
            result, error = eval_temporal_model(model_dir, config_name, model_type)
            if error:
                print(f"[SKIP] {config_name} ({model_type}): {error}")
            else:
                all_results.append(result)
                print(f"\n[RESULT] {config_name} ({model_type}):")
                print(f"  Macro RelMSE: {result.get('macro_relmse', 'N/A')}")
                print(f"  Decoded RelMSE: {result.get('decoded_rel_mse', 'N/A')}")
        except Exception as e:
            print(f"[ERROR] {config_name} ({model_type}): {e}")
    
    if all_results:
        outpath = os.path.join(output_dir, 'all_new_dataset_results.json')
        with open(outpath, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n\nSaved all results to {outpath}")
    
    print("\n\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    datasets = set(r['dataset'] for r in all_results)
    for ds in sorted(datasets):
        print(f"\n{ds}:")
        ds_results = [r for r in all_results if r['dataset'] == ds]
        for r in ds_results:
            macro = r.get('macro_relmse', 'N/A')
            print(f"  {r['model']}: macro_relmse={macro}")
        
        proposed = [r for r in ds_results if r['model'] == 'GraphSpectralFormer']
        baseline = [r for r in ds_results if r['model'] == 'SEA_baseline']
        if proposed and baseline and proposed[0].get('macro_relmse') and baseline[0].get('macro_relmse'):
            gain = (baseline[0]['macro_relmse'] - proposed[0]['macro_relmse']) / baseline[0]['macro_relmse'] * 100
            print(f"  Improvement: {gain:.1f}% {'(PROPOSED WINS)' if gain > 0 else '(BASELINE WINS)'}")


if __name__ == '__main__':
    main()
