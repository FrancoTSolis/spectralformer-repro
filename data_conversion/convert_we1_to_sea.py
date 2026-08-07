#!/usr/bin/env python3
"""Convert WE1 (wave equation) from PyG per-graph format to SEA-compatible numpy arrays."""

import os
import torch
import numpy as np

def load_split(proc_dir):
    """Load all samples from a processed split directory."""
    files = sorted([f for f in os.listdir(proc_dir) if f.startswith('data_') and f.endswith('.pt')])
    fields_list = []
    initial_conds = []
    pos = None
    edge_index = None
    edge_attr = None

    for f in files:
        d = torch.load(os.path.join(proc_dir, f), map_location='cpu', weights_only=False)
        y = d.y.numpy()  # (T, N_nodes, 1)
        ic = d.x[:, 3].numpy()  # initial condition at each node
        fields_list.append(y)
        initial_conds.append(ic)
        if pos is None:
            pos = d.pos.numpy()  # (N_nodes, 1)
            edge_index = d.edge_index.numpy()  # (2, E)
            edge_attr = d.edge_attr.numpy()  # (E,)

    fields = np.stack(fields_list, axis=0)  # (N_traj, T, N_nodes, 1)
    initial_conds = np.stack(initial_conds, axis=0)  # (N_traj, N_nodes)
    return fields, initial_conds, pos, edge_index, edge_attr

def main():
    base = './mesh_operator/data/mp_pde/WE1'

    print("Loading train split...")
    train_fields, train_ic, pos, edge_index, edge_attr = load_split(
        f'{base}/train/processed/pde_250-100')
    print(f"  Train fields: {train_fields.shape}")

    print("Loading valid split...")
    valid_fields, valid_ic, _, _, _ = load_split(
        f'{base}/valid/processed/pde_250-100')
    print(f"  Valid fields: {valid_fields.shape}")

    print("Loading test split...")
    test_fields, test_ic, _, _, _ = load_split(
        f'{base}/test/processed/pde_250-100')
    print(f"  Test fields: {test_fields.shape}")

    all_fields = np.concatenate([train_fields, valid_fields, test_fields], axis=0)
    all_ic = np.concatenate([train_ic, valid_ic, test_ic], axis=0)
    print(f"\nCombined fields: {all_fields.shape}")
    print(f"Combined ICs: {all_ic.shape}")
    print(f"Field range: [{all_fields.min():.4f}, {all_fields.max():.4f}]")

    n_train = len(train_fields)
    n_valid = len(valid_fields)
    n_test = len(test_fields)
    print(f"Split: train={n_train}, valid={n_valid}, test={n_test}")
    print(f"Total: {n_train + n_valid + n_test}")

    coordinates = np.tile(pos[np.newaxis], (len(all_fields), 1, 1))
    print(f"Coordinates: {coordinates.shape}")

    ic_norm = np.linalg.norm(all_ic, axis=1, keepdims=True)
    ic_norm = np.where(ic_norm == 0, 1.0, ic_norm)
    ic_energy = (all_ic ** 2).sum(axis=1, keepdims=True)
    ic_max = all_ic.max(axis=1, keepdims=True)
    ic_min = all_ic.min(axis=1, keepdims=True)
    input_features = np.concatenate([
        ic_norm / ic_norm.max(),
        ic_energy / ic_energy.max(),
        ic_max / max(abs(ic_max.max()), abs(ic_max.min()), 1e-8),
        ic_min / max(abs(ic_min.max()), abs(ic_min.min()), 1e-8),
    ], axis=1).astype(np.float32)
    print(f"Input features: {input_features.shape}")

    for model_dir in ['SEA', 'SEA-baseline']:
        out_dir = f'./{model_dir}/data_computed/WE1'
        os.makedirs(out_dir, exist_ok=True)

        np.save(f'{out_dir}/field_data.npy', all_fields.astype(np.float32))
        np.save(f'{out_dir}/coordinates.npy', coordinates.astype(np.float32))
        np.save(f'{out_dir}/input_data.npy', input_features)

        T = all_fields.shape[1]
        input_tiled = np.tile(input_features.reshape(-1, 1, 4), (1, T, 1))
        np.save(f'{out_dir}/input_data_tiled.npy', input_tiled.astype(np.float32))

        np.savez(f'{out_dir}/edge_data.npz',
                 edge_index=edge_index, edge_attr=edge_attr)

        split_file = f'{out_dir}/split_info.txt'
        with open(split_file, 'w') as f:
            f.write(f"train: 0-{n_train-1}\n")
            f.write(f"valid: {n_train}-{n_train+n_valid-1}\n")
            f.write(f"test: {n_train+n_valid}-{n_train+n_valid+n_test-1}\n")

        print(f"\nSaved to {out_dir}/")
        print(f"  field_data.npy: {all_fields.shape}")
        print(f"  coordinates.npy: {coordinates.shape}")
        print(f"  input_data.npy: {input_features.shape}")
        print(f"  input_data_tiled.npy: {input_tiled.shape}")
        print(f"  edge_data.npz: edge_index={edge_index.shape}")

if __name__ == '__main__':
    main()
