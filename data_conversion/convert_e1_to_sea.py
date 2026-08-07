#!/usr/bin/env python3
"""Convert E1 (advection equation) from PyG per-graph format to SEA-compatible numpy arrays."""

import os
import torch
import numpy as np

def load_split(proc_dir):
    files = sorted([f for f in os.listdir(proc_dir) if f.startswith('data_') and f.endswith('.pt')])
    fields_list = []
    initial_conds = []
    pos = None
    edge_index = None
    edge_attr = None

    for f in files:
        d = torch.load(os.path.join(proc_dir, f), map_location='cpu', weights_only=False)
        y = d.y.numpy()
        ic = d.x[:, 3].numpy()
        fields_list.append(y)
        initial_conds.append(ic)
        if pos is None:
            pos = d.pos.numpy()
            edge_index = d.edge_index.numpy()
            edge_attr = d.edge_attr.numpy()

    fields = np.stack(fields_list, axis=0)
    initial_conds = np.stack(initial_conds, axis=0)
    return fields, initial_conds, pos, edge_index, edge_attr

def main():
    base = './mesh_operator/data/mp_pde/E1'

    print("Loading train split...")
    train_fields, train_ic, pos, edge_index, edge_attr = load_split(
        f'{base}/train/processed/pde_250-200')
    print(f"  Train fields: {train_fields.shape}")

    print("Loading valid split...")
    valid_fields, valid_ic, _, _, _ = load_split(
        f'{base}/valid/processed/pde_250-200')
    print(f"  Valid fields: {valid_fields.shape}")

    print("Loading test split...")
    test_fields, test_ic, _, _, _ = load_split(
        f'{base}/test/processed/pde_250-200')
    print(f"  Test fields: {test_fields.shape}")

    all_fields = np.concatenate([train_fields, valid_fields, test_fields], axis=0)
    all_ic = np.concatenate([train_ic, valid_ic, test_ic], axis=0)
    print(f"\nCombined fields: {all_fields.shape}")

    n_train = len(train_fields)
    n_valid = len(valid_fields)
    n_test = len(test_fields)

    # Coordinates in (d, N) format
    zeros = np.zeros_like(pos)
    coords_2d = np.concatenate([pos, zeros], axis=-1)  # (200, 2)
    coordinates = coords_2d.T.astype(np.float32)  # (2, 200)

    # Edge attr: (E, 3) with [distance, 0, distance]
    edge_attr_3col = np.stack([edge_attr, np.zeros_like(edge_attr), edge_attr], axis=1)

    # Conditioning features from initial conditions
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

    T = all_fields.shape[1]
    input_tiled = np.tile(input_features.reshape(-1, 1, 4), (1, T, 1))

    for model_dir in ['SEA', 'SEA-baseline']:
        out_dir = f'./{model_dir}/data_computed/E1'
        os.makedirs(out_dir, exist_ok=True)

        np.save(f'{out_dir}/field_data.npy', all_fields.astype(np.float32))
        np.save(f'{out_dir}/coordinates.npy', coordinates)
        np.save(f'{out_dir}/input_data.npy', input_features)
        np.save(f'{out_dir}/input_data_tiled.npy', input_tiled.astype(np.float32))
        np.savez(f'{out_dir}/edge_data.npz', edge_index=edge_index, edge_attr=edge_attr_3col)

        with open(f'{out_dir}/split_info.txt', 'w') as f:
            f.write(f"train: 0-{n_train-1}\n")
            f.write(f"valid: {n_train}-{n_train+n_valid-1}\n")
            f.write(f"test: {n_train+n_valid}-{n_train+n_valid+n_test-1}\n")

        print(f"\nSaved to {out_dir}/")
        print(f"  field_data: {all_fields.shape}, coords: {coordinates.shape}")
        print(f"  input: {input_features.shape}, edge_attr: {edge_attr_3col.shape}")

if __name__ == '__main__':
    main()
