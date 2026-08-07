#!/usr/bin/env python3
"""Compute edge_data.npz for high-resolution cylinder flow mesh using Delaunay triangulation."""

import numpy as np
from scipy.spatial import Delaunay

coords = np.load('./SEA/data/CF/all_data/coordinates.npy')
print(f"Coordinates shape: {coords.shape}")

x = coords[0]
y = coords[1]
points = np.stack([x, y], axis=-1)
print(f"Points shape: {points.shape}, N={len(points)}")

tri = Delaunay(points)
print(f"Number of simplices: {len(tri.simplices)}")

edges_set = set()
for simplex in tri.simplices:
    for i in range(3):
        for j in range(i+1, 3):
            a, b = simplex[i], simplex[j]
            edges_set.add((min(a, b), max(a, b)))

edges = np.array(list(edges_set), dtype=np.int64)
print(f"Number of unique edges: {len(edges)}")

edge_index = np.stack([
    np.concatenate([edges[:, 0], edges[:, 1]]),
    np.concatenate([edges[:, 1], edges[:, 0]])
], axis=0)
print(f"edge_index shape: {edge_index.shape}")

dx = x[edge_index[0]] - x[edge_index[1]]
dy = y[edge_index[0]] - y[edge_index[1]]
dist = np.sqrt(dx**2 + dy**2)
edge_attr = np.stack([dx, dy, dist], axis=-1)
print(f"edge_attr shape: {edge_attr.shape}")

out_path = './SEA/data/CF/all_data/edge_data.npz'
np.savez(out_path, edge_index=edge_index, edge_attr=edge_attr)
print(f"Saved to {out_path}")

edge_data = np.load(out_path)
print(f"Verification - edge_index: {edge_data['edge_index'].shape}, edge_attr: {edge_data['edge_attr'].shape}")
