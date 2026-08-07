import torch
import numpy as np
import os, math
from typing import List, Tuple, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
from models.encoder_decoder import SpatialModel
from utils.modular_testing import unit_test_create_partitions2D, unit_test_create_partitions3D
from models.transforms import GraphFourier
from dataclasses import dataclass
from torch_geometric.utils import get_laplacian
from scipy.sparse.linalg import eigsh
import warnings
import scipy.sparse as sp

class SpectralPartitioner:
    def __init__(self,
                 coords: torch.Tensor,  # [N]
                 edge_index: torch.Tensor,     # [2, E], undirected or directed
                 edge_weight: torch.Tensor,    # [E]
                 num_nodes: int,
                 k: int = 64,                       # embedding dimension (≈ number of parts)
                 normalization: str = "sym",   # 'sym' (Lsym) | 'rw' (Lrw) | 'unnorm'
                 pad_id: int = -1,
                 pad_field_value: float = 0.0,
                 device: str = "cpu"):
        self.device = device
        self.full_coords = coords.to(self.device).float().T           # [N, d]
        self.edge_index = edge_index.long().to(self.device)
        self.edge_weight = edge_weight.float().to(self.device)  # [E]
        self.N = int(num_nodes)
        self.k = int(k)
        self.pad_id = pad_id
        self.pad_field_value = pad_field_value
        self.normalization = normalization
        self.x_coords = self.full_coords[:, 0]
        self.y_coords = self.full_coords[:, 1]


        self.L = self._build_laplacian(self.edge_index, self.edge_weight,
                                       self.N, normalization=self.normalization)
        self.laplacian = 'normalized'
        self.U = self._spectral_embedding(self.L, k=self.k)  # torch.FloatTensor [N, k]
        

    def create_partitions(self, vars, compute_gft=False, k_eigs=None):
        """
        Partition and (optionally) compute per-patch GFT basis U.

        Inputs
        ------
        vars       : list of tensors shaped [T, N] (time, nodes) for each variable
        compute_gft: bool, if True also compute GFT for each patch (returns U, eig)
        k_eigs     : int or None. If set, keep min(k_eigs, n_patch) eigenvectors per patch.
                     If None, keep all (i.e., n_patch) eigenvectors.

        Returns
        -------
        padded_partitions : list[(coords[B=nodes,2], fields[T, nodes, num_vars])]
        padded_index_map  : list[LongTensor[nodes]] with pad_id for padded entries
        U_padded          : FloatTensor [num_patches, max_len, K_max]  (if compute_gft=True)
        eig_padded        : FloatTensor [num_patches, K_max]           (if compute_gft=True)
        """
        self.var_list = [var.to(self.device).float() for var in vars if var is not None]
        if len(self.var_list) == 0:
            raise ValueError("At least one variable must be provided")

        indices = self._partition(self.k, balanced=True)

        partitions = []
        index_map = []
        # For optional GFT
        U_list = []
        eig_list = []

        for i in range(self.k):
            mask = indices == i
            idx = mask.nonzero(as_tuple=False).view(-1)
            index_map.append(idx)

            if torch.any(mask):
                partition_coords = self.full_coords[idx]
                partition_fields = torch.stack([var[:, mask] for var in self.var_list], dim=2)
            else:
                partition_coords = torch.empty((0, 2), dtype=torch.float32, device=self.device)
                partition_fields = torch.empty((self.var_list[0].shape[0], 0, len(self.var_list)),
                                                dtype=torch.float32, device=self.device)

            partitions.append((partition_coords, partition_fields))

            # Compute GFT on the subgraph of VALID nodes only
            if compute_gft:
                U_patch, eig_patch = self._compute_patch_gft(idx, partition_coords, k_eigs)
                U_list.append(U_patch)   # [n_patch, k_eff] or [0, 0]
                eig_list.append(eig_patch)  # [k_eff] or [0]

        self.index_map = index_map
        self.padded_partitions, self.padded_index_map = self.pad_partitions(partitions, index_map)

        if not compute_gft:
            return self.padded_partitions, self.padded_index_map

        # Pad U/eig so they can be stacked
        max_len = max(coords.shape[0] for coords, _ in partitions)
        if k_eigs is None:
            K_max = max_len
        else:
            K_max = int(min(k_eigs, max_len))

        num_patches = len(U_list)
        U_padded = torch.zeros((num_patches, max_len, K_max), device=self.device, dtype=torch.float32)
        eig_padded = torch.zeros((num_patches, K_max), device=self.device, dtype=torch.float32)

        for p, (U_patch, eig_patch) in enumerate(zip(U_list, eig_list)):
            n = U_patch.shape[0]
            k_eff = U_patch.shape[1] if U_patch.ndim == 2 else 0
            if n > 0 and k_eff > 0:
                U_padded[p, :n, :k_eff] = U_patch
                eig_padded[p, :k_eff] = eig_patch

        # Save for later use if needed
        self.U_padded = U_padded
        self.eig_padded = eig_padded

        return self.padded_partitions, self.padded_index_map, U_padded, eig_padded

    def pad_partitions(self, partitions, index_map):
        max_len = max(coords.shape[0] for coords, _ in partitions)

        padded_partitions = []
        padded_index_map  = []
        num_vars = len(self.var_list)

        for (coords, fields), indices in zip(partitions, index_map):
            pad_size = max_len - coords.shape[0]

            if pad_size > 0:
                pad_coords = torch.full((pad_size, 2), self.pad_field_value, dtype=torch.float32, device=self.device)
                padded_coords = torch.cat((coords, pad_coords), dim=0)

                pad_fields = torch.full((self.var_list[0].shape[0], pad_size, num_vars),
                                        self.pad_field_value, dtype=torch.float32, device=self.device)
                padded_fields = torch.cat((fields, pad_fields), dim=1)

                pad_indices = torch.full((pad_size,), self.pad_id, dtype=torch.int64, device=self.device)
                padded_indices = torch.cat((indices, pad_indices), dim=0)
            else:
                padded_coords = coords
                padded_fields = fields
                padded_indices = indices

            padded_partitions.append((padded_coords, padded_fields))
            padded_index_map.append(padded_indices)

        return padded_partitions, padded_index_map

    def inverse_partition(self, external_partitions=None, time_dim=None):
        reconstructed_coords = torch.empty_like(self.full_coords)

        dummy_var = torch.stack(self.var_list, dim=2)
        _, C, F = dummy_var.shape
        B = external_partitions[0][1].shape[0]
        reconstructed_fields = torch.empty((B, C, F), device=self.device)

        time_dim   = time_dim            if time_dim            is not None else reconstructed_fields.shape[0]
        partitions = external_partitions if external_partitions is not None else self.padded_partitions

        reconstructed_fields = reconstructed_fields[:time_dim]

        for idx, (coords, fields) in enumerate(partitions):
            indices         =   self.padded_index_map[idx]
            valid_mask      =   indices != self.pad_id
            valid_indices   =   indices[valid_mask]

            reconstructed_coords[valid_indices, :]     = coords[valid_mask]
            reconstructed_fields[:, valid_indices, :]  = fields[:, valid_mask]

        return reconstructed_coords, reconstructed_fields

    # ----------------- GFT internals -----------------

    def _compute_patch_gft(self, global_indices, coords, k_eigs):
        """
        Build a subgraph on nodes 'global_indices', compute Laplacian eigendecomp.
        Returns (U, eig) where U is [n, k_eff], eig is [k_eff].
        If n==0: returns (zeros[0,0], zeros[0])
        """
        n = coords.shape[0]
        if n == 0:
            return (torch.zeros(0, 0, device=self.device), torch.zeros(0, device=self.device))

        # Compute edges (prefer filtering from global; else kNN)
        if self.edge_index is not None:
            sub_ei, sub_w = self._subgraph_from_global(global_indices)
            if sub_ei is None:  # no edges after filtering, fallback to kNN
                sub_ei, sub_w = self._knn_graph(coords, max(1, min(self.knn_k, n-1)))
        else:
            sub_ei, sub_w = self._knn_graph(coords, max(1, min(self.knn_k, n-1)))

        # Build Laplacian
        L = self._laplacian_from_edges(n, sub_ei, sub_w, self.laplacian)

        # Eigendecomposition (symmetric)
        # eigh returns ascending eigenvalues
        eigvals, eigvecs = torch.linalg.eigh(L)

        # How many to keep
        if k_eigs is None:
            k_eff = n
        else:
            k_eff = int(min(k_eigs, n))

        U = eigvecs[:, :k_eff].contiguous().float()
        lam = eigvals[:k_eff].contiguous().float()
        return U, lam

    def _subgraph_from_global(self, global_indices):
        """
        Filter global edges to those with both endpoints in 'global_indices'.
        Returns local edge_index [2, E_sub] and edge_weight [E_sub].
        """
        if self.edge_index is None:
            return None, None

        N_global = self.full_coords.shape[0]
        idx_map = torch.full((N_global,), -1, dtype=torch.long, device=self.device)
        if global_indices.numel() > 0:
            idx_map[global_indices] = torch.arange(global_indices.numel(), device=self.device, dtype=torch.long)

        e0_local = idx_map[self.edge_index[0]]
        e1_local = idx_map[self.edge_index[1]]
        mask = (e0_local >= 0) & (e1_local >= 0)
        if not torch.any(mask):
            return None, None

        e0 = e0_local[mask]
        e1 = e1_local[mask]
        w  = self.edge_weight[mask] if self.edge_weight is not None else torch.ones_like(e0, dtype=torch.float32)

        # Ensure undirected by adding reverse and coalescing
        ei = torch.cat([torch.stack([e0, e1], dim=0), torch.stack([e1, e0], dim=0)], dim=1)
        w  = torch.cat([w, w], dim=0)

        # Coalesce duplicates by summing weights (simple & deterministic)
        # Build dense accumulation index to unique (i,j)
        key = e0 * (global_indices.numel()) + e1
        key2 = e1 * (global_indices.numel()) + e0
        key_all = torch.cat([key, key2], dim=0)  # already duplicated above; keep as-is
        # A light coalesce without sorting: use scatter_add into dense matrix then read back
        n = global_indices.numel()
        W = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        W[ei[0], ei[1]] += w
        # Extract edges from W
        src, dst = torch.nonzero(W, as_tuple=True)
        weights = W[src, dst]
        return torch.stack([src, dst], dim=0), weights
    
    # def _partition(self, num_parts: int, balanced: bool = False, n_init: int = 10, seed: int = 42):
    #     """
    #     Run k-means on rows of U to get cluster labels.
    #     If balanced=True, do a simple balanced variant (optional).
    #     Returns LongTensor labels [N].
    #     """
    #     import pymetis

    #     if num_parts != self.k:
    #         warnings.warn("Typically set k == num_parts for spectral clustering.")

    #     xadj, adjncy, eweights = self.edge_index_to_metis_csr(
    #         self.edge_index, self.edge_weight, self.N,
    #         make_undirected=True, remove_self_loops=True, weight_scale=1000)

    #     n_cuts, labels = pymetis.part_graph(num_parts, xadj=xadj, adjncy=adjncy, eweights=eweights)

    #     return torch.tensor(labels, dtype=torch.long, device=self.device)
    
    # @staticmethod
    # def edge_index_to_metis_csr(
    #     edge_index: torch.Tensor,            # [2, E] long
    #     edge_weight: Optional[torch.Tensor], # [E] float or int
    #     num_nodes: Optional[int] = None,
    #     make_undirected: bool = True,
    #     remove_self_loops: bool = True,
    #     weight_scale: Optional[int] = 1000,  # scale floats to ints; set None to auto-scale
    # ) -> Tuple[List[int], List[int], List[int]]:
    #     """
    #     Returns:
    #     xadj    : list[int] of length N+1
    #     adjncy  : list[int] of length nnz
    #     eweights: list[int] of length nnz (positive ints)
    #     """
    #     assert edge_index.dim() == 2 and edge_index.size(0) == 2
    #     if num_nodes is None:
    #         num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0

    #     E = edge_index.size(1)
    #     device = edge_index.device

    #     # default weights = 1
    #     if edge_weight is None:
    #         edge_weight = torch.ones(E, dtype=torch.float32, device=device)
    #     else:
    #         edge_weight = edge_weight.to(device)

    #     # optional: drop self-loops
    #     if remove_self_loops and E > 0:
    #         mask = edge_index[0] != edge_index[1]
    #         edge_index = edge_index[:, mask]
    #         edge_weight = edge_weight[mask]

    #     # ensure undirected by mirroring
    #     if make_undirected and edge_index.numel() > 0:
    #         edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    #         edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    #     # coalesce duplicates by summing weights, then convert to CSR
    #     A = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).coalesce()
    #     A_csr = A.to_sparse_csr()
    #     rowptr = A_csr.crow_indices()   # [N+1]
    #     colind = A_csr.col_indices()    # [nnz]
    #     vals   = A_csr.values()         # [nnz]

    #     # METIS requires positive integer edge weights
    #     if torch.is_floating_point(vals):
    #         v = torch.clamp(vals, min=0)  # non-negative
    #         if weight_scale is None:
    #             vmax = float(v.max().item()) if v.numel() else 1.0
    #             scale = (1000.0 / vmax) if vmax > 0 else 1.0
    #             ivals = torch.clamp((v * scale).round(), min=1).to(torch.int64)
    #         else:
    #             ivals = torch.clamp((v * weight_scale).round(), min=1).to(torch.int64)
    #     else:
    #         ivals = torch.clamp(vals.to(torch.int64), min=1)

    #     # PyMETIS expects plain Python lists/ints
    #     xadj    = rowptr.tolist()
    #     adjncy  = colind.tolist()
    #     eweights= ivals.tolist()
    #     return xadj, adjncy, eweights
    
    def _knn_graph(self, coords, k):
        """
        Build a symmetric k-NN graph within the patch using Euclidean distances.
        Weights: Gaussian kernel with sigma = median(nonzero distances) (fallback=1.0).
        """
        # coords: [n,2]
        n = coords.shape[0]
        if n == 1:
            return torch.zeros((2,0), dtype=torch.long, device=self.device), torch.zeros(0, device=self.device)

        D = torch.cdist(coords, coords, p=2)  # [n, n]
        # exclude self by setting diagonal to +inf for topk
        D.fill_diagonal_(float('inf'))
        k = max(1, min(k, n-1))
        knn_dist, knn_idx = torch.topk(D, k=k, largest=False, dim=1)  # neighbors for each node

        # Edges i -> j
        src = torch.arange(n, device=self.device).unsqueeze(1).expand_as(knn_idx).reshape(-1)
        dst = knn_idx.reshape(-1)

        # Make undirected by adding reverse and dedup via dense coalesce
        ei = torch.cat([torch.stack([src, dst], dim=0),
                        torch.stack([dst, src], dim=0)], dim=1)

        # Gaussian weights
        flat_d = torch.cat([knn_dist.reshape(-1), knn_dist.reshape(-1)], dim=0)
        nz = flat_d[flat_d.isfinite() & (flat_d > 0)]
        sigma = nz.median() if nz.numel() > 0 else torch.tensor(1.0, device=self.device)
        # clamp sigma to avoid tiny values
        sigma = torch.clamp(sigma, min=1e-6)
        w = torch.exp(-(flat_d ** 2) / (2.0 * sigma ** 2))
        # coalesce as dense
        W = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        W[ei[0], ei[1]] = torch.maximum(W[ei[0], ei[1]], w)  # take max to avoid multiple edges
        src2, dst2 = torch.nonzero(W, as_tuple=True)
        weights = W[src2, dst2]
        return torch.stack([src2, dst2], dim=0), weights

    def _laplacian_from_edges(self, n, edge_index, edge_weight, mode):
        """
        Build dense Laplacian matrix from edge list.
        """
        W = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        if edge_index is not None and edge_index.numel() > 0:
            W[edge_index[0], edge_index[1]] += edge_weight
            # force symmetric (undirected)
            W = 0.5 * (W + W.t())
            W.fill_diagonal_(0.)

        d = torch.sum(W, dim=1)
        if mode == 'unnormalized':
            D = torch.diag(d)
            L = D - W
        else:  # normalized
            # D^{-1/2} W D^{-1/2}
            inv_sqrt = torch.where(d > 0, d.rsqrt(), torch.zeros_like(d))
            S = inv_sqrt.unsqueeze(1) * W * inv_sqrt.unsqueeze(0)
            L = torch.eye(n, device=self.device, dtype=torch.float32) - S
        return L

    @staticmethod
    def _build_laplacian(edge_index: torch.Tensor,
                         edge_weight: torch.Tensor,
                         N: int,
                         normalization: str = "sym") -> sp.csr_matrix:
        """Return SciPy CSR Laplacian matrix."""
        i0 = edge_index[0].numpy()
        i1 = edge_index[1].numpy()
        w  = edge_weight.numpy()

        # Build symmetric adjacency
        A = sp.coo_matrix((w, (i0, i1)), shape=(N, N))
        A = A.maximum(A.T)  # ensure symmetry

        # Degrees
        d = np.asarray(A.sum(axis=1)).ravel()
        Dinv = sp.diags(1.0 / np.maximum(d, 1e-12))

        if normalization == "unnorm":
            L = sp.diags(d) - A
        elif normalization == "sym":
            # L_sym = I - D^{-1/2} A D^{-1/2}
            Dm12 = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-12)))
            L = sp.eye(N, format="csr") - Dm12 @ A @ Dm12
        elif normalization == "rw":
            # L_rw = I - D^{-1} A
            L = sp.eye(N, format="csr") - Dinv @ A
        else:
            raise ValueError("normalization must be one of {'sym','rw','unnorm'}")
        return L.tocsr()

    @staticmethod
    def _spectral_embedding(L: sp.csr_matrix,
                            k: int,
                            maxiter: int = 2000,
                            tol: float = 1e-4) -> torch.Tensor:
        """
        Compute the k smallest eigenvectors (Ng–Jordan–Weiss).
        Returns a row-normalized torch.FloatTensor [N, k].
        """
        N = L.shape[0]
        if k >= N:
            raise ValueError("k must be smaller than number of nodes")

        try:
            vals, vecs = eigsh(L, k=k, which="SM", maxiter=maxiter, tol=tol)  # float64
        except Exception as e:
            warnings.warn(f"eigsh failed ({e}); retrying with looser tol.")
            vals, vecs = eigsh(L, k=k, which="SM", maxiter=maxiter, tol=max(1e-3, tol))

        U = vecs  # [N, k], numpy float64
        # Row-normalize (Ng–Jordan–Weiss)
        row_norm = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / (row_norm + 1e-12)
        return torch.from_numpy(U.astype(np.float32))  # [N, k]

    def _partition(self, num_parts: int, balanced: bool = False, n_init: int = 10, seed: int = 42):
        """
        Run k-means on rows of U to get cluster labels.
        If balanced=True, do a simple balanced variant (optional).
        Returns LongTensor labels [N].
        """
        from sklearn.cluster import KMeans

        if num_parts != self.k:
            warnings.warn("Typically set k == num_parts for spectral clustering.")

        U_np = self.U.numpy()
        if not balanced:
            km = KMeans(n_clusters=num_parts, n_init=n_init, random_state=seed)
            labels = km.fit_predict(U_np)
        else:
            # Simple balanced k-means: iterative size-capped assignment + Lloyd steps.
            # (Good enough to reduce big imbalances; swap in a better solver later.)
            labels = balanced_kmeans(U_np, num_parts, n_init=n_init, seed=seed)

        return torch.from_numpy(labels.astype(np.int64))

# --- Optional: a tiny balanced k-means helper (placeholder) ---
def balanced_kmeans(X, k, n_init=10, seed=0):
    # Start with vanilla k-means and then greedily rebalance by moving farthest points
    from sklearn.cluster import KMeans
    rng = np.random.RandomState(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        km = KMeans(n_clusters=k, n_init=1, random_state=rng.randint(1<<31))
        labels = km.fit_predict(X)
        centers = km.cluster_centers_
        # target size
        N = X.shape[0]
        tgt = int(np.ceil(N / k))
        # greedy rebalance
        for it in range(4):
            sizes = np.bincount(labels, minlength=k)
            over = np.where(sizes > tgt)[0]
            under = np.where(sizes < tgt)[0]
            if len(over) == 0 or len(under) == 0: break
            D = ((X - centers[labels])**2).sum(1)
            for c in over:
                idx_c = np.where(labels == c)[0]
                # move farthest points in c to their best underfull cluster
                idx_sorted = idx_c[np.argsort(-D[idx_c])]
                for j in idx_sorted:
                    # find nearest underfull center
                    d2 = ((X[j] - centers[under])**2).sum(1)
                    c2 = under[np.argmin(d2)]
                    labels[j] = c2
                    sizes[c] -= 1; sizes[c2] += 1
                    if sizes[c] <= tgt: break
                under = np.where(sizes < tgt)[0]
                if len(under) == 0: break
            # recompute centers
            for c in range(k):
                pts = X[labels == c]
                if len(pts) > 0:
                    centers[c] = pts.mean(0)
        inertia = ((X - centers[labels])**2).sum()
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels


class DataPartitioner2D:
    """
    2D spatial partitioner with optional per-patch GFT.
    If compute_gft=True (in create_partitions), returns padded U and eigenvalues for each patch.

    Parameters
    ----------
    x_coords, y_coords : 1D tensors of length N (float)
    m, n               : number of bins along x and y (grid boundaries = m and n)
    pad_id             : index used when padding index maps
    pad_field_value    : value used when padding coords/fields
    device             : 'cpu' or 'cuda'

    Optional Graph Inputs
    ---------------------
    edge_index : LongTensor [2, E] global edges (undirected or directed)
    edge_weight: FloatTensor [E] weights for edge_index (optional)
    knn_k      : neighbors per node when building per-patch graph if no global edges provided
    laplacian  : 'normalized' (I - D^{-1/2} W D^{-1/2}) or 'unnormalized' (D - W)
    """
    def __init__(
        self,
        x_coords,
        y_coords,
        m=9,
        n=9,
        pad_id=-1,
        pad_field_value=0.0,
        device='cpu',
        edge_index=None,
        edge_weight=None,
        knn_k=6,
        laplacian='normalized',
    ):
        self.device = device
        self.x_coords = x_coords.to(self.device).float()
        self.y_coords = y_coords.to(self.device).float()
        self.full_coords = torch.stack((self.x_coords, self.y_coords), dim=1)

        self.m = m
        self.n = n
        self.pad_id = pad_id
        self.pad_field_value = float(pad_field_value)

        # Optional global graph
        self.edge_index = None if edge_index is None else edge_index.to(self.device).long()
        self.edge_weight = None if edge_weight is None else edge_weight.to(self.device).float()
        self.knn_k = int(knn_k)
        assert laplacian in ('normalized', 'unnormalized')
        self.laplacian = laplacian

    # ----------------- public API -----------------

    def create_partitions(self, vars, compute_gft=False, k_eigs=None):
        """
        Partition and (optionally) compute per-patch GFT basis U.

        Inputs
        ------
        vars       : list of tensors shaped [T, N] (time, nodes) for each variable
        compute_gft: bool, if True also compute GFT for each patch (returns U, eig)
        k_eigs     : int or None. If set, keep min(k_eigs, n_patch) eigenvectors per patch.
                     If None, keep all (i.e., n_patch) eigenvectors.

        Returns
        -------
        padded_partitions : list[(coords[B=nodes,2], fields[T, nodes, num_vars])]
        padded_index_map  : list[LongTensor[nodes]] with pad_id for padded entries
        U_padded          : FloatTensor [num_patches, max_len, K_max]  (if compute_gft=True)
        eig_padded        : FloatTensor [num_patches, K_max]           (if compute_gft=True)
        """
        self.var_list = [var.to(self.device).float() for var in vars if var is not None]
        if len(self.var_list) == 0:
            raise ValueError("At least one variable must be provided")

        x_min, x_max = torch.min(self.x_coords), torch.max(self.x_coords)
        y_min, y_max = torch.min(self.y_coords), torch.max(self.y_coords)

        x_boundary = torch.linspace(x_min, x_max, self.m, device=self.device)
        y_boundary = torch.linspace(y_min, y_max, self.n, device=self.device)

        x_indices = torch.bucketize(self.x_coords, x_boundary, right=True).clamp_(1, self.m - 1)
        y_indices = torch.bucketize(self.y_coords, y_boundary, right=True).clamp_(1, self.n - 1)

        partitions = []
        index_map = []
        # For optional GFT
        U_list = []
        eig_list = []

        for i in range(1, self.m):
            for j in range(1, self.n):
                mask = (x_indices == i) & (y_indices == j)
                indices = mask.nonzero(as_tuple=False).view(-1)
                index_map.append(indices)

                if torch.any(mask):
                    partition_coords = torch.stack((self.x_coords[mask], self.y_coords[mask]), dim=1)
                    partition_fields = torch.stack([var[:, mask] for var in self.var_list], dim=2)
                else:
                    partition_coords = torch.empty((0, 2), dtype=torch.float32, device=self.device)
                    partition_fields = torch.empty((self.var_list[0].shape[0], 0, len(self.var_list)),
                                                   dtype=torch.float32, device=self.device)

                partitions.append((partition_coords, partition_fields))

                # Compute GFT on the subgraph of VALID nodes only
                if compute_gft:
                    U_patch, eig_patch = self._compute_patch_gft(indices, partition_coords, k_eigs)
                    U_list.append(U_patch)   # [n_patch, k_eff] or [0, 0]
                    eig_list.append(eig_patch)  # [k_eff] or [0]

        self.index_map = index_map
        self.padded_partitions, self.padded_index_map = self.pad_partitions(partitions, index_map)

        if not compute_gft:
            return self.padded_partitions, self.padded_index_map

        # Pad U/eig so they can be stacked
        max_len = max(coords.shape[0] for coords, _ in partitions)
        if k_eigs is None:
            K_max = max_len
        else:
            K_max = int(min(k_eigs, max_len))

        num_patches = len(U_list)
        U_padded = torch.zeros((num_patches, max_len, K_max), device=self.device, dtype=torch.float32)
        eig_padded = torch.zeros((num_patches, K_max), device=self.device, dtype=torch.float32)

        for p, (U_patch, eig_patch) in enumerate(zip(U_list, eig_list)):
            n = U_patch.shape[0]
            k_eff = U_patch.shape[1] if U_patch.ndim == 2 else 0
            if n > 0 and k_eff > 0:
                U_padded[p, :n, :k_eff] = U_patch
                eig_padded[p, :k_eff] = eig_patch

        # Save for later use if needed
        self.U_padded = U_padded
        self.eig_padded = eig_padded

        return self.padded_partitions, self.padded_index_map, U_padded, eig_padded

    def pad_partitions(self, partitions, index_map):
        max_len = max(coords.shape[0] for coords, _ in partitions)

        padded_partitions = []
        padded_index_map  = []
        num_vars = len(self.var_list)

        for (coords, fields), indices in zip(partitions, index_map):
            pad_size = max_len - coords.shape[0]

            if pad_size > 0:
                pad_coords = torch.full((pad_size, 2), self.pad_field_value, dtype=torch.float32, device=self.device)
                padded_coords = torch.cat((coords, pad_coords), dim=0)

                pad_fields = torch.full((self.var_list[0].shape[0], pad_size, num_vars),
                                        self.pad_field_value, dtype=torch.float32, device=self.device)
                padded_fields = torch.cat((fields, pad_fields), dim=1)

                pad_indices = torch.full((pad_size,), self.pad_id, dtype=torch.int64, device=self.device)
                padded_indices = torch.cat((indices, pad_indices), dim=0)
            else:
                padded_coords = coords
                padded_fields = fields
                padded_indices = indices

            padded_partitions.append((padded_coords, padded_fields))
            padded_index_map.append(padded_indices)

        return padded_partitions, padded_index_map

    def inverse_partition(self, external_partitions=None, time_dim=None):
        reconstructed_coords = torch.empty_like(self.full_coords)

        dummy_var = torch.stack(self.var_list, dim=2)
        _, C, F = dummy_var.shape
        B = external_partitions[0][1].shape[0]
        reconstructed_fields = torch.empty((B, C, F), device=self.device)

        time_dim   = time_dim            if time_dim            is not None else reconstructed_fields.shape[0]
        partitions = external_partitions if external_partitions is not None else self.padded_partitions

        reconstructed_fields = reconstructed_fields[:time_dim]

        for idx, (coords, fields) in enumerate(partitions):
            indices         =   self.padded_index_map[idx]
            valid_mask      =   indices != self.pad_id
            valid_indices   =   indices[valid_mask]

            reconstructed_coords[valid_indices, :]     = coords[valid_mask]
            reconstructed_fields[:, valid_indices, :]  = fields[:, valid_mask]

        return reconstructed_coords, reconstructed_fields

    # ----------------- GFT internals -----------------

    def _compute_patch_gft(self, global_indices, coords, k_eigs):
        """
        Build a subgraph on nodes 'global_indices', compute Laplacian eigendecomp.
        Returns (U, eig) where U is [n, k_eff], eig is [k_eff].
        If n==0: returns (zeros[0,0], zeros[0])
        """
        n = coords.shape[0]
        if n == 0:
            return (torch.zeros(0, 0, device=self.device), torch.zeros(0, device=self.device))

        # Compute edges (prefer filtering from global; else kNN)
        if self.edge_index is not None:
            sub_ei, sub_w = self._subgraph_from_global(global_indices)
            if sub_ei is None:  # no edges after filtering, fallback to kNN
                sub_ei, sub_w = self._knn_graph(coords, max(1, min(self.knn_k, n-1)))
        else:
            sub_ei, sub_w = self._knn_graph(coords, max(1, min(self.knn_k, n-1)))

        # Build Laplacian
        L = self._laplacian_from_edges(n, sub_ei, sub_w, self.laplacian)

        # Eigendecomposition (symmetric)
        # eigh returns ascending eigenvalues
        eigvals, eigvecs = torch.linalg.eigh(L)

        # How many to keep
        if k_eigs is None:
            k_eff = n
        else:
            k_eff = int(min(k_eigs, n))

        U = eigvecs[:, :k_eff].contiguous().float()
        lam = eigvals[:k_eff].contiguous().float()
        return U, lam

    def _subgraph_from_global(self, global_indices):
        """
        Filter global edges to those with both endpoints in 'global_indices'.
        Returns local edge_index [2, E_sub] and edge_weight [E_sub].
        """
        if self.edge_index is None:
            return None, None

        N_global = self.full_coords.shape[0]
        idx_map = torch.full((N_global,), -1, dtype=torch.long, device=self.device)
        if global_indices.numel() > 0:
            idx_map[global_indices] = torch.arange(global_indices.numel(), device=self.device, dtype=torch.long)

        e0_local = idx_map[self.edge_index[0]]
        e1_local = idx_map[self.edge_index[1]]
        mask = (e0_local >= 0) & (e1_local >= 0)
        if not torch.any(mask):
            return None, None

        e0 = e0_local[mask]
        e1 = e1_local[mask]
        w  = self.edge_weight[mask] if self.edge_weight is not None else torch.ones_like(e0, dtype=torch.float32)

        # Ensure undirected by adding reverse and coalescing
        ei = torch.cat([torch.stack([e0, e1], dim=0), torch.stack([e1, e0], dim=0)], dim=1)
        w  = torch.cat([w, w], dim=0)

        # Coalesce duplicates by summing weights (simple & deterministic)
        # Build dense accumulation index to unique (i,j)
        key = e0 * (global_indices.numel()) + e1
        key2 = e1 * (global_indices.numel()) + e0
        key_all = torch.cat([key, key2], dim=0)  # already duplicated above; keep as-is
        # A light coalesce without sorting: use scatter_add into dense matrix then read back
        n = global_indices.numel()
        W = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        W[ei[0], ei[1]] += w
        # Extract edges from W
        src, dst = torch.nonzero(W, as_tuple=True)
        weights = W[src, dst]
        return torch.stack([src, dst], dim=0), weights

    def _knn_graph(self, coords, k):
        """
        Build a symmetric k-NN graph within the patch using Euclidean distances.
        Weights: Gaussian kernel with sigma = median(nonzero distances) (fallback=1.0).
        """
        # coords: [n,2]
        n = coords.shape[0]
        if n == 1:
            return torch.zeros((2,0), dtype=torch.long, device=self.device), torch.zeros(0, device=self.device)

        D = torch.cdist(coords, coords, p=2)  # [n, n]
        # exclude self by setting diagonal to +inf for topk
        D.fill_diagonal_(float('inf'))
        k = max(1, min(k, n-1))
        knn_dist, knn_idx = torch.topk(D, k=k, largest=False, dim=1)  # neighbors for each node

        # Edges i -> j
        src = torch.arange(n, device=self.device).unsqueeze(1).expand_as(knn_idx).reshape(-1)
        dst = knn_idx.reshape(-1)

        # Make undirected by adding reverse and dedup via dense coalesce
        ei = torch.cat([torch.stack([src, dst], dim=0),
                        torch.stack([dst, src], dim=0)], dim=1)

        # Gaussian weights
        flat_d = torch.cat([knn_dist.reshape(-1), knn_dist.reshape(-1)], dim=0)
        nz = flat_d[flat_d.isfinite() & (flat_d > 0)]
        sigma = nz.median() if nz.numel() > 0 else torch.tensor(1.0, device=self.device)
        # clamp sigma to avoid tiny values
        sigma = torch.clamp(sigma, min=1e-6)
        w = torch.exp(-(flat_d ** 2) / (2.0 * sigma ** 2))
        # coalesce as dense
        W = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        W[ei[0], ei[1]] = torch.maximum(W[ei[0], ei[1]], w)  # take max to avoid multiple edges
        src2, dst2 = torch.nonzero(W, as_tuple=True)
        weights = W[src2, dst2]
        return torch.stack([src2, dst2], dim=0), weights

    def _laplacian_from_edges(self, n, edge_index, edge_weight, mode):
        """
        Build dense Laplacian matrix from edge list.
        """
        W = torch.zeros((n, n), device=self.device, dtype=torch.float32)
        if edge_index is not None and edge_index.numel() > 0:
            W[edge_index[0], edge_index[1]] += edge_weight
            # force symmetric (undirected)
            W = 0.5 * (W + W.t())
            W.fill_diagonal_(0.)

        d = torch.sum(W, dim=1)
        if mode == 'unnormalized':
            D = torch.diag(d)
            L = D - W
        else:  # normalized
            # D^{-1/2} W D^{-1/2}
            inv_sqrt = torch.where(d > 0, d.rsqrt(), torch.zeros_like(d))
            S = inv_sqrt.unsqueeze(1) * W * inv_sqrt.unsqueeze(0)
            L = torch.eye(n, device=self.device, dtype=torch.float32) - S
        return L


class DataPartitioner3D:
    def __init__(self, x_coords, y_coords, z_coords, vars, m=9, n=9, k=9, pad_id=-1, pad_field_value=0, device='cpu'):
        self.device = device
        self.x_coords = x_coords.to(self.device).float()
        self.y_coords = y_coords.to(self.device).float()
        self.z_coords = z_coords.to(self.device).float()
        self.full_coords = torch.stack((self.x_coords, self.y_coords, self.z_coords), dim=1)

        self.var_list = [var.to(self.device).float() for var in vars if var is not None]

        if len(self.var_list) == 0:
            raise ValueError("At least one variable must be provided")

        self.m = m
        self.n = n
        self.k = k
        self.pad_id = pad_id
        self.pad_field_value = pad_field_value
        
    def create_partitions(self):
        x_min, x_max = torch.min(self.x_coords), torch.max(self.x_coords)
        y_min, y_max = torch.min(self.y_coords), torch.max(self.y_coords)
        z_min, z_max = torch.min(self.z_coords), torch.max(self.z_coords)

        x_boundary = torch.linspace(x_min, x_max, self.m, device=self.device)
        y_boundary = torch.linspace(y_min, y_max, self.n, device=self.device)
        z_boundary = torch.linspace(z_min, z_max, self.k, device=self.device)

        x_indices = torch.bucketize(self.x_coords, x_boundary, right=True)
        y_indices = torch.bucketize(self.y_coords, y_boundary, right=True)
        z_indices = torch.bucketize(self.z_coords, z_boundary, right=True)

        x_indices.clamp_(1, self.m - 1)
        y_indices.clamp_(1, self.n - 1)
        z_indices.clamp_(1, self.k - 1)

        partitions = []
        index_map = []

        for i in range(1, self.m):
            for j in range(1, self.n):
                for k in range(1, self.k):
                    mask = (x_indices == i) & (y_indices == j) & (z_indices == k)
                    indices = mask.nonzero(as_tuple=False).view(-1)
                    index_map.append(indices)

                    if torch.any(mask):
                        partition_coords = torch.stack((self.x_coords[mask], self.y_coords[mask], self.z_coords[mask]), dim=1)
                        partition_fields = torch.stack([var[:, mask] for var in self.var_list], dim=2)
                    else:
                        partition_coords = torch.empty((0, 3), dtype=torch.float32, device=self.device)
                        partition_fields = torch.empty((self.var_list[0].shape[0], 0, len(self.var_list)), dtype=torch.float32, device=self.device)

                    partitions.append((partition_coords, partition_fields))

        self.index_map = index_map
        self.padded_partitions, self.padded_index_map = self.pad_partitions(partitions, index_map)
        return self.padded_partitions, self.padded_index_map

    def pad_partitions(self, partitions, index_map):
        max_len = max(coords.shape[0] for coords, _ in partitions)

        padded_partitions = []
        padded_index_map  = []
        num_vars = len(self.var_list)

        for (coords, fields), indices in zip(partitions, index_map):
            pad_size = max_len - coords.shape[0]

            if pad_size > 0:
                pad_coords = torch.full((pad_size, 3), self.pad_field_value, dtype=torch.float32, device=self.device)
                padded_coords = torch.cat((coords, pad_coords), dim=0)

                pad_fields = torch.full((self.var_list[0].shape[0], pad_size, num_vars), self.pad_field_value, dtype=torch.float32, device=self.device)
                padded_fields = torch.cat((fields, pad_fields), dim=1)

                pad_indices = torch.full((pad_size,), self.pad_id, dtype=torch.int64, device=self.device)
                padded_indices = torch.cat((indices, pad_indices), dim=0)
            else:
                padded_coords = coords
                padded_fields = fields
                padded_indices = indices

            padded_partitions.append((padded_coords, padded_fields))
            padded_index_map.append(padded_indices)

        return padded_partitions, padded_index_map

    def inverse_partition(self, external_partitions=None, time_dim=None):
        reconstructed_coords = torch.empty_like(self.full_coords)

        dummy_var = torch.stack(self.var_list, dim=2)
        _,C,F = dummy_var.shape
        B = external_partitions[0][1].shape[0] if external_partitions else dummy_var.shape[0]
        reconstructed_fields = torch.empty((B,C,F), device=self.device)

        time_dim   = time_dim            if time_dim            is not None else reconstructed_fields.shape[0]
        partitions = external_partitions if external_partitions is not None else self.padded_partitions

        reconstructed_fields = reconstructed_fields[:time_dim]

        for idx, (coords, fields) in enumerate(partitions):
            indices         =   self.padded_index_map[idx]
            valid_mask      =   indices != self.pad_id
            valid_indices   =   indices[valid_mask]

            reconstructed_coords[valid_indices,:]     = coords[valid_mask]
            reconstructed_fields[:, valid_indices, :] = fields[:, valid_mask]

        return reconstructed_coords, reconstructed_fields




class MinMaxScaler:
    def __init__(self,
                 feature_range=(-1, 1),
                 name='scaler',
                 save_dir='.',
                 use_quantiles=False, q_low=1.0, q_high=99.0,
                 log_mode='off',            # 'off' | 'signed' | 'pos'
                 slog_scale='median',        # 'median' | 'mean' | float
                 eps=1e-12):
        if isinstance(feature_range, dict):
            feature_range = feature_range.get('feature_range', (-1, 1))
        if not (isinstance(feature_range, (list, tuple)) and len(feature_range) == 2):
            raise ValueError(f"feature_range must be a 2-tuple/list, got {feature_range}")
        if log_mode not in ('off', 'signed', 'pos'):
            raise ValueError("log_mode must be 'off', 'signed', or 'pos'")
        self.feature_range = tuple(feature_range)
        self.use_quantiles = use_quantiles
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.log_mode = log_mode
        self.slog_scale = slog_scale
        self.eps = eps

        self.min_val = None   # CPU tensors
        self.max_val = None
        self.s_log = None     # per-feature scale used in (signed) log, CPU
        self.name = name
        self.save_file = os.path.join(save_dir, f'{name}_min_max_values.pt')

    def _as_2d(self, data: torch.Tensor) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        G = data.shape[-1]
        return data.reshape(-1, G)

    # -------- log helper(s) (forward & inverse), per-feature --------
    def _compute_s_log(self, X: torch.Tensor) -> torch.Tensor:
        """Return per-feature s (CPU or device of X) used for log transforms."""
        if isinstance(self.slog_scale, (int, float)):
            s = torch.tensor(self.slog_scale, dtype=X.dtype, device=X.device).clamp(min=self.eps)
            return s.expand(X.shape[1])
        if self.slog_scale == 'median':
            s = X.abs().median(dim=0).values.clamp(min=self.eps) if self.log_mode == 'signed' \
                else X.clamp_min(0).median(dim=0).values.clamp(min=self.eps)
            return s
        if self.slog_scale == 'mean':
            s = X.abs().mean(dim=0).clamp(min=self.eps) if self.log_mode == 'signed' \
                else X.clamp_min(0).mean(dim=0).clamp(min=self.eps)
            return s
        raise ValueError("slog_scale must be 'median' | 'mean' | float")

    def _log_forward(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.log_mode == 'off':
            return X
        if self.log_mode == 'signed':
            return X.sign() * torch.log1p(X.abs() / s)
        # pos mode
        Xp = X.clamp_min(0.0)
        return torch.log1p(Xp / s)

    def _log_inverse(self, Z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.log_mode == 'off':
            return Z
        if self.log_mode == 'signed':
            # sign preserved by forward
            return Z.sign() * (torch.expm1(Z.abs()) * s)
        # pos mode
        return torch.expm1(Z).clamp_min(0.0) * s

    # ---------------- core API ----------------
    def fit(self, data: torch.Tensor):
        X = self._as_2d(data).to(torch.float64)  # fit in float64 for stable stats
        # compute and cache s_log if needed
        if self.log_mode != 'off':
            s = self._compute_s_log(X)
        else:
            s = None

        # transform to the working space
        Xw = self._log_forward(X, s) if s is not None else X

        if self.use_quantiles:
            ql = torch.quantile(Xw, self.q_low / 100.0, dim=0)
            qh = torch.quantile(Xw, self.q_high / 100.0, dim=0)
            self.min_val = ql
            self.max_val = qh
        else:
            self.min_val = Xw.min(dim=0).values
            self.max_val = Xw.max(dim=0).values

        # avoid degenerate ranges
        same = (self.max_val - self.min_val) <= 1e-12
        if same.any():
            self.max_val[same] = self.min_val[same] + 1.0

        # persist on CPU
        self.min_val = self.min_val.cpu()
        self.max_val = self.max_val.cpu()
        self.s_log = (s.detach().cpu() if s is not None else None)

        os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
        torch.save({
            'min_val': self.min_val,
            'max_val': self.max_val,
            's_log': self.s_log,
            'feature_range': self.feature_range,
            'use_quantiles': self.use_quantiles,
            'q_low': self.q_low,
            'q_high': self.q_high,
            'log_mode': self.log_mode,
            'slog_scale': self.slog_scale,
            'eps': self.eps,
        }, self.save_file)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler not fitted")
        X = self._as_2d(data).to(torch.float32)
        # ensure s on device
        if self.log_mode != 'off':
            if self.s_log is None:
                raise ValueError("s_log missing — fit() must be run before transform() with log_mode!=off.")
            s = self.s_log.to(X.device, dtype=X.dtype)
            X = self._log_forward(X, s)

        minv = self.min_val.to(X.device, dtype=X.dtype)
        maxv = self.max_val.to(X.device, dtype=X.dtype)
        # robust clip to learned bounds
        X = torch.clamp(X, min=minv, max=maxv)

        rng = (maxv - minv).clamp_min(self.eps)
        std = (X - minv) / rng
        a, b = self.feature_range
        Xs = std * (b - a) + a
        return Xs.reshape_as(data).to(data.dtype)

    def inverse_transform(self, scaled_data: torch.Tensor) -> torch.Tensor:
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler not fitted")
        Ys = self._as_2d(scaled_data).to(torch.float32)

        minv = self.min_val.to(Ys.device, dtype=Ys.dtype)
        maxv = self.max_val.to(Ys.device, dtype=Ys.dtype)
        a, b = self.feature_range
        std = (Ys - a) / (b - a)
        Xw = std * (maxv - minv) + minv  # back to working (possibly log) space

        if self.log_mode == 'off':
            X = Xw
        else:
            if self.s_log is None:
                raise ValueError("s_log missing — cannot invert log transform without the stored scales.")
            s = self.s_log.to(Ys.device, dtype=Ys.dtype)
            X = self._log_inverse(Xw, s)

        return X.reshape_as(scaled_data).to(scaled_data.dtype)

    def load_values(self, path=None):
        load_file = path or self.save_file
        if not os.path.exists(load_file):
            raise FileNotFoundError(f"No saved values at {load_file}")
        saved = torch.load(load_file, map_location='cpu')
        self.min_val = saved['min_val']
        self.max_val = saved['max_val']
        self.s_log = saved.get('s_log', None)
        # restore config too (useful if you only call transform/inverse later)
        self.feature_range = tuple(saved.get('feature_range', self.feature_range))
        self.use_quantiles = saved.get('use_quantiles', self.use_quantiles)
        self.q_low = saved.get('q_low', self.q_low)
        self.q_high = saved.get('q_high', self.q_high)
        self.log_mode = saved.get('log_mode', self.log_mode)
        self.slog_scale = saved.get('slog_scale', self.slog_scale)
        self.eps = saved.get('eps', self.eps)
        
class ProcessData:
    def __init__(self, n_inp, U_patch, config):
        self.config = config
        self.model_path = config['encoder_decoder_path']
        self.batch_size = config['spatial_batch_size']
        self.device = config['device']
        self.n_inp = n_inp
        self.embed_dim = config['embed_dim_spatial']
        self.U_patch = U_patch

        if config['dimension'] == '3D':
            self.P = (config['m']-1) * (config['n']-1) * (config['k']-1)
        else:
            self.P = (config['m']-1) * (config['n']-1)

    def initialize_spatial_model(self):
        return SpatialModel(
            field_groups=self.config['field_groups'],
            n_inp=self.n_inp,
            MLP_hidden=self.config['MLP_hidden_spatial'],
            num_layers=self.config['num_layers_spatial'],
            embed_dim=self.config['embed_dim_spatial'],
            n_heads=self.config['n_heads_spatial'],
            max_len=self.config['block_size_spatial'],
            src_len=self.config['src_len_spatial'],
            variational=self.config['variational_spatial'],
            U_patch=self.U_patch,
            dropout=self.config['dropout_spatial']
        ).to(self.device)

    def load_model(self):
        state_dict = torch.load(self.model_path, map_location=self.device)
        new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        self.model_spatial.load_state_dict(new_state_dict)
        self.model_spatial.eval()

    def initialize_and_process_data(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            dataloader = data
        else:
            data_spatial_dataset = EncoderDecoderDataset(data)
            dataloader = DataLoader(data_spatial_dataset, batch_size=1000, shuffle=False)
        
        processed_data = self.process_data(dataloader)
        return processed_data

    def process_data(self, dataloader):
        self.model_spatial = self.initialize_spatial_model()
        self.load_model()
        self.model_spatial.to(self.device)
        processed_chunks = []

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                data = self.model_spatial.generate_padding_mask(data)
                if self.config['variational_spatial']:
                    z, _, _ = self.model_spatial.encode(data)
                else:
                    z = self.model_spatial.encode(data)
                processed_chunks.append(z.cpu())

        #self.clear_gpu_memory()
        return torch.cat(processed_chunks, dim=0)

    def decode_data(self, data):
        self.model_spatial = self.initialize_spatial_model()
        self.load_model()
        self.model_spatial.to(self.device)
        data = data.to(self.device)
        with torch.no_grad():
            decoded = self.model_spatial.decode(data)
            result = decoded.cpu()

        return result

    def clear_gpu_memory(self):
        if self.device != 'cpu':
            torch.cuda.empty_cache()
            self.model_spatial.cpu()
            for param in self.model_spatial.parameters():
                param.data = param.data.cpu()
                if param.grad is not None:
                    param.grad.data = param.grad.data.cpu()
            print("GPU memory cleared")


class EncoderDecoderDataset(Dataset):
    def __init__(self, precomputed_data):
        self.data = precomputed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_tensor = self.data[idx]
        return data_tensor  # Return the data as both input and target

class TemporalDataset(Dataset):
    def __init__(self, data_list, data_list_original, field_ib, src_len=64, overlap=0, device='cpu', time_shifting_flag=False):
        self.device              =    device
        self.data_list           =    data_list
        self.data_list_original  =    data_list_original
        self.field_ib            =    field_ib
        self.src_len             =    src_len
        self.overlap             =    overlap
        self.step                =    src_len - overlap
        self.time_shifting_flag  =    time_shifting_flag

        self.num_samples = 0
        self.segment_samples = []

        # Calculate the number of valid src in the data
        # this is for the model that does not use tgt, instead plays with mask to reveal information
        for data in data_list:
            num_pairs = data.shape[0] // self.step
            self.segment_samples.append(num_pairs)
            self.num_samples += num_pairs

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        src: Time index fed to model
        tgt: Time index past src the model should estimate
        tgt_original: Original spatially unprocessed tgt
        field_ib_out: Field input/boundary
        """

        # Determine which data segment the index falls into
        segment_index = 0
        cumulative_samples = 0

        for i, samples in enumerate(self.segment_samples):
            cumulative_samples += samples
            if idx < cumulative_samples:
                segment_index = i
                break

        if idx < cumulative_samples:
            data_idx = idx - (cumulative_samples - self.segment_samples[segment_index])
        else:
            raise IndexError("Index out of range")


        if self.time_shifting_flag:
            shift_idx = np.random.randint(0, self.data_list[segment_index].shape[0] - self.step)
        else:
            shift_idx = 0
        active_data = self.data_list[segment_index]
        active_data_original = self.data_list_original[segment_index]
        active_field_ib = self.field_ib[segment_index]

        start_idx = data_idx * self.step
        end_idx = start_idx + self.src_len

        src = active_data[start_idx + shift_idx :  end_idx + shift_idx]
        tgt = active_data[start_idx+1+ shift_idx :  end_idx+1+ shift_idx]
        tgt_original = active_data_original[start_idx+1+ shift_idx : end_idx+1+ shift_idx]
        field_ib_out = active_field_ib[start_idx + shift_idx : end_idx + shift_idx]

        return src, tgt, tgt_original, field_ib_out

class MeshProcessor:
    def __init__(self, config: Dict[str, Any], coordinates: Tuple[torch.Tensor, ...]): # coordinates: [3, C]
        self.config = config
        self.dimension = config.get('dimension', '3D')

        if 'field_groups' not in config:
            raise ValueError("'field_groups' must be specified in the config dictionary")
        self.field_groups = config['field_groups']
        self.scale_feature_range = config.get('scale_feature_range')
        self.save_dir = config['save_dir']
        self.csv_scale_name = config.get('csv_scale_name', 'scaler')
        
        if self.dimension == '3D' and coordinates.shape[0] is None:
            raise ValueError("3D processing requires x, y, and z coordinates")
        elif self.dimension == '2D' and coordinates.shape[0] is None:
            raise ValueError("2D processing requires x and y coordinates")
        
        self.coordinates = coordinates.T
        
        self.scalers = []
        if self.scale_feature_range is not None:
            for i, group in enumerate(self.field_groups):
                scaler_config = {
                    'feature_range': self.scale_feature_range,
                    'name': f"{self.csv_scale_name}-group{i}",
                    'save_dir': self.save_dir,
                    'use_quantiles': config.get('use_quantiles', False),
                    'q_low': config.get('q_low', 1.0),
                    'q_high': config.get('q_high', 99.0),
                    'log_mode': config.get('log_mode', 'off'),
                    'slog_scale': config.get('slog_scale', 'median'),
                    'eps': config.get('scaler_eps', 1e-12)
                }
                scaler = MinMaxScaler(**scaler_config)  # ✅
                self.scalers.append(scaler)

    def patchify_and_scale(self, data: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, train_indices: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...], Any]:
        T, N, F = data.shape
        batch_stacked_fields = []
        batch_stacked_coords = []
        self.U_patch = None

        # Scale the data before patchifying
        if self.scale_feature_range is not None:
            if train_indices is not None:
                for i, (scaler, group) in enumerate(zip(self.scalers, self.field_groups)):
                    scaler.fit(data[:, :, group])
            else:
                try:
                    for i, scaler in enumerate(self.scalers):
                        scaler_file = os.path.join(self.save_dir, f"{self.csv_scale_name}-group{i}_min_max_values.pt")
                        scaler.load_values(scaler_file)
                    print(f"Loaded scaler values from {self.save_dir}")
                except FileNotFoundError as e:
                    raise ValueError(f"No saved scaler values found and train_indices is None. Error: {str(e)}")

        scaled_data = self._scale_fields(data)
        
        m, n = self.config['m'], self.config['n']
        k = self.config['k'] if self.dimension == '3D' else None

        if self.dimension == '3D':
            self.partitioner = DataPartitioner3D(x_coords=self.coordinates[0], 
                                               y_coords=self.coordinates[1], 
                                               z_coords=self.coordinates[2],
                                               m=m, n=n, k=k, 
                                               pad_id=-1, pad_field_value=0)
        else:
            # self.partitioner = DataPartitioner2D(x_coords=self.coordinates[0], 
            #                                    y_coords=self.coordinates[1],
            #                                    m=m, n=n, 
            #                                    pad_id=-1, pad_field_value=0,
            #                                    edge_index=edge_index, edge_weight=edge_weight)

            print('edge_index shape:', edge_index.shape)
            self.partitioner = SpectralPartitioner(self.coordinates,
                                                   edge_index=edge_index,
                                                   edge_weight=edge_weight,
                                                   num_nodes=N,
                                                   k=(m-1) * (n-1),
                                                   pad_id=-1,
                                                   pad_field_value=0)

        # Patchify the scaled data
        for idx in range(0, len(data), 2048):
            batch_data = scaled_data[idx:idx+2048]
            var_list = [batch_data[:,:,i].to(torch.float32) for i in range(batch_data.shape[-1])]
            # patched_data, patched_index_map = self.partitioner.create_partitions(var_list)

            
            if self.U_patch is None:
                patched_data, patched_index_map, U_padded, eig_padded = self.partitioner.create_partitions(var_list, compute_gft=True)
                self.U_patch = U_padded
            else:
                patched_data, patched_index_map = self.partitioner.create_partitions(var_list, compute_gft=False)

            fields_list = [part[1] for part in patched_data]
            coords_list = [part[0] for part in patched_data]

            stacked_fields = torch.stack(fields_list, dim=1)  # [B, P, C, F]
            stacked_coords = torch.stack(coords_list, dim=1)  # [T, P, C, 3] or [T, P, C, 2]

            batch_stacked_fields.append(stacked_fields)
            #batch_stacked_coords.append(stacked_coords)

        if self.config.get('perform_initial_test', True):
            self._perform_initial_test(patched_data)

        # Concatenate all batches
        final_stacked_fields = torch.cat(batch_stacked_fields, dim=0)  # [T, P, C, F]
        self.stacked_coords = stacked_coords  # [T, P, C, 3] or [T, P, C, 2]

        return self.stacked_coords, final_stacked_fields, self.U_patch

    def _scale_fields(self, fields: torch.Tensor) -> torch.Tensor:
        if self.scale_feature_range is None:
            return fields
        
        scaled_fields = torch.zeros_like(fields)
        for scaler, group in zip(self.scalers, self.field_groups):
            scaled_fields[..., group] = scaler.transform(fields[..., group])
        return scaled_fields

    def inverse_scale_and_unpatch(self, scaled_fields: torch.Tensor) -> torch.Tensor:  # [T, P, C, F]
        T, P, C, F = scaled_fields.shape
        final_reconstructed_fields = []
        for idx in range(0, T, 2048):
            coords_process = self.stacked_coords
            scaled_fields_process = scaled_fields[idx:idx+2048]
            unpatched_data = [(coords_process[:,i].to(torch.float32), scaled_fields_process[:,i].to(torch.float32)) for i in range(P)]
            reconstructed_coords, reconstructed_fields = self.partitioner.inverse_partition(unpatched_data, time_dim=T)
            final_reconstructed_fields.append(reconstructed_fields)

        final_reconstructed_fields = torch.cat(final_reconstructed_fields, dim=0)

        # Inverse scale
        if self.scale_feature_range is not None:
            unscaled_fields = torch.zeros_like(final_reconstructed_fields)
            for scaler, group in zip(self.scalers, self.field_groups):
                unscaled_fields[..., group] = scaler.inverse_transform(final_reconstructed_fields[..., group])
        else:
            unscaled_fields = final_reconstructed_fields

        return unscaled_fields
    
    def _perform_initial_test(self, patched_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        reconstructed_coordinations, reconstructed_fields = self.partitioner.inverse_partition(external_partitions=patched_data)
        print('Results of simple initial test:')
        if self.dimension == '3D':
            unit_test_create_partitions3D(
                truth_fields=self.partitioner.var_list,
                coordx=self.partitioner.x_coords,
                coordy=self.partitioner.y_coords,
                coordz=self.partitioner.z_coords,
                inversed_fields=reconstructed_fields,
                inversed_coordx=reconstructed_coordinations[:, 0],
                inversed_coordy=reconstructed_coordinations[:, 1],
                inversed_coordz=reconstructed_coordinations[:, 2]
            )
        else:
            unit_test_create_partitions2D(
                truth_fields=self.partitioner.var_list,
                coordx=self.partitioner.x_coords,
                coordy=self.partitioner.y_coords,
                inversed_fields=reconstructed_fields,
                inversed_coordx=reconstructed_coordinations[:, 0],
                inversed_coordy=reconstructed_coordinations[:, 1]
            )