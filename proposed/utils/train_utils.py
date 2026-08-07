import torch.nn.functional as FF
import torch
import os
from abc import ABC, abstractmethod
from typing import Any, Dict
import torch.nn as nn
from utils.modular_testing import plot_all_fields_2d, plot_all_fields_3d
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv
from scipy.spatial import cKDTree, Delaunay

class Vloss:
    def __init__(self, KL_weight_min, KL_weight_max, total_steps):
        self.recon_loss = None
        self.KL_loss = None
        self.total_loss = None
        self.KL_weight_min = KL_weight_min
        self.KL_weight_max = KL_weight_max
        self.total_steps = total_steps

    def __call__(self, x, z_mu, z_logvar, mu_recon, iteration, sigma_recon=None):
        x_recon = mu_recon
        self.KL_weight = self.KL_weight_min + (self.KL_weight_max - self.KL_weight_min) * (iteration / self.total_steps)
        self.recon_loss = FF.mse_loss(x_recon, x)
        self.KL_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        self.total_loss = self.recon_loss + self.KL_weight*self.KL_loss
        return self.total_loss
    

def initialize_optimizer(model, config): # config
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=config.get('weight_decay', 0.0)) # config
    if config.get('scheduler', None) == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config['epoch_num']) # config
        return optimizer, scheduler
    else:
        return optimizer


def calculate_R2(prediction, labels):
    prediction = prediction.reshape(-1)  # Flatten the tensor
    labels = labels.reshape(-1)  # Flatten the tensor
    residual = torch.sum((prediction - labels) ** 2)
    total = torch.sum((labels - torch.mean(labels)) ** 2)
    R2 = 1-(residual/total)
    return R2

class BaseErrorTracker(ABC):
    @abstractmethod
    def record_error(self, phase, epoch, metrics):
        pass

    @abstractmethod
    def log_model(self, model, criterion, optimizer):
        pass

    @abstractmethod
    def finish(self):
        pass

class WandbErrorTracker(BaseErrorTracker):
    def __init__(self, project_name, run_name=None, config=None):
        import wandb
        self.wandb = wandb
        self.run = self.wandb.init(project=project_name, name=run_name, config=config)

    def record_error(self, phase, epoch, metrics):
        log_dict = {"epoch": epoch}
        for key, value in metrics.items():
            log_dict[f"{phase}/{key}"] = value
        self.wandb.log(log_dict)

    def log_model(self, model, criterion, optimizer):
        self.wandb.watch(model, criterion, log="all", log_freq=10)

    def finish(self):
        self.wandb.finish()

class NoOpErrorTracker(BaseErrorTracker):
    def __init__(self, *args, **kwargs):
        pass

    def record_error(self, phase, epoch, metrics):
        pass

    def log_model(self, model, criterion, optimizer):
        pass

    def finish(self):
        pass

def create_error_tracker(use_wandb, project_name, run_name=None, config=None):
    if use_wandb:
        try:
            import wandb
            WANDB_API_KEY = config.get("WANDB_API_KEY", None)
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
            wandb.login()
            print("Successfully logged in to Weights & Biases!")
            return WandbErrorTracker(project_name, run_name, config)
        except ImportError:
            print("Wandb not installed. Using NoOpErrorTracker.")
            return NoOpErrorTracker()
        except Exception as e:
            print(f"Error initializing Wandb: {str(e)}. Using NoOpErrorTracker.")
            return NoOpErrorTracker()
    else:
        return NoOpErrorTracker()

def relativeMSE(predictions: torch.Tensor, truth: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mse = torch.sum((predictions - truth) ** 2, dim=dim)
    normalization = torch.sum(truth ** 2, dim=dim)
    epsilon = 1e-8
    return mse / (normalization + epsilon)

def relativeMSE_recon(predictions: torch.Tensor, truth: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mse = torch.sum((predictions - truth) ** 2, dim=dim)
    normalization = torch.sum(truth ** 2, dim=dim)
    epsilon = 1e-8
    return mse / (normalization + epsilon)

def relativeMSE_with_time(predictions: torch.Tensor, truth: torch.Tensor, dim = 2) -> torch.Tensor:
    """
    Calculate the Relative Mean Squared Error, preserving the time dimension.
    
    Args:
    predictions (torch.Tensor): Predicted values with shape [trajectory, time, cell, field]
    truth (torch.Tensor): Ground truth values with shape [trajectory, time, cell, field]
    
    Returns:
    torch.Tensor: RelMSE for each field and time step, shape [time, field]
    """
    assert predictions.shape == truth.shape, "Predictions and truth must have the same shape"
    
    # Calculate squared differences
    squared_diff = (predictions - truth)**2
    
    # Sum over cell dimensions
    sum_squared_diff = torch.sum(squared_diff, dim=(dim))
    sum_squared_truth = torch.sum(truth**2, dim=(dim))
    
    # Avoid division by zero
    epsilon = 1e-8
    
    # Calculate RelMSE for each field and time step
    rel_mse = sum_squared_diff / (sum_squared_truth + epsilon)
    
    return rel_mse



def autoregressive_validation(model, validationLoader, loss_fn, device):
    """
    Perform autoregressive validation on 1 sample of the validation set.
    In order to reduce the validation time.
    """
    model.eval()
    autoreg_val_loss = 0.0
    autoreg_val_rel_mse = 0.0
    
    with torch.no_grad():
        # Get only the first batch
        data, target, _, ib = next(iter(validationLoader))
        
        # Select only the first sample
        data, target, ib = data[0:1].to(device), target[0:1].to(device), ib[0:1].to(device)
        
        autoreg_input = data[:, 0:1, :, :]
        for i in range(target.shape[1]):
            ib_slice = ib[:, :i+1, :]
            output = model(autoreg_input, ib_slice)
            next_step = output[:, -1:, :, :]
            autoreg_input = torch.cat((autoreg_input, next_step), dim=1)
        
        autoregressive_output = autoreg_input[:, 1:, :, :]
        v_loss = loss_fn(autoregressive_output, target)
        v_rel_mse = relativeMSE_with_time(autoregressive_output, target, dim=3).mean()
        
        autoreg_val_loss = v_loss.item()
        autoreg_val_rel_mse = v_rel_mse.item()
    
    return autoreg_val_loss, autoreg_val_rel_mse

def full_autoregressive_evaluation(model, 
                                   dataLoader, 
                                   loss_fn, 
                                   device, 
                                   processor, 
                                   mesh_processor, 
                                   config,
                                   epoch,
                                   plot_traj=True):
    model.eval()
    encoded_rel_mse_return = 0.0
    decoded_rel_mse_return = 0.0
    num_batches = 0
    with torch.no_grad():
        for data, target, original_data, ib in dataLoader:
            data, target, original_data, ib = data.to(device), target.to(device), original_data.to(device), ib.to(device)
            # ib should be [B, L, ib_num]
            B, L, F, D = data.shape
            if ib.dim() == 2:
                # static per-trajectory vector -> tile over time
                ib = ib.unsqueeze(1).expand(B, L, -1)
            elif ib.shape[1] == 1:
                # time-invariant but already 3D -> tile
                ib = ib.expand(B, L, -1)
            elif ib.shape[1] == F:
                # COMMON MISTAKE: ib laid out per-variable; collapse or map to per-time
                # simplest: average across variables, then tile across time
                ib = ib.mean(dim=1, keepdim=True).expand(B, L, -1)
            elif ib.shape[1] != L:
                raise ValueError(f"ib must be [B,L,ib_num]; got {ib.shape} with L={L}")

            autoreg_input = data[:, 0:1, :, :]
            for i in range(target.shape[1]):
                ib_slice = ib[:, :i+1, :]
                output = model(autoreg_input, ib_slice)
                next_step = output[:, -1:, :, :]
                autoreg_input = torch.cat((autoreg_input, next_step), dim=1)
            
            autoregressive_output = autoreg_input[:, 1:, :, :]
            
            # Calculate encoded RelMSE
            encoded_rel_mse = relativeMSE(autoregressive_output, target).mean()
            encoded_rel_mse_return += encoded_rel_mse.mean().item()
            tr, T, _, _ = autoregressive_output.shape
            if config['dimension'] == '3D':
                n_patches = (config['m']-1) * (config['n']-1) * (config['k']-1)
            else:
                n_patches = (config['m']-1) * (config['n']-1)
            
            autoregressive_output_decode = inverse_transform_processed_data(autoregressive_output, tr, T, n_patches, len(config['field_groups']))
            autoregressive_output_decode = processor.decode_data(autoregressive_output_decode)
            if config['SEA_mixed']:
                B, P, F, C = autoregressive_output_decode.shape
                autoregressive_output_decode = autoregressive_output_decode.reshape(B, P, C, F)
            elif config['SEA_isolate']:
                autoregressive_output_decode = autoregressive_output_decode.permute(0, 1, 3, 2)
            else:
                assert False, "Invalid SEA data configuration"
            autoregressive_output_decode = autoregressive_output_decode.to('cpu')
            autoregressive_output_decode = mesh_processor.inverse_scale_and_unpatch(autoregressive_output_decode).to(device)
            
            _, C, F = autoregressive_output_decode.shape
            autoregressive_output_decode = autoregressive_output_decode.reshape(tr, T, C, F)
            
            # Calculate RelMSE for each time step and field (decoded)
            decoded_rel_mse = relativeMSE_with_time(autoregressive_output_decode, original_data).mean(dim=0)
            decoded_rel_mse_return += decoded_rel_mse.mean().item()
            decoded_rel_mse_avg = decoded_rel_mse.mean(dim=1)

            print(f'This is the average relative MSE per field: {decoded_rel_mse.mean(dim=0)}')

            # Plotting the fields for 5 random samples
            num_samples = 5
            total_samples = original_data.shape[1]
            np.random.seed(config['random_seed'])
            sample_indices = np.random.choice(total_samples, num_samples, replace=False)
            autoregressive_output_decode_plot = autoregressive_output_decode[0, :, :, :]
            original_data_plot = original_data[0, :, :, :]
            # Get coordinates
            if config['dimension'] == '2D':
                coordx = mesh_processor.coordinates[0].cpu()
                coordy = mesh_processor.coordinates[1].cpu()
            elif config['dimension'] == '3D':
                coordx = mesh_processor.coordinates[0].cpu()
                coordy = mesh_processor.coordinates[1].cpu()
                coordz = mesh_processor.coordinates[2].cpu()
            
            # Plot for each sample index
            original_data_plot = original_data_plot.cpu()
            autoregressive_output_decode_plot = autoregressive_output_decode_plot.cpu()
            csv_filename = f"{config['save_dir']}/rollout_error_{config['case_name']}_{config['run_name']}.csv"
            try:
                for idx in sample_indices:
                    if config['dimension'] == '2D':
                        plot_all_fields_2d(original_data_plot, coordx, coordy, idx, 
                                           filename=f"{config['save_dir']}/temporal_original_data_{idx}_{epoch}.png", 
                                           show=False)
                        plot_all_fields_2d(autoregressive_output_decode_plot, coordx, coordy, idx, 
                                           filename=f"{config['save_dir']}/temporal_decoded_data_{idx}_{epoch}.png", 
                                           show=False)
                    elif config['dimension'] == '3D':
                        plot_all_fields_3d(original_data_plot, coordx, coordy, coordz, idx, 
                                           filename=f"{config['save_dir']}/temporal_original_data_{idx}_{epoch}.png", 
                                           show=False)
                        plot_all_fields_3d(autoregressive_output_decode_plot, coordx, coordy, coordz, idx, 
                                           filename=f"{config['save_dir']}/temporal_decoded_data_{idx}_{epoch}.png", 
                                           show=False)
            except Exception as e:
                print(f"Warning: plotting failed ({e}), continuing with CSV save...")
            with open(csv_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # Write header
                csvwriter.writerow(['Time Step'] + [f'Field {i+1}' for i in range(decoded_rel_mse.shape[1])])
                # Write data
                for i, row in enumerate(decoded_rel_mse.cpu().numpy()):
                    csvwriter.writerow([i+1] + list(row))
            
            print(f"Decoded relative MSE saved to {csv_filename}")
            if plot_traj:
                T, F = decoded_rel_mse.shape
                plt.figure(figsize=(10, 6))
                time_steps = range(1, T + 1)
                for f in range(F):
                    field_relMSE = decoded_rel_mse[:, f]
                    plt.plot(time_steps, field_relMSE.cpu().numpy(), label=f'Field {f+1}')
                
                plt.plot(time_steps, decoded_rel_mse_avg.cpu().numpy(), label='average Relative MSE')
                plt.xlabel('Time Step')
                plt.ylabel('Relative MSE')
                plt.title('Rollout Error: Relative MSE over Time for Each Field (Single Sample)')
                plt.legend()
                plt.grid(True, which="both", ls="-", alpha=0.2)
                plt.savefig(f"{config['save_dir']}/rollout_error_{config['case_name']}_{config['run_name']}.png")
                plt.close()

            num_batches += 1

        # Calculate and return the average error across all batches
        if num_batches > 0:
            return {
                'encoded_rel_mse': encoded_rel_mse_return / num_batches,
                'decoded_rel_mse': decoded_rel_mse_return / num_batches
            }
        else:
            return None


def transform_processed_data(processed_data, tr, T, n_patches, num_field_groups):
    """
    Transform the processed data into the required shape.
    
    Args:
    processed_data (torch.Tensor): Input tensor of shape [B, P, EF, D]
    tr (int): Number of trajectories
    T (int): Number of time steps
    n_patches (int): Number of patches
    num_field_groups (int): Number of field groups
    
    Returns:
    torch.Tensor: Transformed tensor of shape [tr, T, num_field_groups, P*D]
    """
    D = processed_data.shape[-1]
    
    transformed = processed_data.reshape(tr, T, n_patches, num_field_groups, D)
    
    transformed = transformed.permute(0, 1, 3, 2, 4)
    
    transformed = transformed.reshape(tr, T, num_field_groups, -1)
    
    return transformed

def inverse_transform_processed_data(transformed_data, tr, T, n_patches, num_field_groups):
    """
    Inverse transform the processed data back to its original shape.
    
    Args:
    transformed_data (torch.Tensor): Input tensor of shape [tr, T, num_field_groups, P*D]
    tr (int): Number of trajectories
    T (int): Number of time steps
    n_patches (int): Number of patches
    num_field_groups (int): Number of field groups
    processor (ProcessData): Processor object for decoding
    
    Returns:
    torch.Tensor: Inverse transformed tensor of shape [B, P, F, C]
    """
    D = transformed_data.shape[-1] // n_patches
    
    inversed = transformed_data.reshape(tr, T, num_field_groups, n_patches, D)
    
    inversed = inversed.permute(0, 1, 3, 2, 4)
    
    inversed = inversed.reshape(tr*T, n_patches, num_field_groups, D)
    
    return inversed

def test_encoder_decoder(processor, testLoader, mesh_processor, U_patch, config):
    device = torch.device(config['device'])
    # Get original data from testLoader
    original_data = torch.cat([batch for batch in testLoader], dim=0)

    # encode and decode data
    encoded_data = processor.initialize_and_process_data(testLoader)
    decoded_data = processor.decode_data(encoded_data)

    # get loss before inverse scaling and unpatching
    loss = FF.mse_loss(decoded_data, original_data)
    print(f"Test Loss before inverse scaling and unpatching: {loss.item():.6f}")

    # inverse scaling and unpatching
    if config['SEA_mixed']:
        B, P, F, C = decoded_data.shape
        decoded_data = decoded_data.reshape(-1,P,F,C)
        original_data = original_data.reshape(-1,P,F,C)
    elif config['SEA_isolate']:
        decoded_data = decoded_data.permute(0, 1, 3, 2)
        original_data = original_data.permute(0, 1, 3, 2)
    else:
        assert False, "Invalid SEA data configuration"
    
    decoded_data = mesh_processor.inverse_scale_and_unpatch(decoded_data)
    original_data = mesh_processor.inverse_scale_and_unpatch(original_data)
    
    # get loss after inverse scaling and unpatching
    loss = FF.mse_loss(decoded_data, original_data)
    print(f"Test Loss after inverse scaling and unpatching: {loss.item():.6f}")

    rel_mse = relativeMSE_recon(decoded_data, original_data, dim=1)
    rel_mse = rel_mse.mean()
    print(f"Test Relative MSE after inverse scaling and unpatching: {rel_mse.item():.6f}")
    
    # Select random samples for visualization
    num_samples = 5
    total_samples = original_data.shape[0]
    sample_indices = np.random.choice(total_samples, num_samples, replace=False)
    
    # Get coordinates
    if config['dimension'] == '2D':
        coordx = mesh_processor.coordinates[0].cpu()
        coordy = mesh_processor.coordinates[1].cpu()
    elif config['dimension'] == '3D':
        coordx = mesh_processor.coordinates[0].cpu()
        coordy = mesh_processor.coordinates[1].cpu()
        coordz = mesh_processor.coordinates[2].cpu()
    
    # Plot for each sample index
    original_data = original_data.cpu()
    decoded_data = decoded_data.cpu()

    for idx in sample_indices:
        if config['dimension'] == '2D':
            plot_all_fields_2d(original_data, coordx, coordy, idx, 
                               filename=f"{config['save_dir']}/original_data_{idx}.png", 
                               show=False)
            plot_all_fields_2d(decoded_data, coordx, coordy, idx, 
                               filename=f"{config['save_dir']}/decoded_data_{idx}.png", 
                               show=False)
        elif config['dimension'] == '3D':
            plot_all_fields_3d(original_data, coordx, coordy, coordz, idx, 
                               filename=f"{config['save_dir']}/original_data_{idx}.png", 
                               show=False)
            plot_all_fields_3d(decoded_data, coordx, coordy, coordz, idx, 
                               filename=f"{config['save_dir']}/decoded_data_{idx}.png", 
                               show=False)
    
def gft_patches(x: torch.Tensor, U_patch: torch.Tensor):
    """
    Applies a patch-wise Graph Fourier Transform.

    Args:
        x (torch.Tensor): The input signal with shape [B, P, C, F].
                          (Batch, Patches, Nodes, Features)
        U_patch (torch.Tensor): The eigenvectors with shape [P, C, K].
                                (Patches, Nodes, Modes)

    Returns:
        torch.Tensor: The transformed signal in the spectral domain,
                      with shape [B, P, F, K].
                      (Batch, Patches, Features, Modes)
    """
    # For each batch (b) and patch (p), we project the signal (c)
    # onto the eigenvectors (k). The 'c' dimension is summed out.
    # The output has dimensions for batch, patch, features, and modes.
    C_hat = torch.einsum('bpfc,pck->bpfk', x, U_patch)
    return C_hat

def igft_patches(C_hat: torch.Tensor, U_patch: torch.Tensor):
    """
    Applies a patch-wise inverse Graph Fourier Transform.

    Args:
        C_hat (torch.Tensor): The signal in the spectral domain,
                              with shape [B, P, F, K].
                              (Batch, Patches, Features, Modes)
        U_patch (torch.Tensor): The eigenvectors with shape [P, C, K].
                                (Patches, Nodes, Modes)

    Returns:
        torch.Tensor: The reconstructed signal in the node domain,
                      with shape [B, P, C, F].
                      (Batch, Patches, Nodes, Features)
    """
    # For each batch (b) and patch (p), we reconstruct the signal (c)
    # by summing the eigenvectors (k) weighted by the spectral coefficients.
    # The 'k' dimension is summed out.
    x_rec = torch.einsum('bpfk,pck->bpfc', C_hat, U_patch)
    return x_rec

def _to_undirected_unique(pairs):
    """Converts directed pairs to unique undirected pairs using PyTorch."""
    # Ensure pairs is a torch tensor of type long
    pairs = torch.as_tensor(pairs, dtype=torch.long)
    
    # Separate the pairs into two tensors
    i, j = pairs[:, 0], pairs[:, 1]
    
    # Sort pairs to ensure (i, j) and (j, i) are treated the same
    i2, j2 = torch.minimum(i, j), torch.maximum(i, j)
    
    # Stack them back into a (N, 2) tensor
    pairs = torch.stack([i2, j2], dim=1)
    
    # Return unique pairs
    return torch.unique(pairs, dim=0)

def _edge_lengths(coords, pairs):
    """Calculates the Euclidean distance for each pair of coordinates using PyTorch."""
    # Calculate vector differences between paired coordinates
    diffs = coords[pairs[:, 0]] - coords[pairs[:, 1]]
    
    # Compute the L2 norm (Euclidean distance) for each difference vector
    return torch.linalg.norm(diffs, dim=1)

def build_edges(coords,
                method="knn",
                k=8,
                radius=None,
                max_edge_len=None,
                adaptive_factor=None,
                adaptive_knn_k=6):
    """
    Builds graph edges from coordinates using PyTorch operations.

    Args:
        coords (Tensor): A (N, 2) float tensor of node coordinates.
        method (str): 'knn', 'radius', or 'delaunay'.
        k (int): Neighbors per node for 'knn'.
        radius (float): Connection radius for 'radius'.
        max_edge_len (float): Global cutoff. Keep edges with length <= max_edge_len.
        adaptive_factor (float): Adaptive cutoff multiplier.
        adaptive_knn_k (int): k used to estimate local spacing.
        
    Returns:
        (Tensor, Tensor): A tuple containing the edge_index (2, E) and edge weights (E,).
    """
    # Ensure coords is a PyTorch tensor on the correct device
    coords = torch.as_tensor(coords, dtype=torch.float64)
    N = coords.shape[0]
    device = coords.device

    # --- Build candidate edges ---
    if method in ("knn", "radius") or (adaptive_factor is not None):
        # Compute the full pairwise distance matrix.
        # This can be memory-intensive for large N.
        dists_matrix = torch.cdist(coords, coords)

    if method == "knn":
        # Find the k+1 smallest distances for each point (includes self)
        _, idxs = torch.topk(dists_matrix, k=min(k + 1, N), dim=1, largest=False)
        
        # Create pairs by repeating node indices and flattening neighbor indices
        rows = torch.arange(N, device=device).repeat_interleave(idxs.shape[1] - 1)
        cols = idxs[:, 1:].reshape(-1)
        pairs = torch.stack([rows, cols], dim=1)

    elif method == "radius":
        if radius is None:
            raise ValueError("radius must be provided for method='radius'.")
        # Find pairs where distance is within radius (and not zero, i.e., not self-loops)
        mask = (dists_matrix > 0) & (dists_matrix < radius)
        pairs = torch.nonzero(mask, as_tuple=False)

    elif method == "delaunay":
        # Delaunay is a specialized algorithm not in PyTorch.
        # We convert to NumPy, compute, then convert back to a tensor.
        coords_np = coords.cpu().numpy()
        tri = Delaunay(coords_np)
        edges = set()
        for a, b, c in tri.simplices:
            edges.update({tuple(sorted((a, b))),
                          tuple(sorted((b, c))),
                          tuple(sorted((a, c)))})
        # Convert the resulting list of tuples to a PyTorch tensor
        if not edges: # Handle case with no edges
             pairs = torch.empty((0, 2), dtype=torch.long, device=device)
        else:
             pairs = torch.tensor(list(edges), dtype=torch.long, device=device)

    else:
        raise ValueError("method must be 'knn', 'radius', or 'delaunay'.")

    # Make undirected + unique
    pairs = _to_undirected_unique(pairs)

    # Return early if no edges were formed
    if pairs.shape[0] == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(0, dtype=torch.float32, device=device)

    # Compute edge lengths once
    lens = _edge_lengths(coords, pairs)

    # --- Global cutoff ---
    if max_edge_len is not None:
        keep = lens <= max_edge_len
        pairs, lens = pairs[keep], lens[keep]

    # --- Adaptive local cutoff ---
    if adaptive_factor is not None and len(pairs) > 0:
        # Find the k+1 nearest neighbors for the adaptive cutoff calculation
        d, _ = torch.topk(dists_matrix, k=min(adaptive_knn_k + 1, N), dim=1, largest=False)
        
        # Calculate local spacing as the median distance to neighbors (excluding self)
        neighbor_dists = d[:, 1:]
        local_spacing = torch.median(neighbor_dists, dim=1).values if neighbor_dists.numel() > 0 else torch.zeros(N, device=device)
        
        # Calculate adaptive threshold for each edge
        thr = adaptive_factor * 0.5 * (local_spacing[pairs[:, 0]] + local_spacing[pairs[:, 1]])
        keep = lens <= thr
        pairs, lens = pairs[keep], lens[keep]

    # Final weights and edge_index format
    weights = lens.to(torch.float32)
    edge_index = pairs.T  # Shape (2, E)
    
    return edge_index, weights