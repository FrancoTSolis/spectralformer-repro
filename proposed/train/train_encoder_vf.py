import torch
from torch.utils.data import DataLoader
import numpy as np
import h5py
from typing import Dict, Any, Tuple, Optional, List
from utils.data_processors import EncoderDecoderDataset, MeshProcessor, SpectralProcessor
from utils.train_utils import Vloss, initialize_optimizer, calculate_R2, build_edges
from models.encoder_decoder import SpatialModel
from utils.modular_testing import test_mesh_processor_2d, test_mesh_processor_3d
import time
import sys
import random

def load_and_convert(config: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
    graphs: List[Dict[str, torch.Tensor]] = []
    for i in range(21):
        # If these files actually contain multiple named arrays, prefer .npz.
        # With .npy this only works if you saved an object/structured array.
        data = np.load(f"{config['field_data_path']}/rawData{i}.npy", allow_pickle=True)

        x = torch.as_tensor(data['x'], dtype=torch.float32)                 # e.g., [T,N,F] or [N,F]
        para = torch.as_tensor(data['para'], dtype=torch.float32)           # any shape
        ei = torch.as_tensor(data['edge_index'], dtype=torch.long)          # [2, E]
        edge_attr = torch.as_tensor(data['edge_attr'], dtype=torch.float32) # [E, ...]
        dis = edge_attr[:, 2]                                               # [E]

        sigma = dis.mean()                                                  # scalar tensor
        w = torch.exp(-(dis * dis) / (2.0 * sigma * sigma + 1e-12))         # [E]

        graphs.append(dict(x=x, para=para, edge_index=ei, edge_weight=w))
    return graphs

def get_datasets(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, MeshProcessor]:
    train_sources, val_sources, test_sources, mesh_processor, U_patch = process_data(config)

    dataset_train = EncoderDecoderDataset(train_sources)
    dataset_validation = EncoderDecoderDataset(val_sources)
    dataset_test = EncoderDecoderDataset(test_sources)

    batch_size = config['batch_size']
    shuffle = True

    trainLoader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    validationLoader = DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)
    testLoader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return trainLoader, validationLoader, testLoader, mesh_processor, U_patch

def _pad_list_to_tensor(ragged: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """
    ragged list of [N_i, C] -> [B, N_max, C]
    """
    if len(ragged) == 0:
        return torch.empty(0, 0, 0)
    B = len(ragged)
    N_max = max(t.shape[0] for t in ragged)
    C = ragged[0].shape[1]
    out = ragged[0].new_full((B, N_max, C), pad_value)
    for i, t in enumerate(ragged):
        n = t.shape[0]
        out[i, :n, :] = t
    return out  # [B, N_max, C]

def process_data(config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
    """
    Load graphs (variable N), pad per split, and return SEA-ready tensors.
    No MeshProcessor; coordinates are unused here (dummy coords can be added if needed downstream).
    Returns:
      train_sources_tokenized      [B_tr, 1, F=Nmax_tr, C]
      validation_sources_tokenized [B_va, 1, Nmax_va,   C]
      test_sources_tokenized       [B_te, 1, Nmax_te,   C]
      mesh_processor = None
      U_patch = None
    """
    # ---- Seeds ----
    seed = config.get('random_seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---- Load ragged graphs ----
    graphs = load_and_convert(config)  # list of dicts: each has 'x' [T,N,F] or [N,F], etc.
    assert isinstance(graphs, list) and len(graphs) > 0, "No graphs loaded."

    # ---- Make sample list (flatten time into batch) ----
    # Each sample is a tensor [N, C]. If x is [T,N,F], we create T samples.
    samples: List[torch.Tensor] = []
    for g in graphs:
        x = g['x']
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x)
        if x.dim() == 3:
            T, N, C = x.shape
            for t in range(T):
                samples.append(x[t].to(torch.float32))   # [N, C]
        elif x.dim() == 2:
            samples.append(x.to(torch.float32))          # [N, C]
        else:
            raise ValueError(f"Unsupported x dim: {x.dim()}")

    total_samples = len(samples)
    if total_samples == 0:
        raise ValueError("No samples after flattening time.")

    # ---- Split indices ----
    train_fraction = float(config['train_fraction'])
    val_fraction   = float(config['val_fraction'])

    idx = np.arange(total_samples)
    rng = np.random.default_rng(config.get('random_seed', 42))
    rng.shuffle(idx)

    train_len = int(np.round(total_samples * train_fraction))
    val_len   = int(np.round(total_samples * val_fraction))
    test_len  = total_samples - train_len - val_len

    train_idx = idx[:train_len]
    val_idx   = idx[train_len:train_len + val_len]
    test_idx  = idx[train_len + val_len:]

    if config.get('print_split_sizes', True):
        print(f"Total samples: {total_samples}")
        print(f"Train length: {train_len}")
        print(f"Val length:   {val_len}")
        print(f"Test length:  {test_len}")

    config['train_size'] = int(train_len)

    # ---- Build per-split padded tensors ----
    def build_split(idxs: np.ndarray) -> torch.Tensor:
        if idxs.size == 0:
            return torch.empty(0, 1, 0, 0)
        xs = [samples[i] for i in idxs]              # list of [N_i, C]
        X = _pad_list_to_tensor(xs, pad_value=0.0)   # [B, N_max, C]
        X = X.unsqueeze(1)                           # [B, 1, N_max, C] -> SEA expects [B,P,F,C] with P=1
        return X

    train_sources_tokenized      = build_split(train_idx)
    validation_sources_tokenized = build_split(val_idx)
    test_sources_tokenized       = build_split(test_idx)

    # ---- SEA "isolate" vs "mixed" switch (P=1 in this path) ----
    if config.get('SEA_isolate', True):
        # already [B, 1, F, C]; nothing to change
        pass
    elif config.get('SEA_mixed', False):
        # Merge batch into one long batch over P=1 -> no-op aside from reshape
        # Keep the same shape semantics (still [B, 1, F, C]) for downstream compatibility.
        pass
    else:
        raise AssertionError("Invalid SEA data configuration")

    # ---- n_inp (channel dimension) ----
    if train_sources_tokenized.numel() > 0:
        config['n_inp'] = int(train_sources_tokenized.shape[-1])

    if config.get('print_split_sizes', True):
        print(f"Train shape: {tuple(train_sources_tokenized.shape)}")
        print(f"Val   shape: {tuple(validation_sources_tokenized.shape)}")
        print(f"Test  shape: {tuple(test_sources_tokenized.shape)}")

    mesh_processor = None
    U_patch = None
    return train_sources_tokenized, validation_sources_tokenized, test_sources_tokenized, mesh_processor, U_patch

def get_model(config: Dict[str, Any], U_patch, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer]:
    model = SpatialModel(
        field_groups=config['field_groups'],
        n_inp=config['n_inp'],  # Use the extracted n_inp here
        MLP_hidden=config['MLP_hidden'],
        num_layers=config['num_layers'],
        embed_dim=config['embed_dim'],
        n_heads=config['n_heads'],
        max_len=config['block_size'],
        src_len=config.get('src_len', 0),
        variational=config['variational'],
        U_patch=U_patch,
        dropout=config['dropout'],
    )

    if config.get('load_pretrained', False):
        model_path = config['pretrained_model_path']
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pre-trained model from {model_path}")

    model = model.to(device)
    optimizer = initialize_optimizer(model, config)

    if config['variational']:
        total_steps = round(config['epoch_num'] * config['train_size'] // config['batch_size'])
        loss_fn = Vloss(config['KL_weight_min'], config['KL_weight_max'], total_steps)
    else:
        loss_fn = torch.nn.MSELoss()

    return model, loss_fn, optimizer

def pre_train_encoder(config: Dict[str, Any]):
    device = torch.device(config['device'])
    trainLoader, validationLoader, testLoader, mesh_processor, U_patch = get_datasets(config)
    model, loss_fn, optimizer = get_model(config, U_patch, device)

    return model, optimizer, trainLoader, validationLoader, testLoader, loss_fn, mesh_processor


def train(config: Dict[str, Any], error_tracker):
    model, optimizer, trainLoader, validationLoader, testLoader, loss_fn, mesh_processor = pre_train_encoder(config)
    device = torch.device(config['device'])
    model.to(device)
    model.train()
    
    start_time = time.time()
    prev_error = float('inf')
    iter = 0

    error_tracker.log_model(model, loss_fn, optimizer)

    for epoch in range(1, config['epoch_num'] + 1):
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_r2_sum = 0.0
        num_train_batches = 0

        for data in trainLoader:
            data = data.to(device)
            optimizer.zero_grad()
            
            if config['variational']:
                outputs, mu, logvar = model(data)
                loss = loss_fn(x=data, z_mu=mu, z_logvar=logvar, mu_recon=outputs, sigma_recon=None, iteration=iter)
            else:
                outputs = model(data)
                loss = loss_fn(outputs, data)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if config['variational']:
                train_recon_loss += loss_fn.recon_loss.item()
                train_kl_loss += loss_fn.KL_loss.item()
            else:
                train_recon_loss += loss.item()
            train_r2_sum += calculate_R2(outputs.detach(), data.detach()).item()
            num_train_batches += 1
            iter += 1

        # Calculate average losses and R2 score
        train_loss /= num_train_batches
        train_recon_loss /= num_train_batches
        if config['variational']:
            train_kl_loss /= num_train_batches
        train_r2 = train_r2_sum / num_train_batches

        # Log train errors
        train_metrics = {
            "Loss": train_loss,
            "Recon_Loss": train_recon_loss,
            "R2": train_r2
        }
        if config['variational']:
            train_metrics["KL_Loss"] = train_kl_loss
        error_tracker.record_error("train", epoch, train_metrics)

        if epoch % config.get('validation_interval', 1) == 0 or epoch == config['epoch_num']:
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_r2_sum = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for v_data in validationLoader:
                    v_data = v_data.to(device)
                    if config['variational']:
                        v_outputs, v_mu, v_logvar = model(v_data)
                        v_loss = loss_fn(v_data, v_mu, v_logvar, v_outputs, sigma_recon=None, iteration=iter)
                    else:
                        v_outputs = model(v_data)
                        v_loss = loss_fn(v_outputs, v_data)

                    val_loss += v_loss.item()
                    if config['variational']:
                        val_recon_loss += loss_fn.recon_loss.item()
                        val_kl_loss += loss_fn.KL_loss.item()
                    else:
                        val_recon_loss += v_loss.item()
                    val_r2_sum += calculate_R2(v_outputs, v_data).item()
                    num_val_batches += 1

            # Calculate average validation losses and R2 score
            val_loss /= num_val_batches
            val_recon_loss /= num_val_batches
            if config['variational']:
                val_kl_loss /= num_val_batches
            val_r2 = val_r2_sum / num_val_batches

            # Log validation errors
            val_metrics = {
                "Loss": val_loss,
                "Recon_Loss": val_recon_loss,
                "R2": val_r2
            }
            if config['variational']:
                val_metrics["KL_Loss"] = val_kl_loss
            error_tracker.record_error("val", epoch, val_metrics)

            print(f"\nEpoch: {epoch}/{config['epoch_num']}")
            if config['variational']:
                print(f"Train - Total Loss: {train_loss:.8f}, Recon Loss: {train_recon_loss:.8f}, KL Loss: {train_kl_loss:.8f}, R^2: {train_r2:.8f}")
                print(f"Val   - Total Loss: {val_loss:.8f}, Recon Loss: {val_recon_loss:.8f}, KL Loss: {val_kl_loss:.8f}, R^2: {val_r2:.8f}")
            else:
                print(f"Train - Loss: {train_loss:.8f}, R^2: {train_r2:.8f}")
                print(f"Val   - Loss: {val_loss:.8f}, R^2: {val_r2:.8f}")

            # Check if current model is the best so far
            if val_recon_loss < prev_error:
                prev_error = val_recon_loss
                print("--- New Best Model Saved ---")
                model.to('cpu')
                model_path = f"{config['save_dir']}/encoder_decoder_{config['case_name']}_{config['run_name']}.pt"
                torch.save(model.state_dict(), model_path)
                model.to(device)
            else:
                print("--- No Improvement, Best Model Retained ---")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    # Finish error tracking
    error_tracker.finish()

    return model