import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
from utils.data_processors import MeshProcessor, ProcessData, EncoderDecoderDataset, TemporalDataset
from utils.modular_testing import test_mesh_processor_2d, test_mesh_processor_3d
from utils.train_utils import Vloss, initialize_optimizer, autoregressive_validation, full_autoregressive_evaluation, transform_processed_data, inverse_transform_processed_data, build_edges
from models.temporal import TemporalModel
import time 
import sys
import random

def load_and_convert(config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    def load_single_file(path: str) -> torch.Tensor:
        if path.endswith('.npy'):
            data = np.load(path)
            if config.get('convert_to_torch', True):
                data = torch.from_numpy(data)
        elif path.endswith('.pt'):
            data = torch.load(path)
        else:
            raise ValueError(f"Unsupported file format for {path}. Only .npy and .pt are supported.")

        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor.")

        return data.to(config['device'])

    # Load field data
    field_data_path = config['field_data_path']
    field_data= load_single_file(field_data_path)

    # Load coordinates
    coordinates_path = config['coordinates_path']
    if config['static_mesh']:
        coordinates = load_single_file(coordinates_path).T
    else:
        coordinates = load_single_file(coordinates_path).transpose(0, 2, 1)  # (tr, P, d) -> (tr, d, P)
        
    if 'edge_path' in config and config['edge_path'] is not None:
        edge_data_path = config['edge_path']
        edge_data = np.load(edge_data_path)
        print(edge_data.keys())
        edge_index = torch.from_numpy(edge_data['edge_index'])
        edge_attr = torch.from_numpy(edge_data['edge_attr'])
        if edge_attr.dim() == 2:
            edge_attr = edge_attr.unsqueeze(0)
        dis = edge_attr[0, :, 2]
        if config.get('edge_weight_mode') == 'ones':
            edge_weight = torch.ones_like(dis)
        else:
            sigma = torch.mean(dis)
            edge_weight = torch.exp(-(dis * dis) / (2.0 * (sigma ** 2) + 1e-12))
    else:
        edge_index, dis = build_edges(coordinates, method='delaunay', max_edge_len=0.06)   # or "knn", "radius"
        sigma = torch.mean(dis)
        edge_weight = torch.exp(-(dis * dis) / (2.0 * (sigma ** 2) + 1e-12))

    # Load input data if path is provided in config
    input_data = None
    if 'input_path' in config and config['input_path'] is not None:
        input_data_path = config['input_path']
        input_data = load_single_file(input_data_path).float()
    if input_data.dim() == 1:
        input_data = input_data.unsqueeze(-1)

    return field_data, coordinates, input_data, edge_index, edge_weight

def get_datasets(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, MeshProcessor, ProcessData]:
    train_sources, val_sources, test_sources, train_original, val_original, test_original, data_input_train, data_input_val, data_input_test, _, _, mesh_processor, processor = process_data(config)

    dataset_train = TemporalDataset(
        train_sources, 
        train_original, 
        data_input_train, 
        config['dataset_src_len'], 
        config['dataset_overlap'], 
        config['device'], 
        config['dataset_time_shifting_flag']
    )
    dataset_validation = TemporalDataset(
        val_sources, 
        val_original, 
        data_input_val, 
        config['dataset_src_len'], 
        config['dataset_overlap'], 
        config['device'], 
        config['dataset_time_shifting_flag']
    )
    dataset_test = TemporalDataset(
        test_sources, 
        test_original, 
        data_input_test, 
        config['dataset_src_len'], 
        config['dataset_overlap'], 
        config['device'], 
        config['dataset_time_shifting_flag']
    )

    batch_size = config['batch_size']
    seed = config.get('random_seed', 42)

    # Use a fixed seed for the DataLoader
    g = torch.Generator()
    g.manual_seed(seed)

    trainLoader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g)
    validationLoader = DataLoader(dataset_validation, batch_size=8, shuffle=False, generator=g)
    testLoader = DataLoader(dataset_test, batch_size=8, shuffle=False, generator=g)

    return trainLoader, validationLoader, testLoader, mesh_processor, processor


def process_data(config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load data, process it, create train/val/test splits, and apply MeshProcessor.
    """
    seed = config.get('random_seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data
    field_data, coordinates, data_input, edge_index, edge_weight = load_and_convert(config)
    print(f"Field data shape: {field_data.shape}")
    print(f"Input data shape: {data_input.shape}")

    tr, T, C, F = field_data.shape
    field_data = field_data.reshape(tr*T, C, F)

    # Create train/val/test splits
    train_fraction = config['train_fraction']
    val_fraction = config['val_fraction']
    total_samples = tr
    
    np.random.seed(config.get('random_seed', 42))
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    train_length = int(np.round(tr * train_fraction))
    val_length = int(np.round(tr * val_fraction))
    test_length = tr - train_length - val_length

    print(f"Train length: {train_length}")
    print(f"Val length: {val_length}")
    print(f"Test length: {test_length}")
    
    train_indices = indices[:train_length]
    val_indices = indices[train_length:train_length + val_length]
    test_indices = indices[train_length + val_length:]

    # Apply MeshProcessor
    mesh_processor = MeshProcessor(config, coordinates)
    stacked_coords, scaled_fields, U_patch = mesh_processor.patchify_and_scale(field_data, edge_index, edge_weight, train_indices=train_indices)

    # Optionally test mesh structure
    if config['test_mesh_structure']:
        reconstructed = mesh_processor.inverse_scale_and_unpatch(scaled_fields)
        if config['dimension'] == '2D':
            test_results = test_mesh_processor_2d(field_data.float(), reconstructed, coordinates)
        else:
            test_results = test_mesh_processor_3d(field_data.float(), reconstructed, coordinates)
        print(test_results)
    
    n_inp = scaled_fields.shape[2]
    config['n_inp'] = n_inp
    n_patches = scaled_fields.shape[1]

    #---------------------------------------------------------------------------------------------------
    # This is a switch to allow mixing while keeping the exact structure and dimensions
    if config['SEA_isolate']:    
        scaled_fields = scaled_fields.permute(0, 1, 3, 2)
    elif config['SEA_mixed']:   
        B, P, C_patched, F = scaled_fields.shape
        scaled_fields = scaled_fields.reshape(B,P,F,C_patched) #.permute(0, 1, 3, 2)
    else:
        assert False, "Invalid SEA data configuration"
    #---------------------------------------------------------------------------------------------------
    
    print(f"Scaled fields shape: {scaled_fields.shape}")
    processor = ProcessData(n_inp, U_patch, config)
    processed_data = processor.initialize_and_process_data(scaled_fields) # [B, P, EF, D]
    processed_data = transform_processed_data(processed_data, tr, T, n_patches, len(config['field_groups']))
    # ----------------------------------------------------------------------------------
    # Prepare data for model
    train_sources_tokenized = processed_data[train_indices, ...]
    validation_sources_tokenized = processed_data[val_indices, ...]
    test_sources_tokenized = processed_data[test_indices, ...]

    data_input_train = data_input[train_indices, ...]
    data_input_val = data_input[val_indices, ...]
    data_input_test = data_input[test_indices, ...]

    field_data = field_data.reshape(tr, T, C, F)
    train_original = field_data[train_indices, ...]
    validation_original = field_data[val_indices, ...]
    test_original = field_data[test_indices, ...]

    if config.get('print_split_sizes', True):
        print(f'Train size: {train_sources_tokenized.shape[0]}')
        print(f'Validation size: {validation_sources_tokenized.shape[0]}')
        print(f'Test size: {test_sources_tokenized.shape[0]}')
        print(f'Train shape: {train_sources_tokenized.shape}')
        print(f'Validation shape: {validation_sources_tokenized.shape}')
        print(f'Test shape: {test_sources_tokenized.shape}')

    return train_sources_tokenized, validation_sources_tokenized, test_sources_tokenized, train_original, validation_original, test_original, data_input_train, data_input_val, data_input_test, stacked_coords, scaled_fields, mesh_processor, processor
def get_model(config: Dict[str, Any], device: torch.device) -> Tuple[TemporalModel, torch.nn.Module, torch.optim.Optimizer]:
    model = TemporalModel(config['num_layers'], 
                          config['embed_dim'], 
                          config['n_heads'], 
                          config['block_size'], 
                          config['scale_ratio'], 
                          config['src_len'],
                          config['num_fields'], 
                          config['down_proj'], 
                          config['dropout'], 
                          config['exchange_mode'], 
                          config['pos_encoding_mode'],
                          config['ib_scale_mode'], 
                          config['ib_addition_mode'], 
                          config['ib_mlp_layers'],
                          config['ib_num'],
                          config['add_info_after_cross'],
                          config['LN_type'])

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

def pre_train_temporal(config: Dict[str, Any]) -> Tuple[TemporalModel, torch.optim.Optimizer, DataLoader, DataLoader, DataLoader, torch.nn.Module, MeshProcessor, ProcessData]:
    device = torch.device(config['device'])
    trainLoader, validationLoader, testLoader, mesh_processor, processor = get_datasets(config)
    model, loss_fn, optimizer = get_model(config, device)

    return model, optimizer, trainLoader, validationLoader, testLoader, loss_fn, mesh_processor, processor
    
def train(config: Dict[str, Any], error_tracker):
    model, optimizer, trainLoader, validationLoader, testLoader, loss_fn, mesh_processor, processor = pre_train_temporal(config)
    device = torch.device(config['device'])
    model.to(device)
    
    start_time = time.time()
    prev_error = float('inf')
    prev_error_autoreg = float('inf')
    full_val_decoded = 10000
    iter = 0

    error_tracker.log_model(model, loss_fn, optimizer)

    full_eval_interval = config.get('full_eval_interval', 50)

    for epoch in range(1, config['epoch_num'] + 1):
        model.train()
        train_loss = 0.0
        num_train_batches = 0

        for data, target, _, ib in trainLoader:
            data, target, ib = data.to(device), target.to(device), ib.to(device)
            optimizer.zero_grad()
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

            outputs = model(data, ib)
            field_weights = config.get('field_loss_weights')
            if field_weights is None:
                loss = loss_fn(outputs, target)
            else:
                weights = torch.as_tensor(
                    field_weights, dtype=outputs.dtype, device=device
                )
                if weights.numel() != F:
                    raise ValueError(
                        f"field_loss_weights has {weights.numel()} entries, expected {F}"
                    )
                field_mse = torch.square(outputs - target).mean(dim=(0, 1, 3))
                loss = (field_mse * weights).sum() / weights.sum()

            rollout_weight = config.get('rollout_loss_weight', 0.0)
            rollout_start = config.get('rollout_loss_start_epoch', 1)
            if rollout_weight > 0 and epoch >= rollout_start:
                rollout_horizon = min(
                    int(config.get('rollout_loss_horizon', 4)), target.shape[1]
                )
                rollout_batch = min(
                    int(config.get('rollout_loss_batch_size', 1)), B
                )
                autoreg = data[:rollout_batch, :1]
                rollout_terms = []
                for rollout_step in range(rollout_horizon):
                    rollout_output = model(
                        autoreg,
                        ib[:rollout_batch, : rollout_step + 1],
                    )
                    next_step = rollout_output[:, -1:]
                    expected = target[
                        :rollout_batch,
                        rollout_step : rollout_step + 1,
                    ]
                    if field_weights is None:
                        rollout_term = loss_fn(next_step, expected)
                    else:
                        rollout_field_mse = torch.square(
                            next_step - expected
                        ).mean(dim=(0, 1, 3))
                        rollout_term = (
                            rollout_field_mse * weights
                        ).sum() / weights.sum()
                    rollout_terms.append(rollout_term)
                    autoreg = torch.cat((autoreg, next_step), dim=1)
                rollout_loss = torch.stack(rollout_terms).mean()
                loss = (1.0 - rollout_weight) * loss + rollout_weight * rollout_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_train_batches += 1
            iter += 1

        # Calculate average loss
        train_loss /= len(trainLoader)#num_train_batches

        # Log train errors
        train_metrics = {
            "Loss": train_loss,
        }
        error_tracker.record_error("train", epoch, train_metrics)

        if epoch % config.get('validation_interval', 1) == 0 or epoch == config['epoch_num']:
            model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for v_data, v_target, _, v_ib in validationLoader:
                    v_data, v_target, v_ib = v_data.to(device), v_target.to(device), v_ib.to(device)
                                # ib should be [B, L, ib_num]
                    B, L, F, D = v_data.shape
                    if v_ib.dim() == 2:
                        # static per-trajectory vector -> tile over time
                        v_ib = v_ib.unsqueeze(1).expand(B, L, -1)
                    elif v_ib.shape[1] == 1:
                        # time-invariant but already 3D -> tile
                        v_ib = v_ib.expand(B, L, -1)
                    elif v_ib.shape[1] == F:
                        # COMMON MISTAKE: ib laid out per-variable; collapse or map to per-time
                        # simplest: average across variables, then tile across time
                        v_ib = v_ib.mean(dim=1, keepdim=True).expand(B, L, -1)
                    elif v_ib.shape[1] != L:
                        raise ValueError(f"ib must be [B,L,ib_num]; got {v_ib.shape} with L={L}")
                    v_outputs = model(v_data, v_ib)
                    v_loss = loss_fn(v_outputs, v_target)

                    val_loss += v_loss.item()
                    num_val_batches += 1

            # Calculate average validation loss
            val_loss /= num_val_batches

            val_metrics = {
                "Loss": val_loss
            }

            # Perform full autoregressive evaluation at specified intervals
            if epoch % full_eval_interval == 0:
                full_eval_results = full_autoregressive_evaluation(
                    model, validationLoader, loss_fn, device, 
                    processor, mesh_processor,
                    plot_traj=True,
                    epoch=epoch,
                    config=config
                )
                
                # Merge full evaluation metrics into val_metrics
                val_metrics.update({
                    "Full_Encoded_Rel_MSE": full_eval_results['encoded_rel_mse'],
                    "Full_Decoded_Rel_MSE": full_eval_results['decoded_rel_mse'],
                })

                full_val_decoded = full_eval_results['decoded_rel_mse']
                # Checkpoint save
                if full_val_decoded < prev_error_autoreg:
                    prev_error_autoreg = full_val_decoded
                    print("--- Checkpoint Model Saved ---")
                    model.to('cpu')
                    model_path = f"{config['save_dir']}/temporal_Checkpoint_{config['case_name']}_{config['run_name']}.pt"
                    torch.save(model.state_dict(), model_path)
                    model.to(device)
                else:
                    print("--- No Improvement, Checkpoint Model Retained ---")

            # Log validation errors (now includes full eval metrics when performed)
            error_tracker.record_error("val", epoch, val_metrics)

            print(f"\nEpoch: {epoch}/{config['epoch_num']}")
            print(f"Train Loss: {train_loss:.8f}")
            for key, value in val_metrics.items():
                print(f"{key}: {value:.8f}")

            if val_loss < prev_error:
                prev_error = val_loss
                prev_error_autoreg = full_val_decoded
                print("--- New Best Model Saved ---")
                model.to('cpu')
                model_path = f"{config['save_dir']}/temporal_{config['case_name']}_{config['run_name']}.pt"
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