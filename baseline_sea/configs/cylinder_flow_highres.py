"""High-resolution cylinder flow config for baseline model."""
import torch

def get_config_spatial():
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './checkpoints-cylinder-highres',
        'field_data_path': './data/CF/all_data/field_data.npy',
        'input_path': './data/CF/all_data/input_data.npy',
        'coordinates_path': './data/CF/all_data/coordinates.npy',

        'train_fraction': 0.6,
        'val_fraction': 0.2,
        'random_seed': 42,
        'dimension': '2D',
        'field_groups': [[0, 1], [2]],
        'scale_feature_range': None,
        'use_quantiles': False,
        'q_low': 1.0,
        'q_high': 99.0,
        'log_mode': 'off',
        'slog_scale': 'median',
        'scaler_eps': 1e-12,
        'csv_scale_name': 'scaler',
        'm': 14,
        'n': 14,
        'k': None,
        'pad_id': -1,
        'pad_field_value': 0,
        'MLP_hidden': 480,
        'num_layers': 12,
        'embed_dim': 16,
        'n_heads': 8,
        'block_size': 2024,
        'src_len': 0,
        'dropout': 0.0,
        'variational': False,
        'test_mesh_structure': False,
        'perform_initial_test': True,
        'validation_interval': 10,
        'final_save': False,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'KL_weight_min': 0,
        'KL_weight_max': 0,
        'epoch_num': 3000,
        'use_wandb': False,
        'run_name': 'run1',
        'case_name': 'cylinder_flow_highres',
        'project_name': 'spectralformer_spatial',
        'spatial_batch_size': 500,
        'SEA_isolate': True,
        'SEA_mixed': False
    }
    config['embed_dim_spatial'] = config['embed_dim']
    config['n_heads_spatial'] = config['n_heads']
    config['block_size_spatial'] = config['block_size']
    config['dropout_spatial'] = config['dropout']
    config['MLP_hidden_spatial'] = config['MLP_hidden']
    config['num_layers_spatial'] = config['num_layers']
    config['src_len_spatial'] = config['src_len']
    config['variational_spatial'] = config['variational']
    return config

def get_config_temporal():
    spatial_config = get_config_spatial()
    return {
        'device': spatial_config['device'],
        'save_dir': spatial_config['save_dir'],
        'field_data_path': spatial_config['field_data_path'],
        'input_path': spatial_config['input_path'],
        'coordinates_path': spatial_config['coordinates_path'],
        'static_mesh': True,

        'train_fraction': 0.6,
        'val_fraction': 0.2,
        'random_seed': 42,

        'dimension': spatial_config['dimension'],
        'field_groups': spatial_config['field_groups'],
        'scale_feature_range': spatial_config['scale_feature_range'],
        'use_quantiles': spatial_config.get('use_quantiles', False),
        'q_low': spatial_config.get('q_low', 1.0),
        'q_high': spatial_config.get('q_high', 99.0),
        'log_mode': spatial_config.get('log_mode', 'off'),
        'slog_scale': spatial_config.get('slog_scale', 'median'),
        'scaler_eps': spatial_config.get('scaler_eps', 1e-12),
        'csv_scale_name': spatial_config['csv_scale_name'],
        'm': spatial_config['m'],
        'n': spatial_config['n'],
        'k': spatial_config['k'],
        'pad_id': spatial_config['pad_id'],
        'pad_field_value': spatial_config['pad_field_value'],

        'MLP_hidden_spatial': spatial_config['MLP_hidden'],
        'num_layers_spatial': spatial_config['num_layers'],
        'embed_dim_spatial': spatial_config['embed_dim'],
        'n_heads_spatial': spatial_config['n_heads'],
        'block_size_spatial': spatial_config['block_size'],
        'dropout_spatial': spatial_config['dropout'],
        'variational_spatial': spatial_config['variational'],
        'src_len_spatial': spatial_config['src_len'],
        'encoder_decoder_path': f"{spatial_config['save_dir']}/encoder_decoder_{spatial_config['case_name']}_{spatial_config['run_name']}.pt",
        'spatial_batch_size': spatial_config['batch_size'],

        'num_layers': 1,
        'embed_dim': 1024,
        'n_heads': 8,
        'block_size': 2024,
        'scale_ratio': 8,
        'src_len': 0,
        'num_fields': len(spatial_config['field_groups']),
        'down_proj': 2,
        'dropout': 0.1,
        'exchange_mode': 'sea',
        'pos_encoding_mode': 'learnable',
        'ib_scale_mode': 'mlp',
        'ib_addition_mode': 'add',
        'ib_mlp_layers': 1,
        'ib_num': 1,
        'add_info_after_cross': True,
        'LN_type': 'adaln',

        'test_mesh_structure': False,
        'perform_initial_test': True,

        'validation_interval': 10,
        'full_eval_interval': 100,
        'final_save': False,

        'batch_size': 2,
        'dataset_src_len': 399,
        'dataset_overlap': 0,
        'dataset_time_shifting_flag': False,

        'variational': False,
        'learning_rate': 1e-4,
        'KL_weight_min': 0,
        'KL_weight_max': 0,
        'epoch_num': 3000,

        'use_wandb': False,
        'run_name': 'run1',
        'case_name': 'cylinder_flow_highres',
        'project_name': 'spectralformer_temporal',

        'SEA_isolate': spatial_config['SEA_isolate'],
        'SEA_mixed': spatial_config['SEA_mixed']
    }
