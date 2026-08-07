"""Cylinder baseline with the proposed model's stage-specific splits."""
import torch

def get_config_spatial():
    from configs.cylinder_flow import get_config_spatial as base
    config = base()
    config['train_fraction'] = 0.8
    config['val_fraction'] = 0.1
    config['save_dir'] = './checkpoints-cylinder-retrain'
    config['run_name'] = 'retrain1'
    config['use_wandb'] = False
    config['epoch_num'] = 500
    config['batch_size'] = 512
    config['validation_interval'] = 5
    return config

def get_config_temporal():
    from configs.cylinder_flow import get_config_temporal as base
    config = base()
    # Match the canonical temporal evaluation manifest: 61/20/20 trajectories.
    config['train_fraction'] = 0.6
    config['val_fraction'] = 0.2
    # Give ViT-SEA the same six deterministic Re/inverse-Re features used by
    # the canonical GraphSpectralFormer checkpoint.
    config['input_path'] = './data/CF/all_data/input_data_6feat.npy'
    config['ib_num'] = 6
    config['save_dir'] = './checkpoints-cylinder-retrain'
    config['encoder_decoder_path'] = './checkpoints-cylinder-retrain/encoder_decoder_cylinder_flow_retrain1.pt'
    config['run_name'] = 'retrain1'
    config['use_wandb'] = False
    config['epoch_num'] = 1500
    config['full_eval_interval'] = 200
    return config
