"""Multiphase flow baseline retrained with proposed model's data split (80/10/10)."""
import torch

def get_config_spatial():
    from configs.multiphase_flow import get_config_spatial as base
    config = base()
    config['save_dir'] = './checkpoints-multiphase-retrain'
    config['run_name'] = 'retrain1'
    config['use_wandb'] = False
    config['epoch_num'] = 1500
    return config

def get_config_temporal():
    from configs.multiphase_flow import get_config_temporal as base
    config = base()
    config['save_dir'] = './checkpoints-multiphase-retrain'
    config['encoder_decoder_path'] = './checkpoints-multiphase-retrain/encoder_decoder_multiphase_flow_retrain1.pt'
    config['run_name'] = 'retrain1'
    config['use_wandb'] = False
    config['epoch_num'] = 1500
    return config
