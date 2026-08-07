"""WE1 ViT-SEA baseline with the proposed model's split and preprocessing."""
import torch

def get_config_spatial():
    from configs.wave_equation import get_config_spatial as base
    config = base()
    config['save_dir'] = './checkpoints-wave-equation-matched'
    config['run_name'] = 'matched1'
    config['use_quantiles'] = True
    config['q_low'] = 1.0
    config['q_high'] = 99.0
    config['use_wandb'] = False
    config['epoch_num'] = 500
    config['batch_size'] = 2048
    config['validation_interval'] = 5
    return config

def get_config_temporal():
    from configs.wave_equation import get_config_temporal as base
    config = base()
    config['save_dir'] = './checkpoints-wave-equation-matched'
    config['encoder_decoder_path'] = './checkpoints-wave-equation-matched/encoder_decoder_wave_equation_matched1.pt'
    config['run_name'] = 'matched1'
    config['use_quantiles'] = True
    config['q_low'] = 1.0
    config['q_high'] = 99.0
    config['use_wandb'] = False
    config['epoch_num'] = 2000
    config['full_eval_interval'] = 200
    return config
