"""Advection Equation (E1) v3 config - same arch as v1 but lower LR."""
import torch

def get_config_spatial():
    from configs.advection_equation import get_config_spatial as base
    config = base()
    config['save_dir'] = './checkpoints-advection-equation-v3'
    config['run_name'] = 'run3'
    return config

def get_config_temporal():
    from configs.advection_equation import get_config_temporal as base
    config = base()
    config['save_dir'] = './checkpoints-advection-equation-v3'
    config['encoder_decoder_path'] = './checkpoints-advection-equation/encoder_decoder_advection_equation_run1.pt'
    config['run_name'] = 'run3'
    config['learning_rate'] = 3e-5
    config['dropout'] = 0.15
    config['epoch_num'] = 3000
    return config
