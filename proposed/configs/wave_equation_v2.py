"""Wave Equation (WE1) v2 config - more temporal capacity, lower LR."""
import torch

def get_config_spatial():
    from configs.wave_equation import get_config_spatial as base
    config = base()
    config['save_dir'] = './checkpoints-wave-equation-v2'
    config['run_name'] = 'run2'
    return config

def get_config_temporal():
    from configs.wave_equation import get_config_temporal as base
    config = base()
    config['save_dir'] = './checkpoints-wave-equation-v2'
    config['encoder_decoder_path'] = './checkpoints-wave-equation/encoder_decoder_wave_equation_run1.pt'
    config['run_name'] = 'run2'
    config['num_layers'] = 2
    config['learning_rate'] = 5e-5
    config['dropout'] = 0.15
    config['epoch_num'] = 3000
    return config
