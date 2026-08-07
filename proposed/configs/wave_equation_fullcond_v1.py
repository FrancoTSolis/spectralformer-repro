"""WE1 GraphSpectralFormer with the complete initial-condition profile."""

from configs.wave_equation import get_config_spatial as _spatial_base
from configs.wave_equation import get_config_temporal as _temporal_base


def get_config_spatial():
    config = _spatial_base()
    config["save_dir"] = "./checkpoints-wave-equation-fullcond-v1-aligned"
    config["run_name"] = "fullcond1a"
    return config


def get_config_temporal():
    config = _temporal_base()
    config["save_dir"] = "./checkpoints-wave-equation-fullcond-v1-aligned"
    config["run_name"] = "fullcond1a"
    config["input_path"] = "./data_computed/WE1/input_data_full_tiled.npy"
    config["ib_num"] = 104
    config["edge_weight_mode"] = "ones"
    config["encoder_decoder_path"] = (
        "./checkpoints-wave-equation/encoder_decoder_wave_equation_run1.pt"
    )
    config["num_layers"] = 1
    config["dropout"] = 0.1
    config["learning_rate"] = 1e-4
    config["epoch_num"] = 3000
    config["full_eval_interval"] = 500
    config["perform_initial_test"] = False
    config["use_wandb"] = False
    return config
