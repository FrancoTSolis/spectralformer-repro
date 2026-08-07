"""Two-layer, six-condition cylinder GraphSpectralFormer temporal variant."""

from configs.cylinder_flow import get_config_spatial as _spatial_base
from configs.cylinder_flow import get_config_temporal as _temporal_base


def get_config_spatial():
    config = _spatial_base()
    config["save_dir"] = "./checkpoints-cylinder-capacity-v1"
    config["run_name"] = "capacity1"
    return config


def get_config_temporal():
    config = _temporal_base()
    config["save_dir"] = "./checkpoints-cylinder-capacity-v1"
    config["run_name"] = "capacity1"
    config["encoder_decoder_path"] = (
        "./checkpoints-cylinder-flow-spectral-clustering/"
        "encoder_decoder_cylinder_flow_run1.pt"
    )
    config["input_path"] = "./data_old/CF/all_data/input_data_6feat.npy"
    config["ib_num"] = 6
    config["edge_weight_mode"] = "ones"
    config["num_layers"] = 2
    config["dropout"] = 0.15
    config["learning_rate"] = 5e-5
    config["epoch_num"] = 2000
    config["full_eval_interval"] = 500
    config["perform_initial_test"] = False
    config["use_wandb"] = False
    return config
