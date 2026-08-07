"""Two-layer, edge-aligned multiphase GraphSpectralFormer temporal model."""

from configs.multiphase_flow import get_config_spatial as _spatial_base
from configs.multiphase_flow import get_config_temporal as _temporal_base


def get_config_spatial():
    config = _spatial_base()
    config["save_dir"] = "./checkpoints-multiphase-depth2"
    config["run_name"] = "depth2"
    return config


def get_config_temporal():
    config = _temporal_base()
    config["save_dir"] = "./checkpoints-multiphase-depth2"
    config["run_name"] = "depth2"
    config["encoder_decoder_path"] = (
        "./checkpoints-multiphase-flow-spectral-clustering-2/"
        "encoder_decoder_multiphase_flow_run1.pt"
    )
    config["edge_weight_mode"] = "ones"
    config["num_layers"] = 2
    config["dropout"] = 0.15
    config["learning_rate"] = 5e-5
    config["epoch_num"] = 3000
    config["full_eval_interval"] = 500
    config["perform_initial_test"] = False
    config["use_wandb"] = False
    return config
