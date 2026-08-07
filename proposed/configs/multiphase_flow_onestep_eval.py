"""Evaluation config for the unevaluated one-step-best multiphase checkpoint."""

from configs.multiphase_flow import get_config_spatial as _spatial_base
from configs.multiphase_flow import get_config_temporal as _temporal_base


def get_config_spatial():
    config = _spatial_base()
    config["save_dir"] = "./checkpoints-multiphase-flow-spectral-clustering-2"
    config["use_wandb"] = False
    return config


def get_config_temporal():
    config = _temporal_base()
    config["save_dir"] = "./checkpoints-multiphase-flow-spectral-clustering-2"
    config["encoder_decoder_path"] = (
        "./checkpoints-multiphase-flow-spectral-clustering-2/"
        "encoder_decoder_multiphase_flow_run1.pt"
    )
    config["use_wandb"] = False
    config["perform_initial_test"] = False
    return config
