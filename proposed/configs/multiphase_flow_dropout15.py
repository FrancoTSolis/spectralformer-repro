"""One-layer multiphase GraphSpectralFormer with stronger regularization."""

from configs.multiphase_flow_depth2 import get_config_spatial
from configs.multiphase_flow_depth2 import get_config_temporal as _base


def get_config_temporal():
    config = _base()
    config["save_dir"] = "./checkpoints-multiphase-dropout15"
    config["run_name"] = "dropout15"
    config["num_layers"] = 1
    config["dropout"] = 0.15
    config["learning_rate"] = 8e-5
    return config
