"""Rollout-aware multiphase GraphSpectralFormer with pressure weighting."""

from configs.multiphase_flow_depth2 import get_config_spatial
from configs.multiphase_flow_depth2 import get_config_temporal as _base


def get_config_temporal():
    config = _base()
    config["save_dir"] = "./checkpoints-multiphase-rollout-v1"
    config["run_name"] = "rollout1"
    config["num_layers"] = 1
    config["dropout"] = 0.1
    config["learning_rate"] = 8e-5
    config["field_loss_weights"] = [1.0, 2.0]
    config["rollout_loss_weight"] = 0.4
    config["rollout_loss_horizon"] = 4
    config["rollout_loss_batch_size"] = 1
    config["rollout_loss_start_epoch"] = 250
    return config
