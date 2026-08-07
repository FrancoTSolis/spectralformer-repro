"""Pressure-aware fine-tune from the canonical multiphase checkpoint."""

from configs.multiphase_flow_finetune_v1 import get_config_spatial
from configs.multiphase_flow_finetune_v1 import get_config_temporal as _base


def get_config_temporal():
    config = _base()
    config["save_dir"] = "./checkpoints-multiphase-finetune-v2"
    config["run_name"] = "finetune2"
    config["rollout_loss_weight"] = 0.05
    config["rollout_loss_horizon"] = 8
    config["field_loss_weights"] = [1.0, 1.2]
    return config
