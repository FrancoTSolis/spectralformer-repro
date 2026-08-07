"""Low-rate rollout fine-tune from the canonical multiphase checkpoint."""

from configs.multiphase_flow import get_config_spatial as _spatial_base
from configs.multiphase_flow import get_config_temporal as _temporal_base


def get_config_spatial():
    config = _spatial_base()
    config["save_dir"] = "./checkpoints-multiphase-finetune-v1"
    config["run_name"] = "finetune1"
    return config


def get_config_temporal():
    config = _temporal_base()
    config["save_dir"] = "./checkpoints-multiphase-finetune-v1"
    config["run_name"] = "finetune1"
    config["encoder_decoder_path"] = (
        "./checkpoints-multiphase-flow-spectral-clustering-2/"
        "encoder_decoder_multiphase_flow_run1.pt"
    )
    config["learning_rate"] = 1e-5
    config["epoch_num"] = 1000
    config["full_eval_interval"] = 100
    config["rollout_loss_weight"] = 0.1
    config["rollout_loss_horizon"] = 4
    config["rollout_loss_batch_size"] = 1
    config["rollout_loss_start_epoch"] = 1
    config["perform_initial_test"] = False
    config["use_wandb"] = False
    return config
