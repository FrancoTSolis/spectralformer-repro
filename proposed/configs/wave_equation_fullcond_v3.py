"""Four-layer WE1 full-initial-condition GraphSpectralFormer."""

from configs.wave_equation_fullcond_v1 import get_config_spatial
from configs.wave_equation_fullcond_v1 import get_config_temporal as _base


def get_config_temporal():
    config = _base()
    config["save_dir"] = "./checkpoints-wave-equation-fullcond-v3"
    config["run_name"] = "fullcond3"
    config["num_layers"] = 4
    config["dropout"] = 0.15
    config["learning_rate"] = 3e-5
    return config
