"""Two-layer WE1 full-initial-condition GraphSpectralFormer."""

from configs.wave_equation_fullcond_v1 import get_config_spatial
from configs.wave_equation_fullcond_v1 import get_config_temporal as _base


def get_config_temporal():
    config = _base()
    config["save_dir"] = "./checkpoints-wave-equation-fullcond-v2"
    config["run_name"] = "fullcond2"
    config["num_layers"] = 2
    config["learning_rate"] = 5e-5
    return config
