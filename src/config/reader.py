import os
import toml

from src.config.constants import CONF_KEYS, CONF_DIR_NAMES


def get_configs(exp_config_file):
    experiment_config = toml.load(os.path.join(*CONF_DIR_NAMES[CONF_KEYS.CONF_DIRS.EXP], exp_config_file))

    return experiment_config
