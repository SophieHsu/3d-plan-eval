import os
import toml

from lsi_3d.config.constants import CONF_KEYS, CONF_DIR_NAMES


def get_configs(exp_config_file):
    experiment_config = toml.load(os.path.join(*CONF_DIR_NAMES[CONF_KEYS.CONF_DIRS.ALGO], exp_config_file))
    map_config = toml.load(
        os.path.join(*CONF_DIR_NAMES[CONF_KEYS.CONF_DIRS.MAP], experiment_config[CONF_KEYS.MAP_CONFIG])
    )

    return experiment_config, map_config
