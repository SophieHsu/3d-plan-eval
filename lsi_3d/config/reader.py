import os
import toml

LSI_CONFIG_ALGO_DIR = 'lsi_3d/config/algorithm'
LSI_CONFIG_MAP_DIR = 'lsi_3d/config/map'
LSI_CONFIG_AGENT_DIR = 'lsi_3d/config/agent'
LSI_CONFIG_EXP_DIR = 'lsi_3d/config/experiment'

def read_in_lsi_config(exp_config_file):
    experiment_config = toml.load(
        os.path.join(LSI_CONFIG_EXP_DIR, exp_config_file))
    algorithm_config = toml.load(
        os.path.join(LSI_CONFIG_ALGO_DIR,
                     experiment_config["algorithm_config"]))
    map_config = toml.load(
        os.path.join(LSI_CONFIG_MAP_DIR, experiment_config["map_config"]))
    agent_configs = []
    for agent_config_file in experiment_config["agent_config"]:
        agent_config = toml.load(
            os.path.join(LSI_CONFIG_AGENT_DIR, agent_config_file))
        agent_configs.append(agent_config)
    return experiment_config, algorithm_config, map_config, agent_configs