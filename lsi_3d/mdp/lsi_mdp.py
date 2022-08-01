from this import d
from pygame import init


class LsiMdp(object):
    def __init__(self, map, start_locations):
        self.map = map
        self.start_locations = start_locations
        self.num_items_for_soup = 3
        self.delivery_reward = 20

    @staticmethod
    def from_config(map_config, agent_configs):
        map = map_config['layout']
        #a1_loc = (agent_configs[1]['start_x'], agent_configs[1]['start_y'])
        #a2_loc = (agent_configs[0]['start_x'], agent_configs[0]['start_y'])

        start_locations = [(agent_config['start_x'], agent_config['start_y']) for agent_config in agent_configs]

        return LsiMdp(map, start_locations)