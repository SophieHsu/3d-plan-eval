from this import d
from pygame import init


class LsiMdp(object):
    def __init__(self, map, start_locations, hl_start_state):
        self.map = map
        self.start_locations = start_locations
        self.num_items_for_soup = 3
        self.delivery_reward = 20
        self.hl_start_state = hl_start_state

    @staticmethod
    def from_config(map_config, agent_configs, exp_config):
        map = map_config['layout']
        #a1_loc = (agent_configs[1]['start_x'], agent_configs[1]['start_y'])
        #a2_loc = (agent_configs[0]['start_x'], agent_configs[0]['start_y'])

        hl_start_state = exp_config['hl_start_state']
        start_locations = [(agent_config['start_x'], agent_config['start_y'], agent_config['start_direction']) for agent_config in agent_configs]

        return LsiMdp(map, start_locations, hl_start_state)

    def get_state_transition(self, state, joint_action):
        """Gets information about possible transitions for the action.

        Returns the next state, sparse reward and reward shaping.
        Assumes all actions are deterministic.

        NOTE: Sparse reward is given only when soups are delivered,
        shaped reward is given only for completion of subgoals
        (not soup deliveries).
        """
        events_infos = {
            event: [False] * self.num_players
            for event in EVENT_TYPES
        }

        assert not self.is_terminal(
            state), "Trying to find successor of a terminal state: {}".format(
                state)
        # for action, action_set in zip(joint_action, self.get_actions(state)):
        #     if action not in action_set:
        #         raise ValueError("Illegal action %s in state %s" %
        #                          (action, state))

        new_state = state.deepcopy()

        # Resolve interacts first
        sparse_reward, shaped_reward = self.resolve_interacts(
            new_state, joint_action, events_infos)

        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations

        # Resolve player movements
        self.resolve_movement(new_state, joint_action)

        # Finally, environment effects
        self.step_environment_effects(new_state)

        # Additional dense reward logic
        # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)

        return new_state, sparse_reward, shaped_reward, events_infos

    def get_dish_dispenser_locations(self):
        return list(self.terrain_pos_dict['D'])

    def get_onion_dispenser_locations(self):
        return [(0,2)]
        #return list(self.terrain_pos_dict['O'])

    def get_tomato_dispenser_locations(self):
        return list(self.terrain_pos_dict['T'])

    def get_serving_locations(self):
        return list(self.terrain_pos_dict['S'])

    def get_pot_locations(self):
        return list(self.terrain_pos_dict['P'])

    def get_counter_locations(self):
        return list(self.terrain_pos_dict['X'])