from collections import defaultdict
from this import d
import numpy as np
from pygame import init


class LsiMdp(object):
    def __init__(self, map, start_locations, hl_start_state):
        self.map = map
        self.start_locations = start_locations
        self.num_items_for_soup = 3
        self.delivery_reward = 20
        self.hl_start_state = hl_start_state
        self.pot_locations = self.get_pot_locations()

    @staticmethod
    def from_config(map_config, agent_configs, exp_config, grid):
        #a1_loc = (agent_configs[1]['start_x'], agent_configs[1]['start_y'])
        #a2_loc = (agent_configs[0]['start_x'], agent_configs[0]['start_y'])
        # rows = map_config['rows']
        # columns = map_config['columns']
        # grid = []
        # split = map.splitlines()
        # for r in range(len(split)):
        #     grid.append([])
        #     for c in range(len(split[r])):
        #         grid[r].append(split[r][c])
        


        hl_start_state = exp_config['hl_start_state']
        start_locations = [(agent_config['start_x'], agent_config['start_y'], agent_config['start_direction']) for agent_config in agent_configs]

        return LsiMdp(grid, start_locations, hl_start_state)

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

    def get_pot_states(self, state):
        """Returns dict with structure:
        {
         empty: [ObjStates]
         onion: {
            'x_items': [soup objects with x items],
            'cooking': [ready soup objs]
            'ready': [ready soup objs],
            'partially_full': [all non-empty and non-full soups]
            }
         tomato: same dict structure as above
        }
        """
        pots_states_dict = {}
        pots_states_dict['empty'] = []
        pots_states_dict['onion'] = defaultdict(list)
        for pot_pos in self.get_pot_locations():
            if not state.has_object(pot_pos):
                pots_states_dict['empty'].append(pot_pos)
            else:
                soup_obj = state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict[soup_type]['{}_items'.format(
                        num_items)].append(pot_pos)
                elif num_items == self.num_items_for_soup:
                    assert cook_time <= self.soup_cooking_time
                    if cook_time == self.soup_cooking_time:
                        pots_states_dict[soup_type]['ready'].append(pot_pos)
                    else:
                        pots_states_dict[soup_type]['cooking'].append(pot_pos)
                else:
                    raise ValueError("Pot with more than {} items".format(
                        self.num_items_for_soup))

                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict[soup_type]['partially_full'].append(
                        pot_pos)

        return pots_states_dict

    def get_dish_dispenser_locations(self):
        return self.where_map_is('B')

    def get_onion_dispenser_locations(self):
        return self.where_map_is('F')
        #return list(self.terrain_pos_dict['O'])

    def get_tomato_dispenser_locations(self):
        return list(self.terrain_pos_dict['T'])

    def get_serving_locations(self):
        return self.where_map_is('T')

    def get_pot_locations(self):
        return self.where_map_is('P')

    def get_counter_locations(self):
        return self.where_map_is('X')

    def where_map_is(self, letter):
        indexes = []
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == letter:
                    indexes.append((i,j))

        return indexes

    def map_action_to_location(self, holding, in_pot, orders, action_obj, p0_obj = None):
        """
        Get the next location the agent will be in based on current world state and medium level actions.
        """
        p0_obj = p0_obj if p0_obj is not None else holding
        action, obj = action_obj
        #pots_states_dict = self.mdp.get_pot_states()
        location = []
        if action == 'pickup' and obj != 'soup':
            if p0_obj != 'None':
                location = self.drop_item(world_state)
            else:
                if obj == 'onion':
                    location = self.get_onion_dispenser_locations()
                elif obj == 'tomato':
                    location = self.get_tomato_dispenser_locations()
                elif obj == 'dish':
                    location = self.get_dish_dispenser_locations()
                else:
                    print(p0_obj, action, obj)
                    ValueError()
        elif action == 'pickup' and obj == 'soup':
            if p0_obj != 'dish' and p0_obj != 'None':
                location = self.drop_item(world_state)
            elif p0_obj == 'None':
                location = self.get_dish_dispenser_locations()
                print(f'Next Dish Location: {location}')
            else:
                location = self.get_pot_locations() # + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)

        elif action == 'drop':
            if obj == 'onion' or obj == 'tomato':
                #location = self.mdp.get_partially_full_pots(pots_states_dict) + self.mdp.get_empty_pots(pots_states_dict)
                location = self.get_pot_locations()
            elif obj == 'dish':
                location = self.drop_item(world_state)
            else:
                print(p0_obj, action, obj)
                ValueError()

        elif action == 'deliver':
            if p0_obj != 'soup':
                location = self.get_empty_counter_locations(world_state)
            else:
                location = self.get_serving_locations()

        else:
            print(p0_obj, action, obj)
            ValueError()

        return location
