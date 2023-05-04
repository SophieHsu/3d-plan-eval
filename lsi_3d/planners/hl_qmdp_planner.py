import copy
import random
import time

import numpy as np
from lsi_3d.mdp.action import Action
from lsi_3d.planners.high_level_mdp import HighLevelMdpPlanner
from lsi_3d.planners.mid_level_motion import AStarMotionPlanner
from lsi_3d.mdp.hl_state import AgentState, WorldState

class HumanSubtaskQMDPPlanner(HighLevelMdpPlanner):
    def __init__(self, mdp, mlp:AStarMotionPlanner):
    
    # , mlp_params, \
    #     state_dict = {}, state_idx_dict = {}, action_dict = {}, action_idx_dict = {}, transition_matrix = None, reward_matrix = None, policy_matrix = None, value_matrix = None, \
    #     num_states = 0, num_rounds = 0, epsilon = 0.01, discount = 0.8):

        super().__init__(mdp)

        self.world_state_cost_dict = {}
        # self.jmp = JointMotionPlanner(mdp, mlp_params)
        # self.mp = self.jmp.motion_planner
        self.subtask_dict = {}
        self.subtask_idx_dict = {}
        
        self.mlp = mlp

    # @staticmethod
    # def from_pickle_or_compute(mdp, mlp_params, custom_filename=None, force_compute_all=False, info=True, force_compute_more=False):

    #     # assert isinstance(mdp, OvercookedGridworld)

    #     filename = custom_filename if custom_filename is not None else mdp.layout_name + '_' + 'human_subtask_aware_qmdp' + '.pkl'

    #     if force_compute_all:
    #         mdp_planner = HumanSubtaskQMDPPlanner(mdp, mlp_params)
    #         mdp_planner.compute_mdp(filename)
    #         return mdp_planner
        
    #     try:
    #         mdp_planner = HumanSubtaskQMDPPlanner.from_mdp_planner_file(filename)
            
    #         if force_compute_more:
    #             print("Stored mdp_planner computed ", str(mdp_planner.num_rounds), " rounds. Compute another " + str(TRAINNINGUNIT) + " more...")
    #             mdp_planner.compute_mdp(filename)
    #             return mdp_planner

    #     except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
    #         print("Recomputing planner due to:", e)
    #         mdp_planner = HumanSubtaskQMDPPlanner(mdp, mlp_params)
    #         mdp_planner.compute_mdp(filename)
    #         return mdp_planner

    #     if info:
    #         print("Loaded HumanSubtaskQMDPPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))

    #     return mdp_planner

    def init_human_aware_states(self, state_idx_dict=None, order_list=None):
        """
        States: agent 0 holding object, number of item in pot, order list, agent 1 (usually human) holding object, agent 1 subtask.
        """

        # set state dict as [p0_obj, num_item_in_pot, order_list]
        self.init_states(order_list=order_list) 

        # add [p1_obj, subtask] to [p0_obj, num_item_in_pot, order_list]
        objects = ['onion', 'soup', 'dish', 'None'] # 'tomato'
        self.subtask_dict = copy.deepcopy(self.action_dict)
        del self.subtask_dict['drop_dish']
        original_state_dict = copy.deepcopy(self.state_dict)
        self.state_dict.clear()
        self.state_idx_dict.clear()
        for i, obj in enumerate(objects):
            for j, subtask in enumerate(self.subtask_dict.items()):
                self.subtask_idx_dict[subtask[0]] = j
                if self._init_is_valid_object_subtask_pair(obj, subtask[0]):
                    for ori_key, ori_value in original_state_dict.items():
                        new_key = ori_key+'_'+obj + '_' + subtask[0]
                        new_obj = original_state_dict[ori_key]+[obj] + [subtask[0]]
                        self.state_dict[new_key] = new_obj # update value
                        self.state_idx_dict[new_key] = len(self.state_idx_dict)
        return

        # print('subtask dict =', self.subtask_dict)

    def init_transition(self, transition_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]
            for action_key, action_idx in self.action_idx_dict.items():
                normalize_count = 0
                
                # decode state information
                p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
                # calculate next states for p1 (a.k.a. human)
                p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)

                # calculate next states for p0 (conditioned on p1 (a.k.a. human))
                for p1_nxt_state in p1_nxt_states:
                    action, next_state_key = self.state_transition(p0_state, p1_nxt_world_info, human_state=p1_nxt_state)
                    # for action, next_state_key in zip(actions, next_state_keys):
                        # print(p0_state, p1_nxt_world_info, p1_nxt_state, action, next_state_keys)
                    if action_key == action:
                        next_state_idx= self.state_idx_dict[next_state_key]
                        self.transition_matrix[action_idx, state_idx, next_state_idx] += 1.0

                if np.sum(self.transition_matrix[action_idx, state_idx]) > 0.0:
                    self.transition_matrix[action_idx, state_idx] /= np.sum(self.transition_matrix[action_idx, state_idx])

        self.transition_matrix[self.transition_matrix == 0.0] = 0.000001

    def get_successor_states(self, start_world_state, start_state_key, debug=False):
        """
        Successor states for qmdp medium-level actions.
        """
        if len(self.state_dict[start_state_key][2:]) <= 2: # [p0_obj, num_item_in_soup, orders, p1_obj, subtask] 
            return []

        ori_state_idx = self.state_idx_dict[start_state_key]
        successor_states = []

        agent_action_idx_arr, next_state_idx_arr = np.where(self.transition_matrix[:, ori_state_idx] > 0.000001) # returns array(action idx), array(next_state_idx)
        start_time = time.time()
        for next_action_idx, next_state_idx in zip(agent_action_idx_arr, next_state_idx_arr):
            next_world_state, cost = self.mdp_action_state_to_world_state(next_action_idx, next_state_idx, start_world_state)
            successor_states.append((self.get_key_from_value(self.state_idx_dict, next_state_idx), next_world_state, cost))
            if debug: print('Action {} from {} to {} costs {} in {} seconds.'.format(self.get_key_from_value(self.action_idx_dict, next_action_idx), self.get_key_from_value(self.state_idx_dict, ori_state_idx), self.get_key_from_value(self.state_idx_dict, next_state_idx), cost, time.time()-start_time))

        return successor_states

    def decode_state_info(self, state_obj):
        return state_obj[0], state_obj[-2:], state_obj[1:-2]

    def _init_is_valid_object_subtask_pair(self, obj, subtask):
        if obj == 'None':
            if subtask == 'pickup_dish':
                return True
            elif subtask == 'pickup_onion':
                return True
            elif subtask == 'pickup_tomato':
                return True
            # elif subtask == 'pickup_soup':
            #     return True
            else:
                return False
        else:
            if obj == 'onion' and subtask == 'drop_onion':
                return True
            elif obj == 'tomato' and subtask == 'drop_tomato':
                return True
            elif (obj == 'dish') and subtask == 'pickup_soup':
                return True
            # elif (obj == 'dish') and subtask == 'drop_dish':
            #     return True
            elif obj == 'soup' and subtask == 'deliver_soup':
                return True
            else:
                return False
        return True

    def _is_valid_object_subtask_pair(self, obj, subtask, soup_finish, greedy=False):
        if obj == 'None':
            if greedy != True and (subtask == 'pickup_dish' or subtask == 'pickup_onion') and soup_finish <= self.mdp.num_items_for_soup:
                return True
            elif greedy == True and subtask == 'pickup_onion' and soup_finish < self.mdp.num_items_for_soup:
                return True
            elif greedy == True and subtask == 'pickup_dish' and soup_finish == self.mdp.num_items_for_soup:
                return True
            elif subtask == 'pickup_tomato':
                return True
            # elif subtask == 'pickup_soup':
            #     return True
            else:
                return False
        else:
            if obj == 'onion' and subtask == 'drop_onion':
                return True
            elif obj == 'tomato' and subtask == 'drop_tomato':
                return True
            elif (obj == 'dish') and subtask == 'pickup_soup':
                return True
            # elif (obj == 'dish') and subtask == 'drop_dish':
            #     return True
            elif obj == 'soup' and subtask == 'deliver_soup':
                return True
            else:
                return False
        return True

    def human_state_subtask_transition(self, human_state, world_info):
        player_obj = human_state[0]; subtask = human_state[1]
        soup_finish = world_info[0]; orders = [] if len(world_info) < 2 else world_info[1:]
        next_obj = player_obj; next_subtasks = []; 
        next_soup_finish = soup_finish

        if player_obj == 'None':
            if subtask == 'pickup_dish':
                next_obj = 'dish'
                next_subtasks = ['pickup_soup']#, 'drop_dish']

            elif subtask == 'pickup_onion':
                next_obj = 'onion'
                next_subtasks = ['drop_onion']
            
            elif subtask == 'pickup_tomato':
                next_obj = 'tomato'
                next_subtasks = ['drop_tomato']
            
            # elif subtask == 'pickup_soup':
            #     next_obj = 'soup'
            #     next_subtasks = ['deliver_soup']

        else:
            if player_obj == 'onion' and subtask == 'drop_onion' and soup_finish < self.mdp.num_items_for_soup:
                next_obj = 'None'
                next_soup_finish += 1
                next_subtasks = ['pickup_onion', 'pickup_dish'] # 'pickup_tomato'
            
            elif player_obj == 'onion' and subtask == 'drop_onion' and soup_finish == self.mdp.num_items_for_soup:
                next_obj = 'onion'
                next_subtasks = ['drop_onion']

            elif player_obj == 'tomato' and subtask == 'drop_tomato':
                next_obj = 'None'
                next_soup_finish += 1
                next_subtasks = ['pickup_onion', 'pickup_dish'] # 'pickup_tomato'

            elif (player_obj == 'dish') and subtask == 'pickup_soup':
                next_obj = 'soup'
                next_soup_finish = 0
                next_subtasks = ['deliver_soup']

            # elif (player_obj == 'dish') and subtask == 'drop_dish':
            #     next_obj = 'None'
            #     next_subtasks = ['pickup_onion', 'pickup_dish'] # 'pickup_tomato'

            elif player_obj == 'soup' and subtask == 'deliver_soup':
                next_obj = 'None'
                next_subtasks = ['pickup_onion', 'pickup_dish'] # 'pickup_tomato'
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                print(player_obj, subtask)
                raise ValueError()

        if next_soup_finish > self.mdp.num_items_for_soup:
            next_soup_finish = self.mdp.num_items_for_soup

        p1_nxt_states = []
        for next_subtask in next_subtasks:
            p1_nxt_states.append([next_obj, next_subtask])

        nxt_world_info = [next_soup_finish]
        for order in orders:
            nxt_world_info.append(order)

        return p1_nxt_states, nxt_world_info

    def state_transition(self, player_obj, world_info, human_state=[None, None]):
        # game logic
        soup_finish = world_info[0]; orders = [] if len(world_info) < 2 else world_info[1:]
        other_obj = human_state[0]; subtask = human_state[1]
        actions = ''; next_obj = player_obj; next_soup_finish = soup_finish

        if player_obj == 'None':
            if (soup_finish == self.mdp.num_items_for_soup) and (other_obj != 'dish' and subtask != 'pickup_dish'):
                actions = 'pickup_dish'
                next_obj = 'dish'
            elif (soup_finish == (self.mdp.num_items_for_soup-1)) and (other_obj == 'onion' and subtask == 'drop_onion'):
                actions = 'pickup_dish'
                next_obj = 'dish'
            else:
                next_order = None
                if len(orders) > 1:
                    next_order = orders[1]

                if next_order == 'onion':
                    actions = 'pickup_onion'
                    next_obj = 'onion'

                elif next_order == 'tomato':
                    actions = 'pickup_tomato' 
                    next_obj = 'tomato'

                else:
                    actions = 'pickup_onion'
                    next_obj = 'onion'

        else:
            if player_obj == 'onion':
                actions = 'drop_onion'
                next_obj = 'None'
                next_soup_finish += 1

            elif player_obj == 'tomato':
                actions = 'drop_tomato'
                next_obj = 'None'
                next_soup_finish += 1

            elif (player_obj == 'dish') and (soup_finish >= self.mdp.num_items_for_soup-1):
                actions = 'pickup_soup'
                next_obj = 'soup'
                next_soup_finish = 0

            elif (player_obj == 'dish') and (soup_finish < self.mdp.num_items_for_soup-1):
                actions = 'drop_dish'
                next_obj = 'None'

            elif player_obj == 'soup':
                actions = 'deliver_soup'
                next_obj = 'None'
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                print(player_obj)
                raise ValueError()

        if next_soup_finish > self.mdp.num_items_for_soup:
            next_soup_finish = self.mdp.num_items_for_soup

        next_state_keys = next_obj + '_' + str(next_soup_finish)

        for order in orders:
            next_state_keys = next_state_keys + '_' + order

        for human_info in human_state:
            next_state_keys = next_state_keys + '_' + human_info

        return actions, next_state_keys

    def world_state_to_mdp_state_key(self, state, player, other_player, subtask):
        # a0 pos, a0 dir, a0 hold, a1 pos, a1 dir, a1 hold, len(order_list)
        player_obj = None; other_player_obj = None
        if player.holding is not None:
            player_obj = player.holding
        if other_player.holding is not None:
            other_player_obj = other_player.holding

        order_str = None if len(state.orders) == 0 else state.orders[0]
        for order in state.orders[1:]:
            order_str = order_str + '_' + str(order)

        # num_item_in_pot = 0
        # if state.objects is not None and len(state.objects) > 0:
        #     for obj_pos, obj_state in state.objects.items():
        #         if obj_state.name == 'soup' and obj_state.state[1] > num_item_in_pot:
        #             num_item_in_pot = obj_state.state[1]
        
        num_item_in_pot = state.in_pot

        state_strs = str(player_obj)+'_'+str(num_item_in_pot)+'_'+ order_str + '_' + str(other_player_obj) + '_' + subtask

        return state_strs

    def get_mdp_state_idx(self, mdp_state_key):
        if mdp_state_key not in self.state_dict:
            return None
        else:
            return self.state_idx_dict[mdp_state_key]

    def gen_state_dict_key(self, p0_obj, p1_obj, num_item_in_pot, orders, subtasks):
        # a0 hold, a1 hold, 

        player_obj = p0_obj if p0_obj is not None else 'None'
        other_player_obj = p1_obj if p1_obj is not None else 'None'

        order_str = None if orders is None else orders
        for order in orders:
            order_str = order_str + '_' + str(order)

        state_strs = []
        for subtask in subtasks:
            state_strs.appedn(str(player_obj)+'_'+str(soup_finish)+'_'+ order_str + '_' + str(other_player_obj) + '_' + subtask)

        return state_strs

    def get_key_from_value(self, dictionary, state_value):
        try: 
            idx = list(dictionary.values()).index(state_value)
        except ValueError:
            return None
        else:
            return list(dictionary.keys())[idx]

    # def map_action_to_location(self, world_state, human_state, action, obj, p0_obj=None, player_idx=None, counter_drop=True, state_dict=None):
    #     """
    #     Get the next location the agent will be in based on current world state, medium level actions, after-action state obj.
    #     """
    #     state_str = human_state.hl_state
    #     state_dict = self.state_dict if state_dict is None else state_dict
    #     p0_obj = p0_obj if p0_obj is not None else state_dict[human_state.hl_state][0]
    #     other_obj = human_state.holding if human_state.holding is not None else 'None'
    #     pots_states_dict = self.mdp.get_pot_states(world_state)
    #     location = []
    #     WAIT = False # If wait becomes true, one player has to wait for the other player to finish its current task and its next task

    #     if action == 'pickup' and obj != 'soup':
    #         if p0_obj != 'None' and counter_drop:
    #             location = self.drop_item(world_state)
    #         else:
    #             if obj == 'onion':
    #                 location = self.mdp.get_onion_dispenser_locations()
    #             elif obj == 'tomato':
    #                 location = self.mdp.get_tomato_dispenser_locations()
    #             elif obj == 'dish':
    #                 location = self.mdp.get_dish_dispenser_locations()
    #             else:
    #                 print(p0_obj, action, obj)
    #                 ValueError()
    #     elif action == 'pickup' and obj == 'soup':
    #         if p0_obj != 'dish' and p0_obj != 'None' and counter_drop:
    #             location = self.drop_item(world_state)
    #         elif p0_obj == 'None':
    #             location = self.mdp.get_dish_dispenser_locations()
    #         else:
    #             if state_str is not None:
    #                 num_item_in_pot = world_state.in_pot
    #                 if num_item_in_pot == 0:
    #                     location = self.mdp.get_empty_pots(pots_states_dict)
    #                     if len(location) > 0: return location, True
    #                 elif num_item_in_pot > 0 and num_item_in_pot < self.mdp.num_items_for_soup:
    #                     location = self.mdp.get_partially_full_pots(pots_states_dict)
    #                     if len(location) > 0: return location, True
    #                 else:
    #                     location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
    #                 if len(location) > 0: return location, WAIT

    #             location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
    #             if len(location) == 0:
    #                 WAIT = True
    #                 # location = self.ml_action_manager.go_to_closest_feature_or_counter_to_goal(location)
    #                 location = self.mdp.get_partially_full_pots(pots_states_dict) + self.mdp.get_empty_pots(pots_states_dict)
    #                 # location = world_state.players[player_idx].pos_and_or
    #                 return location, WAIT

    #     elif action == 'drop':
    #         if obj == 'onion' or obj == 'tomato':

    #             if state_str is not None:
    #                 num_item_in_pot = world_state.in_pot
    #                 if num_item_in_pot == 0:
    #                     location = self.mdp.get_empty_pots(pots_states_dict)
    #                 elif num_item_in_pot > 0 and num_item_in_pot < self.mdp.num_items_for_soup:
    #                     location = self.mdp.get_partially_full_pots(pots_states_dict)
    #                 else:
    #                     location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
                        
    #                 if len(location) > 0: return location, WAIT

    #             location = self.mdp.get_partially_full_pots(pots_states_dict) + self.mdp.get_empty_pots(pots_states_dict)
                
    #             if len(location) == 0:
    #                 if other_obj != 'onion' and other_obj != 'tomato':
    #                     WAIT = True
    #                     location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
    #                     # location = world_state.players[player_idx].pos_and_or
    #                     return location, WAIT
    #                 elif counter_drop:
    #                     location = self.drop_item(world_state)
    #                 else:
    #                     WAIT = True
    #                     location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
    #                     # location = world_state.players[player_idx].pos_and_or
    #                     return location, WAIT

    #         elif obj == 'dish' and player_idx==0 and counter_drop: # agent_index = 0
    #             location = self.drop_item(world_state)
    #         else:
    #             print(p0_obj, action, obj)
    #             ValueError()

    #     elif action == 'deliver':
    #         if p0_obj != 'soup' and p0_obj != 'None' and counter_drop:
    #             location = self.mdp.get_empty_counter_locations(world_state)
    #         elif p0_obj != 'soup':
    #             if state_str is not None:
    #                 num_item_in_pot = world_state.in_pot
    #                 if num_item_in_pot == 0:
    #                     location = self.mdp.get_empty_pots(pots_states_dict)
    #                     if len(location) > 0: return location, True
    #                 elif num_item_in_pot > 0 and num_item_in_pot < self.mdp.num_items_for_soup:
    #                     location = self.mdp.get_partially_full_pots(pots_states_dict)
    #                     if len(location) > 0: return location, True
    #                 else:
    #                     location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
    #                 if len(location) > 0: return location, WAIT

    #             location = self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
    #             if len(location) == 0:
    #                 WAIT = True
    #                 # location = self.ml_action_manager.go_to_closest_feature_or_counter_to_goal(location)
    #                 location = self.mdp.get_partially_full_pots(pots_states_dict) + self.mdp.get_empty_pots(pots_states_dict)
    #                 # location = world_state.players[player_idx].pos_and_or
    #                 return location, WAIT
    #         else:
    #             location = self.mdp.get_serving_locations()

    #     else:
    #         print(p0_obj, action, obj)
    #         ValueError()

    #     return location, WAIT

    def _shift_same_goal_pos(self, new_positions, change_idx):
        
        pos = new_positions[change_idx][0]
        ori = new_positions[change_idx][1]
        new_pos = pos; new_ori = ori
        if self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[0])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[0])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[1])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[1])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[2])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[2])
        elif self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[3])[0] != pos:
            new_pos, new_ori = self.mdp._move_if_direction(pos, ori, Action.ALL_ACTIONS[3])
        else:
            print('pos = ', pos)
            ValueError()
        
        new_positions[change_idx] = (new_pos, new_ori)

        return new_positions[0], new_positions[1]

    def mdp_action_state_to_world_state(self, action_idx, ori_state_idx, ori_world_state, with_argmin=False):
        new_world_state = copy.deepcopy(ori_world_state)
        ori_mdp_state_key = self.get_key_from_value(self.state_idx_dict, ori_state_idx)
        mdp_state_obj = self.state_dict[ori_mdp_state_key]
        action = self.get_key_from_value(self.action_idx_dict, action_idx)

        possible_agent_motion_goals = self.map_action_to_location(ori_world_state, ori_mdp_state_key, self.action_dict[action], p0_obj=mdp_state_obj[0])# [0], self.action_dict[action][1])# , player_idx=0) 
        possible_human_motion_goals = self.map_action_to_location(ori_world_state, ori_mdp_state_key, self.action_dict[mdp_state_obj[-1]], p0_obj=mdp_state_obj[-2]) #, player_idx=1) # get next world state from human subtask info (aka. mdp action translate into medium level goal position)

        # get next position for AI agent
        # agent_cost, agent_feature_pos = self.mp.min_cost_to_feature(ori_world_state.players[0].pos_and_or, possible_agent_motion_goals, with_motion_goal=True) # select the feature position that is closest to current player's position in world state
        agent_cost, agent_feature_pos = self.mlp.min_cost_to_feature(ori_world_state.players[0].ml_state, possible_agent_motion_goals)
        new_agent_pos = agent_feature_pos if agent_feature_pos is not None else new_world_state.players[0].get_pos_and_or()
        human_cost, human_feature_pos = self.mlp.min_cost_to_feature(ori_world_state.players[1].ml_state, possible_human_motion_goals)
        new_human_pos = human_feature_pos if human_feature_pos is not None else new_world_state.players[1].get_pos_and_or()
        # print(new_agent_pos, new_human_pos)

        # TODO: update this
        # if new_agent_pos == new_human_pos:
        #    new_agent_pos, new_human_pos = self._shift_same_goal_pos([new_agent_pos, new_human_pos], np.argmax(np.array([agent_cost, human_cost])))
            # print('after shift =', new_agent_pos, new_human_pos)

        # update next position for AI agent
        if new_world_state.players[0].has_object():
            new_world_state.players[0].holding = 'None'
        if mdp_state_obj[0] != 'None' and mdp_state_obj[0] != 'soup':
            new_world_state.players[0].holding = mdp_state_obj[0] # ObjectState(mdp_state_obj[0], new_agent_pos)
        new_world_state.players[0].ml_state = (new_agent_pos[0], new_agent_pos[1], new_world_state.players[0].ml_state[2])

        # update next position for human
        if new_world_state.players[1].has_object():
            new_world_state.players[1].holding = 'None'
        if mdp_state_obj[-2] != 'None' and mdp_state_obj[-2] != 'soup':
            new_world_state.players[1].holding =  mdp_state_obj[-2]# ObjectState(mdp_state_obj[-2], new_human_pos)
        new_world_state.players[1].ml_state = (new_human_pos[0], new_human_pos[1], new_world_state.players[1].ml_state[2])

        total_cost = max([agent_cost, human_cost]) # in rss paper is max
        # f AI_WAIT or HUMAN_WAIT: # if wait, then cost is sum of current tasks cost and one player's next task cost (est. as half map area length)
        #     total_cost = agent_cost + human_cost + ((self.mdp.width-1)+(self.mdp.height-1))/2

        if with_argmin:
            return new_world_state, total_cost, [new_agent_pos, new_human_pos]

        return new_world_state, total_cost

    def world_to_state_keys(self, world_state, player, other_player, belief):
        mdp_state_keys = []
        for i, b in enumerate(belief):
            mdp_state_key = self.world_state_to_mdp_state_key(world_state, player, other_player, self.get_key_from_value(self.subtask_idx_dict, i))
            if self.get_mdp_state_idx(mdp_state_key) is not None:
                mdp_state_keys.append(self.world_state_to_mdp_state_key(world_state, player, other_player, self.get_key_from_value(self.subtask_idx_dict, i)))
        return mdp_state_keys

    def joint_action_cost(self, world_state, goal_pos_and_or, COST_OF_STAY=1):
        joint_action_plan, end_motion_state, plan_costs = self.jmp.get_low_level_action_plan(world_state.players_pos_and_or, goal_pos_and_or, merge_one=True)
        # joint_action_plan, end_state, plan_costs = self.mlp.get_embedded_low_level_action_plan(world_state, goal_pos_and_or, other_agent, other_agent_idx)
        # print('joint_action_plan =', joint_action_plan, '; plan_costs =', plan_costs)

        if len(joint_action_plan) == 0:
            return (Action.INTERACT, None), 0

        num_of_non_stay_actions = len([a for a in joint_action_plan if a[0] != Action.STAY])
        num_of_stay_actions = len([a for a in joint_action_plan if a[0] == Action.STAY])

        return joint_action_plan[0], max(plan_costs)# num_of_non_stay_actions+num_of_stay_actions*COST_OF_STAY # in rss paper is max(plan_costs)

    def init_cost(self):
        # self.cost_matrix = 
        # curr_state_idx = start_state_idx
        # is_end_state = False

        # human_location = world_state.players[1].ml_state
        # human_state = world_state.players[1]
        # sum = 0

        self.cost_matrix = np.zeros((len(self.action_dict), len(self.state_dict), len(self.state_dict)))

        for action,action_idx in self.action_idx_dict.items():
            for curr_state, curr_state_idx in self.state_idx_dict.items():
                next_states = np.where(self.transition_matrix[action_idx,curr_state_idx] > 0.00001)[0]

                
                for next_state_idx in next_states:
                    
                    human_action_key = self.get_key_from_value(self.action_idx_dict,action_idx)
                    # world_state = WorldState(curr_state)
                    # agent_state = AgentState()
                    # agent_state.parse_hl_state(curr_state, world_state)
                    # next_locations = self.map_action_to_location(world_state, agent_state, self.action_dict[human_action_key])

                    
                    # min_plan, min_location = self.mlp.min_cost_to_feature(, next_locations)

                    self.cost_matrix[action_idx][curr_state_idx][next_state_idx] = 5

                


    def step(self, world_state, mdp_state_keys, belief, agent_idx, low_level_action=False):
        """
        Compute plan cost that starts from the next qmdp state defined as next_state_v().
        Compute the action cost of excuting a step towards the next qmdp state based on the
        current low level state information.

        next_state_v: shape(len(belief), len(action_idx)). If the low_level_action is True, 
            the action_dix will be representing the 6 low level action index (north, south...).
            If the low_level_action is False, it will be the action_dict (pickup_onion, pickup_soup...).
        """
        start_time = time.time()
        next_state_v = np.zeros((len(belief), len(self.action_dict)), dtype=float)
        action_cost = np.zeros((len(belief), len(self.action_dict)), dtype=float)
        qmdp_q = np.zeros((len(self.action_dict), len(belief)), dtype=float)
        # ml_action_to_low_action = np.zeros()

        # for each subtask, obtain next mdp state but with low level location based on finishing excuting current action and subtask
        nxt_possible_mdp_state = []
        nxt_possible_world_state = []
        ml_action_to_low_action = []
        for i, mdp_state_key in enumerate(mdp_state_keys):
            mdp_state_idx = self.get_mdp_state_idx(mdp_state_key)
            if mdp_state_idx is not None:
                agent_action_idx_arr, next_mdp_state_idx_arr = np.where(self.transition_matrix[:, mdp_state_idx] > 0.000001) # returns array(action idx), array(next_state_idx)
                nxt_possible_mdp_state.append([agent_action_idx_arr, next_mdp_state_idx_arr])
                for j, action_idx in enumerate(agent_action_idx_arr): # action_idx is encoded subtask action
                    # print('action_idx =', action_idx)
                    next_state_idx = next_mdp_state_idx_arr[j]
                    after_action_world_state, cost, goals_pos = self.mdp_action_state_to_world_state(action_idx, mdp_state_idx, world_state, with_argmin=True)

                    # calculate value cost from astar rollout of policy
                    # value_cost = self.compute_V(after_action_world_state, self.get_key_from_value(self.state_idx_dict, next_state_idx), search_depth=100)
                    # TODO: index 133 or 
                    value_cost = self.dist_value_matrix[next_state_idx]
                    # value_cost = self.compute_policy_rollout(next_state_idx, world_state)

                    one_step_cost = cost# joint_action, one_step_cost = self.joint_action_cost(world_state, after_action_world_state.players_pos_and_or)  
                    # print('joint_action =', joint_action, 'one_step_cost =', one_step_cost)
                    # print('Action.ACTION_TO_INDEX[joint_action[agent_idx]] =', Action.ACTION_TO_INDEX[joint_action[agent_idx]])
                    if not low_level_action:
                        # action_idx: are subtask action dictionary index
                        next_state_v[i, action_idx] += (value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx])
                        # print(next_state_v[i, action_idx])

                        # normalized_cost = 

                        ## compute one step cost with joint motion considered
                        action_cost[i, action_idx] -= (one_step_cost)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]
                        # action_cost[i, action_idx] -= (1)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]
                    else:
                        next_state_v[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] += (value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx])
                        # print(next_state_v[i, action_idx])

                        ## compute one step cost with joint motion considered
                        action_cost[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] -= (one_step_cost)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]
                    # print('action_idx =', self.get_key_from_value(self.action_idx_dict, action_idx), '; mdp_state_key =', mdp_state_key, '; next_state_key =', self.get_key_from_value(self.state_idx_dict, next_state_idx))
                    # print('next_state_v =', next_state_v[i])
        # print('action_cost =', action_cost)

        q = self.compute_Q(belief, next_state_v, action_cost)
        # print(q)
        action_idx = self.get_best_action(q)
        # print('get_best_action =', action_idx, '=', self.get_key_from_value(self.action_idx_dict, action_idx))
        # print("It took {} seconds for this step".format(time.time() - start_time))
        if low_level_action:
            return Action.INDEX_TO_ACTION[action_idx], None, low_level_action
            
        return action_idx, self.action_dict[self.get_key_from_value(self.action_idx_dict, action_idx)], low_level_action

    def belief_update(self, world_state, human_state, robot_state, belief_vector, prev_dist_to_feature, greedy=False):
        """
        Update belief based on both human player's game logic and also it's current position and action.
        Belief shape is an array with size equal the length of subtask_dict.
        """
        start_time = time.time()

        # what is distance to object
        # closer to onion station, higher belief in subtask pickup onion
        # based on game logic and distance
        # game logic is uniform over number of possible subtasks

        distance_trans_belief = np.zeros((len(belief_vector), len(belief_vector)), dtype=float)
        human_pos_and_or = human_state.ml_state
        agent_pos_and_or = robot_state.ml_state

        subtask_key = np.array([self.get_key_from_value(self.subtask_idx_dict, i) for i in range(len(belief_vector))])

        # get next position for human
        human_obj = human_state.holding if human_state.holding is not None else 'None'
        game_logic_prob = np.zeros((len(belief_vector)), dtype=float)
        dist_belief_prob = np.zeros((len(belief_vector)), dtype=float)
        for i, belief in enumerate(belief_vector):
            ## estimating next subtask based on game logic
            game_logic_prob[i] = self._is_valid_object_subtask_pair(human_obj, subtask_key[i], world_state.in_pot, greedy=greedy)*1.0

            if game_logic_prob[i] < 0.00001:
                continue
    
            ## tune subtask estimation based on current human's position and action (use minimum distance between features)
            possible_motion_goals = self.map_action_to_location(world_state, human_state, self.subtask_dict[subtask_key[i]], human_state.holding)
            # get next world state from human subtask info (aka. mdp action translate into medium level goal position)
            human_dist_cost, feature_pos = self.mlp.min_cost_to_feature(human_pos_and_or, possible_motion_goals) # select the feature position that is closest to current player's position in world state
            
            if str(feature_pos) not in prev_dist_to_feature:
                prev_dist_to_feature[str(feature_pos)] = human_dist_cost

            dist_belief_prob[i] = (self.mdp.height+self.mdp.width) + (prev_dist_to_feature[str(feature_pos)] - human_dist_cost)
            # dist_belief_prob[i] = (self.mdp.height+self.mdp.width) - human_dist_cost if human_dist_cost < np.inf else (self.mdp.height + self.mdp.width)

            # update distance to feature
            prev_dist_to_feature[str(feature_pos)] = human_dist_cost

        # print('dist_belief_prob =', dist_belief_prob)
        # print('prev_dist_to_feature =', prev_dist_to_feature)
        # print('human_dist_cost =', human_dist_cost)

        game_logic_prob /= game_logic_prob.sum()
        dist_belief_prob /= dist_belief_prob.sum()

        game_logic_prob[game_logic_prob == 0.0] = 0.000001
        dist_belief_prob[dist_belief_prob== 0.0] = 0.000001

        new_belief = belief*game_logic_prob
        new_belief = new_belief*0.7 * dist_belief_prob*0.3

        new_belief /= new_belief.sum()
        # print("It took {} seconds for belief update".format(time.time() - start_time))

        return new_belief, prev_dist_to_feature

    def parse_state(self, state_string):
        state_arr = state_string.split('_')
        h_object = state_arr.pop()
        h_action = state_arr.pop()
        h_holding = state_arr.pop()

        orders = []
        order = state_arr.pop()
        while not order.isdigit():
            orders.append(order)
            order = state_arr.pop()

        in_pot = int(order)
        r_holding = state_arr.pop()
        return (r_holding, in_pot, orders, h_holding, h_action, h_object)


    def post_mdp_setup(self):
        # computes an estimated distance cost for each state,action
        # this is later indexed in qmdp for evaluating future cost

        # copy the value matrix, rollout the policy, sum up distances
        self.dist_value_matrix = np.ones(self.value_matrix.shape)*1000000

        for start_state_idx,v in enumerate(self.dist_value_matrix):
            curr_state_idx = start_state_idx
            future_dist_cost = 0
            policy_a = self.policy_matrix[curr_state_idx]
            orders_left = len(self.parse_state(self.get_key_from_value(self.state_idx_dict, curr_state_idx))[2])

            if orders_left == 0:
                future_dist_cost = 1000000
                self.dist_value_matrix[curr_state_idx] = future_dist_cost
                continue
            
            while not self.reward_matrix[policy_a][curr_state_idx] > 0:
                if orders_left == 0:
                    break
                policy_a = self.policy_matrix[curr_state_idx]
                action = self.get_key_from_value(self.action_idx_dict, policy_a)
                state = self.get_key_from_value(self.state_idx_dict, curr_state_idx)
                # print(state)
                action_cost = self.action_to_features(action)
                possible_next_states = np.where(self.transition_matrix[policy_a, curr_state_idx] > 0.000001)[0]
                next_state = random.choice(possible_next_states) # choose first state TODO: review that we don't have more information
                future_dist_cost += action_cost
                curr_state_idx = next_state
                orders_left = len(self.parse_state(self.get_key_from_value(self.state_idx_dict, curr_state_idx))[2])

                
            print(f'future cost {self.get_key_from_value(self.state_idx_dict,curr_state_idx)}: {future_dist_cost}')
            self.dist_value_matrix[start_state_idx] = future_dist_cost

            # curr_state = self.get_key_from_value(self.state_idx_dict, i)
            # parsed_state = self.parse_state(curr_state)
        
        return

    def action_to_features(self, action):
        if action == 'drop_onion':
            # fridge to stove
            feat_key = 'F_S'
        elif action == 'deliver_soup':
            # stove to table
            feat_key = 'S_T'
        elif action == 'pickup_dish':
            # stove, table to dish station
            # go from center
            feat_key = 'S_C'
        elif action == 'drop_soup':
            feat_key = 'Drop'
        elif action == 'drop_dish':
            feat_key = 'Drop'
        elif action == 'pickup_soup':
            # dish to stove
            feat_key = 'B_S'
        elif action == 'pickup_onion':
            # either coming from stove or table
            # use center point
            feat_key = 'F_C'
        
        dist = self.mlp.dist_between[feat_key]
        return dist

            
    def compute_V(self, next_world_state, mdp_state_key, search_depth=100):
        next_world_state_str = str(next_world_state)
        if next_world_state_str not in self.world_state_cost_dict:

            delivery_horizon=2
            debug=False
            h_fn=Heuristic(self.mp).simple_heuristic
            start_world_state = next_world_state.deepcopy()
            if start_world_state.order_list is None:
                start_world_state.order_list = ["any"] * delivery_horizon
            else:
                start_world_state.order_list = start_world_state.order_list[:delivery_horizon]
            
            expand_fn = lambda state, ori_state_key: self.get_successor_states(state, ori_state_key)
            goal_fn = lambda ori_state_key: len(self.state_dict[ori_state_key][2:]) <= 2
            heuristic_fn = lambda state: h_fn(state)

            # TODO: lookup from regular mdp precomputed value iteration
            # cost = self.value_matrix[get_index(start_world_state)] skip next two lines
            search_problem = SearchTree(start_world_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
            path_end_state, cost, over_limit = search_problem.bounded_A_star_graph_search(qmdp_root=mdp_state_key, info=False, cost_limit=search_depth)

            if over_limit:
                cost = self.optimal_plan_cost(path_end_state, cost)

            self.world_state_cost_dict[next_world_state_str] = cost

        # print('self.world_state_cost_dict length =', len(self.world_state_cost_dict))            
        return (self.mdp.height*self.mdp.width)*2 - self.world_state_cost_dict[next_world_state_str]

    def optimal_plan_cost(self, start_world_state, start_cost):
        self.world_state_to_mdp_state_key(start_world_state, start_world_state.players[0], start_world_state.players[1], subtask)

    def compute_Q(self, b, v, c):
        # print('b =', b)
        # print('v =', v)
        # print('c =', c)

        # tmp=input()
        return b@(v+c)

    def get_best_action(self, q):
        return np.argmax(q)

    def init_mdp(self):
        self.init_actions()
        self.init_human_aware_states(order_list=self.mdp.start_order_list)
        self.init_transition()
        self.init_cost()
        self.init_reward()

    def compute_mdp(self, filename):
        start_time = time.time()

        # final_filepath = os.path.join(PLANNERS_DIR, filename)
        self.init_mdp()
        self.num_states = len(self.state_dict)
        self.num_actions = len(self.action_dict)
        # print('Total states =', self.num_states, '; Total actions =', self.num_actions)

        # TODO: index value iteration in step function to get action
        
        self.value_iteration()

        # print("It took {} seconds to create HumanSubtaskQMDPPlanner".format(time.time() - start_time))
        # self.save_to_file(final_filepath)
        # tmp = input()
        # self.save_to_file(output_mdp_path)
        return

    

    

class Heuristic(object):

    def __init__(self, mp):
        self.motion_planner = mp
        self.mdp = mp.mdp
        self.heuristic_cost_dict = self._calculate_heuristic_costs()
    
    def hard_heuristic(self, state, goal_deliveries, time=0, debug=False):
        # NOTE: does not support tomatoes – currently deprecated as harder heuristic
        # does not seem worth the additional computational time

        """
        From a state, we can calculate exactly how many:
        - soup deliveries we need
        - dishes to pots we need
        - onion to pots we need

        We then determine if there are any soups/dishes/onions
        in transit (on counters or on players) than can be 
        brought to their destinations faster than starting off from
        a dispenser of the same type. If so, we consider fulfilling
        all demand from these positions. 

        After all in-transit objects are considered, we consider the
        costs required to fulfill all the rest of the demand, that is 
        given by:
        - pot-delivery trips
        - dish-pot trips
        - onion-pot trips
        
        The total cost is obtained by determining an optimistic time 
        cost for each of these trip types
        """
        forward_cost = 0

        # Obtaining useful quantities
        objects_dict = state.unowned_objects_by_type
        player_objects = state.player_objects_by_type
        pot_states_dict = self.mdp.get_pot_states(state)
        min_pot_delivery_cost = self.heuristic_cost_dict['pot-delivery']
        min_dish_to_pot_cost = self.heuristic_cost_dict['dish-pot']
        min_onion_to_pot_cost = self.heuristic_cost_dict['onion-pot']

        pot_locations = self.mdp.get_pot_locations()
        full_soups_in_pots = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking'] \
                             + pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
        partially_full_soups = pot_states_dict['onion']['partially_full'] + pot_states_dict['tomato']['partially_full']
        num_onions_in_partially_full_pots = sum([state.get_object(loc).state[1] for loc in partially_full_soups])

        # Calculating costs
        num_deliveries_to_go = goal_deliveries - state.num_delivered

        # SOUP COSTS
        total_num_soups_needed = max([0, num_deliveries_to_go])
        
        soups_on_counters = [soup_obj for soup_obj in objects_dict['soup'] if soup_obj.position not in pot_locations]
        soups_in_transit = player_objects['soup'] + soups_on_counters
        soup_delivery_locations = self.mdp.get_serving_locations()
        
        num_soups_better_than_pot, total_better_than_pot_soup_cost = \
            self.get_costs_better_than_dispenser(soups_in_transit, soup_delivery_locations, min_pot_delivery_cost, total_num_soups_needed, state)
        
        min_pot_to_delivery_trips = max([0, total_num_soups_needed - num_soups_better_than_pot])
        pot_to_delivery_costs = min_pot_delivery_cost * min_pot_to_delivery_trips

        forward_cost += total_better_than_pot_soup_cost
        forward_cost += pot_to_delivery_costs

        # DISH COSTS
        total_num_dishes_needed = max([0, min_pot_to_delivery_trips])
        dishes_on_counters = objects_dict['dish']
        dishes_in_transit = player_objects['dish'] + dishes_on_counters
        
        num_dishes_better_than_disp, total_better_than_disp_dish_cost = \
            self.get_costs_better_than_dispenser(dishes_in_transit, pot_locations, min_dish_to_pot_cost, total_num_dishes_needed, state)

        min_dish_to_pot_trips = max([0, min_pot_to_delivery_trips - num_dishes_better_than_disp])
        dish_to_pot_costs = min_dish_to_pot_cost * min_dish_to_pot_trips

        forward_cost += total_better_than_disp_dish_cost
        forward_cost += dish_to_pot_costs

        # ONION COSTS
        num_pots_to_be_filled = min_pot_to_delivery_trips - len(full_soups_in_pots)
        total_num_onions_needed = num_pots_to_be_filled * 3 - num_onions_in_partially_full_pots
        onions_on_counters = objects_dict['onion']
        onions_in_transit = player_objects['onion'] + onions_on_counters

        num_onions_better_than_disp, total_better_than_disp_onion_cost = \
            self.get_costs_better_than_dispenser(onions_in_transit, pot_locations, min_onion_to_pot_cost, total_num_onions_needed, state)

        min_onion_to_pot_trips = max([0, total_num_onions_needed - num_onions_better_than_disp])
        onion_to_pot_costs = min_onion_to_pot_cost * min_onion_to_pot_trips
        
        forward_cost += total_better_than_disp_onion_cost
        forward_cost += onion_to_pot_costs

        # Going to closest feature costs
        # NOTE: as implemented makes heuristic inconsistent
        # for player in state.players:
        #     if not player.has_object():
        #         counter_objects = soups_on_counters + dishes_on_counters + onions_on_counters
        #         possible_features = counter_objects + pot_locations + self.mdp.get_dish_dispenser_locations() + self.mdp.get_onion_dispenser_locations()
        #         forward_cost += self.action_manager.min_cost_to_feature(player.pos_and_or, possible_features)

        heuristic_cost = forward_cost / 2
        
        if debug:
            env = OvercookedEnv.from_mdp(self.mdp)
            env.state = state
            # print("\n" + "#"*35)
            print("Current state: (ml timestep {})\n".format(time))

            print("# in transit: \t\t Soups {} \t Dishes {} \t Onions {}".format(
                len(soups_in_transit), len(dishes_in_transit), len(onions_in_transit)
            ))

            # NOTE Possible improvement: consider cost of dish delivery too when considering if a
            # transit soup is better than dispenser equivalent
            print("# better than disp: \t Soups {} \t Dishes {} \t Onions {}".format(
                num_soups_better_than_pot, num_dishes_better_than_disp, num_onions_better_than_disp
            ))

            print("# of trips: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
                min_pot_to_delivery_trips, min_dish_to_pot_trips, min_onion_to_pot_trips
            ))

            print("Trip costs: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
                pot_to_delivery_costs, dish_to_pot_costs, onion_to_pot_costs
            ))

            print(str(env) + "HEURISTIC: {}".format(heuristic_cost))

        return heuristic_cost

    def get_costs_better_than_dispenser(self, possible_objects, target_locations, baseline_cost, num_needed, state):
        """
        Computes the number of objects whose minimum cost to any of the target locations is smaller than
        the baseline cost (clipping it if greater than the number needed). It also calculates a lower
        bound on the cost of using such objects.
        """
        costs_from_transit_locations = []
        for obj in possible_objects:
            obj_pos = obj.position
            if obj_pos in state.player_positions:
                # If object is being carried by a player
                player = [p for p in state.players if p.position == obj_pos][0]
                # NOTE: not sure if this -1 is justified.
                # Made things work better in practice for greedy heuristic based agents.
                # For now this function is just used from there. Consider removing later if
                # greedy heuristic agents end up not being used.
                min_cost = self.motion_planner.min_cost_to_feature(player.pos_and_or, target_locations) - 1
            else:
                # If object is on a counter
                min_cost = self.motion_planner.min_cost_between_features([obj_pos], target_locations)
            costs_from_transit_locations.append(min_cost)
        
        costs_better_than_dispenser = [cost for cost in costs_from_transit_locations if cost <= baseline_cost]
        better_than_dispenser_total_cost = sum(np.sort(costs_better_than_dispenser)[:num_needed])
        return len(costs_better_than_dispenser), better_than_dispenser_total_cost

    def _calculate_heuristic_costs(self, debug=False):
        """Pre-computes the costs between common trip types for this mdp"""
        pot_locations = self.mdp.get_pot_locations()
        delivery_locations = self.mdp.get_serving_locations()
        dish_locations = self.mdp.get_dish_dispenser_locations()
        onion_locations = self.mdp.get_onion_dispenser_locations()
        tomato_locations = self.mdp.get_tomato_dispenser_locations()

        heuristic_cost_dict = {
            'pot-delivery': self.motion_planner.min_cost_between_features(pot_locations, delivery_locations, manhattan_if_fail=True),
            'dish-pot': self.motion_planner.min_cost_between_features(dish_locations, pot_locations, manhattan_if_fail=True)
        }

        onion_pot_cost = self.motion_planner.min_cost_between_features(onion_locations, pot_locations, manhattan_if_fail=True)
        tomato_pot_cost = self.motion_planner.min_cost_between_features(tomato_locations, pot_locations, manhattan_if_fail=True)

        if debug: print("Heuristic cost dict", heuristic_cost_dict)
        assert onion_pot_cost != np.inf or tomato_pot_cost != np.inf
        if onion_pot_cost != np.inf:
            heuristic_cost_dict['onion-pot'] = onion_pot_cost
        if tomato_pot_cost != np.inf:
            heuristic_cost_dict['tomato-pot'] = tomato_pot_cost
        
        return heuristic_cost_dict
    
    def simple_heuristic(self, state, time=0, debug=False):
        """Simpler heuristic that tends to run faster than current one"""
        # NOTE: State should be modified to have an order list w.r.t. which
        # one can calculate the heuristic
        assert state.order_list is not None
        
        objects_dict = state.unowned_objects_by_type
        player_objects = state.player_objects_by_type
        pot_states_dict = self.mdp.get_pot_states(state)
        num_deliveries_to_go = state.num_orders_remaining
        
        full_soups_in_pots = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking'] \
                             + pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
        partially_full_onion_soups = pot_states_dict['onion']['partially_full']
        partially_full_tomato_soups = pot_states_dict['tomato']['partially_full']
        num_onions_in_partially_full_pots = sum([state.get_object(loc).state[1] for loc in partially_full_onion_soups])
        num_tomatoes_in_partially_full_pots = sum([state.get_object(loc).state[1] for loc in partially_full_tomato_soups])

        soups_in_transit = player_objects['soup']
        dishes_in_transit = objects_dict['dish'] + player_objects['dish']
        onions_in_transit = objects_dict['onion'] + player_objects['onion']
        tomatoes_in_transit = objects_dict['tomato'] + player_objects['tomato']

        num_pot_to_delivery = max([0, num_deliveries_to_go - len(soups_in_transit)])
        num_dish_to_pot = max([0, num_pot_to_delivery - len(dishes_in_transit)])

        num_pots_to_be_filled = num_pot_to_delivery - len(full_soups_in_pots)
        num_onions_needed_for_pots = num_pots_to_be_filled * 3 - len(onions_in_transit) - num_onions_in_partially_full_pots
        num_tomatoes_needed_for_pots = num_pots_to_be_filled * 3 - len(tomatoes_in_transit) - num_tomatoes_in_partially_full_pots
        num_onion_to_pot = max([0, num_onions_needed_for_pots])
        num_tomato_to_pot = max([0, num_tomatoes_needed_for_pots])

        pot_to_delivery_costs = self.heuristic_cost_dict['pot-delivery'] * num_pot_to_delivery
        dish_to_pot_costs = self.heuristic_cost_dict['dish-pot'] * num_dish_to_pot

        items_to_pot_costs = []
        if 'onion-pot' in self.heuristic_cost_dict.keys():
            onion_to_pot_costs = self.heuristic_cost_dict['onion-pot'] * num_onion_to_pot
            items_to_pot_costs.append(onion_to_pot_costs)
        if 'tomato-pot' in self.heuristic_cost_dict.keys():
            tomato_to_pot_costs = self.heuristic_cost_dict['tomato-pot'] * num_tomato_to_pot
            items_to_pot_costs.append(tomato_to_pot_costs)

        # NOTE: doesn't take into account that a combination of the two might actually be more advantageous.
        # Might cause heuristic to be inadmissable in some edge cases.
        items_to_pot_cost = min(items_to_pot_costs)

        heuristic_cost = (pot_to_delivery_costs + dish_to_pot_costs + items_to_pot_cost) / 2

        if debug:
            env = OvercookedEnv.from_mdp(self.mdp)
            env.state = state
            print("\n" + "#" * 35)
            print("Current state: (ml timestep {})\n".format(time))

            print("# in transit: \t\t Soups {} \t Dishes {} \t Onions {}".format(
                len(soups_in_transit), len(dishes_in_transit), len(onions_in_transit)
            ))

            print("Trip costs: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
                pot_to_delivery_costs, dish_to_pot_costs, onion_to_pot_costs
            ))

            print(str(env) + "HEURISTIC: {}".format(heuristic_cost))

        return heuristic_cost
