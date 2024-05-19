import copy
import os
import pickle
import random
import time

import numpy as np

from lsi_3d.planners.mid_level_motion import AStarMotionPlanner
from lsi_3d.planners.steak_mdp_planner import SteakMediumLevelMDPPlanner

PLANNERS_DIR = 'C:\\Users\\icaro\\3d-plan-eval'


class SteakHumanSubtaskQMDPPlanner(SteakMediumLevelMDPPlanner):
    def __init__(self, mdp, mlp: AStarMotionPlanner, vision_limited_human=None):
        super().__init__(mdp, mlp)

        self.world_state_cost_dict = {}
        self.subtask_dict = {}
        self.subtask_idx_dict = {}
        self.sim_human_model = vision_limited_human

    @staticmethod
    def from_qmdp_planner_file(filename):
        with open(os.path.join(PLANNERS_DIR, filename), 'rb') as f:
            return pickle.load(f)

    def init_human_aware_states(self, state_idx_dict=None, order_list=None):
        """
        States: agent 0 holding object, number of item in pan, chopping time, washing time, order list, agent 1 (usually human) holding object, agent 1 subtask.
        """

        # set state dict as [p0_obj, num_item_in_pot, order_list]
        self.init_states(order_list=order_list)

        # add [p1_obj, subtask] to [p0_obj, num_item_in_pot, order_list]
        self.subtask_dict = copy.deepcopy(self.action_dict)
        original_state_dict = copy.deepcopy(self.state_dict)
        self.state_dict.clear()
        self.state_idx_dict.clear()

        for j, subtask in enumerate(self.subtask_dict.items()):
            self.subtask_idx_dict[subtask[0]] = j
            for ori_key, ori_value in original_state_dict.items():
                ori_state_info = [ori_key.split('_')[0]]
                for i, k in enumerate(ori_key.split('_')[1:]):
                    if k == 'plate' and i == 0 and ori_state_info[0] == 'hot':
                        ori_state_info[0] = 'hot_plate'
                    else:
                        if k == 'None':
                            ori_state_info.append(-1)
                        else:
                            ori_state_info.append(k)

                new_key = ori_key + '_' + subtask[0]
                new_obj = original_state_dict[ori_key] + [subtask[0]]
                self.state_dict[new_key] = new_obj  # update value
                self.state_idx_dict[new_key] = len(self.state_idx_dict)

    def init_transition(self, transition_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros(
            (len(self.action_dict), len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]
            for action_key, action_idx in self.action_idx_dict.items():
                # decode state information
                # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
                p0_state, p1_state, world_info = self.decode_state_info(state_obj)
                # calculate next states for p1 (a.k.a. human)
                p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)

                # append original state of p1 (human) to represent unfinished subtask state transition
                p1_nxt_states.append(p1_state)
                p1_nxt_world_info += [world_info]

                # calculate next states for p0 (conditioned on p1 (a.k.a. human))
                for i, p1_nxt_state in enumerate(p1_nxt_states):
                    actions, next_state_keys = self.stochastic_state_transition(
                        p0_state,
                        p1_nxt_world_info[i],
                        human_state=p1_nxt_state
                    )
                    for action, next_state_key in zip(actions, next_state_keys):
                        if action_key == action:
                            next_state_idx = self.state_idx_dict[next_state_key]
                            self.transition_matrix[action_idx, state_idx, next_state_idx] += 1.0

                if np.sum(self.transition_matrix[action_idx, state_idx]) > 0.0:
                    self.transition_matrix[action_idx, state_idx] /= np.sum(
                        self.transition_matrix[action_idx, state_idx])

        self.transition_matrix[self.transition_matrix == 0.0] = 0.000001

    def init_cost(self):
        self.cost_matrix = np.zeros((len(self.action_dict), len(self.state_dict), len(self.state_dict)))

        for action, action_idx in self.action_idx_dict.items():
            for curr_state, curr_state_idx in self.state_idx_dict.items():
                next_states = np.where(
                    self.transition_matrix[action_idx,
                    curr_state_idx] > 0.00001)[0]

                for next_state_idx in next_states:
                    human_action_key = self.get_key_from_value(self.action_idx_dict, action_idx)
                    self.cost_matrix[action_idx][curr_state_idx][
                        next_state_idx] = 5

    def get_successor_states(self, start_world_state, start_state_key, debug=False, add_rewards=False):
        """
        Successor states for qmdp medium-level actions.
        """
        if len(self.state_dict[start_state_key][4:-1]) == 0:  # [p0_obj, num_item_in_soup, orders, p1_obj, subtask]
            return []

        ori_state_idx = self.state_idx_dict[start_state_key]
        successor_states = []

        # returns array(action idx), array(next_state_idx)
        agent_action_idx_arr, next_state_idx_arr = np.where(self.transition_matrix[:, ori_state_idx] > 0.000001)
        start_time = time.time()
        for next_action_idx, next_state_idx in zip(agent_action_idx_arr, next_state_idx_arr):
            next_world_state, cost = \
                self.mdp_action_state_to_world_state(next_action_idx, next_state_idx, start_world_state)
            successor_states.append((
                self.get_key_from_value(self.state_idx_dict, next_state_idx),
                next_world_state,
                cost
            ))
            if debug:
                print('Action {} from {} to {} costs {} in {} seconds.'.format(
                    self.get_key_from_value(self.action_idx_dict, next_action_idx),
                    self.get_key_from_value(self.state_idx_dict, ori_state_idx),
                    self.get_key_from_value(self.state_idx_dict, next_state_idx), cost, time.time() - start_time
                ))

        return successor_states

    def get_key_from_value(self, dictionary, state_value):
        try:
            idx = list(dictionary.values()).index(state_value)
        except ValueError:
            return None
        else:
            return list(dictionary.keys())[idx]

    def decode_state_info(self, state_obj):
        # state_obj = id 0; other_subtask = last element; world info = everything in between
        return state_obj[0], state_obj[-1], state_obj[1:-1]

    def _is_valid_object_subtask_pair(self, subtask, num_item_in_pot, chop_time, wash_time, vision_limit=False,
                                      human_obj=None, other_agent_holding=None, prev_subtask=None):
        """
        Since we do not consider the other agent's action for the human's subtask initialization,
        we mainly just care about whether the object the human is holding pairs with the subtask.
        Also, since the other agent may change the world, we forgo that the world as constraints on the subtasks.
        When considering belief update, we need to consider how the robot's holding object effects the human when
        the human is a robot-aware human model.
        """
        if chop_time == 'None' or chop_time is None:
            chop_time = -1
        if wash_time == 'None' or wash_time is None:
            wash_time = -1

        # map subtask to possible object holding
        if human_obj is None:
            subtask_action = subtask.split('_')[0]
            subtask_obj = '_'.join(subtask.split('_')[1:])
            if (subtask_action in ['pickup', 'chop', 'heat']) and subtask_obj not in ['steak', 'garnish']:
                obj = 'None'
            elif subtask == 'pickup_steak':
                obj = 'hot_plate'
            elif subtask == 'pickup_garnish':
                obj = 'steak'
            else:
                obj = subtask_obj
        else:
            obj = human_obj

        if obj == 'None':
            if subtask == 'chop_onion' and 0 <= chop_time < self.mdp.chopping_time and \
                    prev_subtask in ['drop_onion', 'chop_onion']:
                return True
            elif subtask == 'heat_hot_plate' and 0 <= wash_time < self.mdp.wash_time and \
                    prev_subtask in ['drop_plate', 'heat_hot_plate']:
                return True
            elif subtask == 'pickup_meat' and num_item_in_pot < self.mdp.num_items_for_steak and \
                    other_agent_holding != 'meat':
                return True
            elif subtask == 'pickup_onion' and chop_time < 0 and other_agent_holding != 'onion':
                return True
            elif subtask == 'pickup_plate' and wash_time < 0 and other_agent_holding != 'plate':
                return True
            elif subtask == 'pickup_hot_plate' and (chop_time >= 0 or other_agent_holding == 'onion') and \
                    wash_time >= self.mdp.wash_time and other_agent_holding != 'hot_plate' and \
                    other_agent_holding != 'steak':
                return True
            elif subtask == 'pickup_hot_plate' and (chop_time >= 0 or other_agent_holding == 'onion') and \
                    wash_time < self.mdp.wash_time and num_item_in_pot > 0 and other_agent_holding != 'hot_plate' and \
                    other_agent_holding != 'steak':
                return True
            else:  # this is an instance that will be triggered when there are no other things to pick up.
                return False
        else:
            if obj == 'onion' and subtask == 'drop_onion':
                return True
            elif obj == 'meat' and subtask == 'drop_meat':
                return True
            elif obj == 'plate' and subtask == 'drop_plate':
                return True
            elif obj == 'hot_plate' and subtask == 'pickup_steak':
                return True
            elif obj == 'steak' and (subtask == 'pickup_garnish' or subtask == 'drop_steak'):
                return True
            elif obj == 'hot_plate' and subtask == 'drop_hot_plate':
                return True
            elif obj == 'dish' and subtask == 'deliver_dish':
                return True
            else:
                return False

    def human_state_subtask_transition(self, subtask, world_info):
        # player_obj = human_state[0]
        num_item_in_pot = int(world_info[0])
        chop_time = world_info[1]
        wash_time = world_info[2]
        orders = [] if len(world_info) < 4 else world_info[3:]

        if chop_time == 'None':
            chop_time = -1
        else:
            chop_time = int(chop_time)
        if wash_time == 'None':
            wash_time = -1
        else:
            wash_time = int(wash_time)

        next_subtasks = []
        nxt_world_info = []
        next_num_item_in_pot = num_item_in_pot
        next_chop_time = chop_time
        next_wash_time = wash_time
        next_orders = orders.copy()

        subtask_action = subtask.split('_')[0]
        subtask_obj = '_'.join(subtask.split('_')[1:])
        if (subtask_action in ['pickup', 'chop', 'heat']) and subtask_obj not in ['steak', 'garnish']:
            player_obj = 'None'
        elif subtask == 'pickup_steak':
            player_obj = 'hot_plate'
        elif subtask == 'pickup_garnish':
            player_obj = 'steak'
        else:
            player_obj = subtask_obj

        if player_obj == 'None':
            if subtask == 'pickup_meat':
                next_obj = 'meat'
                next_subtasks = ['drop_meat']
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'pickup_onion':
                next_obj = 'onion'
                next_subtasks = ['drop_onion']
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'chop_onion' and self.mdp.chopping_time - 1 > chop_time >= 0:
                next_obj = 'None'
                next_chop_time = chop_time + 1
                if next_chop_time > self.mdp.chopping_time:
                    next_chop_time = self.mdp.chopping_time
                next_subtasks = ['chop_onion']
                nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'chop_onion' and chop_time >= self.mdp.chopping_time - 1:
                next_obj = 'None'
                next_chop_time = chop_time + 1
                if next_chop_time > self.mdp.chopping_time:
                    next_chop_time = self.mdp.chopping_time

                if wash_time < 0:
                    next_subtasks.append('pickup_plate')
                    nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
                elif 0 <= wash_time < self.mdp.wash_time:
                    next_subtasks.append('heat_hot_plate')
                    nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
                elif wash_time >= self.mdp.wash_time:
                    next_subtasks.append('pickup_hot_plate')
                    nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'chop_onion' and chop_time < 0:
                next_subtasks = ['pickup_onion']
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'pickup_plate':  # and wash_time < 0:
                next_obj = 'plate'
                next_subtasks = ['drop_plate']
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'heat_hot_plate' and self.mdp.wash_time - 1 > wash_time >= 0:
                next_obj = 'None'
                next_wash_time = wash_time + 1
                if next_wash_time > self.mdp.wash_time:
                    next_wash_time = self.mdp.wash_time
                next_subtasks = ['heat_hot_plate']
                nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)

            elif subtask == 'heat_hot_plate' and wash_time >= self.mdp.wash_time - 1:
                next_obj = 'None'
                next_wash_time = wash_time + 1
                if next_wash_time > self.mdp.wash_time:
                    next_wash_time = self.mdp.wash_time
                next_subtasks = ['pickup_hot_plate']
                nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)

            elif subtask == 'heat_hot_plate' and wash_time < 0:
                next_subtasks = ['pickup_plate']
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'pickup_hot_plate' and num_item_in_pot >= self.mdp.num_items_for_steak:
                next_obj = 'hot_plate'
                next_subtasks = ['pickup_steak']
                nxt_world_info += self.gen_world_info_list(chop_time, -1, num_item_in_pot, orders)

            elif subtask == 'pickup_hot_plate' and num_item_in_pot < self.mdp.num_items_for_steak:
                next_obj = 'hot_plate'
                next_subtasks = ['pickup_hot_plate']
                nxt_world_info += self.gen_world_info_list(chop_time, -1, num_item_in_pot, orders)

            elif subtask == 'pickup_steak':
                next_num_item_in_pot = 0
                if chop_time >= self.mdp.chopping_time:
                    next_subtasks = ['pickup_garnish']
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
                else:
                    next_subtasks = ['pickup_steak']
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
            else:
                print(subtask, world_info)
                raise ValueError()
        else:
            if subtask == 'drop_meat':
                next_obj = 'None'
                if num_item_in_pot < self.mdp.num_items_for_steak: next_num_item_in_pot = 1

                if chop_time < 0:
                    next_subtasks.append('pickup_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
                elif 0 <= chop_time < self.mdp.chopping_time:
                    next_subtasks.append('chop_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)

                if wash_time < 0:
                    next_subtasks.append('pickup_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
                elif 0 <= wash_time < self.mdp.wash_time:
                    next_subtasks.append('heat_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
                elif wash_time >= self.mdp.wash_time:
                    next_subtasks.append('pickup_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)

            elif subtask == 'drop_onion':
                next_obj = 'None'
                if chop_time < 0:
                    next_chop_time = 0

                if chop_time < self.mdp.chopping_time:
                    next_subtasks.append('chop_onion')
                    nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
                else:
                    if wash_time < 0:
                        next_subtasks.append('pickup_plate')
                        nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
                    elif 0 <= wash_time < self.mdp.wash_time:
                        next_subtasks.append('heat_hot_plate')
                        nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
                    elif wash_time >= self.mdp.wash_time:
                        next_subtasks.append('pickup_hot_plate')
                        nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'drop_plate':
                next_obj = 'None'
                if wash_time < 0:
                    next_wash_time = 0

                if wash_time < self.mdp.wash_time:
                    next_subtasks.append('heat_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)
                else:
                    next_subtasks.append('pickup_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)

            elif subtask == 'drop_hot_plate':
                next_obj = 'None'
                if num_item_in_pot < self.mdp.num_items_for_steak:
                    next_subtasks.append('pickup_meat')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
                else:
                    next_subtasks.append('pickup_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'drop_steak':
                next_obj = 'None'
                if chop_time < 0:
                    next_subtasks.append('pickup_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
                elif 0 <= chop_time < self.mdp.chopping_time:
                    next_subtasks.append('chop_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
                else:
                    next_subtasks.append('pickup_steak')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'pickup_garnish' and chop_time >= self.mdp.chopping_time:
                next_obj = 'dish'
                next_subtasks.append('deliver_dish')
                nxt_world_info += self.gen_world_info_list(-1, wash_time, num_item_in_pot, orders)

            elif subtask == 'pickup_garnish' and chop_time < self.mdp.chopping_time:
                next_obj = 'None'
                next_subtasks.append('pickup_garnish')  # next_subtasks.append('drop_steak')
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)

            elif subtask == 'pickup_steak' and num_item_in_pot >= self.mdp.num_items_for_steak:
                next_num_item_in_pot = 0
                next_obj = 'steak'
                next_subtasks.append('pickup_garnish')
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)

            elif subtask == 'pickup_steak' and num_item_in_pot < self.mdp.num_items_for_steak:
                next_obj = 'None'
                next_subtasks.append('drop_hot_plate')
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)

            elif subtask == 'deliver_dish':
                if len(orders) >= 1:
                    next_orders.pop(0)

                next_obj = 'None'
                if wash_time < 0:
                    next_subtasks.append('pickup_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
                elif chop_time < 0:
                    next_subtasks.append('pickup_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
                elif num_item_in_pot < self.mdp.num_items_for_steak:
                    next_subtasks.append('pickup_meat')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
                elif 0 <= chop_time < self.mdp.chopping_time:
                    next_subtasks.append('chop_onion')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
                elif 0 <= wash_time < self.mdp.wash_time:
                    next_subtasks.append('heat_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
                elif wash_time >= self.mdp.wash_time:
                    next_subtasks.append('pickup_hot_plate')
                    nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)

            else:
                print(player_obj, subtask)
                raise ValueError()

        return next_subtasks, nxt_world_info

    def world_based_human_state_subtask_transition(self, subtask, world_info, other_agent_obj='None'):
        num_item_in_pot = int(world_info[0])
        chop_time = world_info[1]
        wash_time = world_info[2]
        orders = [] if len(world_info) < 4 else world_info[3:]

        if chop_time == 'None':
            chop_time = -1
        else:
            chop_time = int(chop_time)
        if wash_time == 'None':
            wash_time = -1
        else:
            wash_time = int(wash_time)

        next_subtasks = []
        nxt_world_info = []
        next_num_item_in_pot = num_item_in_pot
        next_chop_time = chop_time
        next_wash_time = wash_time
        next_orders = orders.copy()

        subtask_action = subtask.split('_')[0]
        subtask_obj = '_'.join(subtask.split('_')[1:])
        player_obj = 'None'

        # update world info
        if (subtask_action in ['pickup']) and subtask_obj not in ['hot_plate', 'steak', 'garnish']:
            nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
            player_obj = subtask_obj
        elif subtask == 'pickup_hot_plate':
            if wash_time >= self.mdp.wash_time:
                nxt_world_info += self.gen_world_info_list(chop_time, -1, num_item_in_pot, orders)
                player_obj = 'hot_plate'
            else:
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
                player_obj = 'None'
        elif subtask == 'pickup_steak':
            if num_item_in_pot > 0:
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, 0, orders)
                player_obj = 'steak'
            else:
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
                player_obj = 'hot_plate'
        elif subtask == 'pickup_garnish':
            if chop_time >= self.mdp.chopping_time:
                nxt_world_info += self.gen_world_info_list(-1, wash_time, num_item_in_pot, orders)
                player_obj = 'dish'
            else:
                nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
                player_obj = 'steak'
        elif subtask == 'drop_meat':
            if num_item_in_pot == 0:
                # meaning you drop at the right location instead of just on the counter
                next_num_item_in_pot = 1
            nxt_world_info += self.gen_world_info_list(chop_time, wash_time, next_num_item_in_pot, orders)
        elif subtask == 'drop_onion':
            if chop_time < 0:
                # meaning you drop at the right location instead of just on the counter
                next_chop_time = 0
            nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
        elif subtask == 'drop_plate':
            if wash_time < 0:
                # meaning you drop at the right location instead of just on the counter
                next_wash_time = 0
            nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)
        elif subtask == 'chop_onion':
            next_chop_time = min(chop_time + 1, self.mdp.chopping_time)
            nxt_world_info += self.gen_world_info_list(next_chop_time, wash_time, num_item_in_pot, orders)
        elif subtask == 'heat_hot_plate':
            next_wash_time = min(wash_time + 1, self.mdp.wash_time)
            nxt_world_info += self.gen_world_info_list(chop_time, next_wash_time, num_item_in_pot, orders)
        elif subtask == 'deliver_dish':
            if len(orders) >= 1:
                next_orders.pop(0)
            nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, next_orders)
        elif subtask in ['drop_hot_plate', 'drop_steak', 'drop_dish']:
            nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
        else:
            nxt_world_info += self.gen_world_info_list(chop_time, wash_time, num_item_in_pot, orders)
            print(subtask, world_info, other_agent_obj)
            raise ValueError()

        # decide next subtask based on the environment
        if player_obj == 'None':
            if 0 <= next_chop_time < self.mdp.chopping_time and (subtask == 'chop_onion' or subtask == 'drop_onion'):
                next_subtasks += ['chop_onion']
            elif 0 <= next_wash_time < self.mdp.wash_time and (subtask == 'heat_hot_plate' or subtask == 'drop_plate'):
                next_subtasks += ['heat_hot_plate']
            elif next_num_item_in_pot == 0 and len(next_orders) > 0 and other_agent_obj != 'meat':
                next_subtasks += ['pickup_meat']
            elif next_chop_time < 0 and other_agent_obj != 'onion':
                next_subtasks += ['pickup_onion']
            elif next_wash_time < 0 and other_agent_obj != 'plate':
                next_subtasks += ['pickup_plate']
            elif (next_chop_time >= self.mdp.chopping_time or other_agent_obj == 'onion') and \
                    next_wash_time >= self.mdp.wash_time and next_num_item_in_pot > 0 and not \
                    (other_agent_obj == 'hot_plate' or other_agent_obj == 'steak'):
                next_subtasks += ['pickup_hot_plate']
            else:
                next_subtasks += ['pickup_plate']
        else:
            if player_obj == 'onion':
                next_subtasks = ['drop_onion']
            elif player_obj == 'meat':
                next_subtasks = ['drop_meat']
            elif player_obj == 'plate':
                next_subtasks = ['drop_plate']
            elif player_obj == 'hot_plate':
                next_subtasks = ['pickup_steak']
            elif player_obj == 'steak':
                next_subtasks = ['pickup_garnish']
            elif player_obj == 'dish':
                next_subtasks = ['deliver_dish']

        if len(next_subtasks) == 0:
            next_subtasks = [subtask]

        return next_subtasks, nxt_world_info

    def state_transition(self, player_obj, world_info, human_state=None, human_obj=None):
        # game logic: but consider that the human subtask in human_state is not yet executed
        num_item_in_pot = world_info[0]
        chop_time = world_info[1]
        wash_time = world_info[2]
        orders = [] if len(world_info) < 4 else world_info[3:]

        if human_state is not None:
            subtask = human_state[0]

            subtask_action = subtask.split('_')[0]
            subtask_obj = '_'.join(subtask.split('_')[1:])

        if human_obj is None:
            if (subtask_action in ['pickup', 'chop', 'heat']) and (subtask_obj not in ['steak', 'garnish']):
                human_obj = 'None'
            elif subtask == 'pickup_steak':
                human_obj = 'hot_plate'
            elif subtask == 'pickup_garnish':
                human_obj = 'steak'
            else:
                human_obj = subtask_obj

        if wash_time == 'None':
            wash_time = -1
        if chop_time == 'None':
            chop_time = -1

        actions = '';
        next_obj = player_obj
        next_num_item_in_pot = num_item_in_pot
        next_chop_time = chop_time
        next_wash_time = wash_time

        if player_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (human_obj != 'meat'):
                actions = 'pickup_meat'
                next_obj = 'meat'
            elif (chop_time < 0) and (human_obj != 'onion'):
                actions = 'pickup_onion'
                next_obj = 'onion'
            elif (wash_time < 0) and (human_obj != 'plate'):
                actions = 'pickup_plate'
                next_obj = 'plate'
            elif ((chop_time >= 0) and (chop_time < self.mdp.chopping_time) and (subtask != 'chop_onion')) or (
                    (chop_time < 0) and (human_obj == 'onion')):
                actions = 'chop_onion'
                next_obj = 'None'
                next_chop_time += 1
            elif ((wash_time >= 0) and (wash_time < self.mdp.wash_time) and (subtask != 'heat_hot_plate')) or (
                    (wash_time < 0) and (human_obj == 'plate')):
                actions = 'heat_hot_plate'
                next_obj = 'None'
                next_wash_time += 1
            elif ((chop_time == self.mdp.chopping_time) or (subtask == 'chop_onion')) and (
                    (wash_time == self.mdp.wash_time) or (subtask == 'heat_hot_plate')) and (
                    subtask != 'pickup_hot_plate'):
                actions = 'pickup_hot_plate'
                next_obj = 'hot_plate'
                next_wash_time = -1
            else:
                actions = 'pickup_meat'
                next_obj = 'meat'

        else:
            if player_obj == 'onion':
                actions = 'drop_onion'
                next_obj = 'None'
                if chop_time < 0: next_chop_time = 0  # doesn't change since no available board to drop

            elif player_obj == 'meat':
                actions = 'drop_meat'
                next_obj = 'None'
                next_num_item_in_pot = 1

            elif player_obj == 'plate':
                actions = 'drop_plate'
                next_obj = 'None'
                if wash_time < 0: next_wash_time = 0  # doesn't change since no available sink to drop

            elif (player_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions = 'pickup_steak'
                next_obj = 'steak'
                next_num_item_in_pot = 0

            elif (player_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions = 'drop_hot_plate'
                next_obj = 'None'

            elif (player_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions = 'pickup_garnish'
                next_obj = 'dish'
                next_chop_time = -1

            elif (player_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                # actions = 'drop_steak'
                # next_obj = 'None'
                actions = 'pickup_garnish'
                next_obj = 'dish'
                next_chop_time = -1

            elif player_obj == 'dish':
                actions = 'deliver_dish'
                next_obj = 'None'
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                print(player_obj, world_info, subtask)
                raise ValueError()

        if next_chop_time < 0:
            next_chop_time = 'None'
        elif next_chop_time > self.mdp.chopping_time:
            next_chop_time = self.mdp.chopping_time

        if next_wash_time < 0:
            next_wash_time = 'None'
        elif next_wash_time > self.mdp.wash_time:
            next_wash_time = self.mdp.wash_time

        next_state_keys = next_obj + '_' + str(next_num_item_in_pot) + '_' + str(next_chop_time) + '_' + str(
            next_wash_time)
        for order in orders:
            next_state_keys = next_state_keys + '_' + order

        next_state_keys = next_state_keys + '_' + subtask

        return actions, next_state_keys

    def stochastic_state_transition(self, player_obj, world_info, human_state=None):
        # game logic: but consider that the human subtask in human_state is not yet executed
        num_item_in_pot = world_info[0]
        chop_time = world_info[1]
        wash_time = world_info[2]
        orders = [] if len(world_info) < 4 else world_info[3:]
        subtask = human_state
        next_subtask = human_state

        if wash_time == 'None':
            wash_time = -1
        if chop_time == 'None':
            chop_time = -1

        actions = []
        next_state_keys = []
        next_obj = player_obj
        next_num_item_in_pot = num_item_in_pot
        next_chop_time = chop_time
        next_wash_time = wash_time

        subtask_action = subtask.split('_')[0]
        subtask_obj = '_'.join(subtask.split('_')[1:])
        if (subtask_action in ['pickup', 'chop', 'heat']) and (subtask_obj not in ['steak', 'garnish']):
            human_obj = 'None'
        elif subtask == 'pickup_steak':
            human_obj = 'hot_plate'
        elif subtask == 'pickup_garnish':
            human_obj = 'steak'
        else:
            human_obj = subtask_obj

        if player_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (human_obj != 'meat'):
                actions += ['pickup_meat']
                next_state_keys += self.gen_state_key('meat', next_chop_time, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)
            if (chop_time < 0) and (human_obj != 'onion'):
                actions += ['pickup_onion']
                next_state_keys += self.gen_state_key('onion', next_chop_time, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)
            if (wash_time < 0) and (human_obj != 'plate' or human_obj != 'hot_plate'):
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)
            if (chop_time >= 0) and (chop_time < self.mdp.chopping_time) and (subtask != 'chop_onion'):
                actions += ['chop_onion']
                next_state_keys += self.gen_state_key('None', next_chop_time + 1, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)

            if (chop_time < 0) and (human_obj == 'onion'):
                pass

            if (wash_time >= 0) and (wash_time < self.mdp.wash_time) and (subtask != 'heat_hot_plate'):
                actions += ['heat_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time + 1,
                                                      next_num_item_in_pot, orders, next_subtask)

            # this is added with the assumption that you take the heat_hot_plate action while the human has not
            # finished their subtask, therefore, the state should not change to the state of after completing
            # head_hot_plate action.
            if (wash_time < 0) and (human_obj == 'plate'):
                pass
                # since usually after dropping the plate in the sink, the person who dropped it will continue to heat it

            # Note: removed the condition that the robot can still pick up the hot_plate when the human has not
            # finished heating the last step
            if ((chop_time >= self.mdp.chopping_time) or (subtask == 'chop_onion')) and \
                    wash_time >= self.mdp.wash_time and (subtask != 'pickup_hot_plate'):
                actions += ['pickup_hot_plate']
                next_state_keys += self.gen_state_key('hot_plate', next_chop_time, -1,
                                                      next_num_item_in_pot, orders, next_subtask)

            if len(actions) == 0:
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)

        else:
            if player_obj == 'onion':
                actions += ['drop_onion']
                if chop_time < 0:  # doesn't change since no available board to drop
                    next_state_keys += self.gen_state_key('None', 0, next_wash_time,
                                                          next_num_item_in_pot, orders, next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time,
                                                          next_num_item_in_pot, orders, next_subtask)

            elif player_obj == 'meat':
                actions += ['drop_meat']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time,
                                                      1, orders, next_subtask)

            elif player_obj == 'plate':
                actions += ['drop_plate']
                if wash_time < 0:  # doesn't change since no avaliable sink to drop
                    next_state_keys += self.gen_state_key('None', next_chop_time, 0,
                                                          next_num_item_in_pot, orders, next_subtask)
                elif wash_time > 0 and chop_time >= self.mdp.chopping_time and \
                        num_item_in_pot >= self.mdp.num_items_for_steak:  # do not drop plate since we are in the plating stage and no other actions are availiable
                    next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time,
                                                          next_num_item_in_pot, orders, next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time,
                                                          next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions += ['pickup_steak']
                next_state_keys += self.gen_state_key('steak', next_chop_time, next_wash_time,
                                                      0, orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions += ['drop_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)

            elif (player_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)

            elif player_obj == 'dish':
                actions += ['deliver_dish']
                if len(orders) >= 1:
                    orders.pop(0)
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)

            else:
                print(player_obj, world_info, next_subtask)
                raise ValueError()

        return actions, next_state_keys

    def consider_subtask_stochastic_state_transition(self, player_obj, world_info, human_state=None):
        # game logic: but consider that the human subtask in human_state is not yet executed
        num_item_in_pot = world_info[0]
        chop_time = world_info[1]
        wash_time = world_info[2]
        orders = [] if len(world_info) < 4 else world_info[3:]
        subtask = human_state
        next_subtask = human_state

        if wash_time == 'None':
            wash_time = -1
        if chop_time == 'None':
            chop_time = -1

        actions = []
        next_state_keys = []
        next_obj = player_obj
        next_num_item_in_pot = num_item_in_pot
        next_chop_time = chop_time
        next_wash_time = wash_time

        subtask_action = subtask.split('_')[0]
        subtask_obj = '_'.join(subtask.split('_')[1:])
        if (subtask_action in ['pickup', 'chop', 'heat']) and (subtask_obj not in ['steak', 'garnish']):
            human_obj = 'None'
        elif subtask == 'pickup_steak':
            human_obj = 'hot_plate'
        elif subtask == 'pickup_garnish':
            human_obj = 'steak'
        else:
            human_obj = subtask_obj

        if player_obj == 'None':
            if ((chop_time >= self.mdp.chopping_time) or (subtask in ['chop_onion'])) and \
                    wash_time >= self.mdp.wash_time and (subtask != 'pickup_hot_plate'):
                actions += ['pickup_hot_plate']
                next_state_keys += self.gen_state_key('hot_plate', next_chop_time, -1,
                                                      next_num_item_in_pot, orders, next_subtask)

            if (chop_time >= 0) and (chop_time < self.mdp.chopping_time) and (subtask != 'chop_onion'):
                actions += ['chop_onion']
                next_state_keys += self.gen_state_key('None', next_chop_time + 1, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)

            # this is added with the assumption that you take the chop_onion action while the human has not finished
            # their subtask, therefore, the state should not change to the state of after completing chop_onion action.
            if (chop_time < 0) and (human_obj == 'onion'):
                # since usually after dropping the onion on the board, the person who dropped it will continue to chop it
                pass

            if ((wash_time >= 0) and (wash_time < self.mdp.wash_time) and (
                    subtask != 'heat_hot_plate')):  # or ((wash_time < 0) and (human_obj == 'plate')):
                actions += ['heat_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time + 1, next_num_item_in_pot,
                                                      orders, next_subtask)

            # this is added with the assumption that you take the heat_hot_plate action while the human has not
            # finished their subtask, therefore, the state should not change to the state of after completing
            # head_hot_plate action.
            if (wash_time < 0) and (human_obj == 'plate'):
                # since usually after dropping the plate in the sink, the person who dropped it will continue to heat it
                pass

                # Note: removed the condition that the robot can still pick up the hot_plate when the human has not
            # finished heating the last step
            if num_item_in_pot < self.mdp.num_items_for_steak and human_obj != 'meat' and subtask != 'pickup_meat':
                actions += ['pickup_meat']
                next_state_keys += self.gen_state_key('meat', next_chop_time, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)
            if (chop_time < 0) and (human_obj != 'onion') and (subtask != 'pickup_onion'):
                actions += ['pickup_onion']
                next_state_keys += self.gen_state_key('onion', next_chop_time, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)
            if (wash_time < 0) and (human_obj != 'plate' or human_obj != 'hot_plate') and (subtask != 'pickup_plate'):
                # consider the human_object not hot plate since not priority
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)

            if len(actions) == 0:
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time,
                                                      next_num_item_in_pot, orders, next_subtask)

        else:
            if player_obj == 'onion':
                actions += ['drop_onion']
                if chop_time < 0:  # doesn't change since no available board to drop
                    next_state_keys += self.gen_state_key('None', 0, next_wash_time,
                                                          next_num_item_in_pot, orders, next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time,
                                                          next_num_item_in_pot, orders, next_subtask)

            elif player_obj == 'meat':
                actions += ['drop_meat']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, 1, orders, next_subtask)

            elif player_obj == 'plate':
                actions += ['drop_plate']
                if wash_time < 0:  # doesn't change since no avaliable sink to drop
                    next_state_keys += self.gen_state_key('None', next_chop_time, 0, next_num_item_in_pot, orders,
                                                          next_subtask)
                elif wash_time > 0 and chop_time >= self.mdp.chopping_time and \
                        num_item_in_pot >= self.mdp.num_items_for_steak:  # do not drop plate since we are in the plating stage and no other actions are availiable
                    next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                          orders, next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                          orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions += ['pickup_steak']
                next_state_keys += self.gen_state_key('steak', next_chop_time, next_wash_time, 0, orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions += ['drop_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                      orders, next_subtask)

            elif (player_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time, next_num_item_in_pot, orders,
                                                      next_subtask)

            elif (player_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time, next_num_item_in_pot, orders,
                                                      next_subtask)

            elif player_obj == 'dish':
                actions += ['deliver_dish']
                if len(orders) >= 1:
                    orders.pop(0)
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                      orders, next_subtask)

            else:
                print(player_obj, world_info, next_subtask)
                raise ValueError()

        return actions, next_state_keys

    def non_subtask_stochastic_state_transition(self, player_obj, world_info, human_state=None):
        # game logic: but consider that the human subtask in human_state is not yet executed
        num_item_in_pot = world_info[0]
        chop_time = world_info[1]
        wash_time = world_info[2]
        orders = [] if len(world_info) < 4 else world_info[3:]
        subtask = human_state
        next_subtask = human_state

        if wash_time == 'None':
            wash_time = -1
        if chop_time == 'None':
            chop_time = -1

        actions = []
        next_state_keys = []
        next_obj = player_obj
        next_num_item_in_pot = num_item_in_pot
        next_chop_time = chop_time
        next_wash_time = wash_time

        subtask_action = subtask.split('_')[0]
        subtask_obj = '_'.join(subtask.split('_')[1:])
        if (subtask_action in ['pickup', 'chop', 'heat']) and (subtask_obj not in ['steak', 'garnish']):
            human_obj = 'None'
        elif subtask == 'pickup_steak':
            human_obj = 'hot_plate'
        elif subtask == 'pickup_garnish':
            human_obj = 'steak'
        else:
            human_obj = subtask_obj

        if player_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (human_obj != 'meat'):
                actions += ['pickup_meat']
                next_state_keys += self.gen_state_key('meat', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                      orders, next_subtask)
            if (chop_time < 0) and (human_obj != 'onion'):
                actions += ['pickup_onion']
                next_state_keys += self.gen_state_key('onion', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                      orders, next_subtask)
            if (wash_time < 0) and (human_obj != 'plate' or human_obj != 'hot_plate'):
                # consider the human_object not hot plate since not priority
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                      orders, next_subtask)
            if (chop_time >= 0) and (
                    chop_time < self.mdp.chopping_time):
                actions += ['chop_onion']
                next_state_keys += self.gen_state_key('None', next_chop_time + 1, next_wash_time, next_num_item_in_pot,
                                                      orders, next_subtask)

            # this is added with the assumption that you take the chop_onion action while the human has not finished
            # their subtask, therefore, the state should not change to the state of after completing chop_onion action.
            if (chop_time < 0) and (human_obj == 'onion'):
                pass
            if (wash_time >= 0) and (wash_time < self.mdp.wash_time):
                actions += ['heat_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time + 1, next_num_item_in_pot,
                                                      orders, next_subtask)

            # this is added with the assumption that you take the heat_hot_plate action while the human has not
            # finished their subtask, therefore, the state should not change to the state of after completing
            # head_hot_plate action.
            if (wash_time < 0) and (human_obj == 'plate'):
                pass

            # Note: removed the condition that the robot can still pick up the hot_plate when the human has not
            # finished heating the last step
            if ((chop_time >= self.mdp.chopping_time) or (subtask == 'chop_onion')) and \
                    wash_time >= self.mdp.wash_time:
                actions += ['pickup_hot_plate']
                next_state_keys += self.gen_state_key('hot_plate', next_chop_time, -1, next_num_item_in_pot, orders,
                                                      next_subtask)

            if len(actions) == 0:
                actions += ['pickup_plate']
                next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                      orders, next_subtask)

        else:
            if player_obj == 'onion':
                actions += ['drop_onion']
                if chop_time < 0:  # doesn't change since no available board to drop
                    next_state_keys += self.gen_state_key('None', 0, next_wash_time, next_num_item_in_pot, orders,
                                                          next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                          orders, next_subtask)

            elif player_obj == 'meat':
                actions += ['drop_meat']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, 1, orders, next_subtask)

            elif player_obj == 'plate':
                actions += ['drop_plate']
                if wash_time < 0:  # doesn't change since no available sink to drop
                    next_state_keys += self.gen_state_key('None', next_chop_time, 0, next_num_item_in_pot, orders,
                                                          next_subtask)
                elif wash_time > 0 and chop_time >= self.mdp.chopping_time and \
                        num_item_in_pot >= self.mdp.num_items_for_steak:
                    # do not drop plate since we are in the plating stage and no other actions are available
                    next_state_keys += self.gen_state_key('plate', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                          orders, next_subtask)
                else:
                    next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                          orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions += ['pickup_steak']
                next_state_keys += self.gen_state_key('steak', next_chop_time, next_wash_time, 0, orders, next_subtask)

            elif (player_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions += ['drop_hot_plate']
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                      orders, next_subtask)

            elif (player_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time, next_num_item_in_pot, orders,
                                                      next_subtask)

            elif (player_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                actions += ['pickup_garnish']
                next_state_keys += self.gen_state_key('dish', -1, next_wash_time, next_num_item_in_pot, orders,
                                                      next_subtask)

            elif player_obj == 'dish':
                actions += ['deliver_dish']
                if len(orders) >= 1:
                    orders.pop(0)
                next_state_keys += self.gen_state_key('None', next_chop_time, next_wash_time, next_num_item_in_pot,
                                                      orders, next_subtask)

            else:
                print(player_obj, world_info, next_subtask)
                raise ValueError()

        return actions, next_state_keys

    def gen_world_info_list(self, next_chop_time, next_wash_time, next_num_item_in_pot, orders):
        if next_chop_time < 0:
            next_chop_time = 'None'
        elif next_chop_time > self.mdp.chopping_time:
            next_chop_time = self.mdp.chopping_time

        if next_wash_time < 0:
            next_wash_time = 'None'
        elif next_wash_time > self.mdp.wash_time:
            next_wash_time = self.mdp.wash_time

        nxt_world = [next_num_item_in_pot, next_chop_time, next_wash_time]
        for order in orders:
            nxt_world.append(order)

        return [nxt_world]

    def gen_state_key(self, next_obj, next_chop_time, next_wash_time, next_num_item_in_pot, orders, subtask):
        if next_chop_time != 'None':
            next_chop_time = int(next_chop_time)
        else:
            next_chop_time = -1

        if next_wash_time != 'None':
            next_wash_time = int(next_wash_time)
        else:
            next_wash_time = -1

        next_num_item_in_pot = int(next_num_item_in_pot)

        if next_chop_time < 0:
            next_chop_time = 'None'
        elif next_chop_time > self.mdp.chopping_time:
            next_chop_time = self.mdp.chopping_time

        if next_wash_time < 0:
            next_wash_time = 'None'
        elif next_wash_time > self.mdp.wash_time:
            next_wash_time = self.mdp.wash_time

        next_state_keys = next_obj + '_' + str(next_num_item_in_pot) + '_' + str(next_chop_time) + '_' + str(
            next_wash_time)
        for order in orders:
            next_state_keys = next_state_keys + '_' + order

        next_state_keys = next_state_keys + '_' + subtask

        return [next_state_keys]

    def world_state_to_mdp_state_key(self, state, player, other_player, env, subtask=None, RETURN_NON_SUBTASK=False,
                                     RETURN_OBJ=False):
        state_str = super().gen_state_dict_key(state, player, other_player=other_player, env=env, RETURN_OBJ=RETURN_OBJ)

        if RETURN_NON_SUBTASK:
            return state_str

        state_str = state_str + '_' + subtask

        return state_str

    def get_mdp_state_idx(self, mdp_state_key):
        if mdp_state_key not in self.state_dict:
            return None
        else:
            return self.state_idx_dict[mdp_state_key]

    def gen_state_dict_key(self, p0_obj, p1_obj, num_item_in_pot, chop_time, wash_time, orders, subtasks):

        player_obj = p0_obj if p0_obj is not None else 'None'

        order_str = None if orders is None else orders
        for order in orders:
            order_str = order_str + '_' + str(order)

        state_strs = []
        for subtask in subtasks:
            state_strs.append(str(player_obj) + '_' + str(num_item_in_pot) + '_' + str(chop_time) + '_' + str(
                wash_time) + '_' + order_str + '_' + subtask)

        return state_strs

    def map_action_to_location(self, world_state, action, obj, p0_obj=None, player_idx=None, counter_drop=True,
                               state_dict=None, occupied_goal=True):

        other_obj = world_state.players[1 - player_idx].held_object.name if world_state.players[
                                                                                1 - player_idx].held_object is not None else 'None'
        pots_states_dict = self.mdp.get_pot_states(world_state)
        location = []

        # If wait becomes true, one player has to wait for the other player to finish its current task and its next task
        WAIT = False
        counter_obj = self.mdp.get_counter_objects_dict(world_state, list(self.mdp.terrain_pos_dict['X']))
        if action == 'pickup' and obj in ['onion', 'plate', 'meat', 'hot_plate']:
            if p0_obj != 'None' and p0_obj != obj and counter_drop:
                WAIT = True
                location += self.drop_item(world_state)
            else:
                location += counter_obj[obj]
                if obj == 'onion':
                    location += self.mdp.get_onion_dispenser_locations()
                elif obj == 'plate':
                    location += self.mdp.get_dish_dispenser_locations()
                elif obj == 'meat':
                    location += self.mdp.get_meat_dispenser_locations()
                elif obj == 'hot_plate':
                    location += (self.mdp.get_sink_status(world_state)['full'] + self.mdp.get_sink_status(world_state)[
                        'ready'])

                    if len(location) == 0:
                        WAIT = True
                        location += self.mdp.get_sink_status(world_state)['empty']

        elif action == 'pickup' and obj == 'garnish':
            if p0_obj != 'steak' and p0_obj != 'None' and counter_drop:
                WAIT = True
                location += self.drop_item(world_state)
            else:
                location += counter_obj[obj]
                location += (self.mdp.get_chopping_board_status(world_state)['full'] +
                             self.mdp.get_chopping_board_status(world_state)['ready'])

                if len(location) == 0:
                    WAIT = True
                    location += self.mdp.get_chopping_board_status(world_state)['empty']
                    return location, WAIT

        elif action == 'pickup' and obj == 'steak':
            if p0_obj != 'hot_plate' and p0_obj != 'None' and counter_drop:
                WAIT = True
                location += self.drop_item(world_state)
            else:
                location += counter_obj[obj]
                location += (self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_ready_pots(
                    pots_states_dict) + self.mdp.get_full_pots(pots_states_dict))

                if len(location) == 0:
                    WAIT = True
                    location += self.mdp.get_empty_pots(pots_states_dict)
                    return location, WAIT

        elif action == 'drop':
            if p0_obj == obj or p0_obj == 'None':
                if obj == 'meat':
                    location += self.mdp.get_empty_pots(pots_states_dict)
                    if len(location) == 0 and other_obj != 'meat' and occupied_goal:
                        WAIT = True
                        location += self.mdp.get_ready_pots(pots_states_dict) + self.mdp.get_cooking_pots(
                            pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)

                elif obj == 'onion':
                    location += self.mdp.get_chopping_board_status(world_state)['empty']
                    if len(location) == 0 and other_obj != 'onion' and occupied_goal:
                        WAIT = True
                        location += self.mdp.get_chopping_board_status(world_state)['ready'] + \
                                    self.mdp.get_chopping_board_status(world_state)['full']

                elif obj == 'plate':
                    location += self.mdp.get_sink_status(world_state)['empty']
                    if len(location) == 0 and other_obj != 'plate' and occupied_goal:
                        WAIT = True
                        location += self.mdp.get_sink_status(world_state)['ready'] + \
                                    self.mdp.get_sink_status(world_state)['full']

                elif (obj == 'hot_plate' or obj == 'steak') and counter_drop:
                    WAIT = True
                    location += self.drop_item(world_state)

            else:
                WAIT = True

            if (len(location) == 0 or WAIT) and counter_drop:
                location += self.drop_item(world_state)
            if (len(location) == 0 or WAIT) and not counter_drop:
                location += world_state.players[player_idx].position

        elif action == 'deliver':
            if p0_obj != 'dish' and p0_obj != 'None':
                WAIT = True
                location += self.drop_item(world_state)
            elif p0_obj == 'None':
                WAIT = True
                location += counter_obj[obj]
                if len(location) == 0:
                    location += self.mdp.get_key_objects_locations()
            else:
                location += self.mdp.get_serving_locations()

        elif action == 'chop' and obj == 'onion':
            if p0_obj != 'None':
                WAIT = True
                location += self.drop_item(world_state)
            else:
                location += self.mdp.get_chopping_board_status(world_state)['full']
                if len(location) == 0:
                    WAIT = True
                    location += self.mdp.get_chopping_board_status(world_state)['empty'] + \
                                self.mdp.get_chopping_board_status(world_state)['ready']

        elif action == 'heat' and obj == 'hot_plate':
            if p0_obj != 'None':
                WAIT = True
                location += self.drop_item(world_state)
            else:
                location += self.mdp.get_sink_status(world_state)['full']
                if len(location) == 0:
                    WAIT = True
                    location += self.mdp.get_sink_status(world_state)['empty'] + self.mdp.get_sink_status(world_state)[
                        'ready']
        else:
            print(p0_obj, action, obj)
            ValueError()

        return location, WAIT

    def _shift_same_goal_pos(self, new_positions, change_idx):

        pos = new_positions[change_idx][0]
        ori = new_positions[change_idx][1]
        new_pos = pos
        new_ori = ori
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

    def mdp_action_state_to_world_state(self, action_idx, ori_state_idx, env, with_argmin=False):
        ori_world_state = env.world_state
        new_world_state = ori_world_state.deepcopy()
        ori_mdp_state_key = self.get_key_from_value(self.state_idx_dict, ori_state_idx)
        mdp_state_obj = self.state_dict[ori_mdp_state_key]
        action = self.get_key_from_value(self.action_idx_dict, action_idx)
        robot_obj = env.robot_state.holding
        possible_agent_motion_goals, AI_WAIT = self.map_action_to_location(ori_world_state, self.action_dict[action][0],
                                                                           self.action_dict[action][1],
                                                                           p0_obj=robot_obj,
                                                                           player_idx=self.agent_index)
        if new_world_state.players[(1 - self.agent_index)].held_object is not None:
            # human_obj = env.human_state.holding
            human_obj = new_world_state.players[(1 - self.agent_index)].held_object.name
        else:
            human_obj = 'None'
        possible_human_motion_goals, HUMAN_WAIT = self.map_action_to_location(
            ori_world_state,
            self.action_dict[mdp_state_obj[-1]][0],
            self.action_dict[mdp_state_obj[-1]][1],
            p0_obj=human_obj,
            player_idx=(1 - self.agent_index)
        )
        # get next world state from human subtask info (aka. mdp action translate into medium level goal position)
        # get next position for AI agent
        agent_cost, agent_feature_pos = self.mp.min_cost_to_feature(ori_world_state.players[0].pos_and_or,
                                                                    possible_agent_motion_goals,
                                                                    with_motion_goal=True)
        # select the feature position that is closest to current player's position in world state
        new_agent_pos = agent_feature_pos if agent_feature_pos is not None else new_world_state.players[
            0].get_pos_and_or()
        human_cost, human_feature_pos = self.mp.min_cost_to_feature(ori_world_state.players[1].pos_and_or,
                                                                    possible_human_motion_goals, with_motion_goal=True)
        new_human_pos = human_feature_pos if human_feature_pos is not None else \
            new_world_state.players[1].get_pos_and_or()
        if new_agent_pos == new_human_pos:
            new_agent_pos, new_human_pos = self._shift_same_goal_pos([new_agent_pos, new_human_pos],
                                                                     np.argmax(np.array([agent_cost, human_cost])))

        # update next position for AI agent
        if new_world_state.players[0].has_object():
            new_world_state.players[0].remove_object()
        if mdp_state_obj[0] != 'None' and mdp_state_obj[0] != 'dish':
            new_world_state.players[0].held_object = ObjectState(mdp_state_obj[0], new_agent_pos)
        new_world_state.players[0].update_pos_and_or(new_agent_pos[0], new_agent_pos[1])

        # update next position for human
        if new_world_state.players[1].has_object():
            new_world_state.players[1].remove_object()
        mdp_state_obj_action = mdp_state_obj[-1].split('_')[0]
        mdp_state_obj_obj = '_'.join(mdp_state_obj[-1].split('_')[1:])
        if (
                mdp_state_obj_action != 'pickup' and mdp_state_obj_action != 'chop' and mdp_state_obj_action != 'heat') and mdp_state_obj_obj != 'dish':
            new_world_state.players[1].held_object = ObjectState(mdp_state_obj_obj, new_human_pos)
        new_world_state.players[1].update_pos_and_or(new_human_pos[0], new_human_pos[1])

        total_cost = max([agent_cost, human_cost])  # in rss paper is max
        if AI_WAIT or HUMAN_WAIT:
            # if wait, then cost is sum of current tasks cost and one player's next task cost (est. as half map area
            # length)
            total_cost = agent_cost + human_cost + ((self.mdp.width - 1) + (self.mdp.height - 1)) / 2

        if with_argmin:
            return new_world_state, total_cost, [new_agent_pos, new_human_pos]

        return new_world_state, total_cost

    def map_state_to_subtask(self, mdp_state_obj, next_mdp_state_obj):
        """
        The next state's subtask can be completed or not. The key is to know what the robot's subtasks can be.
        """

        # edge case: if robot keeps holding plate, then return action subtask to pickup plate
        if mdp_state_obj[0] == 'plate' and next_mdp_state_obj[0] == 'plate':
            return ['pickup'], ['plate']

        human_subtask = mdp_state_obj[-1]
        human_action = human_subtask.split('_')[0]
        human_obj = '_'.join(human_subtask.split('_')[1:])
        actions, objs = [], []

        for i in range(4):
            state_obj = mdp_state_obj[i]
            if state_obj != next_mdp_state_obj[i]:
                if i == 0:  # ai agent holding
                    if next_mdp_state_obj[i] == 'None':  # dropped object
                        if state_obj == 'dish':
                            actions.append('deliver')
                        else:
                            actions.append('drop')
                        objs.append(state_obj)

                    elif state_obj == 'None':
                        actions.append('pickup')
                        objs.append(next_mdp_state_obj[i])

                    elif state_obj == 'hot_plate' and next_mdp_state_obj[i] == 'steak':
                        actions.append('pickup')
                        objs.append('steak')

                    elif state_obj == 'steak' and next_mdp_state_obj[i] == 'dish':
                        actions.append('pickup')
                        objs.append('garnish')

                elif i == 2:
                    tmp_state_obj = state_obj if state_obj != 'None' else -1
                    tmp_next_state_obj = next_mdp_state_obj[i] if next_mdp_state_obj[i] != 'None' else -1
                    if (tmp_state_obj < tmp_next_state_obj) and (
                            tmp_state_obj >= 0):  # and (human_subtask != 'chop_onion'): # status of chop board
                        actions.append('chop')
                        objs.append('onion')

                elif i == 3:
                    tmp_state_obj = state_obj if state_obj != 'None' else -1
                    tmp_next_state_obj = next_mdp_state_obj[i] if next_mdp_state_obj[i] != 'None' else -1
                    if (tmp_state_obj < tmp_next_state_obj) and (
                            tmp_state_obj >= 0):  # and (human_subtask != 'heat_hot_plate'): # status of sink
                        actions.append('heat')
                        objs.append('hot_plate')

        if len(actions) > 1:
            if human_obj in objs:
                rmv_idx = objs.index(human_obj)
                objs.pop(rmv_idx)
                actions.pop(rmv_idx)

        if len(actions) == 0:
            agent_actions, _ = self.stochastic_state_transition(mdp_state_obj[0], mdp_state_obj[1:-1],
                                                                human_state=next_mdp_state_obj[-1])

            for agent_action in agent_actions:
                action, obj = agent_action.split('_')[0], '_'.join(agent_action.split('_')[1:])
                actions.append(action)
                objs.append(obj)

        return actions, objs

    def mdp_state_to_world_state(self, ori_state_idx, next_state_idx, ori_world_state, with_argmin=False,
                                 cost_mode='max', consider_wait=True, occupied_goal=True):
        ori_mdp_state_key = self.get_key_from_value(self.state_idx_dict, ori_state_idx)
        mdp_state_obj = self.state_dict[ori_mdp_state_key]

        next_mdp_state_key = self.get_key_from_value(self.state_idx_dict, next_state_idx)
        next_mdp_state_obj = self.state_dict[next_mdp_state_key]

        new_world_states = []
        if mdp_state_obj == next_mdp_state_obj:
            if with_argmin:
                new_world_states.append([ori_world_state, 0, [ori_world_state.players[0].get_pos_and_or(),
                                                              ori_world_state.players[1].get_pos_and_or()]])
            else:
                new_world_states.append([ori_world_state, 0])

            return np.array(new_world_states, dtype=object)

        # compute the robot's action such that we know what the world state should be like as the robot moves to the
        # next world position based on its action. Can have more than one outcome.
        agent_actions, agent_action_objs = self.map_state_to_subtask(mdp_state_obj, next_mdp_state_obj)

        for agent_action, agent_action_obj in zip(agent_actions, agent_action_objs):
            new_world_state = ori_world_state.deepcopy()

            # get the human's action to location
            if new_world_state.players[abs(1 - self.agent_index)].held_object is not None:
                human_obj = new_world_state.players[abs(1 - self.agent_index)].held_object.name
            else:
                human_obj = 'None'

            possible_human_motion_goals, HUMAN_WAIT = self.map_action_to_location(
                ori_world_state,
                self.action_dict[mdp_state_obj[-1]][0],
                self.action_dict[mdp_state_obj[-1]][1],
                p0_obj=human_obj,
                player_idx=(abs(1 - self.agent_index)),
                occupied_goal=occupied_goal
            )  # get next world state from human subtask info (mdp action translate into medium level goal position)
            human_cost, human_feature_pos = self.mp.min_cost_to_feature(
                ori_world_state.players[1 - self.agent_index].pos_and_or, possible_human_motion_goals,
                with_motion_goal=True)
            new_human_pos = human_feature_pos if human_feature_pos is not None else new_world_state.players[
                1 - self.agent_index].get_pos_and_or()

            # carry out the human's subtask and update the world
            if self.agent_index == 0:
                new_world_state = self.jmp.derive_state(ori_world_state, (
                    ori_world_state.players[self.agent_index].pos_and_or, new_human_pos), [('stay', 'interact')])
            else:
                new_world_state = self.jmp.derive_state(ori_world_state, (
                    new_human_pos, ori_world_state.players[self.agent_index].pos_and_or), [('interact', 'stay')])

            # compute the robot's action to location given that the human has taken their action
            possible_agent_motion_goals, AI_WAIT = self.map_action_to_location(
                new_world_state,
                agent_action,
                agent_action_obj,
                p0_obj=(new_world_state.players[self.agent_index].held_object.name if
                        new_world_state.players[self.agent_index].held_object is not None else 'None'),
                player_idx=self.agent_index,
                occupied_goal=occupied_goal
            )
            # get next position for AI agent
            agent_cost, agent_feature_pos = self.mp.min_cost_to_feature(
                new_world_state.players[self.agent_index].pos_and_or,
                possible_agent_motion_goals,
                with_motion_goal=True
            )  # select the feature position that is closest to current player's position in world state
            new_agent_pos = agent_feature_pos if agent_feature_pos is not None else \
                new_world_state.players[self.agent_index].get_pos_and_or()

            # check if the goal overlaps, if so, move the human out of the way as the human action is already performed
            if new_agent_pos == new_human_pos:
                new_agent_pos, new_human_pos = self._shift_same_goal_pos(
                    [new_agent_pos, new_human_pos], 1 - self.agent_index
                )

            # get new world with agent's action performed
            if self.agent_index == 0:
                new_world_state = self.jmp.derive_state(
                    new_world_state,
                    (new_agent_pos, new_human_pos),
                    [('interact', 'stay')]
                )
            else:
                new_world_state = self.jmp.derive_state(
                    new_world_state,
                    (new_human_pos, new_agent_pos),
                    [('stay', 'interact')]
                )

            # add interaction cost or stay cost when goal overlaps
            agent_cost += 1
            human_cost += 1

            if cost_mode == 'average':
                total_cost = sum([agent_cost, human_cost]) / 2
            elif cost_mode == 'sum':
                total_cost = sum([agent_cost, human_cost])
            elif cost_mode == 'max':
                total_cost = max([agent_cost, human_cost])  # in rss paper is max
            elif cost_mode == 'robot':
                total_cost = agent_cost
            elif cost_mode == 'human':
                total_cost = human_cost

            if (AI_WAIT or HUMAN_WAIT) and consider_wait:
                if AI_WAIT:
                    total_cost = human_cost
                if HUMAN_WAIT:
                    total_cost = agent_cost

            if with_argmin:
                new_world_states.append(
                    [new_world_state, total_cost, [new_agent_pos, new_human_pos], [AI_WAIT, HUMAN_WAIT]])
            else:
                new_world_states.append([new_world_state, total_cost])

        return np.array(new_world_states, dtype=object)

    def world_to_state_keys(self, world_state, player, other_player, belief, env):
        mdp_state_keys = []
        used_belief = []
        for i, b in enumerate(self.subtask_dict):
            mdp_state_key = self.world_state_to_mdp_state_key(
                world_state, player, other_player, env, self.get_key_from_value(self.subtask_idx_dict, i)
            )
            if self.get_mdp_state_idx(mdp_state_key) is not None:
                mdp_state_keys.append(self.world_state_to_mdp_state_key(
                    world_state, player, other_player, env, self.get_key_from_value(self.subtask_idx_dict, i))
                )
                used_belief.append(i)
        return [mdp_state_keys, used_belief]

    def joint_action_cost(self, world_state, goal_pos_and_or, COST_OF_STAY=1, RETURN_PLAN=False, PLAN_COST='short'):
        joint_action_plan, end_motion_state, plan_costs = self.jmp.get_low_level_action_plan(
            world_state.players_pos_and_or, goal_pos_and_or, merge_one=True)

        if len(joint_action_plan) == 0:
            return (Action.INTERACT, None), 0

        num_of_non_stay_actions = len([a for a in joint_action_plan if a[0] != Action.STAY])
        num_of_stay_actions = len([a for a in joint_action_plan if a[0] == Action.STAY])

        if PLAN_COST == 'short':
            total_cost = min(plan_costs)
        elif PLAN_COST == 'average':
            total_cost = sum(plan_costs) / 2 if sum(plan_costs) > 0 else 0
        elif PLAN_COST == 'max':
            total_cost = max(plan_costs)
        elif PLAN_COST == 'robot':
            total_cost = plan_costs[0]
        elif PLAN_COST == 'human':
            total_cost = plan_costs[1]
        else:
            total_cost = max(plan_costs)

        if RETURN_PLAN:
            return np.array(joint_action_plan), total_cost

        # num_of_non_stay_actions+num_of_stay_actions*COST_OF_STAY # in rss paper is max(plan_costs)
        return joint_action_plan[0], total_cost

    def step(self, env, mdp_state_keys_and_belief, belief, agent_idx, low_level_action=False, observation=None):
        """
        Compute plan cost that starts from the next qmdp state defined as next_state_v().
        Compute the action cost of excuting a step towards the next qmdp state based on the
        current low level state information.

        next_state_v: shape(len(belief), len(action_idx)). If the low_level_action is True,
            the action_dic will be representing the 6 low level action index (north, south...).
            If the low_level_action is False, it will be the action_dict (pickup_onion, pickup_soup...).
        """
        start_time = time.time()
        if low_level_action:
            next_state_v = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
            action_cost = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
            qmdp_q = np.zeros((Action.NUM_ACTIONS, len(belief)), dtype=float)

        else:
            next_state_v = np.zeros((len(belief), len(self.action_dict)), dtype=float)
            action_cost = np.zeros((len(belief), len(self.action_dict)), dtype=float)
            qmdp_q = np.zeros((len(self.action_dict), len(belief)), dtype=float)
        # ml_action_to_low_action = np.zeros()

        # for each subtask, obtain next mdp state but with low level location based on finishing excuting current
        # action and subtask
        nxt_possible_mdp_state = []
        nxt_possible_world_state = []
        ml_action_to_low_action = []

        mdp_state_keys = mdp_state_keys_and_belief[0]
        used_belief = mdp_state_keys_and_belief[1]
        for i, mdp_state_key in enumerate(mdp_state_keys):
            mdp_state_idx = self.get_mdp_state_idx(mdp_state_key)
            if mdp_state_idx is not None and belief[used_belief[i]] > 0.01:
                # returns array(action idx), array(next_state_idx)
                agent_action_idx_arr, next_mdp_state_idx_arr = \
                    np.where(self.transition_matrix[:, mdp_state_idx] > 0.000001)
                nxt_possible_mdp_state.append([agent_action_idx_arr, next_mdp_state_idx_arr])
                for j, action_idx in enumerate(agent_action_idx_arr):  # action_idx is encoded subtask action
                    next_state_idx = next_mdp_state_idx_arr[j]
                    after_action_world_state, cost, goals_pos = self.mdp_action_state_to_world_state(action_idx,
                                                                                                     mdp_state_idx, env,
                                                                                                     with_argmin=True)
                    value_cost = self.compute_V(after_action_world_state,
                                                self.get_key_from_value(self.state_idx_dict, next_state_idx),
                                                belief_prob=belief, belief_idx=used_belief[i], search_depth=25,
                                                search_time_limit=0.01)
                    joint_action, one_step_cost = self.joint_action_cost(
                        world_state, after_action_world_state.players_pos_and_or
                    )
                    if one_step_cost > 1000000:
                        one_step_cost = 1000000
                    if not low_level_action:
                        # action_idx: are subtask action dictionary index
                        next_state_v[i, action_idx] += \
                            value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]

                        # compute one step cost with joint motion considered
                        action_cost[i, action_idx] -= (one_step_cost) * self.transition_matrix[
                            action_idx, mdp_state_idx, next_state_idx]
                    else:
                        next_state_v[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] += (
                                value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx])

                        # compute one step cost with joint motion considered
                        action_cost[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] -= \
                            one_step_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]

        q = self.compute_Q(belief, next_state_v, action_cost)
        print('q value =', q)
        print('next_state_value:', next_state_v)
        print('action_cost:', action_cost)
        action_idx = self.get_best_action(q)
        print('get_best_action =', action_idx, '=', self.get_key_from_value(self.action_idx_dict, action_idx))
        print("It took {} seconds for this step".format(time.time() - start_time))
        if low_level_action:
            return Action.INDEX_TO_ACTION[action_idx], None, low_level_action
        return action_idx, self.action_dict[self.get_key_from_value(self.action_idx_dict, action_idx)], low_level_action

    def observe(self, world_state, robot_agent, human_agent):
        num_item_in_pot, chop_state, sink_state = 0, -1, -1
        robot_agent_obj, human_agent_obj = None, None
        for obj in world_state.objects.values():
            if obj.name == 'hot_plate' and obj.position in self.mdp.get_sink_locations():
                wash_time = obj.state
                if wash_time > sink_state:
                    sink_state = wash_time
            elif obj.name == 'steak' and obj.position in self.mdp.get_pot_locations():
                _, _, cook_time = obj.state
                if cook_time > 0:
                    num_item_in_pot = 1
            elif obj.name == 'garnish' and obj.position in self.mdp.get_chopping_board_locations():
                chop_time = obj.state
                if chop_time > chop_state:
                    chop_state = chop_time

        if chop_state < 0:
            chop_state = None
        if sink_state < 0:
            sink_state = None

        if robot_agent.held_object is not None:
            robot_agent_obj = robot_agent.held_object.name
        if human_agent.held_object is not None:
            human_agent_obj = human_agent.held_object.name

        return [num_item_in_pot, chop_state, sink_state], robot_agent_obj, human_agent_obj

    def kb_to_state_info(self, kb):
        num_item_in_pot = 0
        pots = kb['pot_states']
        non_emtpy_pots = pots['cooking'] + pots['ready']
        if len(non_emtpy_pots) > 0:
            num_item_in_pot = 1

        chop_time = -1
        non_empty_boards = kb['chop_states']['ready'] + kb['chop_states']['full']
        if len(non_empty_boards) > 0:
            if kb[non_empty_boards[0]] is not None:
                chop_time = kb[non_empty_boards[0]].state
            else:
                raise ValueError()

        wash_time = -1
        non_empty_sink = kb['sink_states']['ready'] + kb['sink_states']['full']
        if len(non_empty_sink) > 0:
            if kb[non_empty_sink[0]] is not None:
                wash_time = kb[non_empty_sink[0]].state
            else:
                raise ValueError()

        robot_obj = kb['other_player']['holding'] if kb['other_player']['holding'] is not None else 'None'

        return num_item_in_pot, chop_time, wash_time, robot_obj

    def belief_update(self, env, belief_vector, prev_dist_to_feature, sim_human_agent, greedy=False, vision_limit=False,
                      prev_max_belief=None):
        """
        Update belief based on both human player's game logic and also it's current position and action.
        Belief shape is an array with size equal the length of subtask_dict.
        human_player is the human agent class that is in the simulator.
        NOTE/TODO: the human_player needs to be simulated when we later use an actual human to run experiments.
        """
        new_prev_dist_to_feature = {}
        num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(
            sim_human_agent.human_sim.knowledge_base)

        print('Robot understanding of human obs = ', num_item_in_pot, chop_time, wash_time)
        start_time = time.time()

        distance_trans_belief = np.zeros((len(belief_vector), len(belief_vector)), dtype=float)
        human_pos_and_or = env.human_state.ml_state

        subtask_key = np.array([self.get_key_from_value(self.subtask_idx_dict, i) for i in range(len(belief_vector))])

        human_obj = env.human_state.holding
        game_logic_prob = np.zeros((len(belief_vector)), dtype=float)
        dist_belief_prob = np.zeros((len(belief_vector)), dtype=float)
        print('subtasks:', self.subtask_dict.keys())
        for i, belief in enumerate(belief_vector):
            # estimating next subtask based on game logic
            game_logic_prob[i] = self._is_valid_object_subtask_pair(subtask_key[i], num_item_in_pot, chop_time,
                                                                    wash_time, vision_limit=vision_limit,
                                                                    human_obj=human_obj, other_agent_holding=robot_obj,
                                                                    prev_subtask=prev_max_belief) * 1.0

            # tune subtask estimation based on current human's position and action (use minimum distance between
            # features)
            feature_pos = env.map_action_to_location(
                (self.subtask_dict[subtask_key[i]][0], self.subtask_dict[subtask_key[i]][1]),
                human_pos_and_or)  # p0_obj=human_obj, player_idx=(1-self.agent_index), counter_drop=True)

            human_dist_cost = len(self.mlp.compute_single_agent_astar_path(env.human_state.ml_state, feature_pos[0:2],
                                                                           end_facing=feature_pos[2]))

            if str(feature_pos) not in prev_dist_to_feature:
                prev_dist_to_feature[str(feature_pos)] = human_dist_cost

            # TODO: the offset of dist_belief_prob to avoid being smaller than 0, is the average distance between
            #  features.
            dist_belief_prob[i] = (prev_dist_to_feature[str(feature_pos)] - human_dist_cost)

            if human_dist_cost == 0:
                dist_belief_prob[i] = 1

            # update distance to feature
            new_prev_dist_to_feature[str(feature_pos)] = human_dist_cost

        prev_dist_to_feature = new_prev_dist_to_feature
        print('prev_dist_to_feature =', prev_dist_to_feature)

        # Note: prev_dist_to_feature is not updating all positions, but only the possible goals. Hence the game logic
        # should multipy the distance first then the distance is normalized
        if game_logic_prob.sum() > 0.0:
            game_logic_prob /= game_logic_prob.sum()
        else:
            # since no next subtask for human, set the pickup plate to 1 (this matches the human logic)
            game_logic_prob[2] = 1.0
        game_logic_prob[game_logic_prob <= 0.000001] = 0.000001
        print('game_logic_prob =', game_logic_prob)
        print('dist_belief_prob =', dist_belief_prob)
        # let all dist_belief_prob > 0
        offset = min(dist_belief_prob)
        for i in range(len(dist_belief_prob)):
            if dist_belief_prob[i] != 0.0:
                dist_belief_prob[i] -= offset
        # only update the belief if the distance differences provides information (aka not all zeros)
        dist_belief_prob *= game_logic_prob
        if dist_belief_prob.sum() > 0.0:
            dist_belief_prob[dist_belief_prob <= 0.000001] = 0.000001
            dist_belief_prob /= dist_belief_prob.sum()
            print('dist_belief_prob =', dist_belief_prob)

        print('original belief:', belief_vector)
        new_belief = belief_vector * game_logic_prob
        new_belief /= new_belief.sum()
        new_belief = new_belief * 0.7 + dist_belief_prob * 0.3

        new_belief /= new_belief.sum()
        count_small = len(new_belief[new_belief <= 0.01])
        new_belief[new_belief > 0.01] -= (0.01 * count_small)
        new_belief[new_belief <= 0.01] = 0.01
        print('new_belief =', new_belief)
        print('max belif =', list(self.subtask_dict.keys())[np.argmax(new_belief)])

        return new_belief, prev_dist_to_feature

    def compute_V(self, next_world_state, mdp_state_key, belief_prob=None, belief_idx=None, search_depth=200,
                  search_time_limit=10, add_rewards=False, gamma=False, debug=False):
        start_time = time.time()
        next_world_state_str = str(next_world_state)
        flag = True  #False

        if belief_prob is not None:
            if belief_prob[belief_idx] <= 0.03:
                return 0

        if flag:
            delivery_horizon = 2
            h_fn = Steak_Heuristic(self.mp).simple_heuristic
            start_world_state = next_world_state.deepcopy()
            if start_world_state.order_list is None:
                start_world_state.order_list = ["any"] * delivery_horizon
            else:
                start_world_state.order_list = start_world_state.order_list[:delivery_horizon]

            expand_fn = lambda state, ori_state_key: self.get_successor_states(state, ori_state_key,
                                                                               add_rewards=add_rewards)
            goal_fn = lambda ori_state_key: len(self.state_dict[ori_state_key][4:-1]) == 0
            heuristic_fn = lambda state: h_fn(state)

            search_problem = SearchTree(start_world_state, goal_fn, expand_fn, heuristic_fn, debug=self.debug)
            path_end_state, cost, over_limit, end_state_key = search_problem.bounded_A_star_graph_search(
                qmdp_root=mdp_state_key, info=False, cost_limit=search_depth, time_limit=search_time_limit, gamma=gamma,
                debug=self.debug)

            if over_limit or cost > 10000:
                cost = self.optimal_plan_cost(path_end_state, cost)

            if add_rewards:
                state_obj = self.state_dict[end_state_key].copy()
                player_obj, pot_state, chop_state, sink_state = state_obj[:4]
                remaining_orders = state_obj[4:-1]
                human_subtask = state_obj[-1]
                human_action = human_subtask.split('_')[0]
                human_obj = '_'.join(human_subtask.split('_')[1:])
                if human_action in ['pickup', 'chop', 'heat'] and human_obj not in ['garnish', 'steak']:
                    human_obj = 'None'
                elif human_action == 'pickup' and human_obj == 'garnish':
                    human_obj = 'steak'
                elif human_action == 'pickup' and human_obj == 'steak':
                    human_obj = 'hot_plate'

                delta_cost = (-40) * len(remaining_orders)
                if chop_state == 'None' or chop_state is None:
                    chop_state = 0
                else:
                    chop_state += 3  #1
                if sink_state == 'None' or sink_state is None:
                    sink_state = 0
                else:
                    sink_state += 3  #1

                # the rewards are given in two phases. One where you prep and the other where you collect and plate.
                delta_cost += (5 * pot_state + 1 * chop_state + 1 * sink_state)
                # else:
                if 'hot_plate' in [player_obj, human_obj]:
                    delta_cost += 12  # 9
                if 'steak' in [player_obj, human_obj]:
                    delta_cost += 20  # 18
                if 'dish' in [player_obj, human_obj]:
                    delta_cost += 30

                if player_obj not in ['None', None, 'hot_plate', 'steak', 'dish']:
                    delta_cost += 1
                if human_obj not in ['None', None, 'hot_plate', 'steak', 'dish']:
                    delta_cost += 1

                if len(remaining_orders) == 0:
                    # set to 100 such that would not optimize after termination and focus only on decreasing cost
                    delta_cost = 100

                if self.debug:
                    print('world info:', player_obj, pot_state, chop_state, sink_state, remaining_orders, human_obj)
                    print('delta_cost:cost', delta_cost, cost)

                cost -= delta_cost * 2 * (0.9 ** (2 - len(remaining_orders)))
            self.world_state_cost_dict[(next_world_state_str, mdp_state_key)] = cost

        return max(
            (self.mdp.height * self.mdp.width) * 5 - self.world_state_cost_dict[(next_world_state_str, mdp_state_key)],
            0)

    def optimal_plan_cost(self, start_world_state, start_cost):
        self.world_state_to_mdp_state_key(start_world_state, start_world_state.players[0], start_world_state.players[1],
                                          subtask)

    def compute_Q(self, b, v, c, gamma=0.9):
        print('b =', b)
        print('v =', v)
        print('c =', c)

        return b @ ((v * gamma) + c)

    def get_best_action(self, q):
        return np.argmax(q)

    def save_to_file(self, filename):
        print("In save_to_file")
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def init_mdp(self, orders):
        self.init_actions()
        self.init_human_aware_states(order_list=orders)
        self.init_transition()
        self.init_cost()
        self.init_reward()

    def compute_mdp(self, orders, filename=None, ):
        start_time = time.time()

        # final_filepath = os.path.join(PLANNERS_DIR, filename)
        self.init_mdp(orders)
        self.num_states = len(self.state_dict)
        self.num_actions = len(self.action_dict)

        with open(os.path.join(PLANNERS_DIR, 'value_policy_matrix.pkl'), 'rb') as f:
            loaded = pickle.load(f)
            self.value_matrix = loaded[0]
            self.policy_matrix = loaded[1]

        return

    def precompute_future_cost(self):
        # computes an estimated distance cost for each state,action
        # this is later indexed in qmdp for evaluating future cost

        from_saved = True
        if from_saved:
            with open(os.path.join(PLANNERS_DIR, 'future_dist_saved.pkl'), 'rb') as f:
                loaded = pickle.load(f)
                self.dist_value_matrix = loaded
            return

        # copy the value matrix, rollout the policy, sum up distances
        self.dist_value_matrix = np.ones(self.value_matrix.shape) * 1000000

        for start_state_idx, v in enumerate(self.dist_value_matrix):
            curr_state_idx = start_state_idx
            future_dist_cost = 0
            policy_a = self.policy_matrix[curr_state_idx]
            orders_left = len(self.state_dict[self.get_key_from_value(self.state_idx_dict, curr_state_idx)][4:-1])

            if orders_left == 0:
                future_dist_cost = 0
                self.dist_value_matrix[curr_state_idx] = future_dist_cost
                continue

            while orders_left != 0:
                if orders_left == 0:
                    break
                policy_a = self.policy_matrix[curr_state_idx]
                action = self.get_key_from_value(self.action_idx_dict,
                                                 policy_a)
                state = self.get_key_from_value(self.state_idx_dict,
                                                curr_state_idx)
                human_action_obj = self.state_dict[state][-1]  # self.get_action_obj_from_state(state)
                # take max of human action and robot action to upper bound cost
                human_action_cost = self.action_to_features(human_action_obj)
                robot_action_cost = self.action_to_features(action)
                action_cost = max(human_action_cost, robot_action_cost)

                possible_next_states = np.where(
                    self.transition_matrix[policy_a,
                    curr_state_idx] > 0.000001)[0]
                next_state = random.choice(
                    possible_next_states
                )  # choose first state TODO: review that we don't have more information
                future_dist_cost += action_cost
                curr_state_idx = next_state
                orders_left = len(
                    self.state_dict[
                        self.get_key_from_value(self.state_idx_dict,
                                                curr_state_idx)][4:-1])

            print(
                f'future cost {self.get_key_from_value(self.state_idx_dict, start_state_idx)}: {future_dist_cost}'
            )
            self.dist_value_matrix[start_state_idx] = future_dist_cost

        return

    def action_to_features(self, action):
        feat_key = None
        if action == 'drop_onion':
            # fridge to stove
            feat_key = 'G_K'
        elif action == 'pickup_meat':
            feat_key = 'F_C'
        elif action == 'deliver_dish':
            # garnish to table
            feat_key = 'K_T'
        elif action == 'pickup_steak':
            # stove, table to dish station
            # go from center
            feat_key = 'P_C'
        elif action == 'drop_steak':
            feat_key = 'Drop'
        elif action == 'drop_dish':
            feat_key = 'Drop'
        elif action == 'pickup_soup':
            # dish to stove
            feat_key = 'B_S'
        elif action == 'pickup_onion':
            # use center point
            feat_key = 'G_C'
        elif action == 'pickup_plate':
            feat_key = 'D_C'
        elif action == 'pickup_hot_plate':
            feat_key = 'W_C'
        elif action == 'drop_meat':
            feat_key = 'F_P'
        elif action == 'drop_plate':
            feat_key = 'D_W'
        elif action == 'drop_hot_plate':
            feat_key = 'Drop'
        elif action == 'chop_onion':
            dist = 2
        elif action == 'heat_hot_plate':
            dist = 2
        elif action == 'pickup_garnish':
            feat_key = 'P_K'

        if feat_key is not None:
            dist = self.mlp.dist_between[feat_key]
        return dist
