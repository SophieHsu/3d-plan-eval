import copy
import random
import time

import numpy as np

from src.environment.lsi_env import LsiEnv
from src.mdp.action import Action
from src.planners.high_level_mdp import HighLevelMdpPlanner
from src.planners.mid_level_motion import AStarMotionPlanner


class HumanSubtaskQMDPPlanner(HighLevelMdpPlanner):

    def __init__(self, mdp, mlp: AStarMotionPlanner):
        super().__init__(mdp)

        self.world_state_cost_dict = {}
        self.subtask_dict = {}
        self.subtask_idx_dict = {}
        self.mlp = mlp

    def init_human_aware_states(self, state_idx_dict=None, order_list=None):
        """
        States: agent 0 holding object, number of item in pot, order list, agent 1 (usually human) holding object, agent 1 subtask.
        """

        # set state dict as [p0_obj, num_item_in_pot, order_list]
        self.init_states(order_list=order_list)

        # add [p1_obj, subtask] to [p0_obj, num_item_in_pot, order_list]
        objects = ['onion', 'soup', 'dish', 'None']  # 'tomato'
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
                        new_key = ori_key + '_' + obj + '_' + subtask[0]
                        new_obj = original_state_dict[ori_key] + [obj] + [
                            subtask[0]
                        ]
                        self.state_dict[new_key] = new_obj  # update value
                        self.state_idx_dict[new_key] = len(self.state_idx_dict)
        return

    def init_transition(self, transition_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros(
            (len(self.action_dict), len(self.state_idx_dict),
             len(self.state_idx_dict)),
            dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]
            for action_key, action_idx in self.action_idx_dict.items():
                normalize_count = 0

                # decode state information
                # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
                p0_state, p1_state, world_info = self.decode_state_info(state_obj)
                # calculate next states for p1 (a.k.a. human)
                p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(
                    p1_state, world_info)

                # calculate next states for p0 (conditioned on p1 (a.k.a. human))
                for p1_nxt_state in p1_nxt_states:
                    action, next_state_key = self.state_transition(
                        p0_state, p1_nxt_world_info, human_state=p1_nxt_state)

                    if action_key == action:
                        next_state_idx = self.state_idx_dict[next_state_key]
                        self.transition_matrix[action_idx, state_idx, next_state_idx] += 1.0

                if np.sum(self.transition_matrix[action_idx, state_idx]) > 0.0:
                    self.transition_matrix[action_idx, state_idx] /= np.sum(
                        self.transition_matrix[action_idx, state_idx])

        self.transition_matrix[self.transition_matrix == 0.0] = 0.000001

    def get_successor_states(self,
                             start_world_state,
                             start_state_key,
                             debug=False):
        """
        Successor states for qmdp medium-level actions.
        """
        if len(self.state_dict[start_state_key][2:]) <= 2:  # [p0_obj, num_item_in_soup, orders, p1_obj, subtask]
            return []

        ori_state_idx = self.state_idx_dict[start_state_key]
        successor_states = []

        # returns array(action idx), array(next_state_idx)
        agent_action_idx_arr, next_state_idx_arr = np.where(self.transition_matrix[:, ori_state_idx] > 0.000001)
        start_time = time.time()
        for next_action_idx, next_state_idx in zip(agent_action_idx_arr,
                                                   next_state_idx_arr):
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
                    self.get_key_from_value(self.state_idx_dict, next_state_idx),
                    cost,
                    time.time() - start_time
                ))

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
            else:
                return False
        else:
            if obj == 'onion' and subtask == 'drop_onion':
                return True
            elif obj == 'tomato' and subtask == 'drop_tomato':
                return True
            elif (obj == 'dish') and subtask == 'pickup_soup':
                return True
            elif obj == 'soup' and subtask == 'deliver_soup':
                return True
            else:
                return False

    def _is_valid_object_subtask_pair(self, obj, subtask, soup_finish, greedy=False):
        if obj == 'None':
            if greedy is not True and (
                    subtask == 'pickup_dish' or subtask == 'pickup_onion'
            ) and soup_finish <= self.mdp.num_items_for_soup:
                return True
            elif greedy is True and subtask == 'pickup_onion' and soup_finish < self.mdp.num_items_for_soup:
                return True
            elif greedy is True and subtask == 'pickup_dish' and soup_finish == self.mdp.num_items_for_soup:
                return True
            elif subtask == 'pickup_tomato':
                return True
            else:
                return False
        else:
            if obj == 'onion' and subtask == 'drop_onion':
                return True
            elif obj == 'tomato' and subtask == 'drop_tomato':
                return True
            elif (obj == 'dish') and subtask == 'pickup_soup':
                return True
            elif obj == 'soup' and subtask == 'deliver_soup':
                return True
            else:
                return False

    def human_state_subtask_transition(self, human_state, world_info):
        player_obj = human_state[0]
        subtask = human_state[1]
        soup_finish = world_info[0]
        orders = [] if len(world_info) < 2 else world_info[1:]
        next_obj = player_obj
        next_subtasks = []
        next_soup_finish = soup_finish

        if player_obj == 'None':
            if subtask == 'pickup_dish':
                next_obj = 'dish'
                next_subtasks = ['pickup_soup']

            elif subtask == 'pickup_onion':
                next_obj = 'onion'
                next_subtasks = ['drop_onion']

            elif subtask == 'pickup_tomato':
                next_obj = 'tomato'
                next_subtasks = ['drop_tomato']

        else:
            if player_obj == 'onion' and subtask == 'drop_onion' and soup_finish < self.mdp.num_items_for_soup:
                next_obj = 'None'
                next_soup_finish += 1
                next_subtasks = ['pickup_onion', 'pickup_dish']

            elif player_obj == 'onion' and subtask == 'drop_onion' and soup_finish == self.mdp.num_items_for_soup:
                next_obj = 'onion'
                next_subtasks = ['drop_onion']

            elif player_obj == 'tomato' and subtask == 'drop_tomato':
                next_obj = 'None'
                next_soup_finish += 1
                next_subtasks = ['pickup_onion', 'pickup_dish']

            elif (player_obj == 'dish') and subtask == 'pickup_soup':
                next_obj = 'soup'
                next_soup_finish = 0
                next_subtasks = ['deliver_soup']

            elif player_obj == 'soup' and subtask == 'deliver_soup':
                next_obj = 'None'
                next_subtasks = ['pickup_onion',
                                 'pickup_dish']  # 'pickup_tomato'
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
        soup_finish = world_info[0]
        orders = [] if len(world_info) < 2 else world_info[1:]
        other_obj = human_state[0]
        subtask = human_state[1]
        actions = ''
        next_obj = player_obj
        next_soup_finish = soup_finish

        if player_obj == 'None':
            if soup_finish == self.mdp.num_items_for_soup and other_obj != 'dish' and subtask != 'pickup_dish':
                actions = 'pickup_dish'
                next_obj = 'dish'
            elif soup_finish == self.mdp.num_items_for_soup - 1 and other_obj == 'onion' and subtask == 'drop_onion':
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

            elif player_obj == 'dish' and soup_finish >= self.mdp.num_items_for_soup - 1:
                actions = 'pickup_soup'
                next_obj = 'soup'
                next_soup_finish = 0

            elif player_obj == 'dish' and soup_finish < self.mdp.num_items_for_soup - 1:
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
        player_obj = None
        other_player_obj = None
        if player.holding is not None:
            player_obj = player.holding
        if other_player.holding is not None:
            other_player_obj = other_player.holding

        order_str = None if len(state.orders) == 0 else state.orders[0]
        for order in state.orders[1:]:
            order_str = order_str + '_' + str(order)

        num_item_in_pot = state.in_pot

        if order_str is not None:
            state_strs = str(player_obj) + '_' + str(num_item_in_pot) + '_' + \
                         order_str + '_' + str(other_player_obj) + '_' + subtask
        else:
            state_strs = str(player_obj) + '_' + str(num_item_in_pot) + '_' + str(other_player_obj) + '_' + subtask

        return state_strs

    def get_mdp_state_idx(self, mdp_state_key):
        if mdp_state_key not in self.state_dict:
            return None
        else:
            return self.state_idx_dict[mdp_state_key]

    def gen_state_dict_key(self, p0_obj, p1_obj, num_item_in_pot, orders, subtasks):
        # a0 hold, a1 hold,
        order_str = None if orders is None else orders
        for order in orders:
            order_str = order_str + '_' + str(order)

        state_strs = []
        return state_strs

    def get_key_from_value(self, dictionary, state_value):
        try:
            idx = list(dictionary.values()).index(state_value)
        except ValueError:
            return None
        else:
            return list(dictionary.keys())[idx]

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

    def mdp_action_state_to_world_state(self, action_idx, ori_state_idx, env: LsiEnv, with_argmin=False):
        ori_world_state = env.world_state
        new_world_state = copy.deepcopy(ori_world_state)
        ori_mdp_state_key = self.get_key_from_value(self.state_idx_dict,
                                                    ori_state_idx)
        mdp_state_obj = self.state_dict[ori_mdp_state_key]
        action = self.get_key_from_value(self.action_idx_dict, action_idx)

        agent_motion_goal = env.map_action_to_location(self.action_dict[action], env.robot_state.ml_state)
        human_motion_goal = env.map_action_to_location(self.action_dict[mdp_state_obj[-1]], env.human_state.ml_state)

        new_agent_pos = agent_motion_goal if agent_motion_goal is not None else env.robot_state.ml_state
        agent_cost = len(self.mlp.compute_single_agent_astar_path(env.robot_state.ml_state, new_agent_pos[0:2],
                                                                  end_facing=new_agent_pos[2]))

        new_human_pos = human_motion_goal if human_motion_goal is not None else env.human_state.ml_state
        human_cost = len(self.mlp.compute_single_agent_astar_path(env.human_state.ml_state, new_human_pos[0:2],
                                                                  end_facing=new_human_pos[2]))

        # update next position for AI agent
        if new_world_state.players[0].has_object():
            new_world_state.players[0].holding = 'None'
        if mdp_state_obj[0] != 'None' and mdp_state_obj[0] != 'soup':
            new_world_state.players[0].holding = mdp_state_obj[0]
        new_world_state.players[0].ml_state = (
            new_agent_pos[0],
            new_agent_pos[1],
            new_world_state.players[0].ml_state[2]
        )

        # update next position for human
        if new_world_state.players[1].has_object():
            new_world_state.players[1].holding = 'None'
        if mdp_state_obj[-2] != 'None' and mdp_state_obj[-2] != 'soup':
            new_world_state.players[1].holding = mdp_state_obj[-2]
        new_world_state.players[1].ml_state = (
            new_human_pos[0], new_human_pos[1],
            new_world_state.players[1].ml_state[2])

        total_cost = min([agent_cost, human_cost])  # in rss paper is max

        if with_argmin:
            return new_world_state, total_cost, [new_agent_pos, new_human_pos]

        return new_world_state, total_cost

    def world_to_state_keys(self, world_state, player, other_player):
        mdp_state_keys = []
        for i, b in enumerate(self.subtask_dict):
            mdp_state_key = self.world_state_to_mdp_state_key(
                world_state, player, other_player,
                self.get_key_from_value(self.subtask_idx_dict, i))
            if self.get_mdp_state_idx(mdp_state_key) is not None:
                mdp_state_keys.append(
                    self.world_state_to_mdp_state_key(
                        world_state, player, other_player,
                        self.get_key_from_value(self.subtask_idx_dict, i)))
        return mdp_state_keys

    def joint_action_cost(self, world_state, goal_pos_and_or, COST_OF_STAY=1):
        joint_action_plan, end_motion_state, plan_costs = self.jmp.get_low_level_action_plan(
            world_state.players_pos_and_or, goal_pos_and_or, merge_one=True)

        if len(joint_action_plan) == 0:
            return (Action.INTERACT, None), 0

        return joint_action_plan[0], max(plan_costs)

    def init_cost(self):
        self.cost_matrix = np.zeros(
            (len(self.action_dict), len(self.state_dict),
             len(self.state_dict)))

        for action, action_idx in self.action_idx_dict.items():
            for curr_state, curr_state_idx in self.state_idx_dict.items():
                next_states = np.where(self.transition_matrix[action_idx, curr_state_idx] > 0.00001)[0]

                for next_state_idx in next_states:
                    human_action_key = self.get_key_from_value(self.action_idx_dict, action_idx)
                    self.cost_matrix[action_idx][curr_state_idx][next_state_idx] = 5

    def get_action_obj_from_state(self, state_str):
        split = state_str.split('_')
        action_obj = split[-2:]
        return f'{action_obj[0]}_{action_obj[1]}'

    def step(self, env, mdp_state_keys, belief, agent_idx, low_level_action=False):
        """
        Compute plan cost that starts from the next qmdp state defined as next_state_v().
        Compute the action cost of excuting a step towards the next qmdp state based on the
        current low level state information.

        next_state_v: shape(len(belief), len(action_idx)). If the low_level_action is True, 
            the action_dix will be representing the 6 low level action index (north, south...).
            If the low_level_action is False, it will be the action_dict (pickup_onion, pickup_soup...).
        """
        VALUE_OFFSET = (self.mdp.height * self.mdp.width) * 2
        start_time = time.time()
        next_state_v = np.zeros((len(belief), len(self.action_dict)), dtype=float)
        action_cost = np.zeros((len(belief), len(self.action_dict)), dtype=float)

        # for each subtask, obtain next mdp state but with low level location based on
        # finishing excuting current action and subtask
        nxt_possible_mdp_state = []
        nxt_possible_world_state = []
        ml_action_to_low_action = []
        for i, mdp_state_key in enumerate(mdp_state_keys):
            mdp_state_idx = self.get_mdp_state_idx(mdp_state_key)
            if mdp_state_idx is not None:
                # returns array(action idx), array(next_state_idx)
                agent_action_idx_arr, next_mdp_state_idx_arr = \
                    np.where(self.transition_matrix[:, mdp_state_idx] > 0.000001)
                nxt_possible_mdp_state.append(
                    [agent_action_idx_arr, next_mdp_state_idx_arr])
                for j, action_idx in enumerate(agent_action_idx_arr):  # action_idx is encoded subtask action
                    next_state_idx = next_mdp_state_idx_arr[j]
                    after_action_world_state, cost, goals_pos = self.mdp_action_state_to_world_state(
                        action_idx,
                        mdp_state_idx,
                        env,
                        with_argmin=True
                    )

                    value_cost = self.dist_value_matrix[next_state_idx]
                    # joint_action, one_step_cost = self.joint_action_cost
                    # (world_state, after_action_world_state.players_pos_and_or)
                    one_step_cost = cost
                    value_cost = max(0, VALUE_OFFSET - value_cost)

                    if not low_level_action:
                        # action_idx: are subtask action dictionary index
                        human_action_obj = self.get_action_obj_from_state(mdp_state_key)
                        human_action_idx = self.subtask_idx_dict[human_action_obj]

                        next_state_v[human_action_idx, action_idx] += \
                            value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]

                        # compute one step cost with joint motion considered
                        action_cost[human_action_idx, action_idx] -= \
                            one_step_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]

                    else:
                        next_state_v[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] += \
                            value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]

                        action_cost[i, Action.ACTION_TO_INDEX[joint_action[agent_idx]]] -= \
                            one_step_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]

        q = self.compute_Q(belief, next_state_v, action_cost)
        action_idx = self.get_best_action(q)

        if low_level_action:
            return Action.INDEX_TO_ACTION[action_idx], None, low_level_action

        return action_idx, self.action_dict[self.get_key_from_value(
            self.action_idx_dict, action_idx)], low_level_action

    def belief_update(self,
                      env: LsiEnv,
                      belief_vector,
                      prev_dist_to_feature,
                      greedy=False):
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
        human_pos_and_or = env.human_state.ml_state
        agent_pos_and_or = env.robot_state.ml_state

        subtask_key = np.array([
            self.get_key_from_value(self.subtask_idx_dict, i)
            for i in range(len(belief_vector))
        ])

        # get next position for human
        human_obj = env.human_state.holding if env.human_state.holding is not None else 'None'
        game_logic_prob = np.zeros((len(belief_vector)), dtype=float)
        dist_belief_prob = np.full((len(belief_vector)), (self.mdp.height + self.mdp.width) / 2, dtype=float)

        for i, belief in enumerate(belief_vector):
            # estimating next subtask based on game logic
            game_logic_prob[i] = self._is_valid_object_subtask_pair(
                human_obj, subtask_key[i], env.world_state.in_pot,
                greedy=greedy) * 1.0

            # tune subtask estimation based on current human's position and action
            # (use minimum distance between features)
            motion_goal = env.map_action_to_location(
                self.subtask_dict[subtask_key[i]],
                env.human_state.ml_state,
                is_human=True)
            # get next world state from human subtask info (aka. mdp action translate into medium level goal position)
            feature_pos = motion_goal if motion_goal is not None else env.human_state.ml_state
            human_dist_cost = len(self.mlp.compute_single_agent_astar_path(env.human_state.ml_state, feature_pos[0:2],
                                                                           end_facing=feature_pos[2]))
            if str(feature_pos) not in prev_dist_to_feature:
                prev_dist_to_feature[str(feature_pos)] = human_dist_cost

            dist_belief_prob[i] += prev_dist_to_feature[str(feature_pos)] - human_dist_cost

            # update distance to feature
            prev_dist_to_feature[str(feature_pos)] = human_dist_cost

            if game_logic_prob[i] < 0.00001:
                continue

        game_logic_prob /= game_logic_prob.sum()
        dist_belief_prob /= dist_belief_prob.sum()

        game_logic_prob[game_logic_prob < 0.000001] = 0.000001
        dist_belief_prob[dist_belief_prob < 0.000001] = 0.000001

        new_belief = belief_vector * game_logic_prob

        # preference for old belief vs moving towards new goal
        new_belief = new_belief * 0.4 * dist_belief_prob * 0.6

        new_belief /= new_belief.sum()

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
        return r_holding, in_pot, orders, h_holding, h_action, h_object

    def post_mdp_setup(self):
        # computes an estimated distance cost for each state,action
        # this is later indexed in qmdp for evaluating future cost

        # copy the value matrix, rollout the policy, sum up distances
        self.dist_value_matrix = np.ones(self.value_matrix.shape) * 1000000

        for start_state_idx, v in enumerate(self.dist_value_matrix):
            curr_state_idx = start_state_idx
            future_dist_cost = 0
            policy_a = self.policy_matrix[curr_state_idx]
            orders_left = len(
                self.parse_state(
                    self.get_key_from_value(self.state_idx_dict,
                                            curr_state_idx))[2])

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
                human_action_obj = self.get_action_obj_from_state(state)
                # take max of human action and robot action to upper bound cost
                human_action_cost = self.action_to_features(human_action_obj)
                robot_action_cost = self.action_to_features(action)
                action_cost = max(human_action_cost, robot_action_cost)

                possible_next_states = np.where(self.transition_matrix[policy_a, curr_state_idx] > 0.000001)[0]
                next_state = random.choice(
                    possible_next_states
                )  # choose first state TODO: review that we don't have more information
                future_dist_cost += action_cost
                curr_state_idx = next_state
                orders_left = len(self.parse_state(self.get_key_from_value(self.state_idx_dict, curr_state_idx))[2])

            print(
                f'future cost {self.get_key_from_value(self.state_idx_dict, curr_state_idx)}: {future_dist_cost}'
            )
            self.dist_value_matrix[start_state_idx] = future_dist_cost

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

            delivery_horizon = 2
            debug = False
            h_fn = Heuristic(self.mp).simple_heuristic
            start_world_state = next_world_state.deepcopy()
            if start_world_state.order_list is None:
                start_world_state.order_list = ["any"] * delivery_horizon
            else:
                start_world_state.order_list = start_world_state.order_list[:delivery_horizon]

            expand_fn = lambda state, ori_state_key: self.get_successor_states(state, ori_state_key)
            goal_fn = lambda ori_state_key: len(self.state_dict[ori_state_key][2:]) <= 2
            heuristic_fn = lambda state: h_fn(state)

            # TODO: lookup from regular mdp precomputed value iteration
            search_problem = SearchTree(start_world_state,
                                        goal_fn,
                                        expand_fn,
                                        heuristic_fn,
                                        debug=debug)
            path_end_state, cost, over_limit = search_problem.bounded_A_star_graph_search(
                qmdp_root=mdp_state_key, info=False, cost_limit=search_depth)

            if over_limit:
                cost = self.optimal_plan_cost(path_end_state, cost)

            self.world_state_cost_dict[next_world_state_str] = cost

        return (self.mdp.height * self.mdp.width) * 2 - self.world_state_cost_dict[next_world_state_str]

    def optimal_plan_cost(self, start_world_state, start_cost):
        self.world_state_to_mdp_state_key(
            start_world_state,
            start_world_state.players[0],
            start_world_state.players[1],
            subtask
        )

    def compute_Q(self, b, v, c):
        return b @ (v + c)

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

        self.init_mdp()
        self.num_states = len(self.state_dict)
        self.num_actions = len(self.action_dict)

        # TODO: index value iteration in step function to get action

        self.value_iteration()

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
        partially_full_soups = pot_states_dict['onion'][
                                   'partially_full'] + pot_states_dict['tomato']['partially_full']
        num_onions_in_partially_full_pots = sum(
            [state.get_object(loc).state[1] for loc in partially_full_soups])

        # Calculating costs
        num_deliveries_to_go = goal_deliveries - state.num_delivered

        # SOUP COSTS
        total_num_soups_needed = max([0, num_deliveries_to_go])

        soups_on_counters = [
            soup_obj for soup_obj in objects_dict['soup']
            if soup_obj.position not in pot_locations
        ]
        soups_in_transit = player_objects['soup'] + soups_on_counters
        soup_delivery_locations = self.mdp.get_serving_locations()

        num_soups_better_than_pot, total_better_than_pot_soup_cost = \
            self.get_costs_better_than_dispenser(soups_in_transit, soup_delivery_locations, min_pot_delivery_cost,
                                                 total_num_soups_needed, state)

        min_pot_to_delivery_trips = max(
            [0, total_num_soups_needed - num_soups_better_than_pot])
        pot_to_delivery_costs = min_pot_delivery_cost * min_pot_to_delivery_trips

        forward_cost += total_better_than_pot_soup_cost
        forward_cost += pot_to_delivery_costs

        # DISH COSTS
        total_num_dishes_needed = max([0, min_pot_to_delivery_trips])
        dishes_on_counters = objects_dict['dish']
        dishes_in_transit = player_objects['dish'] + dishes_on_counters

        num_dishes_better_than_disp, total_better_than_disp_dish_cost = \
            self.get_costs_better_than_dispenser(dishes_in_transit, pot_locations, min_dish_to_pot_cost,
                                                 total_num_dishes_needed, state)

        min_dish_to_pot_trips = max(
            [0, min_pot_to_delivery_trips - num_dishes_better_than_disp])
        dish_to_pot_costs = min_dish_to_pot_cost * min_dish_to_pot_trips

        forward_cost += total_better_than_disp_dish_cost
        forward_cost += dish_to_pot_costs

        # ONION COSTS
        num_pots_to_be_filled = min_pot_to_delivery_trips - len(
            full_soups_in_pots)
        total_num_onions_needed = num_pots_to_be_filled * 3 - num_onions_in_partially_full_pots
        onions_on_counters = objects_dict['onion']
        onions_in_transit = player_objects['onion'] + onions_on_counters

        num_onions_better_than_disp, total_better_than_disp_onion_cost = \
            self.get_costs_better_than_dispenser(onions_in_transit, pot_locations, min_onion_to_pot_cost,
                                                 total_num_onions_needed, state)

        min_onion_to_pot_trips = max(
            [0, total_num_onions_needed - num_onions_better_than_disp])
        onion_to_pot_costs = min_onion_to_pot_cost * min_onion_to_pot_trips

        forward_cost += total_better_than_disp_onion_cost
        forward_cost += onion_to_pot_costs

        heuristic_cost = forward_cost / 2

        if debug:
            env = OvercookedEnv.from_mdp(self.mdp)
            env.state = state
            # print("\n" + "#"*35)
            print("Current state: (ml timestep {})\n".format(time))

            print(
                "# in transit: \t\t Soups {} \t Dishes {} \t Onions {}".format(
                    len(soups_in_transit), len(dishes_in_transit),
                    len(onions_in_transit)))

            # NOTE Possible improvement: consider cost of dish delivery too when considering if a
            # transit soup is better than dispenser equivalent
            print("# better than disp: \t Soups {} \t Dishes {} \t Onions {}".
                  format(num_soups_better_than_pot,
                         num_dishes_better_than_disp,
                         num_onions_better_than_disp))

            print("# of trips: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".
                  format(min_pot_to_delivery_trips, min_dish_to_pot_trips,
                         min_onion_to_pot_trips))

            print("Trip costs: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".
                  format(pot_to_delivery_costs, dish_to_pot_costs,
                         onion_to_pot_costs))

            print(str(env) + "HEURISTIC: {}".format(heuristic_cost))

        return heuristic_cost

    def get_costs_better_than_dispenser(self, possible_objects,
                                        target_locations, baseline_cost,
                                        num_needed, state):
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
                min_cost = self.motion_planner.min_cost_to_feature(
                    player.pos_and_or, target_locations) - 1
            else:
                # If object is on a counter
                min_cost = self.motion_planner.min_cost_between_features(
                    [obj_pos], target_locations)
            costs_from_transit_locations.append(min_cost)

        costs_better_than_dispenser = [
            cost for cost in costs_from_transit_locations
            if cost <= baseline_cost
        ]
        better_than_dispenser_total_cost = sum(
            np.sort(costs_better_than_dispenser)[:num_needed])
        return len(
            costs_better_than_dispenser), better_than_dispenser_total_cost

    def _calculate_heuristic_costs(self, debug=False):
        """Pre-computes the costs between common trip types for this mdp"""
        pot_locations = self.mdp.get_pot_locations()
        delivery_locations = self.mdp.get_serving_locations()
        dish_locations = self.mdp.get_dish_dispenser_locations()
        onion_locations = self.mdp.get_onion_dispenser_locations()
        tomato_locations = self.mdp.get_tomato_dispenser_locations()

        heuristic_cost_dict = {
            'pot-delivery':
                self.motion_planner.min_cost_between_features(
                    pot_locations, delivery_locations, manhattan_if_fail=True),
            'dish-pot':
                self.motion_planner.min_cost_between_features(
                    dish_locations, pot_locations, manhattan_if_fail=True)
        }

        onion_pot_cost = self.motion_planner.min_cost_between_features(
            onion_locations, pot_locations, manhattan_if_fail=True)
        tomato_pot_cost = self.motion_planner.min_cost_between_features(
            tomato_locations, pot_locations, manhattan_if_fail=True)

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

        full_soups_in_pots = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking'] + \
                             pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
        partially_full_onion_soups = pot_states_dict['onion']['partially_full']
        partially_full_tomato_soups = pot_states_dict['tomato']['partially_full']
        num_onions_in_partially_full_pots = sum([
            state.get_object(loc).state[1]
            for loc in partially_full_onion_soups
        ])
        num_tomatoes_in_partially_full_pots = sum([
            state.get_object(loc).state[1]
            for loc in partially_full_tomato_soups
        ])

        soups_in_transit = player_objects['soup']
        dishes_in_transit = objects_dict['dish'] + player_objects['dish']
        onions_in_transit = objects_dict['onion'] + player_objects['onion']
        tomatoes_in_transit = objects_dict['tomato'] + player_objects['tomato']

        num_pot_to_delivery = max(
            [0, num_deliveries_to_go - len(soups_in_transit)])
        num_dish_to_pot = max(
            [0, num_pot_to_delivery - len(dishes_in_transit)])

        num_pots_to_be_filled = num_pot_to_delivery - len(full_soups_in_pots)
        num_onions_needed_for_pots = num_pots_to_be_filled * 3 - len(
            onions_in_transit) - num_onions_in_partially_full_pots
        num_tomatoes_needed_for_pots = num_pots_to_be_filled * 3 - len(
            tomatoes_in_transit) - num_tomatoes_in_partially_full_pots
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

        heuristic_cost = (pot_to_delivery_costs + dish_to_pot_costs +
                          items_to_pot_cost) / 2

        if debug:
            env = OvercookedEnv.from_mdp(self.mdp)
            env.state = state
            print("\n" + "#" * 35)
            print("Current state: (ml timestep {})\n".format(time))

            print(
                "# in transit: \t\t Soups {} \t Dishes {} \t Onions {}".format(
                    len(soups_in_transit), len(dishes_in_transit),
                    len(onions_in_transit)))

            print("Trip costs: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
                pot_to_delivery_costs, dish_to_pot_costs,onion_to_pot_costs)
            )

            print(str(env) + "HEURISTIC: {}".format(heuristic_cost))

        return heuristic_cost
