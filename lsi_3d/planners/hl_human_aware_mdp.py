import copy

import numpy as np

from lsi_3d.planners.high_level_mdp import HighLevelMdpPlanner


class HLHumanAwareMDPPlanner(HighLevelMdpPlanner):

    def __init__(self, mdp, hhlp):  # hhlp = None, mlp_params = None, \
        """Initializes HL Human Aware MDP Planner

        Args:
            mdp (Markov Decision Process): Defines MDP
            hhlp (Human High Level Planner): Defines how human next states are anticipated
        """
        self.hhlp = hhlp
        super().__init__(mdp)

    def init_cost(self, cost_matrix=None):
        return super().init_cost(cost_matrix)

    def init_human_aware_states(self, state_idx_dict=None, order_list=None):
        self.init_states(order_list=order_list)

        # add p1_obj to [p0_obj, num_item_in_pot, order_list]
        objects = ['onion', 'soup', 'dish', 'None']
        original_state_dict = copy.deepcopy(self.state_dict)
        self.state_dict.clear()
        self.state_idx_dict.clear()
        for i, obj in enumerate(objects):
            for ori_key, ori_value in original_state_dict.items():
                new_key = ori_key + '_' + obj
                new_obj = original_state_dict[ori_key] + [obj]
                self.state_dict[new_key] = new_obj  # update value
                self.state_idx_dict[new_key] = len(self.state_idx_dict)

    def init_transition_matrix(self, transition_matrix=None):
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros(
            (len(self.action_dict), len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        game_logic_transition = self.transition_matrix.copy()

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            for action_key, action_idx in self.action_idx_dict.items():
                state_idx = self.state_idx_dict[state_key]

                # define state and action game transition logic
                p0_state, p1_obj = self.extract_p0(state_obj)

                # get next step p1 object
                p1_nxt_states = []
                if len(p0_state) > 2:
                    p1_nxt_states = self.hhlp.get_state_trans(p1_obj, p0_state[1], p0_state[2:])
                else:
                    p1_nxt_states = self.hhlp.get_state_trans(p1_obj, p0_state[1], [])

                for [p1_nxt_obj, aft_p1_num_item_in_pot, aft_p1_order_list,
                     p1_trans_prob, p1_pref_prob] in p1_nxt_states:

                    p0_obj, soup_finish, orders = self.ml_state_to_objs(p0_state)
                    p0_ori_key = p0_state[0]
                    for s in p0_state[1:]:
                        p0_ori_key = p0_ori_key + '_' + str(s)

                    p1_ori_action, p0_nxt_p1_ori_p0_key = self.state_action_nxt_state(p0_obj, soup_finish, orders,
                                                                                      p1_obj)

                    # based on action what is probability of transition
                    if action_key == p1_ori_action:  # p0 nxt based on p1_ori
                        p0_nxt_p1_ori_nxt_idx = self.state_idx_dict[p0_nxt_p1_ori_p0_key + '_' + p1_obj]

                        # 1 - p1_trans_prob because based on p1 being in original state
                        game_logic_transition[action_idx][state_idx][p0_nxt_p1_ori_nxt_idx] += \
                            (1.0 - p1_trans_prob) * p1_pref_prob

                    else:  # p0_ori + p1_ori
                        p0_ori_p1_ori_nxt_idx = self.state_idx_dict[p0_ori_key + '_' + p1_obj]
                        game_logic_transition[action_idx][state_idx][p0_ori_p1_ori_nxt_idx] += \
                            (1.0 - p1_trans_prob) * p1_pref_prob

                        if game_logic_transition[action_idx][state_idx][p0_ori_p1_ori_nxt_idx] > 1:
                            game_logic_transition[action_idx][state_idx][p0_ori_p1_ori_nxt_idx] = 1

                    # get next step p0 object based on p1 next state
                    p1_nxt_action, p0_nxt_p1_nxt_p0_key = self.state_action_nxt_state(p0_obj, aft_p1_num_item_in_pot,
                                                                                      aft_p1_order_list, p1_nxt_obj)

                    p1_nxt_key = str(aft_p1_num_item_in_pot)
                    for obj in aft_p1_order_list:
                        p1_nxt_key = p1_nxt_key + '_' + obj
                    p1_nxt_key = p1_nxt_key + '_' + p1_nxt_obj

                    if action_key == p1_nxt_action:  # p0 nxt based on p1 next
                        p0_nxt_p1_nxt_nxt_idx = self.state_idx_dict[p0_nxt_p1_nxt_p0_key + '_' + p1_nxt_obj]
                        game_logic_transition[action_idx][state_idx][
                            p0_nxt_p1_nxt_nxt_idx] += p1_trans_prob * p1_pref_prob

                        # print(state_key,'--', action_key, '-->', p0_nxt_p1_nxt_p0_key + '_' + p1_nxt_obj)

                    else:  # action not matched; thus, p0 ori based on p1 next
                        p0_ori_p1_ori_nxt_idx = self.state_idx_dict[p0_ori_key + '_' + p1_obj]
                        game_logic_transition[action_idx][state_idx][p0_ori_p1_ori_nxt_idx] += \
                            p1_trans_prob * p1_pref_prob

        self.transition_matrix = game_logic_transition

    def init_reward(self, reward_matrix=None):
        # state: obj + action + bool(soup nearly finish) + orders

        self.reward_matrix = reward_matrix if reward_matrix is not None else np.zeros(
            (len(self.action_dict), len(self.state_idx_dict)), dtype=float)

        # when deliver order, pickup onion. probabily checking the change in states to give out rewards:
        # if action is correct, curr_state acts and changes to rewardable next state. Then, we reward.

        for state_key, state_obj in self.state_dict.items():
            # state: obj + action + bool(soup nearly finish) + orders
            player_obj, soup_finish, orders = self.ml_state_to_objs(state_obj[:-1])

            if player_obj == 'soup':
                self.reward_matrix[self.action_idx_dict['deliver_soup'], self.state_idx_dict[
                    state_key]] += self.mdp.delivery_reward / 10.0

            if len(orders) == 0:
                self.reward_matrix[:, self.state_idx_dict[state_key]] += self.mdp.delivery_reward

    def extract_p0(self, state_obj):
        return state_obj[:-1], state_obj[-1]

    def init_mdp(self, order_list):
        self.init_human_aware_states(order_list=order_list)
        self.init_actions()
        self.init_transition_matrix()
        self.init_reward()
        self.init_cost()

    def gen_state_dict_key(self, state, player, soup_finish, other_player):
        # a0 pos, a0 dir, a0 hold, a1 pos, a1 dir, a1 hold, len(order_list)

        player_obj = None
        other_player_obj = None
        if player.held_object is not None:
            player_obj = player.held_object.name
        if other_player.held_object is not None:
            other_player_obj = other_player.held_object.name

        order_str = None if state.order_list is None else state.order_list[0]
        for order in state.order_list[1:]:
            order_str = order_str + '_' + str(order)

        state_str = str(player_obj) + '_' + str(soup_finish) + '_' + order_str + '_' + str(other_player_obj)

        return state_str

    def human_state_subtask_transition(self, human_state, world_info):
        # From human state calculate manhattan / A* path and generate a probability distribution
        # over this distance for each motion goal
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
                next_subtasks = ['pickup_onion', 'pickup_dish']  # 'pickup_tomato'

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
                next_subtasks = ['pickup_onion', 'pickup_dish']
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

    def get_mdp_key_from_state(self, world_state, robot_state, human_state):
        key = f"{robot_state.holding}_{world_state.in_pot}"
        for order in world_state.orders:
            key += f'_{order}'

        if human_state.holding is not None:
            key = key + '_' + human_state.holding
        return key
