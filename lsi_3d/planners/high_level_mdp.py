import itertools
import time

import numpy as np


class HighLevelMdpPlanner(object):

    def __init__(self, mdp, num_rounds = 0, epsilon = 0.01, discount = 0.8):
        self.mdp = mdp
        self.state_dict = {}
        self.state_idx_dict = {}
        self.action_dict = {}
        self.action_idx_dict = {}
        self.discount = discount
        self.epsilon = epsilon
        self.num_rounds = num_rounds
        self.transition_matrix = None
        self.value_matrix = None
        self.policy_matrix = None
        self.num_states = None
        self.num_actions = None

    def init_states(self, order_list=None):
        """
        Initializes the states of mdp. Agent may be holding 1 item
        there may be between 0 and 3 onions on the stove and there
        may be between 0 and order_list number of dishes already delivered
        """

        # objects agent is holding or not
        objects = ['onion', 'soup', 'dish', 'None']
        max_in_soup = 3

        for order_num in range(len(order_list)+1): # +1 for case where nothing is held
            for onion_count in range(max_in_soup+1):
                for object in objects:
                    key = f'{object}_{onion_count}'
                    value = [object, onion_count]

                    for order in order_list[:order_num]:
                        key += f'_{order}'
                        value.append(order)

                    self.state_dict[key] = value

        self.state_idx_dict = {k:i for i, k in enumerate(self.state_dict.keys())}
        return

    def init_actions(self, actions=None):
        objects = ['onion', 'dish'] # 'tomato'
        common_actions = ['pickup', 'drop']
        addition_actions = [['deliver','soup'], ['pickup', 'soup']]

        common_action_obj_pair = list(itertools.product(common_actions, objects))
        common_action_obj_pair = [list(i) for i in common_action_obj_pair]
        actions = common_action_obj_pair + addition_actions
        self.action_dict = {action[0]+'_'+action[1]:action for action in actions}
        self.action_idx_dict = {action[0]+'_'+action[1]:i for i, action in enumerate(actions)}

    def init_transition_matrix(self, transition_matrix=None):
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        game_logic_transition = self.transition_matrix.copy()
        # distance_transition = self.transition_matrix.copy()

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            for action_key, action_idx in self.action_idx_dict.items():
                state_idx = self.state_idx_dict[state_key]
                next_state_idx = state_idx
                next_action_idx = action_idx
        
                # define state and action game transition logic
                player_obj, soup_finish, orders = self.ml_state_to_objs(state_obj)
                next_actions, next_state_keys = self.state_action_nxt_state(player_obj, soup_finish, orders)

                if next_actions == action_key:
                    next_state_idx = self.state_idx_dict[next_state_keys]

                game_logic_transition[next_action_idx][state_idx][next_state_idx] += 1.0

            # print(state_key)
        # print(game_logic_transition[:, 25])
        # tmp = input()

        self.transition_matrix = game_logic_transition
        return

    def ml_state_to_objs(self, state_obj):
        # state: obj + action + bool(soup nearly finish) + orders
        player_obj = state_obj[0]; soup_finish = state_obj[1];
        orders = []
        if len(state_obj) > 2:
            orders = state_obj[2:]

        return player_obj, soup_finish, orders
        
    def state_action_nxt_state(self, player_obj, soup_finish, orders, other_obj=''):
        # game logic
        actions = ''; next_obj = player_obj; next_soup_finish = soup_finish
        if player_obj == 'None':
            if (soup_finish == self.mdp.num_items_for_soup) and (other_obj != 'dish'):
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

            elif (player_obj == 'dish') and (soup_finish == self.mdp.num_items_for_soup):
                actions = 'pickup_soup'
                next_obj = 'soup'
                next_soup_finish = 0

            elif (player_obj == 'dish') and (soup_finish != self.mdp.num_items_for_soup):
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

        return actions, next_state_keys

    def init_reward(self, reward_matrix=None):
        # state: obj + action + bool(soup nearly finish) + orders

        self.reward_matrix = reward_matrix if reward_matrix is not None else np.zeros((len(self.action_dict), len(self.state_idx_dict)), dtype=float)

        # when deliver order, pickup onion. probabily checking the change in states to give out rewards: if action is correct, curr_state acts and changes to rewardable next state. Then, we reward.

        for state_key, state_obj in self.state_dict.items():
            # state: obj + action + bool(soup nearly finish) + orders
            player_obj = state_obj[0]; soup_finish = state_obj[1]
            orders = []
            if len(state_obj) > 2:
                orders = state_obj[2:]

            if player_obj == 'soup':
                self.reward_matrix[self.action_idx_dict['deliver_soup']][self.state_idx_dict[state_key]] += self.mdp.delivery_reward
        
            if len(orders) == 0:
                self.reward_matrix[:,self.state_idx_dict[state_key]] += self.mdp.delivery_reward


    def init_mdp(self, order_list):
        self.init_states(order_list)
        self.init_actions()
        self.init_transition_matrix()
        self.init_reward()

    def bellman_operator(self, V=None):
        if V is None:
            V = self.value_matrix

        Q = np.zeros((self.num_actions, self.num_states))
        for a in range(self.num_actions):
            Q[a] = self.reward_matrix[a] + self.discount * self.transition_matrix[a].dot(V)

        return Q.max(axis=0), Q.argmax(axis=0)

    @staticmethod
    def get_span(arr):
        # print('in get span arr.max():', arr.max(), ' - arr.min():', arr.min(), ' = ', (arr.max()-arr.min()))
        return arr.max()-arr.min()

    def value_iteration(self, value_matrix=None):
        self.value_matrix = value_matrix if value_matrix is not None else np.zeros((self.num_states), dtype=float)
        self.policy_matrix = value_matrix if value_matrix is not None else np.zeros((self.num_states), dtype=float)

        # computation of threshold of variation for V for an epsilon-optimal policy
        if self.discount < 1.0:
            thresh = self.epsilon * (1 - self.discount) / self.discount
        else:
            thresh = self.epsilon

        iter_count = 0
        while True:
            V_prev = self.value_matrix.copy()

            self.value_matrix, self.policy_matrix = self.bellman_operator()

            variation = self.get_span(self.value_matrix-V_prev)

            if variation < thresh:
                #self.log_value_iter(iter_count)
                break
            #elif iter_count % LOGUNIT == 0:
                #self.log_value_iter(iter_count)
            else:
                pass
            iter_count += 1

    def compute_mdp_policy(self, order_list):
        start_time = time.time()
        self.init_mdp(order_list)
        #self.init_mdp()
        self.num_states = len(self.state_dict)
        self.num_actions = len(self.action_dict)

        self.value_iteration()

        print("It took {} seconds to create MediumLevelMdpPlanner".format(time.time() - start_time))
        return 

    def map_action_to_location(self, world_state, agent_state, action_obj, p0_obj = None):
        """
        Get the next location the agent will be in based on current world state and medium level actions.
        """
        p0_obj = p0_obj if p0_obj is not None else agent_state.holding
        action, obj = action_obj
        #pots_states_dict = self.mdp.get_pot_states()
        location = []
        if action == 'pickup' and obj != 'soup':
            if p0_obj != 'None':
                location = self.drop_item(world_state)
            else:
                if obj == 'onion':
                    location = self.mdp.get_onion_dispenser_locations()
                elif obj == 'tomato':
                    location = self.mdp.get_tomato_dispenser_locations()
                elif obj == 'dish':
                    location = self.mdp.get_dish_dispenser_locations()
                else:
                    print(p0_obj, action, obj)
                    ValueError()
        elif action == 'pickup' and obj == 'soup':
            if p0_obj != 'dish' and p0_obj != 'None':
                location = self.drop_item(world_state)
            elif p0_obj == 'None':
                location = self.mdp.get_dish_dispenser_locations()
                print(f'Next Dish Location: {location}')
            else:
                location = self.mdp.get_pot_locations() # + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)

        elif action == 'drop':
            if obj == 'onion' or obj == 'tomato':
                #location = self.mdp.get_partially_full_pots(pots_states_dict) + self.mdp.get_empty_pots(pots_states_dict)
                location = self.mdp.get_pot_locations()
            elif obj == 'dish':
                location = self.drop_item(world_state, agent_state)
            else:
                print(p0_obj, action, obj)
                ValueError()

        elif action == 'deliver':
            if p0_obj != 'soup':
                location = self.mdp.get_empty_counter_locations(world_state)
            else:
                location = self.mdp.get_serving_locations()

        else:
            print(p0_obj, action, obj)
            ValueError()

        return location

    def get_mdp_key_from_state(self, world_state, robot_state, human_state):
        key = f"{robot_state.holding}_{world_state.in_pot}"
        for order in world_state.orders:
            key += f'_{order}'

        return key

    def drop_item(self, world_state, agent_state):
        agent_state.holding = None
        return agent_state.ml_state[0:2]

