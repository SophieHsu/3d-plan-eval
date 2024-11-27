import itertools

import numpy as np
from src.planners.high_level_mdp import HighLevelMdpPlanner
from src.planners.mid_level_motion import AStarMotionPlanner


class SteakMediumLevelMDPPlanner(HighLevelMdpPlanner):
    def __init__(self, mdp, mlp: AStarMotionPlanner):
        super().__init__(mdp)

        self.world_state_cost_dict = {}
        self.subtask_dict = {}
        self.subtask_idx_dict = {}

        self.mlp = mlp

    def gen_state_dict_key(self, world_state, player, other_player=None, env=None, RETURN_OBJ=False):
        player_obj = 'None'
        if player.holding is not None:
            player_obj = player.holding

        order_str = '' if len(world_state.orders) == 0 else world_state.orders[0]
        for order in world_state.orders[1:]:
            order_str = order_str + '_' + str(order)

        pot_state, chop_state, sink_state = 0, -1, -1

        if len(env.kitchen.ready_sinks) > 0:
            sink_state = 2
        elif len(env.kitchen.rinsing_sinks) > 0:
            sink_state = 1
        elif len(world_state.state_dict['sink_states']['full']) > 0:
            sink_state = 0
        elif len(world_state.state_dict['sink_states']['empty']) > 0:
            sink_state = 'None'

        # steaks
        if len(world_state.state_dict['pot_states']['empty']) > 0:
            pot_state = 0
        else:
            pot_state = 1

        if len(world_state.state_dict['chop_states']['empty']) > 0:
            chop_state = 'None'
        elif len(world_state.state_dict['chop_states']['full']) > 0:
            if player_obj == 'knife' or other_player.holding == 'knife':
                chop_state = 1
            else:
                chop_state = 0
        else:
            chop_state = 2

        if RETURN_OBJ:
            return [player_obj, pot_state, chop_state, sink_state, world_state.orders]

        state_str = str(player_obj) + '_' + str(pot_state) + '_' + str(chop_state) + '_' + str(sink_state)
        if order_str != '':
            state_str = state_str + '_' + order_str

        return state_str

    def init_states(self, state_idx_dict=None, order_list=None):
        if state_idx_dict is None:
            objects = ['meat', 'onion', 'plate', 'hot_plate', 'steak', 'dish', 'None']

            state_keys = []
            state_obj = []
            tmp_state_obj = []

            for obj in objects:
                tmp_state_obj.append(([obj]))

            # include key object state 
            objects_only_arr = [obj.copy() for obj in tmp_state_obj]
            for i in range(self.mdp.num_items_for_steak + 1):
                tmp_keys = [val + '_' + str(i) for val in objects]

                state_keys = state_keys + tmp_keys
                tmp_state_obj = [obj.copy() for obj in objects_only_arr]
                for obj in tmp_state_obj:
                    obj.append(i)
                state_obj = state_obj + [obj for obj in tmp_state_obj]

            tmp_state_key = state_keys
            prev_state_obj = [obj.copy() for obj in state_obj]
            tmp_state_obj = [obj.copy() for obj in state_obj]
            prev_keys = tmp_state_key.copy()
            tmp_state_key = []
            state_obj = []

            for i in range(self.mdp.chopping_time + 1):
                tmp_keys = [k + '_' + str(i) for k in prev_keys]
                tmp_state_key += tmp_keys

                for obj in tmp_state_obj:
                    obj.append(i)
                state_obj = state_obj + [obj for obj in tmp_state_obj]
                tmp_state_obj = [obj.copy() for obj in prev_state_obj]

            tmp_keys = [k + '_None' for k in prev_keys]
            tmp_state_key += tmp_keys
            prev_keys = tmp_state_key.copy()

            for obj in tmp_state_obj:
                obj.append('None')
            state_obj = state_obj + [obj for obj in tmp_state_obj]

            prev_state_obj = [obj.copy() for obj in state_obj]
            tmp_state_obj = [obj.copy() for obj in state_obj]
            prev_keys = tmp_state_key.copy()
            tmp_state_key = []
            state_obj = []

            for i in range(self.mdp.wash_time + 1):
                tmp_keys = [k + '_' + str(i) for k in prev_keys]
                tmp_state_key += tmp_keys

                for obj in tmp_state_obj:
                    obj.append(i)
                state_obj = state_obj + [obj for obj in tmp_state_obj]
                tmp_state_obj = [obj.copy() for obj in prev_state_obj]

            tmp_keys = [k + '_None' for k in prev_keys]
            tmp_state_key += tmp_keys
            for obj in tmp_state_obj:
                obj.append('None')
            state_obj = state_obj + [obj for obj in tmp_state_obj]
            # tmp_state_key = state_keys
            prev_state_obj = [obj.copy() for obj in state_obj]
            tmp_state_obj = [obj.copy() for obj in state_obj]
            state_obj = []

            # include order list items in state
            for order in order_list:
                prev_keys = tmp_state_key.copy()
                tmp_keys = [i + '_' + order for i in prev_keys]
                # state_keys = state_keys + tmp_keys
                tmp_state_key += tmp_keys

                for obj in tmp_state_obj:
                    obj.append(order)
                tmp_state_obj = prev_state_obj + [obj for obj in tmp_state_obj]
                prev_state_obj = [obj.copy() for obj in tmp_state_obj]

            self.state_idx_dict = {k: i for i, k in enumerate(tmp_state_key)}
            self.state_dict = {key: obj for key, obj in zip(tmp_state_key, tmp_state_obj)}

        else:
            self.state_idx_dict = state_idx_dict
            self.state_dict = state_dict
        return

    def init_actions(self, actions=None, action_dict=None, action_idx_dict=None):
        if actions is None:
            objects = ['meat', 'onion', 'plate', 'hot_plate', 'steak']
            common_actions = ['pickup', 'drop']
            addition_actions = [['chop', 'onion'], ['heat', 'hot_plate'], ['pickup', 'garnish'], ['deliver', 'dish']]

            common_action_obj_pair = list(itertools.product(common_actions, objects))
            common_action_obj_pair = [list(i) for i in common_action_obj_pair]
            actions = common_action_obj_pair + addition_actions
            self.action_dict = {action[0] + '_' + action[1]: action for action in actions}
            self.action_idx_dict = {action[0] + '_' + action[1]: i for i, action in enumerate(actions)}

        else:
            self.action_dict = action_dict
            self.action_idx_dict = action_idx_dict

        return

    def init_transition_matrix(self, transition_matrix=None):
        self.transition_matrix = transition_matrix if transition_matrix is not None else np.zeros(
            (len(self.action_dict), len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        game_logic_transition = self.transition_matrix.copy()
        distance_transition = self.transition_matrix.copy()

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            for action_key, action_idx in self.action_idx_dict.items():
                state_idx = self.state_idx_dict[state_key]
                next_state_idx = state_idx
                next_action_idx = action_idx

                # define state and action game transition logic
                player_obj, num_item_in_pot, chop_time, wash_time, orders = self.ml_state_to_objs(state_obj)
                next_actions, next_state_keys = self.state_action_nxt_state(player_obj, num_item_in_pot, chop_time,
                                                                            wash_time, orders)

                if next_actions == action_key:
                    next_state_idx = self.state_idx_dict[next_state_keys]

                game_logic_transition[next_action_idx][state_idx][next_state_idx] += 1.0

        self.transition_matrix = game_logic_transition

    def ml_state_to_objs(self, state_obj):
        # state: obj + action + bool(soup nearly finish) + orders
        player_obj = state_obj[0];
        num_item_in_pot = state_obj[1];
        chop_time = state_obj[2];
        wash_time = state_obj[3];
        orders = []
        if len(state_obj) > 4:
            orders = state_obj[4:]

        return player_obj, num_item_in_pot, chop_time, wash_time, orders

    def state_action_nxt_state(self, player_obj, num_item_in_pot, chop_time, wash_time, orders, other_obj=''):
        # game logic
        actions = ''
        next_obj = player_obj
        next_num_item_in_pot = num_item_in_pot
        next_chop_time = chop_time
        next_wash_time = wash_time
        if wash_time == 'None':
            wash_time = -1
        if chop_time == 'None':
            chop_time = -1

        if player_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (other_obj != 'meat'):
                actions = 'pickup_meat'
                next_obj = 'meat'
            elif (chop_time < 0) and (other_obj != 'onion'):
                actions = 'pickup_onion'
                next_obj = 'onion'
            elif (chop_time > 0) and (chop_time < self.mdp.chopping_time) and (wash_time < self.mdp.wash_time):
                actions = 'chop_onion'
                next_obj = 'None'
            elif (wash_time < 0) and (other_obj != 'plate'):
                actions = 'pickup_plate'
                next_obj = 'plate'
            elif (wash_time > 0) and (wash_time < self.mdp.wash_time):
                actions = 'heat_hot_plate'
                next_obj = 'None'
            elif (chop_time == self.mdp.chopping_time) and (wash_time == self.mdp.wash_time):
                actions = 'pickup_hot_plate'
                next_obj = 'hot_plate'
            else:
                next_order = None
                if len(orders) > 1:
                    next_order = orders[1]

                if next_order == 'steak':
                    actions = 'pickup_meat'
                    next_obj = 'meat'

                else:
                    actions = 'pickup_meat'
                    next_obj = 'meat'

        else:
            if player_obj == 'onion':
                actions = 'drop_onion'
                next_obj = 'None'
                next_chop_time = 0

            elif player_obj == 'meat':
                actions = 'drop_meat'
                next_obj = 'None'
                next_num_item_in_pot = 1

            elif player_obj == 'plate':
                actions = 'drop_plate'
                next_obj = 'None'
                next_wash_time = 0

            elif (player_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions = 'pickup_steak'
                next_obj = 'steak'
                next_num_item_in_pot = 0

            elif (player_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions = 'drop_hot_plate'
                next_obj = 'None'
                next_wash_time = 'None'

            elif (player_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions = 'pickup_garnish'
                next_obj = 'dish'
                next_chop_time = 'None'

            elif (player_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                actions = 'drop_steak'
                next_obj = 'None'

            elif player_obj == 'dish':
                actions = 'deliver_dish'
                next_obj = 'None'
                if len(orders) >= 1:
                    orders.pop(0)
            else:
                print(player_obj)
                raise ValueError()

        next_state_keys = next_obj + '_' + str(next_num_item_in_pot) + '_' + str(next_chop_time) + '_' + str(
            next_wash_time)
        for order in orders:
            next_state_keys = next_state_keys + '_' + order

        return actions, next_state_keys

    def map_action_to_location(self, world_state, state_obj, action, obj, p0_obj=None):
        """
        Get the next location the agent will be in based on current world state and medium level actions.
        """
        p0_obj = p0_obj if p0_obj is not None else self.state_dict[state_obj][0]
        pots_states_dict = self.mdp.get_pot_states(world_state)
        location = []
        if action == 'pickup':
            if obj == 'onion':
                location = self.mdp.get_onion_dispenser_locations()
            elif obj == 'plate':
                location = self.mdp.get_dish_dispens()
            elif obj == 'meat':
                location = self.mdp.get_meat_dispenser_locations()
            elif obj == 'hot_plate':
                location = self.mdp.get_sink_status(world_state)['full'] + self.mdp.get_sink_status(world_state)[
                    'ready']
            elif obj == 'garish':
                location = self.mdp.get_chopping_board_status(world_state)['full'] + \
                           self.mdp.get_chopping_board_status(world_state)['ready']
            elif obj == 'steak':
                location = self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_ready_pots(
                    pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)
            else:
                print(p0_obj, action, obj)
                ValueError()

        elif action == 'drop':
            if obj == 'meat':
                location = self.mdp.get_empty_pots(pots_states_dict)
            elif obj == 'onion':
                location = self.mdp.get_chopping_board_status(world_state)['empty']
            elif obj == 'plate':
                location = self.mdp.get_sink_status(world_state)['empty']
            elif obj == 'hot_plate' or obj == 'steak':
                location = self.drop_item(world_state)
            else:
                print(p0_obj, action, obj)
                ValueError()

        elif action == 'deliver':
            if p0_obj != 'dish':
                location = self.drop_item(world_state)
            else:
                location = self.mdp.get_serving_locations()

        else:
            print(p0_obj, action, obj)
            ValueError()

        return location

    def init_reward(self, reward_matrix=None):
        self.reward_matrix = reward_matrix if reward_matrix is not None else np.zeros(
            (len(self.action_dict), len(self.state_idx_dict)), dtype=float)

        for state_key, state_obj in self.state_dict.items():
            # state: obj + action + bool(soup nearly finish) + orders
            player_obj = state_obj[0];
            soup_finish = state_obj[1]
            orders = []
            if len(state_obj) > 5:
                orders = state_obj[4:-1]

            if player_obj == 'dish':
                self.reward_matrix[self.action_idx_dict['deliver_dish']][
                    self.state_idx_dict[state_key]] += self.mdp.delivery_reward

            if len(orders) == 0:
                self.reward_matrix[:, self.state_idx_dict[state_key]] += self.mdp.delivery_reward

        return

    def get_mdp_key_from_state(self, env):
        key = f"{env.robot_state.holding}_{len(env.world_state.state_dict['pot_states'])}"
        for order in env.world_state.orders:
            key += f'_{order}'

        return key
