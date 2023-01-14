import itertools
import time
import numpy as np
from lsi_3d.utils.grid_helpers import GridHelper

class GridAvoidanceMDP():
    """
    2d-Grid MDP for path finding with a mobile agent that must be avoided
    """
    def __init__(self, grid, directions, terminal_state, start_state, discount=0.9):
        self.grid = grid
        self.directions = directions
        self.discount = discount
        self.states = []
        self.actions = []
        self.rewards = []
        self.transitions = []

    def init_mdp(self, terminal_state, start_state):
        self.init_states()
        self.init_actions()
        self.init_transitions()
        self.init_rewards(terminal_state)

    def init_states(self):
        """
        Initializes the states of mdp. Agent may be in any square on map and facing in
        1 of 4 directions
        """
        for (row, col),_ in np.ndenumerate(self.grid):
            for dire in self.directions:
                self.states.append((row,col,dire))

    def get_state_index(self,row,col,dire):
        # get 1-d array index for r,c,d of state array
        return np.ravel_multi_index((row,col,dire), (len(self.grid),len(self.grid[0]),len(self.directions)))
    
    def init_actions(self):
        # Turn north, east, south, or west. iDle, or Interact
        self.actions = ['F','N','E','S','W','D','I']

    def get_action_index(self, action):
        return self.actions.index('I')

    def init_transitions(self):
        self.transitions = np.zeros((len(self.actions), len(self.states), len(self.states)))

        game_logic_transition = self.transitions.copy()

        # state transition calculation
        for state_idx, state in enumerate(self.states):
            for action_idx, action in self.actions:
                next_state_prob, next_state_idx = self.get_trans_prob(state, action)
                game_logic_transition[action_idx][state_idx][next_state_idx] += next_state_prob

        self.transitions = game_logic_transition
        return

    def get_trans_prob(self, state, action):
        next_state = GridHelper.transition(*state,action)
        is_valid = GridHelper.valid_pos(self.grid, *next_state)
        return is_valid, next_state

    def init_rewards(self, terminal_state):
        self.rewards = np.zeros((len(self.actions), len(self.states)))

        # when deliver order, pickup onion. probabily checking the change in states
        self.rewards[(self.get_action_index('I'), *terminal_state)] = 1

    def bellman_operator(self, V=None):
        if V is None:
            V = self.value_matrix

        Q = np.zeros((self.num_actions, self.num_states))
        for a in range(self.num_actions):
            Q[a] = self.rewards[a] + self.discount * self.transitions[a].dot(V)

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
        self.num_states = len(self.state_dict)
        self.num_actions = len(self.action_dict)

        self.value_iteration()

        print("It took {} seconds to create MediumLevelMdpPlanner".format(time.time() - start_time))
        return 

    def map_action_to_location(self, state_obj, action_obj, p0_obj = None):
        """
        Get the next location the agent will be in based on current world state and medium level actions.
        """
        p0_obj = p0_obj if p0_obj is not None else state_obj.holding
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
                location = state_obj.get_ready_pots()[0] # + self.mdp.get_cooking_pots(pots_states_dict) + self.mdp.get_full_pots(pots_states_dict)

        elif action == 'drop':
            if obj == 'onion' or obj == 'tomato':
                #location = self.mdp.get_partially_full_pots(pots_states_dict) + self.mdp.get_empty_pots(pots_states_dict)
                location = self.mdp.get_pot_locations()
            elif obj == 'dish':
                location = self.drop_item(world_state)
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
