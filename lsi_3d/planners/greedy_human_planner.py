import numpy as np
from agent import Agent
from lsi_3d.mdp.lsi_env import LsiEnv
from lsi_3d.mdp.hl_state import AgentState, WorldState
from lsi_3d.utils.functions import find_nearby_open_space

class HLGreedyHumanPlanner(object):
    def __init__(self, mdp, mlp) -> None:
        self.mdp = mdp
        self.mlp = mlp

    def get_state_trans(self, holding, in_pot, orders):
        # goes from fridge to onion

        action,object = 'stay', holding
        if in_pot < 3 and holding == 'None':
            action,object = ('pickup', 'onion')
            next_hl_state = f'onion_{in_pot}'
            next_hl_holding = 'onion'
        elif holding == 'onion':
            action,object = ('drop','onion')
            next_hl_state = f'None_{in_pot+1}'
            next_hl_holding = 'None'
        elif in_pot <= 3 and holding == 'None':
            action,object = ('pickup','dish')
            next_hl_state = f'dish_{in_pot}'
            next_hl_holding = 'dish'
        elif holding == 'dish' and in_pot == 3:
            action,object = ('pickup','soup')
            in_pot = 0
            next_hl_state = f'soup_{in_pot}'
            next_hl_holding = 'soup'
        elif holding == 'soup':
            action,object = ('deliver','soup')
            next_hl_state = f'None_{in_pot}'
            next_hl_holding = 'None'
        else:
            next_hl_holding = holding
            next_hl_state = f'{holding}_{in_pot}'
        
        for order in orders:
            next_hl_state += f'_{order}'

        possible_motion_goals = self.mdp.map_action_to_location(holding, in_pot, orders, (action,object))
        goals = possible_motion_goals
        pref_prob = 1

        probs = self.get_trans_probabilities(next_hl_holding, in_pot, orders, goals, holding, pref_prob)

        # get motion goals
        # get the probability of transition based on distance 
        # return [p1_nxt_obj, aft_p1_num_item_in_pot, aft_p1_order_list, p1_trans_prob, p1_pref_prob] in p1_nxt_states:


        #start = ml_state[0] + ml_state[1]

        # should this happen outside
        #paths = self.mlp.compute_motion_plan(state.ml_state, (goal,state.ml_state[0]))

        # path = self.mlp.compute_single_agent_astar_path(agent_state.ml_state, goal)

        #path = convert_path_to_mla(path)

        #pref_prob = 1

        return probs

    def get_trans_probabilities(self, next_hl_holding, in_pot, orders, ml_goals, holding, pref_prob):
        next_states = []
        for i, ml_goal in enumerate(ml_goals):
            # WAIT = ml_goal[4]
            min_distance = np.Inf
            # if not WAIT:
            start_locations = self.start_location_from_object(holding)
            open_start_locations = [find_nearby_open_space(self.mlp.map, loc) for loc in start_locations]

            for start_loc in open_start_locations:
                if start_loc == None:
                    print('No start location for human planner')
                

                # min_distance = self.ml_action_manager.motion_planner.min_cost_between_features(start_locations, ml_goal[1])
            
                # TODO: add in start facing direction intelligence from start_location from object to be object facing appliance
                # iterate over start locations
                plan = self.mlp.compute_motion_plan([(start_loc)], [ml_goal])
                
                if len(plan) > 0:
                    trans_prob = 1/len(plan)
                else:
                    trans_prob = 0.00001

            #else:
            #    min_distance = 1.0
            #next_states.append([ml_goal[0], ml_goal[2], ml_goal[3], 1.0/min_distance, curr_p[i]])
                next_states.append([next_hl_holding, in_pot, orders, trans_prob, pref_prob])
        
        next_states = np.array(next_states, dtype=object)

        return next_states

    def start_location_from_object(self, obj):
        """ 
        Calculate the starting location based on the object in the human's hand. The feature tile bellowing to the held object will be used as the start location.

        Return: list(starting locations)
        """
        if obj == 'None':
            # default to have dropped item
            start_locations = self.mdp.get_pot_locations() + self.mdp.get_serving_locations()
        elif obj == 'onion':
            start_locations = self.mdp.get_onion_dispenser_locations()
        elif obj == 'tomato':
            start_locations = self.mdp.get_tomato_dispenser_locations()
        elif obj == 'dish':
            start_locations = self.mdp.get_dish_dispenser_locations()
        elif obj == 'soup':
            start_locations = self.mdp.get_pot_locations()
        else:
            ValueError()

        return start_locations



            


            



            

