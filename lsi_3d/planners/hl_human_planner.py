import numpy as np
from lsi_3d.utils.functions import find_nearby_open_space


class HLHumanPlanner(object):
    def __init__(self, mdp, mlp): #, ml_action_manager, goal_preference, adaptiveness):
        self.mdp = mdp
        self.mlp = mlp
        #self.ml_action_manager = ml_action_manager
        
        self.sub_goals = {'Onion cooker':0, 'Soup server':1}
        #self.adaptiveness = adaptiveness
        #self.goal_preference = np.array(goal_preference)
        #self.prev_goal_dstb = self.goal_preference

    # def get_state_trans(self, obj, num_item_in_pot, order_list):
    #     # return probability distribution based on A* distance
    #     # Should Human HL Planner be subclass

    #     ml_goals, curr_p = self.human_ml_motion_goal(obj, num_item_in_pot, order_list)
        
    #     next_states = []
    #     for i, ml_goal in enumerate(ml_goals):
    #         WAIT = ml_goal[4]
    #         min_distance = np.Inf
    #         if not WAIT:
    #             start_locations = self.start_location_from_object(obj)
    #             # min_distance = self.ml_action_manager.motion_planner.min_cost_between_features(start_locations, ml_goal[1])
    #             # min_distance = self.mlp.compute_motion_plan(start_locations, ml_goal[1])
    #             min_distance = 2.0
    #         else:
    #             min_distance = 1.0
    #         next_states.append([ml_goal[0], ml_goal[2], ml_goal[3], 1.0/min_distance, curr_p[i]])
        
    #     next_states = np.array(next_states, dtype=object)

    #     return next_states

    def get_state_trans(self, obj, num_item_in_pot, order_list):
        ml_goals, curr_p = self.human_ml_motion_goal(obj, num_item_in_pot, order_list)
        
        next_states = []
        for i, ml_goal in enumerate(ml_goals):
            WAIT = ml_goal[4]
            min_distance = np.Inf
            if not WAIT:
                start_locations = self.start_location_from_object(obj)
                open_start_locations = [find_nearby_open_space(self.mlp.map, loc) for loc in start_locations]
                
                # use a* planner
                plans = [self.mlp.compute_motion_plan([start], ml_goal[1]) for start in open_start_locations]
                min_distances = [len(plan) for plan in plans]
                min_distance = min(min_distances)
                #min_distance = 2.0
            else:
                min_distance = 1.0
            next_states.append([ml_goal[0], ml_goal[2], ml_goal[3], 1.0/min_distance, curr_p[i]])
        
        next_states = np.array(next_states, dtype=object)

        return next_states

    def human_ml_motion_goal(self, obj, num_item_in_pot, order_list):
        """ 
        Get the human's motion goal based on its held object. The return can be multiple location since there can be multiple same feature tiles.

        Return: next object, list(motion goals)
        """
        self.sub_goals = {'Onion cooker': 0, 'Soup server': 1}

        ml_logic_goals = self.logic_ml_action(obj, num_item_in_pot, order_list)

        self.adaptiveness = 0.5
        self.prev_goal_dstb = np.array([0.5,0.5])

        curr_p = ((1.0-self.adaptiveness)*self.prev_goal_dstb + self.adaptiveness*ml_logic_goals)   
        # print(self.adaptiveness, self.prev_goal_dstb, ml_logic_goals, curr_p)
        task = np.random.choice(len(self.sub_goals), p=curr_p)
        self.prev_goal_dstb = curr_p

        ml_goals = []
        ml_goals.append(self.onion_cooker_ml_goal(obj, num_item_in_pot, order_list))
        ml_goals.append(self.soup_server_ml_goal(obj, num_item_in_pot, order_list))
        ml_goals = np.array(ml_goals, dtype=object)

        return ml_goals, curr_p

    def onion_cooker_ml_goal(self, obj, num_item_in_pot, order_list):
        """
        Player action logic as an onion cooker.

        Return: a list of motion goals
        """
        motion_goal = []; next_obj = ''; WAIT = False
        if obj == 'None':
            motion_goal = self.mdp.get_onion_dispenser_locations()
            next_obj = 'onion'
        elif obj == 'onion':
            motion_goal = self.mdp.get_pot_locations()
            next_obj = 'None'
            num_item_in_pot += 1
        else:
            # drop the item in hand
            motion_goal = self.mdp.get_counter_locations()
            next_obj = 'None'
            # next_obj = obj
            # WAIT = True

        if num_item_in_pot > self.mdp.num_items_for_soup:
            num_item_in_pot = self.mdp.num_items_for_soup

        return next_obj, motion_goal, num_item_in_pot, order_list, WAIT

    def soup_server_ml_goal(self, obj, num_item_in_pot, order_list):
        motion_goal = []; WAIT = False; next_obj = ''
        if obj == 'None':
            motion_goal = self.mdp.get_dish_dispenser_locations()
            next_obj = 'dish'
        elif obj == 'dish' and num_item_in_pot == self.mdp.num_items_for_soup:
            motion_goal = self.mdp.get_pot_locations()
            next_obj = 'soup'
            num_item_in_pot = 0
        elif obj == 'dish' and num_item_in_pot != self.mdp.num_items_for_soup:
            motion_goal = None
            next_obj = obj
            WAIT = True
        elif obj == 'soup':
            motion_goal = self.mdp.get_serving_locations()
            order_list = [] if len(order_list) <= 1 else order_list[1:]
            next_obj = 'None'
        else:
            # drop the item in hand
            motion_goal = self.mdp.get_counter_locations()
            next_obj = 'None'
            # next_obj = obj
            # WAIT = True

        if num_item_in_pot > self.mdp.num_items_for_soup:
            num_item_in_pot = self.mdp.num_items_for_soup

        return next_obj, motion_goal, num_item_in_pot, order_list, WAIT
    
    def logic_ml_action(self, player_obj, num_item_in_pot, order_list):
        """
        """
        env_pref = np.zeros(len(self.sub_goals))

        if player_obj == 'None':

            if num_item_in_pot == self.mdp.num_items_for_soup:
                env_pref[1] += 1
            else:
                next_order = None
                if len(order_list) > 1:
                    next_order = order_list[1]

                if next_order == 'onion':
                    env_pref[0] += 1
                elif next_order == 'tomato':
                    # env_pref[self.sub_goals['Tomato cooker']] += 1
                    pass
                elif next_order is None or next_order == 'any':
                    env_pref[0] += 1
                    # env_pref[self.sub_goals['Tomato cooker']] += 1

        else:
            if player_obj == 'onion':
                env_pref[0] += 1

            elif player_obj == 'tomato':
                # env_pref[self.sub_goals['Tomato cooker']] += 1
                pass

            elif player_obj == 'dish':
                env_pref[1] += 1

            elif player_obj == 'soup':
                env_pref[1] += 1
            else:
                raise ValueError()

        if np.sum(env_pref) > 0.0:
            env_pref = env_pref/np.sum(env_pref)
        else:
            env_pref = np.ones((len(env_pref)))/len(env_pref)

        return env_pref

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