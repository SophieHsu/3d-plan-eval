import itertools
import time


class HighLevelActionManager(object):
    """
    Manager for medium level actions (specific joint motion goals). 
    Determines available medium level actions for each state.
    
    Args:
        mdp (OvercookedGridWorld): gridworld of interest
        start_orientations (bool): whether the JointMotionPlanner should store plans for 
                                   all starting positions & orientations or just for unique 
                                   starting positions
    """

    def __init__(self, mdp, params):
        start_time = time.time()
        self.mdp = mdp
        
        self.params = params
        self.wait_allowed = params['wait_allowed']
        self.counter_drop = params["counter_drop"]
        self.counter_pickup = params["counter_pickup"]
        
        self.joint_motion_planner = JointMotionPlanner(mdp, params)
        self.motion_planner = self.joint_motion_planner.motion_planner
        print("It took {} seconds to create MediumLevelActionManager".format(time.time() - start_time))

    def joint_ml_actions(self, state):
        """Determine all possible joint medium level actions for a certain state"""
        agent1_actions, agent2_actions = tuple(self.get_medium_level_actions(state, player) for player in state.players)
        joint_ml_actions = list(itertools.product(agent1_actions, agent2_actions))
        
        # ml actions are nothing but specific joint motion goals
        valid_joint_ml_actions = list(filter(lambda a: self.is_valid_ml_action(state, a), joint_ml_actions))

        # HACK: Could cause things to break.
        # Necessary to prevent states without successors (due to no counters being allowed and no wait actions)
        # causing A* to not find a solution
        if len(valid_joint_ml_actions) == 0:
            agent1_actions, agent2_actions = tuple(self.get_medium_level_actions(state, player, waiting_substitute=True) for player in state.players)
            joint_ml_actions = list(itertools.product(agent1_actions, agent2_actions))
            valid_joint_ml_actions = list(filter(lambda a: self.is_valid_ml_action(state, a), joint_ml_actions))
            if len(valid_joint_ml_actions) == 0:
                print("WARNING: Found state without valid actions even after adding waiting substitute actions. State: {}".format(state))
        return valid_joint_ml_actions

    def is_valid_ml_action(self, state, ml_action):
        return self.joint_motion_planner.is_valid_jm_start_goal_pair(state.players_pos_and_or, ml_action)

    def get_medium_level_actions(self, state, player, waiting_substitute=False):
        """
        Determine valid medium level actions for a player.
        
        Args:
            state (OvercookedState): current state
            waiting_substitute (bool): add a substitute action that takes the place of 
                                       a waiting action (going to closest feature)
        
        Returns:
            player_actions (list): possible motion goals (pairs of goal positions and orientations)
        """
        player_actions = []
        counter_pickup_objects = self.mdp.get_counter_objects_dict(state, self.counter_pickup)
        if not player.has_object():
            onion_pickup = self.pickup_onion_actions(counter_pickup_objects)
            tomato_pickup = self.pickup_tomato_actions(counter_pickup_objects)
            dish_pickup = self.pickup_dish_actions(counter_pickup_objects)
            soup_pickup = self.pickup_counter_soup_actions(counter_pickup_objects)
            player_actions.extend(onion_pickup + tomato_pickup + dish_pickup + soup_pickup)

        else:
            player_object = player.get_object()
            pot_states_dict = self.mdp.get_pot_states(state)

            # No matter the object, we can place it on a counter
            if len(self.counter_drop) > 0:
                player_actions.extend(self.place_obj_on_counter_actions(state))

            if player_object.name == 'soup':
                player_actions.extend(self.deliver_soup_actions())
            elif player_object.name == 'onion':
                player_actions.extend(self.put_onion_in_pot_actions(pot_states_dict))
            elif player_object.name == 'tomato':
                player_actions.extend(self.put_tomato_in_pot_actions(pot_states_dict))
            elif player_object.name == 'dish':
                # Not considering all pots (only ones close to ready) to reduce computation
                # NOTE: could try to calculate which pots are eligible, but would probably take
                # a lot of compute
                player_actions.extend(self.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=False))
            else:
                raise ValueError("Unrecognized object")

        if self.wait_allowed:
            player_actions.extend(self.wait_actions(player))

        if waiting_substitute:
            # Trying to mimic a "WAIT" action by adding the closest allowed feature to the avaliable actions
            # This is because motion plans that aren't facing terrain features (non counter, non empty spots)
            # are not considered valid
            player_actions.extend(self.go_to_closest_feature_actions(player))

        is_valid_goal_given_start = lambda goal: self.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, goal)    
        player_actions = list(filter(is_valid_goal_given_start, player_actions))
        return player_actions

    def pickup_onion_actions(self, counter_objects, only_use_dispensers=False):
        """If only_use_dispensers is True, then only take onions from the dispensers"""
        onion_pickup_locations = self.mdp.get_onion_dispenser_locations()
        if not only_use_dispensers:
            onion_pickup_locations += counter_objects['onion']
        return self._get_ml_actions_for_positions(onion_pickup_locations)

    def pickup_tomato_actions(self, counter_objects):
        tomato_dispenser_locations = self.mdp.get_tomato_dispenser_locations()
        tomato_pickup_locations = tomato_dispenser_locations + counter_objects['tomato']
        return self._get_ml_actions_for_positions(tomato_pickup_locations)

    def pickup_dish_actions(self, counter_objects, only_use_dispensers=False):
        """If only_use_dispensers is True, then only take dishes from the dispensers"""
        dish_pickup_locations = self.mdp.get_dish_dispenser_locations()
        if not only_use_dispensers:
            dish_pickup_locations += counter_objects['dish']
        return self._get_ml_actions_for_positions(dish_pickup_locations)

    def pickup_counter_soup_actions(self, counter_objects):
        soup_pickup_locations = counter_objects['soup']
        return self._get_ml_actions_for_positions(soup_pickup_locations)

    def place_obj_on_counter_actions(self, state):
        all_empty_counters = set(self.mdp.get_empty_counter_locations(state))
        valid_empty_counters = [c_pos for c_pos in self.counter_drop if c_pos in all_empty_counters]
        return self._get_ml_actions_for_positions(valid_empty_counters)

    def deliver_soup_actions(self):
        serving_locations = self.mdp.get_serving_locations()
        return self._get_ml_actions_for_positions(serving_locations)

    def put_onion_in_pot_actions(self, pot_states_dict):
        partially_full_onion_pots = pot_states_dict['onion']['partially_full']
        fillable_pots = partially_full_onion_pots + pot_states_dict['empty']
        return self._get_ml_actions_for_positions(fillable_pots)

    def put_tomato_in_pot_actions(self, pot_states_dict):
        partially_full_tomato_pots = pot_states_dict['tomato']['partially_full']
        fillable_pots = partially_full_tomato_pots + pot_states_dict['empty']
        return self._get_ml_actions_for_positions(fillable_pots)
    
    def pickup_soup_with_dish_actions(self, pot_states_dict, only_nearly_ready=False):
        ready_pot_locations = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
        nearly_ready_pot_locations = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
        if not only_nearly_ready:
            partially_full_pots = pot_states_dict['tomato']['partially_full'] + pot_states_dict['onion']['partially_full']
            nearly_ready_pot_locations = nearly_ready_pot_locations + pot_states_dict['empty'] + partially_full_pots
        return self._get_ml_actions_for_positions(ready_pot_locations + nearly_ready_pot_locations)

    def go_to_closest_feature_actions(self, player):
        feature_locations = self.mdp.get_onion_dispenser_locations() + self.mdp.get_tomato_dispenser_locations() + \
                            self.mdp.get_pot_locations() + self.mdp.get_dish_dispenser_locations()
        closest_feature_pos = self.motion_planner.min_cost_to_feature(player.pos_and_or, feature_locations, with_argmin=True)[1]
        return self._get_ml_actions_for_positions([closest_feature_pos])

    def go_to_closest_feature_or_counter_to_goal(self, goal_pos_and_or, goal_location):
        """Instead of going to goal_pos_and_or, go to the closest feature or counter to this goal, that ISN'T the goal itself"""
        valid_locations = self.mdp.get_onion_dispenser_locations() + \
                                    self.mdp.get_tomato_dispenser_locations() + self.mdp.get_pot_locations() + \
                                    self.mdp.get_dish_dispenser_locations() + self.counter_drop
        valid_locations.remove(goal_location)
        closest_non_goal_feature_pos = self.motion_planner.min_cost_to_feature(
                                            goal_pos_and_or, valid_locations, with_argmin=True)[1]
        return self._get_ml_actions_for_positions([closest_non_goal_feature_pos])

    def wait_actions(self, player):
        waiting_motion_goal = (player.position, player.orientation)
        return [waiting_motion_goal]

    def _get_ml_actions_for_positions(self, positions_list):
        """Determine what are the ml actions (joint motion goals) for a list of positions
        
        Args:
            positions_list (list): list of target terrain feature positions
        """
        possible_motion_goals = []
        for pos in positions_list:
            # All possible ways to reach the target feature
            for motion_goal in self.joint_motion_planner.motion_planner.motion_goals_for_pos[pos]:
                possible_motion_goals.append(motion_goal)
        return possible_motion_goals