import time
import numpy as np
from lsi_3d.mdp.action import Action
from lsi_3d.planners.steak_human_subtask_qmdp_planner import SteakHumanSubtaskQMDPPlanner


class SteakKnowledgeBasePlanner(SteakHumanSubtaskQMDPPlanner):
    def __init__(self, mdp, mlp, state_dict={}, state_idx_dict={}, action_dict={}, action_idx_dict={}, transition_matrix=None, reward_matrix=None, policy_matrix=None, value_matrix=None, num_states=0, num_rounds=0, epsilon=0.01, discount=0.8, jmp=None, vision_limited_human=None, debug=False, search_depth=5, kb_search_depth=3):
        super().__init__(mdp, mlp, vision_limited_human)

        self.list_objs = ['None', 'meat', 'onion', 'plate', 'hot_plate', 'steak', 'dish']
        self.kb_space = (self.mdp.num_items_for_steak+1) * (self.mdp.chopping_time+1) * (self.mdp.wash_time+1) * len(self.list_objs) # num_in_pot_item * chop_time * wash_time * holding
        self.init_kb_idx_dict()
        self.debug = debug
        self.search_depth = search_depth
        self.kb_search_depth = kb_search_depth
    
    @staticmethod
    def from_pickle_or_compute(mdp, mlp_params, custom_filename=None, force_compute_all=False, info=True, force_compute_more=False, jmp=None, vision_limited_human=None, debug=False, search_depth=5, kb_search_depth=3):

        assert isinstance(mdp, OvercookedGridworld)

        filename = custom_filename if custom_filename is not None else mdp.layout_name + '_' + 'steak_knowledge_aware_qmdp' + '.pkl'

        if force_compute_all:
            mdp_planner = SteakKnowledgeBasePlanner(mdp, mlp_params, vision_limited_human=vision_limited_human, debug=debug, search_depth=search_depth, kb_search_depth=kb_search_depth)
            mdp_planner.compute_mdp(filename)
            return mdp_planner
        
        try:
            mdp_planner = SteakKnowledgeBasePlanner.from_qmdp_planner_file(filename)
            
            if force_compute_more:
                print("Stored mdp_planner computed ", str(mdp_planner.num_rounds), " rounds. Compute another " + str(TRAINNINGUNIT) + " more...")
                mdp_planner.compute_mdp(filename)
                return mdp_planner

        except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
            print("Recomputing planner due to:", e)
            mdp_planner = SteakKnowledgeBasePlanner(mdp, mlp_params, jmp=jmp, vision_limited_human=vision_limited_human, debug=debug, search_depth=search_depth, kb_search_depth=kb_search_depth)
            mdp_planner.compute_mdp(filename)
            return mdp_planner

        if info:
            print("Loaded SteakKnowledgeBasePlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))

        return mdp_planner
    
    def init_kb_idx_dict(self):
        self.kb_idx_dict = {}
        count = 0
        for num_item_for_steak in range(self.mdp.num_items_for_steak+1):
            for chop_time in range(-1, self.mdp.chopping_time+1):
                for wash_time in range(-1, self.mdp.wash_time+1):
                    for obj in self.list_objs:
                        kb_key = '.'.join([str(num_item_for_steak), str(chop_time), str(wash_time), obj])
                        self.kb_idx_dict[kb_key] = count
                        count += 1

    def init_s_kb_trans_matrix(self, s_kb_trans_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        # self.s_kb_trans_matrix = s_kb_trans_matrix if s_kb_trans_matrix is not None else np.identity((len(self.state_idx_dict)), dtype=float)
        self.s_kb_trans_matrix = s_kb_trans_matrix if s_kb_trans_matrix is not None else np.zeros((len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]

            # decode state information
            p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
            # calculate next states for p1 (a.k.a. human)
            # p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)
            p1_nxt_states, p1_nxt_world_info = self.world_based_human_state_subtask_transition(p1_state, world_info, other_agent_obj=p0_state)

            # append original state of p1 (human) to represent unfinished subtask state transition
            p1_nxt_states.append(p1_state)
            p1_nxt_world_info += [world_info]

            # calculate next states for p0 (conditioned on p1 (a.k.a. human))
            for i, p1_nxt_state in enumerate(p1_nxt_states):
                _, next_state_keys = self.consider_subtask_stochastic_state_transition(p0_state, p1_nxt_world_info[0], human_state=p1_nxt_state)
                # _, next_state_keys = self.stochastic_state_transition(p0_state, p1_nxt_world_info[i], human_state=p1_nxt_state)
                # consider the next state where agent 0 does not complete action execution
                p0_not_complete_key = self.get_key_from_value(self.state_dict, [state_obj[0]]+p1_nxt_world_info[i]+[p1_nxt_state])
                if (p0_not_complete_key not in next_state_keys) and p0_not_complete_key != state_key:
                    next_state_keys.append(p0_not_complete_key)
                for next_state_key in next_state_keys:
                    next_state_idx = self.state_idx_dict[next_state_key]
                    self.s_kb_trans_matrix[state_idx, next_state_idx] += 1.0

            if np.sum(self.s_kb_trans_matrix[state_idx]) > 0.0:
                self.s_kb_trans_matrix[state_idx] /= np.sum(self.s_kb_trans_matrix[state_idx])

        self.s_kb_trans_matrix[self.s_kb_trans_matrix == 0.0] = 0.000001

    def init_sprim_s_kb_trans_matrix(self, sprim_s_kb_trans_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.sprim_s_kb_trans_matrix = sprim_s_kb_trans_matrix if sprim_s_kb_trans_matrix is not None else np.zeros((len(self.state_idx_dict)), dtype=object)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]
            self.sprim_s_kb_trans_matrix[state_idx] = {}

            # decode state information
            p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
            # calculate next states for p1 (a.k.a. human)
            # p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)
            p1_nxt_states, p1_nxt_world_info = self.world_based_human_state_subtask_transition(p1_state, world_info, other_agent_obj=p0_state)

            # append original state of p1 (human) to represent unfinished subtask state transition
            p1_nxt_states.append(p1_state)
            p1_nxt_world_info += [world_info]

            # calculate next states for p0 (conditioned on p1 (a.k.a. human))
            for i, p1_nxt_state in enumerate(p1_nxt_states):
                # _, next_state_keys = self.stochastic_state_transition(p0_state, p1_nxt_world_info[i], human_state=p1_nxt_state)
                _, next_state_keys = self.consider_subtask_stochastic_state_transition(p0_state, p1_nxt_world_info[0], human_state=p1_nxt_state)
                
                for next_state_key in next_state_keys:
                    needed_kb_keys_for_next = self.get_needed_kb_key(next_state_key, p1_nxt_state, state_key)
                    next_state_idx = self.state_idx_dict[next_state_key]
                    for needed_kb_key_for_next in needed_kb_keys_for_next:
                        if (self.kb_idx_dict[needed_kb_key_for_next], next_state_idx) in self.sprim_s_kb_trans_matrix[state_idx].keys():
                            self.sprim_s_kb_trans_matrix[state_idx][(self.kb_idx_dict[needed_kb_key_for_next], next_state_idx)] += 1.0
                        else:
                            self.sprim_s_kb_trans_matrix[state_idx][(self.kb_idx_dict[needed_kb_key_for_next], next_state_idx)] = 1.0
                
                # add for self trainsition probability
                needed_kb_keys_for_next = self.get_needed_kb_key(state_key, p1_state, state_key)
                for needed_kb_key_for_next in needed_kb_keys_for_next:
                    self.sprim_s_kb_trans_matrix[state_idx][(self.kb_idx_dict[needed_kb_key_for_next], state_idx)] = 1.0
  
        if len(self.sprim_s_kb_trans_matrix[state_idx]) > 0:
            sum_count = np.sum(list(self.sprim_s_kb_trans_matrix[state_idx].values()))
            for k,_ in self.sprim_s_kb_trans_matrix[state_idx].items():
                self.sprim_s_kb_trans_matrix[state_idx][k] /= sum_count

    def init_optimal_s_kb_trans_matrix(self, optimal_s_kb_trans_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.optimal_s_kb_trans_matrix = optimal_s_kb_trans_matrix if optimal_s_kb_trans_matrix is not None else np.zeros((len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)#np.identity((len(self.state_idx_dict)), dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]

            # decode state information
            p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
            # calculate next states for p1 (a.k.a. human)
            # p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)
            p1_nxt_states, p1_nxt_world_info = self.world_based_human_state_subtask_transition(p1_state, world_info, other_agent_obj=p0_state)

            # append original state of p1 (human) to represent unfinished subtask state transition
            # p1_nxt_states.append(p1_state)
            # p1_nxt_world_info += [world_info]

            # calculate next states for p0 (conditioned on p1 (a.k.a. human))
            for i, p1_nxt_state in enumerate(p1_nxt_states):
                _, next_state_keys = self.consider_subtask_stochastic_state_transition(p0_state, p1_nxt_world_info[0], human_state=p1_nxt_state)
                # _, next_state_keys = self.stochastic_state_transition(p0_state, p1_nxt_world_info[i], human_state=p1_nxt_state)#, human_state=p1_nxt_state)
                # consider the next state where agent 0 does not complete action execution
                # p0_not_complete_key = self.get_key_from_value(self.state_dict, [state_obj[0]]+p1_nxt_world_info[i]+[p1_nxt_state])
                # if (p0_not_complete_key not in next_state_keys) and p0_not_complete_key != state_key:
                    # next_state_keys.append(p0_not_complete_key)
                for next_state_key in next_state_keys:
                    next_state_idx = self.state_idx_dict[next_state_key]
                    self.optimal_s_kb_trans_matrix[state_idx, next_state_idx] += 1.0

            if np.sum(self.optimal_s_kb_trans_matrix[state_idx]) > 0.0:
                self.optimal_s_kb_trans_matrix[state_idx] /= np.sum(self.optimal_s_kb_trans_matrix[state_idx])

        self.optimal_s_kb_trans_matrix[self.optimal_s_kb_trans_matrix == 0.0] = 0.000001

    def init_optimal_non_subtask_s_kb_trans_matrix(self, optimal_non_subtask_s_kb_trans_matrix=None):
        """
        This transition matrix needs to include subtask tranistion for both robot and human. Humans' state transition is conditioned on the subtask.
        """
        self.optimal_non_subtask_s_kb_trans_matrix = optimal_non_subtask_s_kb_trans_matrix if optimal_non_subtask_s_kb_trans_matrix is not None else np.zeros((len(self.state_idx_dict), len(self.state_idx_dict)), dtype=float)#np.identity((len(self.state_idx_dict)), dtype=float)

        # state transition calculation
        for state_key, state_obj in self.state_dict.items():
            state_idx = self.state_idx_dict[state_key]

            # decode state information
            p0_state, p1_state, world_info = self.decode_state_info(state_obj) # p0_obj; p1_obj, p1_subtask; num_item_in_pot, order_list;
            # calculate next states for p1 (a.k.a. human)
            # p1_nxt_states, p1_nxt_world_info = self.human_state_subtask_transition(p1_state, world_info)
            p1_nxt_states, p1_nxt_world_info = self.world_based_human_state_subtask_transition(p1_state, world_info, other_agent_obj=p0_state)

            # append original state of p1 (human) to represent unfinished subtask state transition
            # p1_nxt_states.append(p1_state)
            # p1_nxt_world_info += [world_info]

            # calculate next states for p0 (conditioned on p1 (a.k.a. human))
            for i, p1_nxt_state in enumerate(p1_nxt_states):
                _, next_state_keys = self.non_subtask_stochastic_state_transition(p0_state, p1_nxt_world_info[0], human_state=p1_nxt_state)#, human_state=p1_nxt_state)
                # consider the next state where agent 0 does not complete action execution
                # p0_not_complete_key = self.get_key_from_value(self.state_dict, [state_obj[0]]+p1_nxt_world_info[i]+[p1_nxt_state])
                # if (p0_not_complete_key not in next_state_keys) and p0_not_complete_key != state_key:
                    # next_state_keys.append(p0_not_complete_key)
                for next_state_key in next_state_keys:
                    next_state_idx = self.state_idx_dict[next_state_key]
                    self.optimal_non_subtask_s_kb_trans_matrix[state_idx, next_state_idx] += 1.0

            if np.sum(self.optimal_non_subtask_s_kb_trans_matrix[state_idx]) > 0.0:
                self.optimal_non_subtask_s_kb_trans_matrix[state_idx] /= np.sum(self.optimal_non_subtask_s_kb_trans_matrix[state_idx])

        self.optimal_non_subtask_s_kb_trans_matrix[self.optimal_non_subtask_s_kb_trans_matrix == 0.0] = 0.000001

    
    def get_successor_states(self, start_world_state, start_state_key, debug=False, add_rewards=False):
        """
        Successor states for qmdp medium-level actions.
        """
        if len(self.state_dict[start_state_key][4:-1]) == 0: # [p0_obj, num_item_in_soup, orders, p1_obj, subtask] 
            return []

        ori_state_idx = self.state_idx_dict[start_state_key]
        successor_states = []

        # next_state_idx_arr = np.where(self.s_kb_trans_matrix[ori_state_idx] > 0.000001)[0]
        next_state_idx_arr = np.where(self.optimal_s_kb_trans_matrix[ori_state_idx] > 0.000001)[0]
        
        start_time = time.time()
        for next_state_idx in next_state_idx_arr:
            # Note: don't go to occupied goals since we asuume these action have to success
            next_world_states_info = self.mdp_state_to_world_state(ori_state_idx, next_state_idx, start_world_state, consider_wait=True, occupied_goal=False)
            for next_world_state, cost in next_world_states_info:
                if add_rewards:
                    next_state_obj = self.state_dict[self.get_key_from_value(self.state_idx_dict, next_state_idx)].copy()
                    player_obj, pot_state, chop_state, sink_state = next_state_obj[:4]
                    remaining_orders = next_state_obj[4:-1]
                    human_subtask = next_state_obj[-1]
                    human_action = human_subtask.split('_')[0]
                    human_obj = '_'.join(human_subtask.split('_')[1:])

                    if human_action in ['pickup', 'chop', 'heat'] and human_obj not in ['garnish', 'steak']:
                        human_obj = 'None'
                    elif human_action == 'pickup' and human_obj == 'garnish':
                        human_obj = 'steak'
                    elif human_action == 'pickup' and human_obj == 'steak':
                        human_obj = 'hot_plate'
                    
                    delta_cost = (-7)*len(remaining_orders)
                    if chop_state == 'None' or chop_state == None:
                        chop_state = 0
                    else:
                        chop_state += 1
                    if sink_state == 'None' or sink_state == None:
                        sink_state = 0
                    else:
                        sink_state += 1
                    # the rewards are given in two phases. One where you prep and the other where you collect and plate.
                    # print('world info:', player_obj, pot_state, chop_state, sink_state, remaining_orders)
                    # if len(remaining_orders) > 0:
                    # if player_obj not in ['hot_plate', 'dish', 'steak'] and human_obj not in ['hot_plate', 'dish', 'steak']:
                    delta_cost += ((1.5)*pot_state + (0.4)*chop_state + (0.4)*sink_state)
                    # else:
                    if 'hot_plate' in [player_obj, human_obj]:
                        delta_cost += 2.5
                    if 'steak' in [player_obj, human_obj]:
                        delta_cost += 4.5
                    if 'dish' in [player_obj, human_obj]:
                        delta_cost += 6.5
                        # print('delta_cost:cost', delta_cost, cost)
                    # cost -= ((delta_cost*(3-len(remaining_orders)))/10)
                    # cost -= (delta_cost)*(1.1**(2-len(remaining_orders)))/5
                successor_states.append((self.get_key_from_value(self.state_idx_dict, next_state_idx), next_world_state, cost))
                if debug: print('From {} to {} costs {} in {} seconds.'.format(self.get_key_from_value(self.state_idx_dict, ori_state_idx), self.get_key_from_value(self.state_idx_dict, next_state_idx), cost, time.time()-start_time))

        return successor_states

    # def next_state_prob(self, world_state, agent_player, observed_info, human_player, vision_limit=True):
    #     """
    #     Update belief based on both human player's game logic and also it's current position and action.
    #     Belief shape is an array with size equal the length of subtask_dict.
    #     human_player is the human agent class that is in the simulator.
    #     NOTE/TODO: the human_player needs to be simulated when we later use an actual human to run experiments.
    #     """
    #     [world_num_item_in_pot, world_chop_time, world_wash_time] = observed_info
    #     belief_vector = np.zeros(len(self.subtask_dict))

    #     # human knowledge base: the observed information should be updated according to the human's vision limitation.
    #     # if vision_limit:
    #     new_knowledge_base = self.sim_human_model.get_knowledge_base(world_state, rollout_kb=True)

    #     num_item_in_pot = 0
    #     pots = new_knowledge_base['pot_states']['steak']
    #     non_emtpy_pots = pots['cooking'] + pots['ready']
    #     if len(non_emtpy_pots) > 0:
    #         num_item_in_pot = 1
        
    #     chop_time = -1
    #     non_empty_boards = new_knowledge_base['chop_states']['ready'] + new_knowledge_base['chop_states']['full']
    #     if len(non_empty_boards) > 0:
    #         chop_time = new_knowledge_base[non_empty_boards[0]].state
        
    #     wash_time = -1
    #     non_empty_sink = new_knowledge_base['sink_states']['ready'] + new_knowledge_base['sink_states']['full']
    #     if len(non_empty_sink) > 0:
    #         if new_knowledge_base[non_empty_sink[0]] is not None:
    #             wash_time = new_knowledge_base[non_empty_sink[0]].state
    #         else:
    #             wash_time = self.mdp.wash_time

    #     robot_obj = new_knowledge_base['other_player'].held_object.name if new_knowledge_base['other_player'].held_object is not None else 'None'
        
    #     print('Robot understanding of human obs = ', num_item_in_pot, chop_time, wash_time)
    #     # else:
    #     #     num_item_in_pot = world_num_item_in_pot
    #     #     chop_time = world_chop_time
    #     #     wash_time = world_wash_time
    #     #     robot_obj = agent_player.held_object.name if agent_player.held_object is not None else 'None'

    #     subtask_key = np.array([self.get_key_from_value(self.subtask_idx_dict, i) for i in range(len(belief_vector))])

    #     # get next position for human
    #     human_obj = human_player.held_object.name if human_player.held_object is not None else 'None'
    #     game_logic_prob = np.zeros((len(belief_vector)), dtype=float)

    #     print('subtasks:', self.subtask_dict.keys())
    #     for i in range(len(belief_vector)):
    #         ## estimating next subtask based on game logic
    #         game_logic_prob[i] = self._is_valid_object_subtask_pair(subtask_key[i], num_item_in_pot, chop_time, wash_time, vision_limit=vision_limit, human_obj=human_obj, other_agent_holding=robot_obj)*1.0
    
    #     game_logic_prob /= game_logic_prob.sum()
    #     game_logic_prob[game_logic_prob == 0.0] = 0.000001
    #     print('game_logic_prob =', game_logic_prob)

    #     return game_logic_prob
    
    # def cond_on_high_step(self, world_state, mdp_state_keys_and_belief, belief, agent_idx, low_level_action=True, observation=None, explicit_communcation=False):
    #     """
    #     Compute plan cost that starts from the next qmdp state defined as next_state_v().
    #     Compute the action cost of excuting a step towards the next qmdp state based on the
    #     current low level state information.

    #     next_state_v: shape(len(belief), len(action_idx)). If the low_level_action is True, 
    #         the action_dic will be representing the 6 low level action index (north, south...).
    #         If the low_level_action is False, it will be the action_dict (pickup_onion, pickup_soup...).
    #     """
    #     start_time = time.time()
    #     next_state_v = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
    #     action_cost = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)

    #     # for each subtask, obtain next mdp state but with low level location based on finishing excuting current action and subtask
    #     nxt_possible_mdp_state = []

    #     mdp_state_keys = mdp_state_keys_and_belief[0]
    #     used_belief = mdp_state_keys_and_belief[1]
    #     for i, mdp_state_key in enumerate(mdp_state_keys):
    #         mdp_state_idx = self.get_mdp_state_idx(mdp_state_key)
    #         if mdp_state_idx is not None and belief[used_belief[i]] > 0.01:
    #             agent_action_idx_arr, next_mdp_state_idx_arr = np.where(self.transition_matrix[:, mdp_state_idx] > 0.000001) # returns array(action idx), array(next_state_idx)
    #             nxt_possible_mdp_state.append([agent_action_idx_arr, next_mdp_state_idx_arr])
    #             track_laction_freq = {}
    #             track_trans = []

    #             for j, action_idx in enumerate(agent_action_idx_arr): # action_idx is encoded subtask action
    #                 next_state_idx = next_mdp_state_idx_arr[j] # high-level transition probability
    #                 after_action_world_state, cost, goals_pos = self.mdp_action_state_to_world_state(action_idx, mdp_state_idx, world_state, with_argmin=True)
    #                 value_cost = self.compute_V(after_action_world_state, self.get_key_from_value(self.state_idx_dict, next_state_idx), belief_prob=belief, belief_idx=used_belief[i], search_depth=25, search_time_limit=0.01)
    #                 joint_action, one_step_cost = self.joint_action_cost(world_state, after_action_world_state.players_pos_and_or)
    #                 nxt_obs_info = self.observe(after_action_world_state, after_action_world_state.players[agent_idx], after_action_world_state.players[abs(agent_idx-1)])
    #                 nxt_state_kb_prob = self.next_state_prob(after_action_world_state, after_action_world_state.players[agent_idx], nxt_obs_info, after_action_world_state.players[abs(agent_idx-1)], vision_limit=True)

    #                 if one_step_cost > 1000000:
    #                     one_step_cost = 1000000

    #                 if joint_action[0] in track_laction_freq.keys():
    #                     track_laction_freq[Action.ACTION_TO_INDEX[joint_action[agent_idx]]] += 1
    #                 else:
    #                     track_laction_freq[Action.ACTION_TO_INDEX[joint_action[agent_idx]]] = 1

    #                 # compute the probability of low-level action to high-level action
    #                 track_trans.append([joint_action, action_idx, next_state_idx, nxt_state_kb_prob, value_cost, one_step_cost])

    #                 # print('joint_action =', joint_action, 'one_step_cost =', one_step_cost)
    #                 # print('Action.ACTION_TO_INDEX[joint_action[agent_idx]] =', Action.ACTION_TO_INDEX[joint_action[agent_idx]])
                
    #             for trans in track_trans:
    #                 [joint_action, action_idx, next_state_idx, nxt_state_kb_prob, value_cost, one_step_cost] = trans
    #                 laction = Action.ACTION_TO_INDEX[joint_action[agent_idx]]
    #                 prob_high_cond_low_action = track_laction_freq[laction]/sum(list(track_laction_freq.values()))
    #                 next_state_v[i, laction] += (value_cost * self.transition_matrix[action_idx, mdp_state_idx, next_state_idx] * nxt_state_kb_prob[self.subtask_idx_dict[self.state_dict[self.get_key_from_value(self.state_idx_dict, next_state_idx)][-1]]] * prob_high_cond_low_action)
                    
    #                 # if explicit_communcation:
    #                 #     nxt_state_kb_prob
    #                 # print(next_state_v[i, action_idx])

    #                 ## compute one step cost with joint motion considered
    #                 action_cost[i, laction] -= (one_step_cost)*self.transition_matrix[action_idx, mdp_state_idx, next_state_idx]

    #     q = self.compute_Q(belief, next_state_v, action_cost)
    #     print('q value =', q)
    #     print('next_state_value:', next_state_v)
    #     print('action_cost:', action_cost)
    #     action_idx = self.get_best_action(q)
    #     print('get_best_action =', action_idx, '=', Action.INDEX_TO_ACTION[action_idx])
    #     print("It took {} seconds for this step".format(time.time() - start_time))
        
    #     return Action.INDEX_TO_ACTION[action_idx], None, low_level_action

    def kb_based_state(self, state_obj_key, kb, kb_key=False):
        new_state_obj = self.state_dict[state_obj_key].copy()
        
        if kb_key:
            num_item_in_pot, chop_time, wash_time, robot_obj = kb.split('.')
            num_item_in_pot = int(num_item_in_pot)
            if chop_time != 'None':
                chop_time = int(chop_time)
            if wash_time != 'None':
                wash_time = int(wash_time)
        else:
            # update state info with kb
            num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(kb)
        
        new_state_obj[0] = robot_obj
        new_state_obj[1] = num_item_in_pot
        new_state_obj[2] = chop_time
        new_state_obj[3] = wash_time

        return new_state_obj 

    def kb_based_human_subtask_state(self, curr_human_subtask, kb, kb_key=False, human_obj=None):
        
        if human_obj is None:
            curr_subtask_action = curr_human_subtask.split('_')[0]
            curr_subtask_obj = '_'.join(curr_human_subtask.split('_')[1:])
            if (curr_subtask_action in ['pickup', 'chop', 'heat']) and curr_subtask_obj not in ['steak', 'garnish']:
                human_obj = 'None'
            elif curr_human_subtask == 'pickup_steak':
                human_obj = 'hot_plate'
            elif curr_human_subtask == 'pickup_garnish':
                human_obj = 'steak'
            else:
                human_obj = curr_subtask_obj

        if kb_key:
            num_item_in_pot, chop_time, wash_time, robot_obj = kb.split('.')
            num_item_in_pot = int(num_item_in_pot)
            if chop_time != 'None':
                chop_time = int(chop_time)
            else:
                chop_time = -1
            if wash_time != 'None':
                wash_time = int(wash_time)
            else:
                wash_time = -1
        else:
            # update state info with kb
            num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(kb)
        
        actions = []
        next_world_infos = []
        if human_obj == 'None':
            if (num_item_in_pot < self.mdp.num_items_for_steak) and (human_obj != 'meat') and (robot_obj != 'meat'):
                actions.append('pickup_meat')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
            
            if (chop_time < 0) and (human_obj != 'onion') and (robot_obj != 'onion'):
                actions.append('pickup_onion')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
            
            if (wash_time < 0) and (human_obj != 'plate') and (robot_obj != 'plate'):
                actions.append('pickup_plate')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
            
            if ((chop_time >= 0) and (chop_time < self.mdp.chopping_time)) or ((chop_time < 0) and (human_obj == 'onion')):
                actions.append('chop_onion')
                next_chop_time = chop_time + 1
                if next_chop_time > self.mdp.chopping_time:
                    next_chop_time = self.mdp.chopping_time
                next_world_infos.append([robot_obj, num_item_in_pot, next_chop_time, wash_time])
            
            if ((wash_time >= 0) and (wash_time < self.mdp.wash_time)) or ((wash_time < 0) and (human_obj == 'plate')):
                actions.append('heat_hot_plate')
                next_wash_time = wash_time + 1
                if next_wash_time > self.mdp.wash_time:
                    next_wash_time = self.mdp.wash_time
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, next_wash_time])

            if (chop_time == self.mdp.chopping_time) and (wash_time == self.mdp.wash_time):
                actions.append('pickup_hot_plate')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, 'None'])
            
            if len(actions) == 0:
                actions.append('pickup_meat')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

        else:
            if human_obj == 'onion':
                actions.append('drop_onion')
                next_chop_time = chop_time
                if chop_time < 0: next_chop_time = 0 # doesn't change since no avaliable board to drop
                next_world_infos.append([robot_obj, num_item_in_pot, next_chop_time, wash_time])

            elif human_obj == 'meat':
                actions.append('drop_meat')
                next_num_item_in_pot = 1
                next_world_infos.append([robot_obj, next_num_item_in_pot, chop_time, wash_time])

            elif human_obj == 'plate':
                actions.append('drop_plate')
                next_wash_time = wash_time
                if wash_time < 0: next_wash_time = 0 # doesn't change since no avaliable sink to drop
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, next_wash_time])

            elif (human_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
                actions.append('pickup_steak')
                next_num_item_in_pot = 0
                next_world_infos.append([robot_obj, next_num_item_in_pot, chop_time, wash_time])

            elif (human_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
                actions.append('drop_hot_plate')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

            elif (human_obj == 'steak') and (chop_time == self.mdp.chopping_time):
                actions.append('pickup_garnish')
                next_world_infos.append([robot_obj, num_item_in_pot, 'None', wash_time])

            elif (human_obj == 'steak') and (chop_time < self.mdp.chopping_time):
                # actions = 'drop_steak'
                # next_obj = 'None'
                actions.append('pickup_garnish')
                next_world_infos.append([robot_obj, num_item_in_pot, 'None', wash_time])

            elif (human_obj == 'dish'):
                actions.append('deliver_dish')
                next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

            else:
                print(human_obj, [robot_obj, num_item_in_pot, chop_time, wash_time])
                raise ValueError()
            
        return actions, next_world_infos
    
    def det_kb_based_human_subtask_state(self, curr_human_subtask, kb, kb_key=False, player_obj='None'):
        
        subtask_action = curr_human_subtask.split('_')[0]
        subtask_obj = '_'.join(curr_human_subtask.split('_')[1:])
            # if (curr_subtask_action in ['pickup', 'chop', 'heat']) and curr_subtask_obj not in ['steak', 'garnish']:
            #     human_obj = 'None'
            # elif curr_human_subtask == 'pickup_steak':
            #     human_obj = 'hot_plate'
            # elif curr_human_subtask == 'pickup_garnish':
            #     human_obj = 'steak'
            # else:
            #     human_obj = curr_subtask_obj

        if kb_key:
            num_item_in_pot, chop_time, wash_time, robot_obj = kb.split('.')
            num_item_in_pot = int(num_item_in_pot)
            if chop_time != 'None':
                chop_time = int(chop_time)
            else:
                chop_time = -1
            if wash_time != 'None':
                wash_time = int(wash_time)
            else:
                wash_time = -1
        else:
            # update state info with kb
            num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(kb)

        next_num_item_in_pot = num_item_in_pot; next_chop_time = chop_time; next_wash_time = wash_time
        
        # next_world_infos = []
        # # update world info: [robot_obj, num_item_in_pot, next_chop_time, wash_time]
        # if (subtask_action in ['pickup']) and subtask_obj not in ['hot_plate', 'steak', 'garnish']:
        #     next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
        # elif curr_human_subtask == 'pickup_hot_plate':
        #     if wash_time >= self.mdp.wash_time:
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, -1])
        #     else:
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
        # elif curr_human_subtask == 'pickup_steak':
        #     if num_item_in_pot == 1:
        #         next_world_infos.append([robot_obj, 0, chop_time, wash_time])
        #     else:
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
        # elif curr_human_subtask == 'pickup_garnish':
        #     if chop_time >= self.mdp.chopping_time:
        #         next_world_infos.append([robot_obj, num_item_in_pot, -1, wash_time])
        #     else:
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
        # elif curr_human_subtask == 'drop_meat':
        #     if num_item_in_pot == 0: next_num_item_in_pot = 1 # meaning you drop at the right location instead of just on the counter
        #     next_world_infos.append([robot_obj, next_num_item_in_pot, chop_time, wash_time])
        # elif curr_human_subtask == 'drop_onion':
        #     if chop_time < 0: next_chop_time = 0 # meaning you drop at the right location instead of just on the counter
        #     next_world_infos.append([robot_obj, num_item_in_pot, next_chop_time, wash_time])
        # elif curr_human_subtask == 'drop_plate':
        #     if wash_time < 0: next_wash_time = 0 # meaning you drop at the right location instead of just on the counter
        #     next_world_infos.append([robot_obj, num_item_in_pot, chop_time, next_wash_time])
        # elif curr_human_subtask == 'chop_onion':
        #     next_chop_time = min(chop_time + 1, self.mdp.chopping_time)
        #     next_world_infos.append([robot_obj, num_item_in_pot, next_chop_time, wash_time])
        # elif curr_human_subtask == 'heat_hot_plate':
        #     next_wash_time = min(wash_time + 1, self.mdp.wash_time)
        #     next_world_infos.append([robot_obj, num_item_in_pot, chop_time, next_wash_time])
        # elif curr_human_subtask == 'deliver_dish':
        #     next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
        # elif curr_human_subtask in ['drop_hot_plate', 'drop_steak', 'drop_dish']:
        #     next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
        # else:
        #     next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
        #     print(curr_human_subtask, (num_item_in_pot, chop_time, wash_time), robot_obj)
        #     raise ValueError()
        
        # decide next subtask based on the environment
        next_subtasks = []
        if player_obj == 'None':
            if next_chop_time >= 0 and next_chop_time < self.mdp.chopping_time and (curr_human_subtask in ['chop_onion', 'drop_onion']):
                next_subtasks += ['chop_onion']
            elif next_wash_time >= 0 and next_wash_time < self.mdp.wash_time and (curr_human_subtask in ['heat_hot_plate', 'drop_plate']):
                next_subtasks += ['heat_hot_plate']
            elif next_num_item_in_pot == 0 and robot_obj != 'meat':
                next_subtasks += ['pickup_meat']
            elif next_chop_time < 0 and robot_obj != 'onion':
                next_subtasks += ['pickup_onion']
            elif next_wash_time < 0 and robot_obj != 'plate':
                next_subtasks += ['pickup_plate']
            elif (next_chop_time >= self.mdp.chopping_time or robot_obj == 'onion') and next_wash_time >= self.mdp.wash_time and next_num_item_in_pot > 0 and not (robot_obj == 'hot_plate' or robot_obj == 'steak'):
                next_subtasks += ['pickup_hot_plate']
            # elif (next_chop_time >= self.mdp.chopping_time or robot_obj == 'onion') and next_wash_time < self.mdp.wash_time and robot_obj == 'plate' and not (robot_obj == 'hot_plate' or robot_obj == 'steak'):
            #     next_subtasks += ['pickup_hot_plate']
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
            next_subtasks = [curr_human_subtask]

        # if human_obj == 'None':
        #     if ((chop_time >= 0) and (chop_time < self.mdp.chopping_time)) or ((chop_time < 0) and (human_obj == 'onion')):
        #         actions.append('chop_onion')
        #         next_chop_time = chop_time + 1
        #         if next_chop_time > self.mdp.chopping_time:
        #             next_chop_time = self.mdp.chopping_time
        #         next_world_infos.append([robot_obj, num_item_in_pot, next_chop_time, wash_time])
        #     elif ((wash_time >= 0) and (wash_time < self.mdp.wash_time)) or ((wash_time < 0) and (human_obj == 'plate')):
        #         actions.append('heat_hot_plate')
        #         next_wash_time = wash_time + 1
        #         if next_wash_time > self.mdp.wash_time:
        #             next_wash_time = self.mdp.wash_time
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, next_wash_time])

        #     elif (num_item_in_pot < self.mdp.num_items_for_steak) and (human_obj != 'meat') and (robot_obj != 'meat'):
        #         actions.append('pickup_meat')
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
            
        #     elif (chop_time < 0) and (human_obj != 'onion') and (robot_obj != 'onion'):
        #         actions.append('pickup_onion')
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])
            
        #     elif (wash_time < 0) and (human_obj != 'plate') and (robot_obj != 'plate'):
        #         actions.append('pickup_plate')
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

        #     elif (chop_time == self.mdp.chopping_time) and (wash_time == self.mdp.wash_time):
        #         actions.append('pickup_hot_plate')
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, 'None'])
            
        #     if len(actions) == 0:
        #         actions.append('pickup_plate')
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

        # else:
        #     if human_obj == 'onion':
        #         actions.append('drop_onion')
        #         next_chop_time = chop_time
        #         if chop_time < 0: next_chop_time = 0 # doesn't change since no avaliable board to drop
        #         next_world_infos.append([robot_obj, num_item_in_pot, next_chop_time, wash_time])

        #     elif human_obj == 'meat':
        #         actions.append('drop_meat')
        #         next_num_item_in_pot = 1
        #         next_world_infos.append([robot_obj, next_num_item_in_pot, chop_time, wash_time])

        #     elif human_obj == 'plate':
        #         actions.append('drop_plate')
        #         next_wash_time = wash_time
        #         if wash_time < 0: next_wash_time = 0 # doesn't change since no avaliable sink to drop
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, next_wash_time])

        #     elif (human_obj == 'hot_plate') and (num_item_in_pot == self.mdp.num_items_for_steak):
        #         actions.append('pickup_steak')
        #         next_num_item_in_pot = 0
        #         next_world_infos.append([robot_obj, next_num_item_in_pot, chop_time, wash_time])

        #     elif (human_obj == 'hot_plate') and (num_item_in_pot < self.mdp.num_items_for_steak):
        #         actions.append('drop_hot_plate')
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

        #     elif (human_obj == 'steak') and (chop_time == self.mdp.chopping_time):
        #         actions.append('pickup_garnish')
        #         next_world_infos.append([robot_obj, num_item_in_pot, 'None', wash_time])

        #     elif (human_obj == 'steak') and (chop_time < self.mdp.chopping_time):
        #         # actions = 'drop_steak'
        #         # next_obj = 'None'
        #         actions.append('pickup_garnish')
        #         next_world_infos.append([robot_obj, num_item_in_pot, 'None', wash_time])

        #     elif (human_obj == 'dish'):
        #         actions.append('deliver_dish')
        #         next_world_infos.append([robot_obj, num_item_in_pot, chop_time, wash_time])

        #     else:
        #         print(human_obj, [robot_obj, num_item_in_pot, chop_time, wash_time])
        #         raise ValueError()
            
        return next_subtasks, None
    
    def kb_compare(self, kb1, kb2):
        # kb1['pot_states']
        # kb1['sink_states']
        # kb1['chop_states']
        # kb1['other_player']

        # diff = DeepDiff(kb1, kb2)
        # return len(diff['values_changed']) == 0

        return all((kb2.get(k) == v for k, v in kb1.items()))

    # def next_kb_prob(self, start_world_state, goal_kb, h_fn=None, delivery_horizon=4, debug=False, search_time=0.01, other_agent_plan=None):
    #     """
    #     Solves A* Search problem to find sequence of low-level actions and observe the knowledge base of the new world state.

    #     Returns:
    #         ml_plan (list): plan not including starting state in form
    #             [(joint_action, successor_state), ..., (joint_action, goal_state)]
    #         cost (int): A* Search cost
    #     """
    #     start_state = start_world_state.deepcopy()
    #     start_kb = self.sim_human_model.get_knowledge_base(start_state, rollout_kb=True)
    #     if self.kb_compare(start_kb, goal_kb):
    #         return None, 0, 1

    #     if start_state.order_list is None:
    #         start_state.order_list = ["any"] * delivery_horizon
    #     else:
    #         start_state.order_list = start_state.order_list[:delivery_horizon]
        
    #     expand_fn = lambda state, depth: self.get_kb_successor_states(state, other_agent_plan[depth])
    #     goal_fn = lambda state: self.kb_compare(self.sim_human_model.get_knowledge_base(state, rollout_kb=True), goal_kb)
    #     heuristic_fn = Steak_Heuristic(self.mp).simple_heuristic

    #     search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
    #     # path_end_state, cost, over_limit = search_problem.bounded_A_star_graph_search(qmdp_root=mdp_state_key, info=False, cost_limit=search_depth, time_limit=search_time_limit)
    #     ml_plan, cost = search_problem.coupled_A_star_graph_search(info=False, time_limit=search_time, path_limit=len(other_agent_plan)-1)
    #     prob = 1
    #     if cost > 0:
    #         if other_agent_plan is not None:
    #             prob = 1/pow(Action.NUM_ACTIONS, cost)
    #         else:
    #             prob = 1/pow(Action.NUM_ACTIONS*Action.NUM_ACTIONS, cost)
        
    #     if len(ml_plan) > 1:
    #         action_plan, _ = zip(*ml_plan)
    #         return (action_plan[1][0][-1], action_plan[1][1][-1]), cost, prob
    #     else:
    #         return None, 0, 1

    def get_kb_key(self, kb):
        num_item_in_pot, chop_time, wash_time, robot_obj = self.kb_to_state_info(kb)
        kb_key = '.'.join([str(num_item_in_pot), str(chop_time), str(wash_time), str(robot_obj)])
        return kb_key

    def get_needed_kb_key(self, next_state_key, next_human_subtask, ori_state_key):
        ori_state_obj = ori_state_key.split('_')
        next_state_obj = next_state_key.split('_')

        kb_keys = []
        # Possibility 1: the next state key directly reflects the needed kb, so all values come from the next key values
        if next_state_obj[0] == 'hot' and next_state_obj[1] == 'plate':
            next_robot_obj = 'hot_plate'
            next_num_item_in_pot, next_chop_time, next_wash_time = next_state_obj[2:5]
        else:
            next_robot_obj = next_state_obj[0]
            next_num_item_in_pot, next_chop_time, next_wash_time = next_state_obj[1:4]

        if next_chop_time == 'None':
            next_chop_time = -1
        if next_wash_time == 'None':
            next_wash_time = -1

        kb_keys.append('.'.join([str(next_num_item_in_pot), str(next_chop_time), str(next_wash_time), str(next_robot_obj)]))

        # Possibility 2: robot changes world based
        kb_robot_obj = next_robot_obj
        if ori_state_obj[0] == 'hot' and ori_state_obj[1] == 'plate':
            ori_robot_obj = 'hot_plate'
            ori_num_item_in_pot, ori_chop_time, ori_wash_time = ori_state_obj[2:5]
        else:
            ori_robot_obj = ori_state_obj[0]
            ori_num_item_in_pot, ori_chop_time, ori_wash_time = ori_state_obj[1:4]

        if ori_chop_time == 'None':
            ori_chop_time = -1
        if ori_wash_time == 'None':
            ori_wash_time = -1

        if next_robot_obj == 'None': # note that we only consider changes in one step, so we assume the previous holding object is reasonable
            if int(ori_num_item_in_pot) < int(next_num_item_in_pot):
                kb_robot_obj = 'meat'
                kb_keys.append('.'.join([str(ori_num_item_in_pot), str(next_chop_time), str(next_wash_time), str(kb_robot_obj)]))

            if int(ori_chop_time) < int(next_chop_time) and int(ori_chop_time) < 0:
                kb_robot_obj = 'onion'
                kb_keys.append('.'.join([str(next_num_item_in_pot), str(ori_chop_time), str(next_wash_time), str(kb_robot_obj)]))

            if int(ori_wash_time) < int(next_wash_time) and int(ori_wash_time) < 0:
                kb_robot_obj = 'plate'
                kb_keys.append('.'.join([str(next_num_item_in_pot), str(next_chop_time), str(ori_wash_time), str(kb_robot_obj)]))

        # we do not need to consider ELSE condition: if the other state changes but the robot has something in hand, then it is assumed to be a change made previously, so we do not consider in the kb as this kb should be the kb of the previous step.

        return kb_keys

    def get_kb_successor_states(self, start_state, kb, other_agent_action=None, explore_interact=False, track_state_kb_map=None):
        successor_kb = []
        joint_motion_actions = []
        if explore_interact: 
            explore_actions = Action.ALL_ACTIONS
        else:
            explore_actions = Action.MOTION_ACTIONS
        if other_agent_action is not None:
            for a in explore_actions:
                joint_motion_action = (a, other_agent_action) if self.agent_index == 0 else (other_agent_action, a)
                joint_motion_actions.append(joint_motion_action)
        # else:
            # joint_motion_actions = itertools.product(explore_actions, explore_actions)
        # dummy_state = start_state.deepcopy()

        for joint_action in joint_motion_actions:
            # dummy_sim_human = self.sim_human_model.deepcopy(start_state)
            # dummy_sim_human.agent_index = abs(1-self.agent_index)
            # dummy_sim_human.update(dummy_state)
            
            new_positions, new_orientations = self.mdp.compute_new_positions_and_orientations(start_state.players, joint_action)
            successor_state = self.jmp.derive_state(start_state, tuple(zip(*[new_positions, new_orientations])), [joint_action])
            
            if (str(successor_state), str(kb)) in track_state_kb_map.keys():
                successor_kb.append((track_state_kb_map[(str(successor_state), str(kb))], successor_state, 1))
            else:
                tmp_kb = self.sim_human_model.get_knowledge_base(successor_state, rollout_kb=kb)
                track_state_kb_map[(str(successor_state), str(kb))] = tmp_kb

                successor_kb.append((tmp_kb, successor_state, 1))
            
            # del dummy_sim_human

        # del dummy_state
        return successor_kb

    def roll_out_for_kb(self, start_world_state, one_step_human_kb, delivery_horizon=4, debug=False, search_time=0.01, search_depth=5, other_agent_plan=None, explore_interact=False):
        start_state = start_world_state.deepcopy()
        start_kb = self.sim_human_model.get_knowledge_base(start_state, rollout_kb=one_step_human_kb)

        if start_state.order_list is None:
            start_state.order_list = ["any"] * delivery_horizon
        else:
            start_state.order_list = start_state.order_list[:delivery_horizon]

        expand_fn = lambda state, kb, depth: self.get_kb_successor_states(state, kb, None if depth > (len(other_agent_plan)-1) else other_agent_plan[depth], explore_interact=explore_interact, track_state_kb_map={})
        heuristic_fn = Steak_Heuristic(self.mp).simple_heuristic

        search_problem = SearchTree(start_state, None, expand_fn, heuristic_fn, debug=debug)
        _, _, kb_prob_dict = search_problem.bfs_track_path(lambda kb: self.get_kb_key(kb), kb_root=start_kb, info=False, time_limit=search_time, path_limit=len(other_agent_plan)-1, search_depth=search_depth)
        # _, _, kb_prob_dict = search_problem.bfs(kb_key_root=self.get_kb_key(start_kb), info=False, time_limit=search_time, path_limit=len(other_agent_plan)-1, search_depth=search_depth)
        
        return kb_prob_dict
    
    def get_human_traj_robot_stays(self, world_state, human_subtask_obj):
        # get human holding object name
        human_obj = 'None' if world_state.players[1-self.agent_index].held_object == None else world_state.players[1-self.agent_index].held_object.name

        # limit the human to take the optimal action to complete its subtask (robot's belief)
        possible_human_motion_goals, HUMAN_WAIT = self.map_action_to_location(world_state, human_subtask_obj[0], human_subtask_obj[1], p0_obj=human_obj, player_idx=abs(1-self.agent_index)) # get next world state from human subtask info (aka. mdp action translate into medium level goal position)

        human_cost, human_feature_pos = self.mp.min_cost_to_feature(world_state.players[abs(1-self.agent_index)].pos_and_or, possible_human_motion_goals, with_motion_goal=True)
        new_human_pos = human_feature_pos if human_feature_pos is not None else world_state.players[(1-self.agent_index)].get_pos_and_or()
        agent_pos = world_state.players[self.agent_index].get_pos_and_or()

        # shift by one grid if goal position overlappes with the robot agent
        if agent_pos == new_human_pos:
            _, new_human_pos = self._shift_same_goal_pos([agent_pos, new_human_pos], np.argmax(np.array([0, human_cost])))

        # get grid path from human's original position to goal
        ori_human_pos = world_state.players[(1-self.agent_index)].get_pos_and_or()
        next_las, _, _ = self.mp.get_plan(ori_human_pos, new_human_pos)

        return next_las

    def subtask_based_next_state(self, subtask, ori_world_info, next_subtask=None):
        ori_state_obj = self.state_dict[ori_world_info+'_'+subtask]
        action = subtask.split('_')[0]
        obj = '_'.join(subtask.split('_')[1:])
        tmp_state_obj = ori_state_obj.copy()
        orders = tmp_state_obj[4:-1]

        # do not change the robot agent's holding object (aka. tmp_state_obj[0])
        if action == 'drop':
            if obj == 'meat':
                if ori_state_obj[1] == 0: 
                    tmp_state_obj[1] = 1
            elif obj == 'onion':
                if ori_state_obj[2] == 'None': 
                    tmp_state_obj[2] = 0
                elif ori_state_obj[2] < self.mdp.chopping_time:
                    tmp_state_obj[2] += 1
            elif obj == 'plate':
                if ori_state_obj[3] == 'None': 
                    tmp_state_obj[3] = 0
                elif ori_state_obj[3] < self.mdp.wash_time:
                    tmp_state_obj[3] += 1
            # tmp_state_obj[0] = 'None' 
        elif action == 'pickup':
            # if obj == 'garnish':
            #     tmp_state_obj[0] = 'dish'
            # else:
            #     tmp_state_obj[0] = obj
            pass
        elif action == 'chop':
            if ori_state_obj[2] != 'None':
                if ori_state_obj[2] < self.mdp.chopping_time:
                    tmp_state_obj[2] += 1
        elif action == 'heat':
            if ori_state_obj[3] != 'None':
                if ori_state_obj[3] < self.mdp.wash_time:
                    tmp_state_obj[3] += 1
        elif action == 'deliver':
            if len(orders) > 0:
                orders.pop()
        
        new_state_obj = tmp_state_obj[:4]
        for o in orders:
            new_state_obj.append(o)
        
        if next_subtask is not None:
            new_state_obj.append(next_subtask)

        return new_state_obj
    
    def old_step(self, world_state, mdp_state_keys_and_belief, belief, agent_idx, low_level_action=True, observation=None, explicit_communcation=False, SEARCH_DEPTH=5, SEARCH_TIME=1, KB_SEARCH_DEPTH=3):
        """
        Compute plan cost that starts from the next qmdp state defined as next_state_v().
        Compute the action cost of excuting a step towards the next qmdp state based on the
        current low level state information.

        next_state_v: shape(len(belief), len(action_idx)). If the low_level_action is True, 
            the action_dic will be representing the 6 low level action index (north, south...).
            If the low_level_action is False, it will be the action_dict (pickup_onion, pickup_soup...).
        """
        start_time = time.time()
        est_next_state_v = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
        action_cost = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
        
        # Get current high-level state based on all possible beliefs of the human subtask
        mdp_state_keys = mdp_state_keys_and_belief[0]
        used_belief = mdp_state_keys_and_belief[1] # idx of the human subtask
        computed_v_cost = {} # a temp storage for computed value cost to save some computation time

        for i, mdp_state_key in enumerate(mdp_state_keys):
            if belief[used_belief[i]] > 0.2 or all(belief < 0.072):#belief[used_belief[i]] > 0.01:
                mdp_state_idx = self.get_mdp_state_idx(mdp_state_key)
                curr_belief = self.get_key_from_value(self.subtask_idx_dict, i)
                if self.sim_human_model is not None:
                    curr_kb = self.get_kb_key(self.sim_human_model.get_knowledge_base(world_state))
                else:
                    curr_kb = self.get_kb_key(world_state)

                # assume the robot doesn't move and compute the human's trajectory to complete its subtask. This is for later speeding up our roll out step.
                human_subtask = self.subtask_dict[self.state_dict[mdp_state_key][-1]]
                next_las = self.get_human_traj_robot_stays(world_state, human_subtask)
                human_next_la = next_las[0]
                next_kb_prob = np.zeros(Action.NUM_ACTIONS)

                for la in Action.ALL_ACTIONS:
                    joint_motion_action = (la, human_next_la) if self.agent_index == 0 else (human_next_la, la)
                    new_positions, new_orientations = self.mdp.compute_new_positions_and_orientations(world_state.players, joint_motion_action)
                    one_la_successor_state = self.jmp.derive_state(world_state, tuple(zip(*[new_positions, new_orientations])), [joint_motion_action])
                    la_step_cost = sum([abs(new_positions[0][0] - world_state.players[0].position[0]), abs(new_positions[0][1] - world_state.players[0].position[1])])

                    # the KB' we want to seek is the one S' needs {NOPE!!!, it is actually that the S, KB' leads to S'}
                    next_kbs_and_prob = self.roll_out_for_kb(one_la_successor_state, search_depth=KB_SEARCH_DEPTH, other_agent_plan=next_las, explore_interact=True)
                    robot_world_state_info = self.world_state_to_mdp_state_key(one_la_successor_state, one_la_successor_state.players[0], one_la_successor_state.players[1], RETURN_NON_SUBTASK=True, RETURN_OBJ=True)
                    
                    # check if human subtask is still the same
                    one_la_human_subtasks = [self.state_dict[mdp_state_key][-1]]
                    human_holding0 = None if world_state.players[1].held_object == None else world_state.players[1].held_object.name
                    human_holding1 = None if one_la_successor_state.players[1].held_object == None else one_la_successor_state.players[1].held_object.name
                    agent_holding0 = None if world_state.players[0].held_object == None else world_state.players[0].held_object.name
                    agent_holding1 = None if one_la_successor_state.players[0].held_object == None else one_la_successor_state.players[0].held_object.name

                    # update the human subtask when the human's holding changes, since this is not shown in the kb, we have a seperate if else statement.
                    

                    # if (human_holding0 != human_holding1 or human_next_la == 'interact'):
                    if (human_next_la == 'interact'):
                        human_changed_world = False
                        i_pos = Action.move_in_direction(one_la_successor_state.players[1].position, one_la_successor_state.players[1].orientation)
                        if world_state.has_object(i_pos) and one_la_successor_state.has_object(i_pos):
                            obj0 = world_state.get_object(i_pos).state
                            obj1 = one_la_successor_state.get_object(i_pos).state
                            if obj0 != obj1:
                                human_changed_world = True
                        elif world_state.has_object(i_pos) or one_la_successor_state.has_object(i_pos):
                            human_changed_world = True

                        if human_changed_world: 
                            # one_la_human_subtasks, _ = self.human_state_subtask_transition(self.state_dict[mdp_state_key][-1], robot_world_state_info[1:])
                            kb_robot_world_state_info = self.get_kb_key(self.sim_human_model.get_knowledge_base(one_la_successor_state, rollout_kb=curr_kb)).split('.')[:-1] + [robot_world_state_info[4]]
                            one_la_human_subtasks, _ = self.human_state_subtask_transition(self.state_dict[mdp_state_key][-1], kb_robot_world_state_info)
                    ## TODO: why not consider the robot changing the world? Only commented out since it seems to work better for initial steps to find the reasonable actions
                    # the idea is update the human subtask when the enivornment changes
                    elif (agent_holding0 != agent_holding1) or (self.state_dict[mdp_state_key][1:4] != robot_world_state_info[1:4]): #or (human_next_la == 'interact'):
                        if self.sim_human_model is not None:
                            one_la_kb_key = self.get_kb_key(self.sim_human_model.get_knowledge_base(one_la_successor_state, rollout_kb=curr_kb))
                        else:
                            one_la_kb_key = self.get_kb_key(one_la_successor_state)

                        # get the next low-level step environment to determine the human's subtask
                        one_la_human_subtasks, _ = np.array(self.kb_based_human_subtask_state(self.state_dict[mdp_state_key][-1], one_la_kb_key, kb_key=True), dtype=object)

                    one_la_human_subtask_count = 0
                    for one_la_human_subtask in one_la_human_subtasks:
                        one_la_state_idx = self.get_mdp_state_idx(self.world_state_to_mdp_state_key(one_la_successor_state, one_la_successor_state.players[0], one_la_successor_state.players[1], one_la_human_subtask))
                        
                        if one_la_state_idx != mdp_state_idx:

                            # since human's holding doesn't change, that means it's subtask goal has not changed, therefore, the comparison should be with the mdp_state_key's human subtask. Keep in mind that the human's subtask goal in mdp_state_key is a goal that is currently being executed and not yet complete.
                            # if (human_holding0 == human_holding1 and human_next_la != 'interact') and self.state_dict[mdp_state_key][-1] != one_la_human_subtask:
                            #     est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] += 0

                            # after_action_world_states = self.mdp_state_to_world_state(one_la_state_idx, one_la_state_idx, one_la_successor_state, with_argmin=True)
                            # else:
                            #     after_action_world_states = self.mdp_state_to_world_state(mdp_state_idx, one_la_state_idx, world_state, with_argmin=True)
                            
                            # if (agent_holding0 != agent_holding1):
                            #     one_step_cost = 1 
                            #     print(one_la_successor_state.players_pos_and_or)
                            #     # total_one_step_cost += one_step_cost
                                
                            #     # V(S')
                            #     # if (s_kb_prim_idx, one_la_state_idx) not in computed_v_cost.keys():
                            #     cost = self.compute_V(one_la_successor_state, self.get_key_from_value(self.state_idx_dict, one_la_state_idx), belief_prob=belief, belief_idx=used_belief[i], search_depth=SEARCH_DEPTH, search_time_limit=SEARCH_TIME, add_rewards=True, gamma=True)
                            #     print('one la state key: cost', self.get_key_from_value(self.state_idx_dict, one_la_state_idx), cost)

                            #     # total_v_cost += cost
                                
                            #     est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] += (cost * (1/len(one_la_human_subtasks))) #* (1/len(after_action_world_states))
                            #     # action_cost[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] -= (one_step_cost) * (1/len(one_la_human_subtasks))# * (1/one_step_cost) 
                            # else:

                            # else:
                            one_la_human_subtask_count += 1
                            after_action_world_states = self.mdp_state_to_world_state(mdp_state_idx, one_la_state_idx, world_state, with_argmin=True)
                            total_v_cost = 0
                            # total_one_step_cost = 0
                            for after_action_world_state in after_action_world_states[:,0]:
                                if self.jmp.is_valid_joint_motion_pair(one_la_successor_state.players_pos_and_or, after_action_world_state.players_pos_and_or):
                                    # already considers the interaction action
                                    _, one_step_cost = self.joint_action_cost(one_la_successor_state, after_action_world_state.players_pos_and_or, PLAN_COST='robot')#'short'
                                else:
                                    one_step_cost = (self.mdp.height*self.mdp.width)*2 
                                
                                # cost = self.compute_V(one_la_successor_state, self.get_key_from_value(self.state_idx_dict, one_la_state_idx), belief_prob=belief, belief_idx=used_belief[i], search_depth=SEARCH_DEPTH, search_time_limit=SEARCH_TIME, add_rewards=True, gamma=True)
                                cost = self.compute_V(after_action_world_state, self.get_key_from_value(self.state_idx_dict, one_la_state_idx), belief_prob=belief, belief_idx=used_belief[i], search_depth=SEARCH_DEPTH, search_time_limit=SEARCH_TIME, add_rewards=True, gamma=True)
                                print('one la state key: cost: one_step_cost: add_cost', self.get_key_from_value(self.state_idx_dict, one_la_state_idx), cost, one_step_cost, (cost/(one_step_cost*200)))
                                
                                est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] += ((cost+(cost/(one_step_cost*200))) * (1/len(after_action_world_states)) * (1/len(one_la_human_subtasks)))
                            
                            # if one_la_human_subtask_count > 0:
                            #     est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] *= (1/one_la_human_subtask_count)

                        else:
                            kb_state_values = {}
                            print(list(next_kbs_and_prob.items()))
                            for next_kb_key, [next_kb_prob, _] in next_kbs_and_prob.items():
                                kb_state_values[next_kb_key] = 0
                                nxt_human_subtasks, _ = np.array(self.kb_based_human_subtask_state(one_la_human_subtask, next_kb_key, kb_key=True), dtype=object)

                                # P(KB'|KB, L_A): for next human subtasks based on kb, we assume the human does complete the task, therefore, the world changes as well
                                # for (nxt_human_subtask, nxt_world_info) in zip(nxt_human_subtasks, nxt_world_infos):
                                    # kb_state_keys = '_'.join([str(i) for i in (self.subtask_based_next_state('_'.join(human_subtask), robot_world_state_info, next_subtask=nxt_human_subtask[0]))])
                                    # kb_state_keys = '_'.join([next_kb_key.split('.')[-1]] + [str(i) for i in nxt_world_info] + nxt_human_subtask)
                                    # s_kb_prim_idx = self.state_idx_dict[kb_state_keys]

                                    # T(S'|S, KB'): all possible next states based on KB' (human new subtask based on completing old subtask with KB)
                                # next_mdp_state_idx_arr = np.where(self.s_kb_trans_matrix[mdp_state_idx] > 0.000001)[0]
                                next_mdp_state_idx_arr = np.where(self.optimal_s_kb_trans_matrix[mdp_state_idx] > 0.000001)[0]

                                # Assuming that the next state constains states where the success is both half for both agents, we then over-write the human -subtask based on the nxt_human_subtasks. This is to match the subtask with the knowledge base the human has but still have the other state information to represent the updated world based on the success probability.
                                kb_based_next_mdp_state_idx_arr = np.array([], dtype=int)
                                # if next_kb_key != curr_kb:
                                for next_mdp_state_idx in next_mdp_state_idx_arr:
                                    kb_state_obj = self.state_dict[self.get_key_from_value(self.state_idx_dict, next_mdp_state_idx)].copy()
                                    for nxt_human_subtask in nxt_human_subtasks:
                                        kb_state_obj[-1] = nxt_human_subtask                                    
                                        kb_based_state_key = self.get_key_from_value(self.state_dict, kb_state_obj)
                                        kb_based_state_idx = self.state_idx_dict[kb_based_state_key]
                                        kb_based_next_mdp_state_idx_arr = np.concatenate((kb_based_next_mdp_state_idx_arr, [kb_based_state_idx]))

                                next_mdp_state_idx_arr = np.unique(kb_based_next_mdp_state_idx_arr)

                                # # next_mdp_state_idx_arr = np.where(self.optimal_s_kb_trans_matrix[mdp_state_idx] > 0.000001)[0]
                                # ## Next states should qualify for two criteria:
                                # # 1. is reachable from current state (If KB explore has found a KB state, then is considered reachable)
                                # if next_kb_key != curr_kb:
                                #     kb_state_obj = self.kb_based_state(mdp_state_key, next_kb_key, kb_key=True)
                                #     for j in range(len(kb_state_obj)):
                                #         if kb_state_obj[j] == -1:
                                #             kb_state_obj[j] = 'None'
                                # # 2. contains the correct next human subtask (keep this even though there will be a gap between the current state to the next state since logically you do not see the transition of the kb change directly in the computation, but it is considered by including the next_kb_prob in the computation)
                                #     next_mdp_state_idx_arr = np.array([], dtype=int)
                                #     for nxt_human_subtask in nxt_human_subtasks:
                                #         kb_state_obj[-1] = nxt_human_subtask                                    
                                #         kb_based_state_key = self.get_key_from_value(self.state_dict, kb_state_obj)
                                #         kb_based_state_idx = self.state_idx_dict[kb_based_state_key]

                                #         # # we do not use the next states of kb_based_state_idx since then we will be considering taking a high-level step in advance (wrong!)
                                #         # next_mdp_state_idx_arr = np.concatenate((next_mdp_state_idx_arr, [kb_based_state_idx]))
                                #         next_mdp_state_idx_arr = np.concatenate((next_mdp_state_idx_arr, np.where(self.s_kb_trans_matrix[kb_based_state_idx] > 0.000001)[0]))
                                #         # next_mdp_state_idx_arr = np.concatenate((next_mdp_state_idx_arr, np.where(self.optimal_s_kb_trans_matrix[kb_based_state_idx] > 0.000001)[0]))

                                # # # else:
                                # #     kb_state_obj[:-1] = self.state_dict[mdp_state_key][:-1]
                                # #     kb_based_state_idx = self.state_idx_dict[self.get_key_from_value(self.state_dict, kb_state_obj)]
                                # #     next_mdp_state_idx_arr = np.concatenate((next_mdp_state_idx_arr, np.where(self.s_kb_trans_matrix[kb_based_state_idx] > 0.000001)[0]))
                                
                                # next_mdp_state_idx_arr = np.unique(next_mdp_state_idx_arr)

                                # only compute human_subtasks that are the same as nxt_human_subtask induced by KB'
                                nxt_state_counter = 0
                                all_nxt_state_value = 0
                                all_nxt_one_step_cost = 0
                                for next_state_idx in next_mdp_state_idx_arr:
                                    # if (self.state_dict[self.get_key_from_value(self.state_idx_dict, next_state_idx)][-1] in nxt_human_subtasks) and (mdp_state_idx != next_state_idx):
                                    if (mdp_state_idx != next_state_idx):
                                        nxt_state_counter+=1
                                        print('nxt_human_subtask', self.state_dict[self.get_key_from_value(self.state_idx_dict, next_state_idx)])
                                        # if one_la_state_idx == mdp_state_idx:
                                        after_action_world_states = self.mdp_state_to_world_state(one_la_state_idx, next_state_idx, one_la_successor_state, with_argmin=True)
                                        # else:
                                        #     after_action_world_states = self.mdp_state_to_world_state(mdp_state_idx, one_la_state_idx, world_state, with_argmin=True)
                                        total_v_cost = 0
                                        total_one_step_cost = 0
                                        for after_action_world_state_info in after_action_world_states:
                                            if self.jmp.is_valid_joint_motion_pair(one_la_successor_state.players_pos_and_or, after_action_world_state_info[0].players_pos_and_or):
                                                [ai_wait, human_wait] = after_action_world_state_info[3]
                                                if ai_wait or human_wait:
                                                    _, one_step_cost = self.joint_action_cost(one_la_successor_state, after_action_world_state_info[0].players_pos_and_or, PLAN_COST= 'human' if ai_wait else 'robot')
                                                else:
                                                    _, one_step_cost = self.joint_action_cost(one_la_successor_state, after_action_world_state_info[0].players_pos_and_or, PLAN_COST='robot')# change to max such that we make sure the state is reached to accuratly represent the probability of obtaining this state value #average
                                                one_step_cost += 1 # consider the current state to the one_la_successor_state action
                                            else:
                                                one_step_cost = (self.mdp.height*self.mdp.width)*2 
                                            
                                            print('(', one_la_successor_state.players_pos_and_or, after_action_world_state_info[0].players_pos_and_or, ') one_step_cost', one_step_cost)
                                            total_one_step_cost += one_step_cost
                                            
                                            # V(S')
                                            # if (s_kb_prim_idx, next_state_idx) not in computed_v_cost.keys():
                                            cost = self.compute_V(after_action_world_state_info[0], self.get_key_from_value(self.state_idx_dict, next_state_idx), belief_prob=belief, belief_idx=used_belief[i], search_depth=SEARCH_DEPTH, search_time_limit=SEARCH_TIME, add_rewards=True, gamma=True)
                                            print('next state key: cost: add_cost', self.get_key_from_value(self.state_idx_dict, next_state_idx), cost, (cost/(one_step_cost*200)))

                                            total_v_cost += cost
                                            kb_state_values[next_kb_key] += (cost + (cost/(one_step_cost*200))) * (1/len(after_action_world_states)) 
                                                                                
                                        kb_state_values[next_kb_key] *= (next_kb_prob * (1/len(one_la_human_subtasks)))
                                        print('(',one_la_state_idx, next_state_idx, ')', 'total_v_cost =', total_v_cost, '; total one step cost:', total_one_step_cost, 'next_kb_prob:', next_kb_prob, 'num_after_action_world:', len(after_action_world_states), 'num_one_la_subtask:', len(one_la_human_subtasks))

                                if nxt_state_counter > 0:
                                    kb_state_values[next_kb_key] *= (1/nxt_state_counter)
                                    print('nxt_state_counter:', nxt_state_counter)

                            est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] += max(kb_state_values.values())
                                        
                                            # action_cost[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] -= (one_step_cost * next_kb_prob * (1/len(after_action_world_states)) * (1/len(one_la_human_subtasks)))# * (1/one_step_cost) 

                                        # if (s_kb_prim_idx, next_state_idx) not in computed_v_cost.keys():
                                        # computed_v_cost[(s_kb_prim_idx, next_state_idx)] = total_v_cost/len(after_action_world_states)
                                        # all_nxt_state_value += total_v_cost/len(after_action_world_states)
                                        # all_nxt_one_step_cost += total_one_step_cost/len(after_action_world_states)

                                        # for value_cost, one_step_cost in computed_v_cost[(s_kb_prim_idx, next_state_idx)]:
                                        # if (self.kb_idx_dict[next_kb_key], next_state_idx) in self.sprim_s_kb_trans_matrix[s_kb_prim_idx]:
                                            # est_next_state_v[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] += (computed_v_cost[(s_kb_prim_idx, next_state_idx)] * self.sprim_s_kb_trans_matrix[s_kb_prim_idx][(self.kb_idx_dict[next_kb_key], next_state_idx)] * next_kb_prob) * (1/len(after_action_world_states))
                                            
                                            # action_cost[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] -= avg_one_step_cost * self.sprim_s_kb_trans_matrix[s_kb_prim_idx][(self.kb_idx_dict[next_kb_key], next_state_idx)] * next_kb_prob
                                            
                                
                                
                                #     action_cost[i, Action.ACTION_TO_INDEX[joint_motion_action[agent_idx]]] -= (all_nxt_one_step_cost+1) * (1/nxt_state_counter) * next_kb_prob

        q = self.compute_Q(belief, est_next_state_v, action_cost)
        action_idx = self.get_best_action(q)

        if self.debug:
            print('q value =', q)
            print('next_state_value:', est_next_state_v)
            print('action_cost:', action_cost)       
            print('get_best_action =', action_idx, '=', Action.INDEX_TO_ACTION[action_idx])
            print("It took {} seconds for this step".format(time.time() - start_time))
        
        return Action.INDEX_TO_ACTION[action_idx], None, low_level_action
    
    def step(self, world_state, belief, SEARCH_DEPTH=5, SEARCH_TIME=1, KB_SEARCH_DEPTH=3, debug=False):
        '''
        The goal is to obtain the possible obtained value of a low-level action and select the one with the highest.
        '''
        start_time = time.time()
        est_next_state_v = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
        action_cost = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
        SEARCH_DEPTH = self.search_depth
        KB_SEARCH_DEPTH = self.kb_search_depth

        # update sim human
        curr_human_kb = self.sim_human_model.get_knowledge_base(world_state)
        curr_kb_key = self.get_kb_key(curr_human_kb)

        ## Reason over all belief over human's subtask
        for curr_subtask, belief_prob in belief.items():
            if belief_prob > 0.2 or all(np.array(list((belief.values()))) < 0.072):
                ## obtain human's next action based on the curr_subtask

                # gets path to complete current subtask
                next_human_las = self.get_human_traj_robot_stays(world_state, self.subtask_dict[curr_subtask])

                # gets the first action to complete subtask
                next_human_la = next_human_las[0]

                # perform one step rollout

                world_state_to_la = {}
                for la in Action.ALL_ACTIONS:
                    joint_motion_action = (la, next_human_la) if self.agent_index == 0 else (next_human_la, la)
                    # get one step roll out world state and its corresponding high-level state representation
                    # gets new position after taking action
                    new_pos, new_ori = self.mdp.compute_new_positions_and_orientations(world_state.players, joint_motion_action)

                    # get new world state after taking the action
                    rollout_world_state = self.jmp.derive_state(world_state, tuple(zip(*[new_pos, new_ori])), [joint_motion_action])
                    if rollout_world_state not in world_state_to_la.keys():
                        # add world state to dictionary with state as key and ac
                        world_state_to_la[rollout_world_state] = [la]
                    else:
                        world_state_to_la[rollout_world_state].append(la)
                
                # Compute value of each one step state
                one_step_state_info_dict = {} # one_step_world_state: [[list of (next_high_state, value of next_high_state, depended next kb, next kb prob)], [list of (one_step_cost from one_step_world_state to next_high_state_world_state)]]
                for one_step_world_state, las in world_state_to_la.items():
                    if len(one_step_world_state.order_list) == 0:
                        return las[0], None, None
                    
                    # maybe dont need this check
                    if one_step_world_state not in one_step_state_info_dict.keys():
                        if self.debug: print('one_step_world_state:', one_step_world_state.players_pos_and_or)
                        ### Map one step world state to a high-level state representation to obtain next high-level state
                        # 1
                        one_step_robot_world_str = self.world_state_to_mdp_state_key(one_step_world_state, one_step_world_state.players[0], one_step_world_state.players[1], RETURN_NON_SUBTASK=True)

                        # one step human subtask (either same subtask or changes due to a change in human's kb)
                        one_step_human_subtasks = [curr_subtask]
                        # 2
                        one_step_human_kb = self.sim_human_model.get_knowledge_base(one_step_world_state, rollout_kb=curr_human_kb)
                        one_step_human_kb_key = self.get_kb_key(one_step_human_kb)
                        if self.debug: print('one step human kb key:', one_step_human_kb_key)
                        
                        human_held_obj = 'None' if one_step_world_state.players[1-self.agent_index].held_object is None else one_step_world_state.players[1-self.agent_index].held_object.name
                        if one_step_human_kb_key != curr_kb_key or next_human_la == 'interact':
                            # get subtask based on current environment (no need to assume task completed)
                            one_step_human_subtasks, _ = self.det_kb_based_human_subtask_state(curr_subtask, one_step_human_kb_key, kb_key=True, player_obj=human_held_obj)

                        one_step_states_keys = ['_'.join([one_step_robot_world_str, one_step_human_subtask]) for one_step_human_subtask in one_step_human_subtasks]
                        if self.debug: print('one_step_human_subtasks:', one_step_human_subtasks)
                        
                        ### Get next possible kbs (roll out for n number of steps)
                        rollout_kbs_and_probs = self.roll_out_for_kb(one_step_world_state, one_step_human_kb, search_depth=KB_SEARCH_DEPTH, other_agent_plan=next_human_las[1:], explore_interact=False)
                        if self.debug: print('rollout_kbs_and_probs:', rollout_kbs_and_probs)
                        
                        ### For each possible one step state and kb, get next high-level state and its value
                        one_step_state_key_info_dict = {}
                        for one_step_state_key in one_step_states_keys:
                            rollout_kb_state_key_cost = {}
                            for rollout_kb_key, [rollout_kb_prob, one_step_to_rollout_world_cost, rollout_world_state] in rollout_kbs_and_probs.items():
                                if self.debug: print('rollout_world_state:', rollout_world_state.players_pos_and_or)
                                if self.debug: print('rollout_kb_key:', rollout_kb_key)
                                rollout_states_keys = [(one_step_human_kb_key, one_step_state_key, 0)]
                                if rollout_kb_key != one_step_human_kb_key: #and self.jmp.is_valid_joint_motion_pair(one_step_world_state.players_pos_and_or, rollout_world_state.players_pos_and_or):
                                    # check if need to update human's subtask state based on next kb
                                    rollout_states_keys = []
                                    ### TODO: check if this plan cost should be just robot 
                                    
                                    # _, one_step_to_rollout_world_cost = self.joint_action_cost(one_step_world_state, rollout_world_state.players_pos_and_or, PLAN_COST='robot')
                                    human_held_obj = 'None' if rollout_world_state.players[1-self.agent_index].held_object is None else rollout_world_state.players[1-self.agent_index].held_object.name

                                    rollout_human_subtasks, _ = self.det_kb_based_human_subtask_state(self.state_dict[one_step_state_key][-1], rollout_kb_key, kb_key=True, player_obj=human_held_obj)
                                    for rollout_human_subtask in rollout_human_subtasks:
                                        rollout_states_keys.append((rollout_kb_key, self.world_state_to_mdp_state_key(rollout_world_state, rollout_world_state.players[0], rollout_world_state.players[1], subtask=rollout_human_subtask), one_step_to_rollout_world_cost))

                                ## get next state based on transition function and average the subtasks' value
                                rollout_state_key_cost = {}
                                for _, rollout_state_key, rollout_cost in rollout_states_keys:
                                    rollout_state_idx = self.state_idx_dict[rollout_state_key]
                                    # next_mdp_state_idx_arr = np.where(self.s_kb_trans_matrix[rollout_state_idx] > 0.000001)[0]
                                    next_mdp_state_idx_arr = np.where(self.optimal_s_kb_trans_matrix[rollout_state_idx] > 0.000001)[0]
                                    # next_mdp_state_idx_arr = np.where(self.optimal_non_subtask_s_kb_trans_matrix[rollout_state_idx] > 0.000001)[0]

                                    # get value of the next states and pick the best performing one
                                    max_cost_next_state_idx = next_mdp_state_idx_arr[0]
                                    next_states_dep_ori_state_kb_prob = {} # next_state_key : [rollout_cost, next_kb_prob, values]
                                    for next_state_idx in next_mdp_state_idx_arr:
                                        next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, next_state_idx)] = [rollout_cost, [], 0]

                                        # map next states to world to get starting world state for the rollout
                                        next_state_world_states = self.mdp_state_to_world_state(rollout_state_idx, next_state_idx, rollout_world_state, with_argmin=True, cost_mode='average', consider_wait=True)
                                        for next_state_world_state in next_state_world_states:
                                            # compute next state reward to rollout n step based on assuming optimal path planning 
                                            ### TODO: check the cost if it has to be positive, reward - steps
                                            cost = self.compute_V(next_state_world_state[0], self.get_key_from_value(self.state_idx_dict, next_state_idx), search_depth=SEARCH_DEPTH, search_time_limit=SEARCH_TIME, add_rewards=True, gamma=0.8, debug=debug, kb_key=rollout_kb_key)
                                            rollout_to_next_world_cost = next_state_world_state[1]

                                            # log the cost and one step cost
                                            next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, next_state_idx)][1].append((cost, rollout_to_next_world_cost))
                                            estimate_cost = cost + (cost/((rollout_to_next_world_cost + rollout_cost)*200))
                                            next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, next_state_idx)][-1] = estimate_cost

                                            # log the max cost state idx
                                            if estimate_cost > next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx)][-1]:
                                                max_cost_next_state_idx = next_state_idx

                                    if self.debug: 
                                        print('all next_mdp_state_idx_arr:', next_states_dep_ori_state_kb_prob)
                                        print('[Best rollout state key --> next world states] ', rollout_state_key, ':', self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx), next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx)], end='\n\n')

                                    rollout_state_key_cost[rollout_state_key] = [self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx), next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx)][-1]]

                                tmp_v = np.array(list(rollout_state_key_cost.values()))
                                # avg_rollout_state_key_value = np.average(np.array(tmp_v[:,-1], dtype=float))
                                max_rollout_state_key = list(rollout_state_key_cost.keys())[np.argmax(np.array(tmp_v[:,-1], dtype=float))]
                                rollout_kb_state_key_cost[rollout_kb_key] = (max_rollout_state_key, list(rollout_state_key_cost.keys()), list(rollout_state_key_cost.values()), rollout_state_key_cost[max_rollout_state_key][-1]) # (kb state key, next state key, next state value)
                                
                                if self.debug: 
                                    print('average all rollout kb subtasks:', rollout_state_key_cost)
                                    print('[Max rollout kb with subtask] ', rollout_kb_key, max_rollout_state_key, rollout_state_key_cost[max_rollout_state_key][-1], end='\n\n')

                            tmp_v = np.array(list(rollout_kb_state_key_cost.values()), dtype=object)
                            max_rollout_kb_key = list(rollout_kb_state_key_cost.keys())[np.argmax(np.array(tmp_v[:,-1], dtype=float))]
                            one_step_state_key_info_dict[one_step_state_key] = (max_rollout_kb_key, rollout_kb_state_key_cost[max_rollout_kb_key][0], rollout_kb_state_key_cost[max_rollout_kb_key][1], rollout_kb_state_key_cost[max_rollout_kb_key][2], rollout_kb_state_key_cost[max_rollout_kb_key][-1]) # (kb key, kb state key, next state key, next state value)
                            
                            if self.debug: 
                                print('all rollout kb:', rollout_kb_state_key_cost)
                                print('[Best rollout kb for one step] ', one_step_state_key, max_rollout_kb_key, rollout_kb_state_key_cost[max_rollout_kb_key][-1], end='\n\n')

                        tmp_v = np.array(list(one_step_state_key_info_dict.values()), dtype=object)
                        # avg_la_state_key = np.average(np.array(tmp_v[:,-1], dtype=float))
                        max_la_state_key = list(one_step_state_key_info_dict.keys())[np.argmax(np.array(tmp_v[:,-1]))]
                        one_step_state_info_dict[one_step_world_state] = (max_la_state_key, list(one_step_state_key_info_dict.keys()), list(one_step_state_key_info_dict.values()), one_step_state_key_info_dict[max_la_state_key][-1])
                        
                        if self.debug: 
                            print('[Max one step state key based on one step subtask] ', max_la_state_key, ':', one_step_state_key_info_dict)
                            print('----------', end='\n\n')

                for one_step_world_state, v in one_step_state_info_dict.items():
                    for la in world_state_to_la[one_step_world_state]:
                        est_next_state_v[self.subtask_idx_dict[curr_subtask]][Action.ACTION_TO_INDEX[la]] = v[-1]

                # pick the la with highest value next state
                tmp_v = np.array(list(one_step_state_info_dict.values()), dtype=object)
                max_one_step_world_state = list(one_step_state_info_dict.keys())[np.argmax(np.array(tmp_v[:,-1], dtype=float))]
                max_la_actions = world_state_to_la[max_one_step_world_state]
                
                if self.debug: print('[Best la actions based on value] ', max_la_actions, max_one_step_world_state, one_step_state_info_dict)

        q = self.compute_Q(list(belief.values()), est_next_state_v, action_cost)
        action_idx = self.get_best_action(q)
        
        if self.debug: 
            print('q value =', q)
            print('next_state_value:', est_next_state_v)
            print('action_cost:', action_cost)       
            print('get_best_action =', action_idx, '=', Action.INDEX_TO_ACTION[action_idx])
        print("It took {} seconds for this step".format(time.time() - start_time))
        

        return Action.INDEX_TO_ACTION[action_idx], None, None

    def step_consider_interact(self, world_state, belief, SEARCH_DEPTH=5, SEARCH_TIME=1, KB_SEARCH_DEPTH=3, debug=False):
        '''
        The goal is to obtain the possible obtained value of a low-level action and select the one with the highest.
        '''
        start_time = time.time()
        est_next_state_v = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)
        action_cost = np.zeros((len(belief), Action.NUM_ACTIONS), dtype=float)

        # update sim human
        curr_human_kb = self.sim_human_model.get_knowledge_base(world_state)
        curr_kb_key = self.get_kb_key(curr_human_kb)

        ## Reason over all belief over human's subtask
        for curr_subtask, belief_prob in belief.items():
            curr_state_key = '_'.join([self.world_state_to_mdp_state_key(world_state, world_state.players[0], world_state.players[1], RETURN_NON_SUBTASK=True), curr_subtask])
            if belief_prob > 0.2 or all(np.array(list((belief.values()))) < 0.072):
                ## obtain human's next action based on the curr_subtask
                next_human_las = self.get_human_traj_robot_stays(world_state, self.subtask_dict[curr_subtask])
                next_human_la = next_human_las[0]

                # perform one step rollout
                world_state_to_la = {}
                for la in Action.ALL_ACTIONS:
                    joint_motion_action = (la, next_human_la) if self.agent_index == 0 else (next_human_la, la)
                    # get one step roll out world state and its corresponding high-level state representation
                    new_pos, new_ori = self.mdp.compute_new_positions_and_orientations(world_state.players, joint_motion_action)
                    rollout_world_state = self.jmp.derive_state(world_state, tuple(zip(*[new_pos, new_ori])), [joint_motion_action])
                    if rollout_world_state not in world_state_to_la.keys():
                        world_state_to_la[rollout_world_state] = [la]
                    else:
                        world_state_to_la[rollout_world_state].append(la)
                
                # Compute value of each one step state
                one_step_state_info_dict = {} # one_step_world_state: [[list of (next_high_state, value of next_high_state, depended next kb, next kb prob)], [list of (one_step_cost from one_step_world_state to next_high_state_world_state)]]
                for one_step_world_state, las in world_state_to_la.items():
                    if len(one_step_world_state.order_list) == 0:
                        return las[0], None, None
                    if one_step_world_state not in one_step_state_info_dict.keys():
                        if self.debug: print('one_step_world_state:', one_step_world_state.players_pos_and_or)
                        ### Map one step world state to a high-level state representation to obtain next high-level state
                        one_step_robot_world_str = self.world_state_to_mdp_state_key(one_step_world_state, one_step_world_state.players[0], one_step_world_state.players[1], RETURN_NON_SUBTASK=True)

                        # one step human subtask (either same subtask or changes due to a change in human's kb)
                        one_step_human_subtasks = [curr_subtask]
                        one_step_human_kb = self.sim_human_model.get_knowledge_base(one_step_world_state, rollout_kb=curr_human_kb)
                        one_step_human_kb_key = self.get_kb_key(one_step_human_kb)
                        if self.debug: print('one step human kb key:', one_step_human_kb_key)
                        
                        human_held_obj = 'None' if one_step_world_state.players[1-self.agent_index].held_object is None else one_step_world_state.players[1-self.agent_index].held_object.name
                        if one_step_human_kb_key != curr_kb_key or next_human_la == 'interact':
                            one_step_human_subtasks, _ = self.det_kb_based_human_subtask_state(curr_subtask, one_step_human_kb_key, kb_key=True, player_obj=human_held_obj)

                        one_step_states_keys = ['_'.join([one_step_robot_world_str, one_step_human_subtask]) for one_step_human_subtask in one_step_human_subtasks]
                        if self.debug: print('one_step_human_subtasks:', one_step_human_subtasks)
                        
                        rollout_kbs_and_probs = []
                        if not (las[0] == 'interact' or next_human_la == 'interact'):
                            ### Get next possible kbs (roll out for n number of steps)
                            rollout_kbs_and_probs = self.roll_out_for_kb(one_step_world_state, one_step_human_kb, search_depth=KB_SEARCH_DEPTH, other_agent_plan=next_human_las[1:], explore_interact=False)
                            if self.debug: print('rollout_kbs_and_probs:', rollout_kbs_and_probs)
                        
                        ### For each possible one step state and kb, get next high-level state and its value
                        one_step_state_key_info_dict = {}
                        for one_step_state_key in one_step_states_keys:
                            if las[0] == 'interact' or next_human_la == 'interact':
                                next_state_world_states = self.mdp_state_to_world_state(self.state_idx_dict[curr_state_key], self.state_idx_dict[one_step_state_key], one_step_world_state, with_argmin=True, cost_mode='average')

                                one_step_world_and_cost = []
                                max_cost = 0
                                for next_state_world_state in next_state_world_states:
                                    # compute next state reward to rollout n step based on assuming optimal path planning 
                                    ### TODO: check the cost if it has to be positive, reward - steps
                                    cost = self.compute_V(next_state_world_state[0], one_step_state_key, search_depth=SEARCH_DEPTH, search_time_limit=SEARCH_TIME, add_rewards=True, gamma=True, debug=debug)
                                    one_step_to_next_world_cost = 1

                                    # log the cost and one step cost
                                    estimate_cost = cost + (cost/(one_step_to_next_world_cost*200))

                                    # log the max cost state idx
                                    if estimate_cost > max_cost:
                                        one_step_world_and_cost = [cost, one_step_to_next_world_cost, estimate_cost]
                                        max_cost = estimate_cost

                                one_step_state_key_info_dict[one_step_state_key] = (one_step_human_kb_key, curr_state_key, one_step_state_key, one_step_world_and_cost, one_step_world_and_cost[-1])
                            else:
                                rollout_kb_state_key_cost = {}
                                for rollout_kb_key, [rollout_kb_prob, one_step_to_rollout_world_cost, rollout_world_state] in rollout_kbs_and_probs.items():
                                    if self.debug: print('rollout_world_state:', rollout_world_state.players_pos_and_or)
                                    if self.debug: print('rollout_kb_key:', rollout_kb_key)
                                    rollout_states_keys = [(one_step_human_kb_key, one_step_state_key, 0)]
                                    if rollout_kb_key != one_step_human_kb_key: #and self.jmp.is_valid_joint_motion_pair(one_step_world_state.players_pos_and_or, rollout_world_state.players_pos_and_or):
                                        # check if need to update human's subtask state based on next kb
                                        rollout_states_keys = []
                                        ### TODO: check if this plan cost should be just robot 
                                        
                                        # _, one_step_to_rollout_world_cost = self.joint_action_cost(one_step_world_state, rollout_world_state.players_pos_and_or, PLAN_COST='robot')
                                        human_held_obj = 'None' if rollout_world_state.players[1-self.agent_index].held_object is None else rollout_world_state.players[1-self.agent_index].held_object.name

                                        rollout_human_subtasks, _ = self.det_kb_based_human_subtask_state(self.state_dict[one_step_state_key][-1], rollout_kb_key, kb_key=True, player_obj=human_held_obj)
                                        for rollout_human_subtask in rollout_human_subtasks:
                                            rollout_states_keys.append((rollout_kb_key, self.world_state_to_mdp_state_key(rollout_world_state, rollout_world_state.players[0], rollout_world_state.players[1], subtask=rollout_human_subtask), one_step_to_rollout_world_cost))

                                    ## get next state based on transition function and average the subtasks' value
                                    rollout_state_key_cost = {}
                                    for _, rollout_state_key, rollout_cost in rollout_states_keys:
                                        rollout_state_idx = self.state_idx_dict[rollout_state_key]
                                        # next_mdp_state_idx_arr = np.where(self.optimal_s_kb_trans_matrix[rollout_state_idx] > 0.000001)[0]
                                        next_mdp_state_idx_arr = np.where(self.optimal_non_subtask_s_kb_trans_matrix[rollout_state_idx] > 0.000001)[0]
                                        
                                        # get value of the next states and pick the best performing one
                                        max_cost_next_state_idx = next_mdp_state_idx_arr[0]
                                        next_states_dep_ori_state_kb_prob = {} # next_state_key : [rollout_cost, next_kb_prob, values]
                                        for next_state_idx in next_mdp_state_idx_arr:
                                            next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, next_state_idx)] = [rollout_cost, [], 0]

                                            # map next states to world to get starting world state for the rollout
                                            next_state_world_states = self.mdp_state_to_world_state(rollout_state_idx, next_state_idx, rollout_world_state, with_argmin=True, cost_mode='average')

                                            for next_state_world_state in next_state_world_states:
                                                # compute next state reward to rollout n step based on assuming optimal path planning 
                                                ### TODO: check the cost if it has to be positive, reward - steps
                                                cost = self.compute_V(next_state_world_state[0], self.get_key_from_value(self.state_idx_dict, next_state_idx), search_depth=SEARCH_DEPTH, search_time_limit=SEARCH_TIME, add_rewards=True, gamma=True, debug=debug)
                                                rollout_to_next_world_cost = next_state_world_state[1]

                                                # log the cost and one step cost
                                                next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, next_state_idx)][1].append((cost, rollout_to_next_world_cost))
                                                estimate_cost = cost + (cost/((rollout_to_next_world_cost + rollout_cost)*200))
                                                next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, next_state_idx)][-1] = estimate_cost

                                                # log the max cost state idx
                                                if estimate_cost > next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx)][-1]:
                                                    max_cost_next_state_idx = next_state_idx

                                        if self.debug: 
                                            print('all next_mdp_state_idx_arr:', next_states_dep_ori_state_kb_prob)
                                            print('[Best rollout state key --> next world states] ', rollout_state_key, ':', self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx), next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx)], end='\n\n')

                                        rollout_state_key_cost[rollout_state_key] = [self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx), next_states_dep_ori_state_kb_prob[self.get_key_from_value(self.state_idx_dict, max_cost_next_state_idx)][-1]]

                                    tmp_v = np.array(list(rollout_state_key_cost.values()))
                                    # avg_rollout_state_key_value = np.average(np.array(tmp_v[:,-1], dtype=float))
                                    max_rollout_state_key = list(rollout_state_key_cost.keys())[np.argmax(np.array(tmp_v[:,-1], dtype=float))]
                                    rollout_kb_state_key_cost[rollout_kb_key] = (max_rollout_state_key, list(rollout_state_key_cost.keys()), list(rollout_state_key_cost.values()), rollout_state_key_cost[max_rollout_state_key][-1]) # (kb state key, next state key, next state value)
                                    
                                    if self.debug: 
                                        print('average all rollout kb subtasks:', rollout_state_key_cost)
                                        print('[Max rollout kb with subtask] ', rollout_kb_key, max_rollout_state_key, rollout_state_key_cost[max_rollout_state_key][-1], end='\n\n')

                                tmp_v = np.array(list(rollout_kb_state_key_cost.values()), dtype=object)
                                max_rollout_kb_key = list(rollout_kb_state_key_cost.keys())[np.argmax(np.array(tmp_v[:,-1], dtype=float))]
                                one_step_state_key_info_dict[one_step_state_key] = (max_rollout_kb_key, rollout_kb_state_key_cost[max_rollout_kb_key][0], rollout_kb_state_key_cost[max_rollout_kb_key][1], rollout_kb_state_key_cost[max_rollout_kb_key][2], rollout_kb_state_key_cost[max_rollout_kb_key][-1]) # (kb key, kb state key, next state key, next state value)
                                
                                if self.debug: 
                                    print('all rollout kb:', rollout_kb_state_key_cost)
                                    print('[Best rollout kb for one step] ', one_step_state_key, max_rollout_kb_key, rollout_kb_state_key_cost[max_rollout_kb_key][-1], end='\n\n')

                        tmp_v = np.array(list(one_step_state_key_info_dict.values()), dtype=object)
                        # avg_la_state_key = np.average(np.array(tmp_v[:,-1], dtype=float))
                        max_la_state_key = list(one_step_state_key_info_dict.keys())[np.argmax(np.array(tmp_v[:,-1]))]
                        one_step_state_info_dict[one_step_world_state] = (max_la_state_key, list(one_step_state_key_info_dict.keys()), list(one_step_state_key_info_dict.values()), one_step_state_key_info_dict[max_la_state_key][-1])
                        
                        if self.debug: 
                            print('[Max one step state key based on one step subtask] ', max_la_state_key, ':', one_step_state_key_info_dict)
                            print('----------', end='\n\n')

                for one_step_world_state, v in one_step_state_info_dict.items():
                    for la in world_state_to_la[one_step_world_state]:
                        est_next_state_v[self.subtask_idx_dict[curr_subtask]][Action.ACTION_TO_INDEX[la]] = v[-1]

                # pick the la with highest value next state
                tmp_v = np.array(list(one_step_state_info_dict.values()), dtype=object)
                max_one_step_world_state = list(one_step_state_info_dict.keys())[np.argmax(np.array(tmp_v[:,-1], dtype=float))]
                max_la_actions = world_state_to_la[max_one_step_world_state]
                
                if self.debug: print('[Best la actions based on value] ', max_la_actions, max_one_step_world_state, one_step_state_info_dict)

        q = self.compute_Q(list(belief.values()), est_next_state_v, action_cost)
        action_idx = self.get_best_action(q)
        
        if self.debug: 
            print('q value =', q)
            print('next_state_value:', est_next_state_v)
            print('action_cost:', action_cost)       
            print('get_best_action =', action_idx, '=', Action.INDEX_TO_ACTION[action_idx])
            print("It took {} seconds for this step".format(time.time() - start_time))
        

        return Action.INDEX_TO_ACTION[action_idx], None, None

    def compute_Q(self, b, v, c, knowledge_gap=0, gamma=0.9):
        '''
        P(H_a|L_a) [vector]: high-level subtask conditioned on low-level action
        T(KB'|KB, L_a) [2D matrix]: transition of the knowledge base
        B(S|o, KB') [2D matrix]: the belief distribution of state S conditioned on the current observation and new knowledge base
        T(S'|S, H_a) [3D matrix=(H_a, S, S')]: the transition matrix of current state to next state given high-level subtask
        V(S') [one value]: value of the next state
        c(S', L_a) [one value]: the cost of reaching the next state given the current low-level action
        KB_cost(KB', o) [one value]: the cost of the knowledge difference the human has with the actual world status
        
        Q value is computed with the equation:
        Q(S', L_a) = [ V(S') * [B(S|o) @ (P(H_a|L_a) * T(S'|S, H_a, KB')*P(KB'|KB, L_a))]# + KB_cost(KB', o)]
        '''
        if self.debug:
            print('b =', b)
            print('v =', v)
            print('c =', c)
        
        return b@((v*gamma)+c)+knowledge_gap
    
    def init_mdp(self, order_list):
        self.init_actions()
        self.init_human_aware_states(order_list=order_list)
        # self.init_s_kb_trans_matrix()
        # self.init_sprim_s_kb_trans_matrix()
        self.init_optimal_s_kb_trans_matrix()
        # self.init_optimal_non_subtask_s_kb_trans_matrix()