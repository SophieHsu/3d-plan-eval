class MediumMdpPlanningAgent(Agent):

    def __init__(self, mdp_planner, delivery_horizon=1, logging_level=0, auto_unstuck=False):
        # self.other_agent = other_agent
        self.delivery_horizon = delivery_horizon
        self.mdp_planner = mdp_planner
        self.logging_level = logging_level
        self.auto_unstuck = auto_unstuck
        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None

    def get_pot_status(self, state):
        pot_states = self.mdp_planner.mdp.get_pot_states(state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]

        return nearly_ready_pots

    def get_ml_states(self, state):
        num_item_in_pot = 0; pot_pos = []
        if state.objects is not None and len(state.objects) > 0:
            for obj_pos, obj_state in state.objects.items():
                # print(obj_state)
                if obj_state.name == 'soup' and obj_state.state[1] > num_item_in_pot:
                    num_item_in_pot = obj_state.state[1]
                    pot_pos = obj_pos

        # print(pot_pos, num_item_in_pot)
        state_str = self.mdp_planner.gen_state_dict_key(state, state.players[1], num_item_in_pot, state.players[0])

        # print('State = ', state_str)

        return state_str

    def action(self, state):
        pot_states = self.mdp_planner.mdp.get_pot_states(state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]

        state_str = self.get_ml_states(state)
        action = []; chosen_action = []
        if state_str not in self.mdp_planner.state_idx_dict:
            # print('State = ', state_str, ';\nNot in dictionary. Action = North')
            action = Action.ALL_ACTIONS[0]#random.choice(Action.ALL_ACTIONS)
            state.players[self.agent_index].active_log += [0]
        
        else:
            # retrieve medium level action from policy
            action_idx = self.mdp_planner.policy_matrix[self.mdp_planner.state_idx_dict[state_str]]
            return action
