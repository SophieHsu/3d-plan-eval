from lsi_3d.agents.agent import Agent
from lsi_3d.planners.mid_level_motion import AStarMotionPlanner

class HlMdpPlanningAgent(Agent):

    def __init__(self, hlp_planner, mlp:AStarMotionPlanner):
        self.mdp_planner = hlp_planner
        self.mlp = mlp

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

    def action(self, hl_state, ml_state):

        state_str = hl_state

        # retrieve high level action from policy
        action_idx = self.mdp_planner.policy_matrix[self.mdp_planner.state_idx_dict[state_str]]

        keys = list(self.mdp_planner.action_idx_dict.keys())
        vals = list(self.mdp_planner.action_idx_dict.values())
        action_object_pair = self.mdp_planner.action_dict[keys[vals.index(action_idx)]]
        # print(self.mdp_planner.state_idx_dict[state_str], action_idx, action_object_pair)

        # map back the medium level action to low level action
        possible_motion_goals = self.mdp_planner.map_action_to_location(state_str, action_object_pair)
        goal = possible_motion_goals[0]
        start = ml_state[0] + ml_state[1]
        paths = self.mlp.compute_motion_plan(start, (ml_state[0],goal))
        
        return paths[1]
