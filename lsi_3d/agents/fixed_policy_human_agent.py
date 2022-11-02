# from agent import Agent
# from lsi_3d.mdp.hl_state import AgentState
# from lsi_3d.utils.enums import HLAction
# from lsi_3d.planners.mid_level_motion import AStarMotionPlanner

# class FixedPolicyAgent(Agent):
#     def __init__(self, hlp, mlp:AStarMotionPlanner) -> None:
#         self.hlp = hlp
#         self.mlp = mlp

#     def action(self, state:AgentState):
#         # goes from fridge to onion
#         action,object = 'stay',state.holding
#         if state.in_pot < 3 and state.holding == 'None':
#             action,object = ('pickup', 'dish')
#             next_hl_state = f'onion_{state.in_pot}_onion_onion'
#         elif state.holding == 'onion':
#             action,object = ('drop','onion')
#             next_hl_state = f'None_{state.in_pot+1}_onion_onion'

#         possible_motion_goals = self.hlp.map_action_to_location(state, (action,object))
#         goal = possible_motion_goals[0]
#         #start = ml_state[0] + ml_state[1]

#         # should this happen outside
#         paths = self.mlp.compute_single_agent_astar_path(state.ml_state[0], goal)
        
#         return next_hl_state, paths, goal

            
from agent import Agent
from lsi_3d.mdp.hl_state import AgentState
from lsi_3d.utils.enums import HLAction

class FixedPolicyAgent(Agent):
    def __init__(self, hlp, mlp) -> None:
        self.hlp = hlp
        self.mlp = mlp

    def action(self, state:AgentState):
        # goes from fridge to onion
        action,object = 'stay',state.holding
        if state.in_pot < 3 and state.holding == 'None':
            action,object = ('pickup', 'onion')
            next_hl_state = f'onion_{state.in_pot}_onion_onion'
        elif state.holding == 'onion':
            action,object = ('drop','onion')
            next_hl_state = f'None_{state.in_pot+1}_onion_onion'
        elif state.in_pot == 3 and state.holding == 'None':
            action,object = ('pickup','dish')
            next_hl_state = f'dish_{state.in_pot}_onion_onion'
        elif state.holding == 'dish' and state.in_pot == 3:
            action,object = ('pickup','soup')
            next_hl_state = f'soup_{state.in_pot}_onion_onion'

        possible_motion_goals = self.hlp.map_action_to_location(state, (action,object))
        goal = possible_motion_goals[0]
        #start = ml_state[0] + ml_state[1]

        # should this happen outside
        paths = self.mlp.compute_motion_plan(state.ml_state, (goal,state.ml_state[0]))
        
        return next_hl_state, paths[0], goal

            


            



            

