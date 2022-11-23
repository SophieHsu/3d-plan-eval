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
from lsi_3d.mdp.lsi_env import LsiEnv
from lsi_3d.mdp.hl_state import AgentState, WorldState

class FixedPolicyAgent(Agent):
    def __init__(self, hlp, mlp) -> None:
        self.hlp = hlp
        self.mlp = mlp

    def action(self, world_state:WorldState, agent_state:AgentState):
        # goes from fridge to onion

        action,object = 'stay',agent_state.holding
        if world_state.in_pot < 3 and agent_state.holding == 'None':
            action,object = ('pickup', 'onion')
            next_hl_state = f'onion_{world_state.in_pot}_onion_onion'
        elif agent_state.holding == 'onion':
            action,object = ('drop','onion')
            next_hl_state = f'None_{world_state.in_pot+1}_onion_onion'
        elif world_state.in_pot == 3 and agent_state.holding == 'None':
            action,object = ('pickup','dish')
            next_hl_state = f'dish_{world_state.in_pot}_onion_onion'
        elif agent_state.holding == 'dish' and world_state.in_pot == 3:
            action,object = ('pickup','soup')
            next_hl_state = f'soup_{world_state.in_pot}_onion_onion'
        elif agent_state.holding == 'soup':
            action,object = ('deliver','soup')
            next_hl_state = f'None_0_onion'

        possible_motion_goals = self.hlp.map_action_to_location(world_state, agent_state, (action,object))
        goal = possible_motion_goals[0]
        #start = ml_state[0] + ml_state[1]

        # should this happen outside
        #paths = self.mlp.compute_motion_plan(state.ml_state, (goal,state.ml_state[0]))
        path = self.mlp.compute_single_agent_astar_path(agent_state.ml_state, goal)
        #path = convert_path_to_mla(path)
        return next_hl_state, path, goal, (action, object)

            


            



            

