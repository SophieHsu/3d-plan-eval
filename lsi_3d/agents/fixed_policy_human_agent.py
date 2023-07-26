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
    def __init__(self, hlp, mlp, onions_for_soup) -> None:
        self.hlp = hlp
        self.mlp = mlp
        self.onions_for_soup = onions_for_soup

    def action(self, world_state:WorldState, agent_state:AgentState, robot_state:AgentState):
        # goes from fridge to onion

        # TODO: Add this code from overcooked greedy agent
        # soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        # other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'

        # if soup_nearly_ready and not other_has_dish:
        #     motion_goals = am.pickup_dish_actions(counter_objects)
        # else:
        #     next_order = None
        #     if state.num_orders_remaining > 1:
        #         next_order = state.next_order

        #     if next_order == 'onion':
        #         motion_goals = am.pickup_onion_actions(counter_objects)
        #     elif next_order == 'tomato':
        #         motion_goals = am.pickup_tomato_actions(counter_objects)
        #     elif next_order is None or next_order == 'any':
        #         motion_goals = am.pickup_onion_actions(counter_objects) + am.pickup_tomato_actions(counter_objects)
        action,object = 'stay',agent_state.holding


        if agent_state.holding == 'None':
            if world_state.in_pot == 2 and robot_state.holding == 'onion':
                    action,object = ('pickup', 'dish')
                    next_hl_state = f'dish_{world_state.in_pot}'
            elif world_state.in_pot == 3 and robot_state.holding != 'dish':
                action,object = ('pickup', 'dish')
                next_hl_state = f'dish_{world_state.in_pot}'
            else:
                action,object = ('pickup', 'onion')
                next_hl_state = f'onion_{world_state.in_pot}'
                agent_state.next_holding = 'onion'
        elif agent_state.holding == 'onion':
            action,object = ('drop','onion')
            next_hl_state = f'None_{world_state.in_pot+1}'
            agent_state.next_holding = 'None'
        elif world_state.in_pot <= 3 and agent_state.holding == 'None':
            action,object = ('pickup','dish')
            next_hl_state = f'dish_{world_state.in_pot}'
            agent_state.next_holding = 'dish'
        elif agent_state.holding == 'dish' and (world_state.in_pot >= self.onions_for_soup-1 or robot_state.holding == 'onion'):
            action,object = ('pickup','soup')
            #world_state.in_pot = 0
            next_hl_state = f'soup_{world_state.in_pot}'
            agent_state.next_holding = 'soup'
        elif agent_state.holding == 'soup':
            action,object = ('deliver','soup')
            next_hl_state = f'None_{world_state.in_pot}'
            agent_state.next_holding = 'None'

        
        for order in world_state.orders:
            next_hl_state += f'_{order}'

        possible_motion_goals = self.hlp.map_action_to_location(world_state, agent_state, (action,object))
        goal = possible_motion_goals[0]
        #start = ml_state[0] + ml_state[1]

        # should this happen outside
        #paths = self.mlp.compute_motion_plan(state.ml_state, (goal,state.ml_state[0]))
        # path = self.mlp.compute_single_agent_astar_path(agent_state.ml_state, goal)
        #path = convert_path_to_mla(path)
        return next_hl_state, goal, (action, object)

            


            



            

