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
from lsi_3d.environment.lsi_env import LsiEnv
from lsi_3d.environment.vision_limit_env import VisionLimitEnv
from lsi_3d.mdp.state import AgentState, WorldState
from utils import grid_to_real_coord

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

        # hack because sometimes when onions are out of bowl the system returns dish
        elif agent_state.holding == 'dish' and world_state.in_pot == 0:
            action,object = ('deliver','soup')
            next_hl_state = f'None_{world_state.in_pot}'
            agent_state.next_holding = 'None'
        
        for order in world_state.orders:
            next_hl_state += f'_{order}'

        possible_motion_goals = self.hlp.map_action_to_location(world_state, agent_state, (action,object))
        goal = possible_motion_goals[0]
        #start = ml_state[0] + ml_state[1]

        # should this happen outside
        # paths = self.mlp.compute_motion_plan(state.ml_state, (goal,state.ml_state[0]))
        # path = self.mlp.compute_single_agent_astar_path(agent_state.ml_state, goal)
        # path = convert_path_to_mla(path)
        return next_hl_state, goal, (action, object)
    
class SteakFixedPolicyHumanAgent(Agent):
    def __init__(self, env:VisionLimitEnv, human_sim) -> None:
        self.env = env
        self.human_sim = human_sim

    def action(self, world_state:WorldState, agent_state:AgentState, robot_state:AgentState):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """

        # TODO: update the state based on kb
        # self.knowledge_base = self.update()
        
        # player = state.players[self.agent_index]
        # other_player = self.knowledge_base['other_player']
        # am = self.mlp.ml_action_manager

        counter_objects = self.env.tracking_env.kitchen.counters
        # sink_status = self.knowledge_base['sink_states']
        # chopping_board_status = self.knowledge_base['chop_states']
        # pot_states_dict = self.knowledge_base['pot_states']
        sink_status = world_state.state_dict['sink_states']
        chopping_board_status = world_state.state_dict['chop_states']
        pot_states_dict = world_state.state_dict['pot_states']
        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = world_state.orders[-1]

        if curr_order == 'any':
            ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
            cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
        else:
            empty_pot = pot_states_dict['empty']
            ready_soups = pot_states_dict['ready']
            cooking_soups = pot_states_dict['cooking']

        other_holding = robot_state.holding

        steak_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        other_has_dish = other_holding == 'dish'
        other_has_hot_plate = other_holding == 'hot_plate'
        other_has_steak = other_holding == 'steak'

        garnish_ready = len(chopping_board_status['ready']) > 0
        chopping = len(chopping_board_status['full']) > 0
        board_empty = len(chopping_board_status['empty']) > 0
        hot_plate_ready = len(sink_status['ready']) > 0
        rinsing = len(sink_status['full']) > 0
        sink_empty = len(sink_status['empty']) > 0
        motion_goals = []

        ready_soups = pot_states_dict['ready']
        cooking_soups = pot_states_dict['cooking']

        steak_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        other_has_dish = robot_state.holding == 'dish'
        other_has_hot_plate = robot_state.holding == 'hot_plate'
        other_has_steak = robot_state.holding == 'steak'
        other_has_meat = robot_state.holding == 'meat'
        other_has_onion = robot_state.holding == 'onion'
        other_has_plate = robot_state.holding == 'plate'

        garnish_ready = len(chopping_board_status['ready']) > 0
        chopping = len(chopping_board_status['full']) > 0
        board_empty = len(chopping_board_status['empty']) > 0
        hot_plate_ready = len(sink_status['ready']) > 0
        rinsing = len(sink_status['full']) > 0
        sink_empty = len(sink_status['empty']) > 0

        if agent_state.holding == 'None':

            if chopping and not garnish_ready:
                # motion_goals = am.chop_onion_on_board_actions(state, knowledge_base=self.knowledge_base)
                action,object = ('chop','onion')
            elif rinsing and not hot_plate_ready:
                action,object = ('heat','hot_plate')
                # motion_goals = am.heat_plate_in_sink_actions(state, knowledge_base=self.knowledge_base)
            elif not steak_nearly_ready and len(world_state.orders) > 0 and not other_has_meat:
                action,object = ('pickup','meat')
                # motion_goals = am.pickup_meat_actions(counter_objects, knowledge_base=self.knowledge_base)
            
            elif not chopping and not garnish_ready and not other_has_onion:
                action,object = ('pickup','onion')
            elif not rinsing and not hot_plate_ready and not other_has_plate:
                action,object = ('pickup','plate')
            
                # motion_goals = am.pickup_plate_actions(counter_objects, state, knowledge_base=self.knowledge_base)
            #elif garnish_ready and hot_plate_ready and not (other_has_hot_plate or other_has_steak):
            elif (garnish_ready or other_has_onion) and hot_plate_ready and not (other_has_hot_plate or other_has_steak):
                action,object = ('pickup','hot_plate')

            elif (garnish_ready or other_has_onion) and steak_nearly_ready and other_has_plate and not (other_has_hot_plate or other_has_steak) and not hot_plate_ready:
                # motion_goals += am.pickup_hot_plate_from_sink_actions(counter_objects, state, knowledge_base=self.knowledge_base)
                action, object = ('pickup','hot_plate')
                # if len(motion_goals) == 0:
                #     if other_has_plate:
                #         motion_goals += self.knowledge_base['sink_states']['full']
                #         if len(motion_goals) == 0:
                #             motion_goals += self.knowledge_base['sink_states']['empty']
                # motion_goals = am.pickup_hot_plate_from_sink_actions(counter_objects, state, knowledge_base=self.knowledge_base)
            else:
                next_order = world_state.orders[-1]

                if next_order == 'steak': #pick up plate first since that is the first empty key object when in the plating stage
                    action,object = ('pickup', 'plate')
                    # motion_goals = am.pickup_plate_actions(counter_objects, knowledge_base=self.knowledge_base)

        else:
            player_obj = agent_state.holding

            if player_obj == 'onion':
                action, object = ('drop', 'onion')
                # motion_goals = am.put_onion_on_board_actions(state, knowledge_base=self.knowledge_base)
            
            elif player_obj == 'meat':
                action,object = ('drop', 'meat')
                # motion_goals = am.put_meat_in_pot_actions(pot_states_dict)

            elif player_obj == "plate":
                action,object = ('drop', 'plate')
                # motion_goals = am.put_plate_in_sink_actions(counter_objects, state, knowledge_base=self.knowledge_base)

            elif player_obj == 'hot_plate' and (other_has_meat or steak_nearly_ready):
                action,object = ('pickup', 'steak')
                # motion_goals = am.pickup_steak_with_hot_plate_actions(pot_states_dict, only_nearly_ready=True)

            elif player_obj == 'steak' and (garnish_ready or other_has_onion or chopping):
                action,object = ('pickup', 'garnish')
                # motion_goals = am.add_garnish_to_steak_actions(state, knowledge_base=self.knowledge_base)

            elif player_obj == 'dish':
                action,object = ('deliver', 'dish')
                # motion_goals = am.deliver_dish_actions()

            # else:
            #     motion_goals += am.place_obj_on_counter_actions(state)


        possible_motion_goals = self.env.map_action_to_location(
            (action, object), agent_state.ml_state)
        goal = possible_motion_goals
        # self.next_hl_state = next_hl_state
        self.action_object = (action, object)
        return None, goal, (action, object)

            


            



            

