from src.agents.agent import Agent
from src.environment.vision_limit_env import VisionLimitEnv
from src.mdp.state import AgentState, WorldState


class FixedPolicyAgent(Agent):
    def __init__(self, hlp, mlp, onions_for_soup) -> None:
        self.hlp = hlp
        self.mlp = mlp
        self.onions_for_soup = onions_for_soup

    def action(self, world_state: WorldState, agent_state: AgentState, robot_state: AgentState):
        action, obj = 'stay', agent_state.holding

        if agent_state.holding is 'None':
            if world_state.in_pot == 2 and robot_state.holding == 'onion':
                action, obj = ('pickup', 'dish')
                next_hl_state = f'dish_{world_state.in_pot}'
            elif world_state.in_pot == 3 and robot_state.holding != 'dish':
                action, obj = ('pickup', 'dish')
                next_hl_state = f'dish_{world_state.in_pot}'
            else:
                action, obj = ('pickup', 'onion')
                next_hl_state = f'onion_{world_state.in_pot}'
                agent_state.next_holding = 'onion'
        elif agent_state.holding == 'onion':
            action, obj = ('drop', 'onion')
            next_hl_state = f'None_{world_state.in_pot + 1}'
            agent_state.next_holding = 'None'
        elif world_state.in_pot <= 3 and agent_state.holding == 'None':
            action, obj = ('pickup', 'dish')
            next_hl_state = f'dish_{world_state.in_pot}'
            agent_state.next_holding = 'dish'
        elif agent_state.holding == 'dish' and (
                world_state.in_pot >= self.onions_for_soup - 1 or robot_state.holding == 'onion'):
            action, obj = ('pickup', 'soup')
            next_hl_state = f'soup_{world_state.in_pot}'
            agent_state.next_holding = 'soup'
        elif agent_state.holding == 'soup':
            action, obj = ('deliver', 'soup')
            next_hl_state = f'None_{world_state.in_pot}'
            agent_state.next_holding = 'None'

        # hack because sometimes when onions are out of bowl the system returns dish
        elif agent_state.holding == 'dish' and world_state.in_pot == 0:
            action, obj = ('deliver', 'soup')
            next_hl_state = f'None_{world_state.in_pot}'
            agent_state.next_holding = 'None'

        for order in world_state.orders:
            next_hl_state += f'_{order}'

        possible_motion_goals = self.hlp.map_action_to_location(world_state, agent_state, (action, obj))
        goal = possible_motion_goals[0]

        return next_hl_state, goal, (action, obj)


class SteakFixedPolicyHumanAgent(Agent):
    def __init__(self, env: VisionLimitEnv, human_sim) -> None:
        self.env = env
        self.human_sim = human_sim

    def action(self, world_state: WorldState, agent_state: AgentState, robot_state: AgentState):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """

        counter_objects = self.env.tracking_env.kitchen.counters
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
                action, obj = ('chop', 'onion')
            elif rinsing and not hot_plate_ready:
                action, obj = ('heat', 'hot_plate')
            elif not steak_nearly_ready and len(world_state.orders) > 0 and not other_has_meat:
                action, obj = ('pickup', 'meat')

            elif not chopping and not garnish_ready and not other_has_onion:
                action, obj = ('pickup', 'onion')
            elif not rinsing and not hot_plate_ready and not other_has_plate:
                action, obj = ('pickup', 'plate')

            elif (garnish_ready or other_has_onion) and hot_plate_ready and not (
                    other_has_hot_plate or other_has_steak):
                action, obj = ('pickup', 'hot_plate')

            elif (garnish_ready or other_has_onion) and steak_nearly_ready and other_has_plate and not (
                    other_has_hot_plate or other_has_steak) and not hot_plate_ready:
                action, obj = ('pickup', 'hot_plate')
            else:
                next_order = world_state.orders[-1]

                if next_order == 'steak':
                    # pick up plate first since that is the first empty key obj when in the plating stage
                    action, obj = ('pickup', 'plate')

        else:
            player_obj = agent_state.holding

            if player_obj == 'onion':
                action, obj = ('drop', 'onion')

            elif player_obj == 'meat':
                action, obj = ('drop', 'meat')

            elif player_obj == "plate":
                action, obj = ('drop', 'plate')

            elif player_obj == 'hot_plate' and (other_has_meat or steak_nearly_ready):
                action, obj = ('pickup', 'steak')

            elif player_obj == 'steak' and (garnish_ready or other_has_onion or chopping):
                action, obj = ('pickup', 'garnish')

            elif player_obj == 'dish':
                action, obj = ('deliver', 'dish')

        possible_motion_goals = self.env.map_action_to_location((action, obj), agent_state.ml_state)
        goal = possible_motion_goals
        self.action_object = (action, obj)
        return None, goal, (action, obj)
