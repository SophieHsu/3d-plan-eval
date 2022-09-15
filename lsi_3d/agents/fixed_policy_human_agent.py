from agent import Agent
from lsi_3d.mdp.hl_state import OvercookedState
from lsi_3d.utils.enums import HLAction

class FixedPolicyAgent(Agent):
    def __init__(self, hlp, mlp) -> None:
        self.hlp = hlp
        self.mlp = mlp

    def action(self, state:OvercookedState):
        # goes from fridge to onion
        action,object = 'stay',state.holding
        if state.in_pot < 3:
            action,object = ('pickup', 'onion')

        possible_motion_goals = self.hlp.map_action_to_location(state, (action,object))
        goal = possible_motion_goals[0]
        #start = ml_state[0] + ml_state[1]

        # should this happen outside
        paths = self.mlp.compute_motion_plan(state.ml_state, (goal,state.ml_state[0]))
        
        return paths[0]

            


            

