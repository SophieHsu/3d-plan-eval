
import string
from tokenize import String

from lsi_3d.utils.enums import ExecutingState

class WorldState(object):
    def __init__(self, start_hl_state):
        self.parse_hl_state(start_hl_state)

    def parse_hl_state(self, hl_state):
        parsed = hl_state.split('_')
        self.in_pot = int(parsed[1])
        self.orders = parsed[2:]

    def update(self, new_hl_state):
        self.parse_hl_state(new_hl_state)

class SoupState(object):
    def __init__(self, location, onions_in_soup) -> None:
        self.onions_in_soup = onions_in_soup
        self.location = location

    def add_onion(self):
        self.onions_in_soup += 1

class AgentState(object):
    def __init__(self, hl_state, ml_state, ll_state = None) -> None:
        self.hl_state = hl_state
        self.ml_state = ml_state
        self.ll_state = ll_state
        self.executing_state = ExecutingState.NO_ML_PATH

        self.holding = 'None'
        self.parse_hl_state(hl_state)
    
    def parse_hl_state(self, hl_state):
        parsed = hl_state.split('_')
        self.holding = parsed[0]
        self.hl_state = hl_state

    def update_hl_state(self, new_hl_state):

        self.hl_state = new_hl_state
        self.parse_hl_state(new_hl_state)

        return self

    def update_ml_state(self, new_ml_state):
        self.ml_state = new_ml_state

    # def get_ready_pots(self):
    #     ready_pots = []
    #     for soup in self.soup_states:
    #         if soup.onions_in_soup == 3:
    #             ready_pots.append(soup.location)

    #     return ready_pots
