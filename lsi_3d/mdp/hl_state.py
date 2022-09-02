
import string
from tokenize import String


class SoupState(object):
    def __init__(self, location, onions_in_soup) -> None:
        self.onions_in_soup = onions_in_soup
        self.location = location

    def add_onion(self):
        self.onions_in_soup += 1

class OvercookedState(object):
    def __init__(self, hl_state, ml_state, ll_state = None, soup_locations = []) -> None:
        self.hl_state = hl_state
        self.ml_state = ml_state
        self.ll_state = ll_state

        if soup_locations != None:
            self.soup_states = [SoupState(location, 0) for location in soup_locations]

        self.holding = 'None'
        self.in_pot = 0
        self.orders = []
        self.parse_hl_state(hl_state)
    
    def parse_hl_state(self, hl_state):
        parsed = hl_state.split('_')
        self.holding = parsed[0]
        self.in_pot = int(parsed[1])
        self.orders = parsed[2:]

    def update(self, new_hl_state, new_ml_state):
        prev_in_pot = self.in_pot

        self.hl_state = new_hl_state
        self.ml_state = new_ml_state

        self.parse_hl_state(new_hl_state)

        new_in_pot = self.in_pot

        if new_in_pot == (prev_in_pot+1):
            self.soup_states[0].add_onion()

        return self

    def get_ready_pots(self):
        ready_pots = []
        for soup in self.soup_states:
            if soup.onions_in_soup == 3:
                ready_pots.append(soup.location)

        return ready_pots
