
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
        self.soup_states = [SoupState(soup_locations, 0)]
