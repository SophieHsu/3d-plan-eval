from lsi_3d.utils.enums import Mode


class WorldState():
    """Keeps track of items in the world that are shared accross agents
    i.e. items in pot, and orders left
    """

    def __init__(self, start_hl_state=None):
        in_pot, orders = self.parse_hl_state(start_hl_state)
        self.in_pot = in_pot
        self.sim_in_pot = in_pot
        self.orders = orders
        self.players = []
        self.max_in_pot = 2

    def parse_hl_state(self, hl_state):
        parsed = hl_state.split('_')
        return (int(parsed[1]), parsed[2:])

    def update(self, action_object):
        #in_pot, orders = self.parse_hl_state(new_hl_state)

        if action_object == ('drop', 'onion') and self.in_pot == self.max_in_pot:
            print("Attempted to add onion to pot, but pot was full. Dropping on floor")
        elif action_object == ('drop', 'onion'):
            self.sim_in_pot += 1
            self.in_pot += 1

        if action_object == ('pickup', 'soup') and self.in_pot == self.max_in_pot:
            self.in_pot = 0

        if action_object == ('deliver', 'soup'):
            self.orders = self.orders[:-1]

    def from_string(self, state_str):
        holding, in_pot, *orders = state_str.split('_')
        self.in_pot = in_pot
        self.orders = orders


class SoupState():

    def __init__(self, location, onions_in_soup) -> None:
        self.onions_in_soup = onions_in_soup
        self.location = location

    def add_onion(self):
        self.onions_in_soup += 1


class AgentState():

    def __init__(self, hl_state=None, ml_state=None, ll_state=None) -> None:
        self.hl_state = hl_state
        self.ml_state = ml_state
        self.ll_state = ll_state
        self.mode = Mode.CALC_HL_PATH

        self.start_state = hl_state

        self.holding = 'None'
        self.next_holding = 'None'

    def parse_hl_state(self, hl_state, world_state: WorldState):
        parsed = hl_state.split('_')
        self.holding = parsed[0]
        self.hl_state = f'{self.holding}_{world_state.in_pot}'

        for order in world_state.orders:
            self.hl_state += f'_{order}'

    def update(self, tracking_env, world_state):
        if len(tracking_env.obj_in_robot_hand()) > 0:
            self.holding = tracking_env.obj_in_robot_hand()[0][0]
        else:
            self.holding = 'None'

        self.hl_state = f'{self.holding}_{world_state.in_pot}'

        for order in world_state.orders:
            self.hl_state += f'_{order}'

    def update_hl_state(self, new_hl_state, world_state):

        self.hl_state = new_hl_state
        self.parse_hl_state(new_hl_state, world_state)

        return self

    def update_ml_state(self, new_ml_state):
        self.ml_state = new_ml_state

    def equal_hl(self, new_hl_state):
        return self.hl_state == new_hl_state

    def equal_ml(self, new_state):
        return (self.ml_state[0], self.ml_state[1]) == (new_state.ml_state[0],
                                                        new_state.ml_state[1])

    def is_start_state(self):
        return self.hl_state == self.start_state

    def has_object(self):
        return not self.holding == None and not self.holding == 'None'

    # def get_ready_pots(self):
    #     ready_pots = []
    #     for soup in self.soup_states:
    #         if soup.onions_in_soup == 3:
    #             ready_pots.append(soup.location)

    #     return ready_pots
