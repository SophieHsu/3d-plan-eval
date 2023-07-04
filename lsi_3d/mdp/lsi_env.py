import math
from igibson.envs.igibson_env import iGibsonEnv
from kitchen import Kitchen
from lsi_3d.agents.agent import Agent
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.mdp.hl_state import AgentState, SoupState, WorldState
from lsi_3d.mdp.lsi_mdp import LsiMdp
from lsi_3d.utils.functions import orn_to_cardinal, quat2euler
from tracking_env import TrackingEnv
from utils import real_to_grid_coord


class LsiEnv(object):
    """
        An environment wrapper for the OvercookedGridworld Markov Decision Process.

        The environment keeps track of the current state of the agent, updates
        it as the agent takes actions, and provides rewards to the agent.

        nav_env: the simulation environment that the lsi_env wraps
    """

    def __init__(self,
                 mdp: LsiMdp,
                 nav_env: iGibsonEnv,
                 tracking_env: TrackingEnv,
                 ig_human: iGibsonAgent,
                 ig_robot: iGibsonAgent,
                 kitchen: Kitchen,
                 recalc_res=None,
                 avoid_radius=None) -> None:

        #self.joint_hl_state = mdp.hl_start_state
        self.kitchen = kitchen
        self.nav_env = nav_env
        self.tracking_env = tracking_env
        self.mdp = mdp
        self.ig_human = ig_human
        self.ig_robot = ig_robot
        self.recalc_res = recalc_res
        self.avoid_radius = avoid_radius
        self.human_state = AgentState(mdp.hl_start_state,
                                      self.mdp.start_locations[0])
        self.robot_state = AgentState(mdp.hl_start_state,
                                      self.mdp.start_locations[1])
        self.world_state = WorldState(mdp.hl_start_state)
        self.world_state.players = [self.robot_state, self.human_state]

    def update_world(self):
        pot_to_onion_dict = self.tracking_env.get_pan_status()
        onions = pot_to_onion_dict[self.kitchen.pans[0]]
        real_onions = len(onions)
        self.world_state.in_pot

        self.world_state.in_pot = real_onions # + self.world_state.sim_in_pot

        if self.world_state.in_pot > self.mdp.num_items_for_soup:
            self.world_state.in_pot = self.mdp.num_items_for_soup

        # update orders left get number of bowls on table remove from original list
        order_list = self.mdp.start_order_list.copy()
        for b in self.kitchen.bowls:
            if self.tracking_env.is_item_on_table(b):
                order_list.pop()
                
        self.world_state.orders = order_list
        return

    def update_robot_hl_state(self, action_object):
        '''
        Update hl state by updated in_pot and orders for world
        and holding for specific agent
        '''
        # if action_object == (
        #         'pickup', 'soup'
        # ) and self.world_state.in_pot == self.mdp.num_items_for_soup:
        #     self.tracking_env.clear_pot()

        self.world_state.update(action_object)
        # self.robot_state.update_hl_state(next_hl_state, self.world_state)
        self.robot_state.update(self.tracking_env, self.world_state)

    def update_human_hl_state(self, next_hl_state, action_object):
        '''
        Update hl state by updated in_pot and orders for world
        and holding for specific agent
        '''
        self.world_state.update(next_hl_state, action_object)
        self.human_state.update_hl_state(next_hl_state, self.world_state)

    def update_human_sim_hl_state(self, next_hl_state, action_object):
        '''
        Update hl state by updated in_pot and orders for world
        and holding for specific agent
        '''
        self.world_state.update(next_hl_state, action_object)
        self.human_sim_state.update_hl_state(next_hl_state, self.world_state)

    def get_human_hl_state(self):
        hl_state = f'{self.human_state.holding}_{self.world_state.in_pot}'
        for order in self.world_state.orders:
            hl_state += f'_{order}'

        return hl_state

    def get_robot_hl_state(self):
        hl_state = f'{self.robot_state.holding}_{self.world_state.in_pot}'
        for order in self.world_state.orders:
            hl_state += f'_{order}'

        return hl_state

    def update_joint_ml_state(self):
        human_ml_state = self.get_ml_state(self.ig_human)
        robot_ml_state = self.get_ml_state(self.ig_robot)
        # human_sim_ml_state = self.get_ml_state(self.ig_human_sim)
        self.robot_state.ml_state = robot_ml_state
        self.human_state.ml_state = human_ml_state
        # self.human_sim_state.ml_state = human_sim_ml_state
        return (human_ml_state, robot_ml_state)

    def get_ml_state(self, agent: iGibsonAgent):
        r_r, r_c, z = agent.object.get_position()
        pos_r = round(r_r + 4.5)
        pos_c = round(r_c + 4.5)

        x, y, z, w = agent.object.get_orientation()
        r, p, y = quat2euler(x, y, z, w)
        facing = orn_to_cardinal(y)

        return (pos_r, pos_c, facing)

    def update(self, new_hl_state, new_ml_state):
        prev_in_pot = self.in_pot

        human_hl, robot_hl = new_hl_state
        human_ml, robot_ml = new_ml_state

        self.human_state.update(human_hl, human_ml)
        self.robot_state.update(robot_hl, robot_ml)

        self.parse_hl_state(new_hl_state)

        new_in_pot = self.in_pot

        if new_in_pot == (prev_in_pot + 1):
            self.soup_states[0].onions_in_soup = self.in_pot

        return self

    def map_action_to_location(self, action_object):
        location = None
        action, object = action_object
        if action == "pickup" and object == "onion":
            # location = self.tracking_env.get_closest_onion().get_position()
            return self.kitchen.get_onion_station()[0]
        elif action == "drop" and object == "onion":
            pan_status = self.tracking_env.get_pan_status()
            min_onions_left = self.mdp.num_items_for_soup + 1 
            best_pan = None
            for pan in self.kitchen.pans:
                onions = len(pan_status[pan])
                onions_left = self.mdp.num_items_for_soup - onions

                if onions_left < min_onions_left and onions_left > 0:
                    best_pan = pan
                    min_onions_left = onions_left

            if best_pan is not None:
                location = best_pan.get_position()

            # location = self.tracking_env.get_closest_pan().get_position()
        elif action == "pickup" and object == "dish":
            # find empty bowl
            for bowl in self.kitchen.bowls:
                items_in_bowl = self.tracking_env.items_in_bowl(bowl)
                if len(items_in_bowl) == 0 and self.tracking_env.is_item_on_counter(bowl):
                    location = bowl.get_position()
                    break

        elif action == "deliver" and object == "soup":
            serving_locations = self.mdp.get_serving_locations()
            for bowl in self.kitchen.bowls:
                bowl_location = real_to_grid_coord(bowl.get_position())
                if bowl_location in serving_locations:
                    serving_locations.remove(bowl_location)
            return serving_locations[0]
        elif action == "pickup" and object == "soup":
            # gets pan closest to done
            for pan in self.kitchen.pans:
                cooked_onions, uncooked_onions = self.tracking_env.is_pan_cooked(pan)
                total_onions = cooked_onions + uncooked_onions
                if total_onions >= self.mdp.num_items_for_soup:
                    location = pan.get_position()
                    break

        if location is not None:
            location = real_to_grid_coord(location)

        return location