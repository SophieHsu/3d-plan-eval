import math

from igibson.envs.igibson_env import iGibsonEnv
from igibson.objects.articulated_object import URDFObject
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.environment.kitchen import Kitchen
from lsi_3d.environment.tracking_env import TrackingEnv
from lsi_3d.mdp.lsi_mdp import LsiMdp
from lsi_3d.mdp.state import AgentState, WorldState
from lsi_3d.utils.functions import find_nearby_open_spaces, norm_orn_to_cardinal, orn_to_cardinal, quat2euler
from utils import grid_to_real_coord, normalize_radians, real_to_grid_coord


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

        # self.joint_hl_state = mdp.hl_start_state
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
        self.is_interacting_with_pot = False

    def update_world(self):
        pans_status = self.tracking_env.get_pan_status().values()
        real_onions = 0
        if len(pans_status) > 0:
            real_onions = max([len(i) for i in list(pans_status)])

        if not any(list(self.tracking_env.kitchen.interact_objs.values())):
            self.world_state.in_pot = real_onions  # + self.world_state.sim_in_pot

        if self.world_state.in_pot > self.mdp.num_items_for_soup:
            self.world_state.in_pot = self.mdp.num_items_for_soup

        # update orders left get number of bowls on table remove from original list
        order_list = self.mdp.start_order_list.copy()
        for b in self.kitchen.bowls:
            if self.tracking_env.is_item_on_table(b):
                order_list.pop()

        self.world_state.orders = order_list

        self.update_joint_ml_state()

        # update holding
        if len(self.tracking_env.obj_in_robot_hand()) > 0:
            self.robot_state.holding = self.tracking_env.obj_in_robot_hand()[0][0]
        else:
            self.robot_state.holding = 'None'

        return

    def update_human_world_state(self):
        self.update_joint_ml_state()
        self.human_state.holding = self.tracking_env.get_human_holding()

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
        """
        Update hl state by updated in_pot and orders for world
        and holding for specific agent
        """
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
        # print('human ml state:', human_ml_state)
        robot_ml_state = self.get_ml_state(self.ig_robot)
        # human_sim_ml_state = self.get_ml_state(self.ig_human_sim)

        if self.kitchen.grid[human_ml_state[0]][human_ml_state[1]] == 'X':
            self.robot_state.ml_state = robot_ml_state
        else:
            pass

        if self.kitchen.grid[human_ml_state[0]][human_ml_state[1]] == 'X':
            self.human_state.ml_state = human_ml_state
        else:
            pass

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

    def transform_end_location(self, loc):
        # objects = self.igibson_env.simulator.scene.get_objects()
        objects = self.tracking_env.kitchen.static_objs
        selected_object = None
        for o in objects.keys():
            if type(o) != URDFObject:
                continue

            if type(objects[o]) == list:
                for o_loc in objects[o]:
                    if o_loc == loc: selected_object = o
                    # pos = grid_to_real_coord(loc)
            elif objects[o] == loc:
                selected_object = o

        pos = list(grid_to_real_coord(loc))
        _, ori = selected_object.get_position_orientation()
        ori = quat2euler(ori[0], ori[1], ori[2], ori[3])[2]
        # absolute transformation
        ori = normalize_radians(ori - 1.57)
        pos[0] = pos[0] + math.cos(ori)
        pos[1] = pos[1] + math.sin(ori)
        opposite_facing = normalize_radians(ori + math.pi)
        row, col = real_to_grid_coord(pos[0:2])
        card_facing = norm_orn_to_cardinal(opposite_facing)

        return row, col, card_facing

    def sort_locations(self, locations, agent_location):
        if len(agent_location) == 3:
            agent_location = agent_location[0:2]
        # euclidean distance may not work for more complicated maps
        return sorted(locations, key=lambda location: math.dist(location[0:2], agent_location))

    def map_action_to_location(self, action_object, agent_location=None, is_human=False):
        # return position and orientation given the agents location and the action object

        location = None
        action, object = action_object
        if action == "pickup" and object == "onion":
            # location = self.tracking_env.get_closest_onion().get_position()
            station_loc = self.kitchen.get_onion_station()[0]
            arrival_loc = self.transform_end_location(station_loc)
            return arrival_loc
        elif action == "drop" and object == "onion":
            pan_status = self.tracking_env.get_pan_status()
            min_onions_left = self.mdp.num_items_for_soup + 1
            agent_location_real = grid_to_real_coord(agent_location)
            pans_sorted = sorted(self.kitchen.pans, key=lambda pan: (
            self.tracking_env.get_pan_enum_status(pan), math.dist(pan.get_position()[0:2], agent_location_real)))
            location = pans_sorted[0].get_position()

        elif action == "pickup" and object == "dish":
            # find empty bowl
            bowls = self.tracking_env.get_bowls_dist_sort(is_human)
            for bowl in bowls:
                items_in_bowl = self.tracking_env.items_in_bowl(bowl)
                if len(items_in_bowl) == 0 and self.tracking_env.is_item_on_counter(bowl):
                    location = bowl.get_position()
                    break

        elif action == "deliver" and object == "soup":
            # serving_locations = self.mdp.get_serving_locations()
            serving_locations = self.tracking_env.get_table_locations()
            for bowl in self.kitchen.bowls:
                bowl_location = real_to_grid_coord(bowl.get_position())
                if bowl_location in serving_locations:
                    serving_locations.remove(bowl_location)

            # sort serving locations by distance from robot
            open_serving_locations = []
            for serving_loc in serving_locations:
                open_serving_locations += find_nearby_open_spaces(self.kitchen.grid, serving_loc)

            sorted_serv_locations = self.sort_locations(open_serving_locations, agent_location)
            return sorted_serv_locations[0] if len(sorted_serv_locations) > 0 else None
        elif action == "pickup" and object == "soup":
            # gets pan closest to done
            for pan in self.tracking_env.dist_sort(self.kitchen.pans, agent_location):
                cooked_onions, uncooked_onions = self.tracking_env.is_pan_cooked(pan)
                total_onions = cooked_onions + uncooked_onions
                if total_onions >= self.mdp.num_items_for_soup:
                    location = pan.get_position()
                    break
        elif action == "drop" and object == "dish":
            empty_counters = self.tracking_env.get_empty_counters()
            sorted_counters = self.tracking_env.dist_sort(empty_counters, grid_to_real_coord(agent_location))
            closest_empty_counter = sorted_counters[0]
            location = closest_empty_counter.get_position()

        if location is not None:
            arrival_loc = real_to_grid_coord(location)
            location = self.transform_end_location(arrival_loc)

        return location
