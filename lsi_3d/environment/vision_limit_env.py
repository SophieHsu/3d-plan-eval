import math
from igibson.envs.igibson_env import iGibsonEnv
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.environment.kitchen import Kitchen
from lsi_3d.environment.lsi_env import LsiEnv
from lsi_3d.environment.tracking_env import TrackingEnv
from lsi_3d.mdp.lsi_mdp import LsiMdp
from lsi_3d.mdp.state import AgentState, WorldState
from utils import grid_to_real_coord

class VisionLimitEnv(LsiEnv):
    def __init__(self, mdp: LsiMdp, nav_env: iGibsonEnv, tracking_env: TrackingEnv, ig_human: iGibsonAgent, ig_robot: iGibsonAgent, kitchen: Kitchen, recalc_res=None, avoid_radius=None) -> None:
        super().__init__(mdp, nav_env, tracking_env, ig_human, ig_robot, kitchen, recalc_res, avoid_radius)

        self.robot_state = AgentState()
        self.human_state = AgentState()
        self.world_state = WorldState(orders=['steak','steak'])

    def update_world(self):
        # pans
        pans_status = self.tracking_env.get_pan_status()
        self.world_state.state_dict['pot_states'] = {}
        self.world_state.state_dict['pot_states']['empty'] = []
        self.world_state.state_dict['pot_states']['ready'] = []
        self.world_state.state_dict['pot_states']['cooking'] = []
        for pan in pans_status:
            if len(pans_status[pan]) == 0:
                self.world_state.state_dict['pot_states']['empty'].append(pan)
            elif len(pans_status[pan]) == self.kitchen.onions_for_soup:
                self.world_state.state_dict['pot_states']['ready'].append(pan)

        # chopping boards
        chop_status = self.tracking_env.get_chopping_board_status()
        self.world_state.state_dict['chop_states'] = {}
        self.world_state.state_dict['chop_states']['empty'] = []
        self.world_state.state_dict['chop_states']['ready'] = []
        self.world_state.state_dict['chop_states']['full'] = []
        for chop in chop_status:
            if len(chop_status[chop]) == 0:
                self.world_state.state_dict['chop_states']['empty'].append(chop)
            else:
                self.world_state.state_dict['chop_states']['ready'].append(chop)

        # sinks
        sink_status = self.tracking_env.get_sink_status()
        self.world_state.state_dict['sink_states'] = {}
        self.world_state.state_dict['sink_states']['empty'] = []
        self.world_state.state_dict['sink_states']['ready'] = []
        self.world_state.state_dict['sink_states']['full'] = []
        for sink in sink_status:
            if len(sink_status[sink]) == 0:
                self.world_state.state_dict['sink_states']['empty'].append(chop)
            else:
                self.world_state.state_dict['sink_states']['ready'].append(chop)


        # update orders left get number of bowls on table remove from original list
        order_list = self.world_state.init_orders.copy()
        for p in self.kitchen.plates:
            if self.tracking_env.is_item_on_table(p):
                order_list.pop()
                
        self.world_state.orders = order_list

        self.update_joint_ml_state()

        # update holding
        if len(self.tracking_env.obj_in_robot_hand()) > 0:
            self.robot_state.holding = self.tracking_env.obj_in_robot_hand()[0][0]
        else:
            self.robot_state.holding = 'None'

        return
    
    def map_action_to_location(self, action_object, agent_location=None, is_human=False):
        # return position and orientation given the agents location and the action object

        location = None
        action, object = action_object
        if action == "pickup" and object == "onion":
            # location = self.tracking_env.get_closest_onion().get_position()
            station_loc = self.kitchen.get_green_onion_station()[0]
            arrival_loc = self.transform_end_location(station_loc)
            return arrival_loc
        elif action == "drop" and object == "onion":
            pan_status = self.tracking_env.get_pan_status()
            agent_location_real = grid_to_real_coord(agent_location)
            pans_sorted = sorted(self.kitchen.pans, key=lambda pan: (self.tracking_env.get_pan_enum_status(pan), math.dist(pan.get_position()[0:2], agent_location_real)))
            location = pans_sorted[0].get_position()
        elif action == "deliver" and object == "dish":
            # dish includes plate, steak, and garnish
            dish = self.tracking_env.get_dish()
            location = dish.get_position()

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
        
        elif action == "pickup" and object == "plate":
            sorted_plates = self.tracking_env.dist_sort(self.kitchen.plates, grid_to_real_coord(agent_location))
            closest_plate = sorted_plates[0]
            location = closest_plate.get_position()

        elif action == "pickup" and object == "meat":
            station_loc = self.kitchen.get_onion_station()[0]
            arrival_loc = self.transform_end_location(station_loc)
            return arrival_loc

        if location is not None:
            arrival_loc = real_to_grid_coord(location)
            location = self.transform_end_location(arrival_loc)

        return location