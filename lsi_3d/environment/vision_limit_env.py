import math
from igibson import object_states
from igibson.envs.igibson_env import iGibsonEnv
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.environment.kitchen import Kitchen
from lsi_3d.environment.lsi_env import LsiEnv
from lsi_3d.environment.tracking_env import TrackingEnv
from lsi_3d.mdp.lsi_mdp import LsiMdp
from lsi_3d.mdp.state import AgentState, WorldState
from lsi_3d.utils.functions import find_nearby_open_spaces
from utils import grid_to_real_coord, real_to_grid_coord

class VisionLimitEnv(LsiEnv):
    def __init__(self, mdp: LsiMdp, nav_env: iGibsonEnv, tracking_env: TrackingEnv, ig_human: iGibsonAgent, ig_robot: iGibsonAgent, kitchen: Kitchen, recalc_res=None, avoid_radius=None) -> None:
        super().__init__(mdp, nav_env, tracking_env, ig_human, ig_robot, kitchen, recalc_res, avoid_radius)

        self.robot_state = AgentState()
        self.human_state = AgentState()
        self.world_state = WorldState(orders=['steak','steak'])
        self.world_state.state_dict['obj_curr_idx'] = 0
        self.world_state.state_dict['objects'] = {}

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
                for item in pans_status[pan]:
                    if item in self.kitchen.steaks:
                        self.world_state.state_dict['pot_states']['ready'].append(pan)
                    else:
                        self.world_state.state_dict['pot_states']['cooking'].append(pan)

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
                for onion in chop_status[chop]:
                    if onion.current_index == 1:
                        self.world_state.state_dict['chop_states']['ready'].append(chop)
                    else:
                        self.world_state.state_dict['chop_states']['full'].append(chop)

        # sinks
        sink_status = self.tracking_env.get_sink_status(same_location=True)
        self.world_state.state_dict['sink_states'] = {}
        self.world_state.state_dict['sink_states']['empty'] = []
        self.world_state.state_dict['sink_states']['ready'] = []
        self.world_state.state_dict['sink_states']['full'] = []
        for sink in sink_status:
            plates = sink_status[sink]

            if len(plates) == 0:
                self.world_state.state_dict['sink_states']['empty'].append(sink)

                if len(self.kitchen.ready_sinks) > 0 or len(self.kitchen.rinsing_sinks) > 0:
                    self.kitchen.ready_sinks.clear()
                    self.kitchen.rinsing_sinks.clear()
            elif self.kitchen.overcooked_object_states[plates[0]]['state'] != 2:
                self.world_state.state_dict['sink_states']['full'].append(sink)
            elif self.kitchen.overcooked_object_states[plates[0]]['state'] == 2:
                self.world_state.state_dict['sink_states']['ready'].append(sink)

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

        if self.tracking_env.obj_in_human_hand() is not None:
            self.human_state.holding = self.tracking_env.get_human_holding()
        else:
            self.human_state.holding = 'None'

        #### The following state dict items are for keeping accurate state for overcooked api
        
        # if plate is now on sink then make it a hot plate in dictionary
        # for plate in self.kitchen.plates:
        #     sink = self.env.tracking_env.get_closest_sink(self.ig_robot.object.get_position())
        #     plate = self.env.tracking_env.get_closest_plate_in_sink(self.ig_robot.object.get_position())
        #     # if plate not in objects
        #     # turn it into a hot plate
        #     if plate is not None:
        #         id = self.env.world_state.state_dict['obj_to_id'][plate]
        #         curr_state = self.env.world_state.state_dict['objects'][plate]['state']
        #         state = 0 if curr_state is None else curr_state + 1
        #         self.env.world_state.state_dict['objects'][plate] = {
        #                 'id': id + 1,
        #                 'name': 'hot_plate',
        #                 'position': self.env.to_overcooked_grid(real_to_grid_coord(plate.get_position())),
        #                 'state':state
        #             }

        for obj, state in self.kitchen.overcooked_object_states.items():
            state['position'] = self.to_overcooked_grid(self.tracking_env.get_position(obj))
                
        for plate in self.kitchen.plates:
            for s,ps in self.tracking_env.get_sink_status().items():
                for p in ps:
                    # if plate is in sink and its state is none, change to 0
                    p_state = self.kitchen.overcooked_object_states[p]
                    if p_state['state'] == None:
                        self.kitchen.drop_plate(p)
                    elif s in self.world_state.state_dict['sink_states']['ready'] and p_state['state']<2:
                        self.kitchen.heat_plate(p)

                if real_to_grid_coord(s.get_position()) == real_to_grid_coord(plate.get_position()) and sink in self.kitchen.ready_sinks:     
                    # if plate is in same location as sink and sink is in ready sinks then heat it
                    p_state = self.kitchen.overcooked_object_states[plate]
                    if p_state['state']<2:
                        self.kitchen.heat_plate(p)

        # if meat is now steak change id and object
        for object, state in self.kitchen.overcooked_object_states.items():
            # meat becomes steak human holding nothing and meat is on
            if state['name'] == 'meat' and self.human_state.holding == 'None':
                # if object is in same location as pan then make it a steak
                pans = self.tracking_env.get_pan_status().items()
                for p,objs in pans:
                    if self.tracking_env.get_position(p) == self.tracking_env.get_position(object):
                    # if p.states[object_states.Inside].get_value(object):

                        self.kitchen.drop_meat(object)

        

        # if onion is now on chopping board change state and position
        for object, state in self.kitchen.overcooked_object_states.items():
            if state['name'] == 'onion' or state['name'] == 'garnish':
                boards = self.tracking_env.get_chopping_board_status().items()
                for b,objs in boards:
                    # for o in objs:
                    o_state = self.kitchen.overcooked_object_states[object]
                    if self.tracking_env.get_position(b) == self.tracking_env.get_position(object):
                        if o_state['state'] == None:
                            self.kitchen.drop_onion(object)
                        elif object.current_index == 1 and o_state['state'] < 2:
                            self.kitchen.chop_onion(object)

        in_human_hand = self.tracking_env.obj_in_human_hand()
        self.kitchen.update_overcooked_robot_holding()
        in_robot_hand = self.kitchen.overcooked_robot_holding
        
        # if hot plate is in same location as the pan then set the overcooked position to be the same as the agent
        # overcooked expects hot plate to be in human hand
        for obj,st in self.kitchen.overcooked_object_states.items():
            for pan in self.kitchen.pans:
                
                if 'hot_plate' in st['name']:
                    obj_position = real_to_grid_coord(obj.get_position())
                    if obj_position == real_to_grid_coord(pan.get_position()):
                        human_dist = math.dist(self.human_state.ml_state[0:2], obj_position)
                        robot_dist = math.dist(self.robot_state.ml_state[0:2], obj_position)
                        if human_dist < robot_dist:
                            in_human_hand = obj
                        else:
                            in_robot_hand = obj

        # need to also make plate with steak when at same position as cutting board to pass holding to overcooked
        for obj,st in self.kitchen.overcooked_object_states.items():
            for chop in self.kitchen.chopping_boards:
                
                if 'steak' in st['name']:
                    obj_position = real_to_grid_coord(obj.get_position())
                    if obj_position == real_to_grid_coord(chop.get_position()):
                        human_dist = math.dist(self.human_state.ml_state[0:2], obj_position)
                        robot_dist = math.dist(self.robot_state.ml_state[0:2], obj_position)
                        if human_dist < robot_dist:
                            in_human_hand = obj
                        else:
                            in_robot_hand = obj

        # keep track of objects and the order they were picked up as they now have an id
        # in_human_hand = self.kitchen.update_overcooked_human_holding(self.human_state.holding, self.tracking_env.obj_in_human_hand())
        
        self.kitchen.overcooked_object_states
        self.human_state.state_dict['human_holding'] = {}
        if in_human_hand is not None:
            if in_human_hand not in self.kitchen.overcooked_object_states.keys():
                self.kitchen.overcooked_object_states[in_human_hand] = {
                    "id": self.kitchen.overcooked_max_id,
                    "name": self.kitchen.get_name(in_human_hand),
                    "position": self.to_overcooked_grid(self.human_state.ml_state[0:2]),
                    "state": None
                }
                self.kitchen.overcooked_obj_to_id[in_human_hand] = self.kitchen.overcooked_max_id
                self.kitchen.overcooked_max_id += 1
            self.human_state.state_dict['human_holding'] = self.kitchen.overcooked_object_states[in_human_hand]
            self.human_state.state_dict['human_holding']['position'] = self.to_overcooked_grid(self.human_state.ml_state[0:2])
            self.kitchen.overcooked_object_states[in_human_hand]['position'] = self.to_overcooked_grid(self.human_state.ml_state[0:2])

        
        # do same for robot
        self.robot_state.state_dict['robot_holding'] = {}
        if in_robot_hand is not None:
            if in_robot_hand not in self.kitchen.overcooked_object_states.keys():
                self.kitchen.overcooked_object_states[in_robot_hand] = {
                    "id": self.kitchen.overcooked_max_id,
                    "name": self.kitchen.get_name(in_robot_hand),
                    "position": self.to_overcooked_grid(self.robot_state.ml_state[0:2]),
                    "state": None
                }
                self.kitchen.overcooked_obj_to_id[in_robot_hand] = self.kitchen.overcooked_max_id
                self.kitchen.overcooked_max_id += 1
            self.robot_state.state_dict['robot_holding'] = self.kitchen.overcooked_object_states[in_robot_hand]
            self.robot_state.state_dict['robot_holding']['position'] = self.to_overcooked_grid(self.robot_state.ml_state[0:2])
            self.kitchen.overcooked_object_states[in_robot_hand]['position'] = self.to_overcooked_grid(self.robot_state.ml_state[0:2])



        # check that objects are not in holding list and object list
        unowned_objects = self.kitchen.overcooked_object_states.copy()
        for obj,st in self.kitchen.overcooked_object_states.items():
            if obj == in_robot_hand:
                unowned_objects.pop(obj)

            if obj == in_human_hand:
                unowned_objects.pop(obj)

            
        
        self.world_state.state_dict['unowned_objects'] = unowned_objects

        self.world_state.state_dict['objects'] = self.kitchen.overcooked_object_states
        return
    
    def map_action_to_location(self, action_object, agent_location=None, is_human=False):
        # return position and orientation given the agents location and the action object
        agent_pos = grid_to_real_coord(agent_location)
        location = None
        action, object = action_object
        if action == "pickup" and object == "onion":
            # location = self.tracking_env.get_closest_onion().get_position()
            station_loc = self.kitchen.get_green_onion_station()[0]
            arrival_loc = self.transform_end_location(station_loc)
            return arrival_loc
        elif action == "drop" and object == "onion":
            station_loc = self.kitchen.get_chopping_station()[0]
            arrival_loc = self.transform_end_location(station_loc)
            return arrival_loc
        elif action == "chop" and object == "onion":
            location = self.tracking_env.get_closest_chopping_board(agent_pos).get_position()
        elif action == "pickup" and object == "garnish":
            chopped_onion = self.tracking_env.get_closest_chopped_onion(agent_pos)
            if chopped_onion is not None:
                location = self.tracking_env.get_position(chopped_onion)
            else:
                location = self.tracking_env.get_closest_chopping_board(agent_pos).get_position()
        elif action == "drop" and object == "meat":
            pan_status = self.tracking_env.get_pan_status()
            agent_location_real = grid_to_real_coord(agent_location)
            pans_sorted = sorted(self.kitchen.pans, key=lambda pan: (self.tracking_env.get_pan_enum_status(pan), math.dist(pan.get_position()[0:2], agent_location_real)))
            location = pans_sorted[0].get_position()


        elif action == "deliver" and (object == "soup" or object == "dish"):
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

        elif action == "drop" and object == "plate":
            sink = self.tracking_env.get_closest_sink(agent_pos)
            location = sink.get_position()

        elif action == "pickup" and object == "meat":
            station_loc = self.kitchen.get_onion_station()[0]
            arrival_loc = self.transform_end_location(station_loc)
            return arrival_loc
        
        elif action == "pickup" and object == "steak":
            ready_pans = self.world_state.state_dict["pot_states"]["ready"]
            if len(ready_pans) == 0:
                location = self.tracking_env.get_closest_pan(agent_pos).get_position()
            else:
                location = ready_pans[0].get_position()
        
        elif action == "pickup" and object == "hot_plate":
            if len(self.kitchen.ready_sinks) > 0:
                sink = self.kitchen.ready_sinks[0]
            else:
                sink = self.tracking_env.get_closest_sink(agent_pos)
            location = sink.get_position()

        elif action == "drop" and (object == "hot_plate" or object == "steak"):
            location = self.tracking_env.get_empty_counters()[0].get_position()

        elif action == "heat" and object == "hot_plate":
            location = self.tracking_env.get_closest_sink(agent_pos).get_position()

        if location is not None:
            arrival_loc = real_to_grid_coord(location)
            location = self.transform_end_location(arrival_loc)

        return location
    
    def to_overcooked_pos_or(self, ml_state):
        r,c = ml_state[0:2]

        # add one for overcooked border padding
        pos = c+1,r+1
        ori = ml_state[2]
        
        if ori == 'N':
            o = (0,-1)
        elif ori == 'E':
            o = (1,0)
        elif ori == 'S':
            o = (0,1)
        elif ori == 'W':
            o = (-1,0)
        else:
            o = None
        return pos, o
    
    def to_overcooked_holding(self, holding):
        if holding == {}:
            return None
        else:
            return holding
        
    def to_overcooked_grid(self, loc):
        pos = loc
        r,c = pos
        x,y = c+1,r+1
        return (x,y)
    
    def to_overcooked_state(self):
        state_dict = {}
        h_p, h_o = self.to_overcooked_pos_or(self.human_state.ml_state)
        r_p, r_o = self.to_overcooked_pos_or(self.robot_state.ml_state)
        
        state_dict['players'] = [
            {
                'position': r_p,
                'orientation': r_o,
                'held_object': self.to_overcooked_holding(self.robot_state.state_dict['robot_holding'])
            },{
                'position': h_p,
                'orientation': h_o,
                'held_object': self.to_overcooked_holding(self.human_state.state_dict['human_holding'])
            },
        ]
        # state_dict['objects'] = [v for k,v in self.world_state.state_dict['objects'].items()]
        state_dict['objects'] = [v for k,v in self.world_state.state_dict['unowned_objects'].items()]
        state_dict['order_list'] = self.world_state.orders

        return state_dict
#
# {
#     "players": [
#         {
#             "position": [
#                 6,
#                 3
#             ],
#             "orientation": [
#                 0,
#                 1
#             ],
#             "held_object": null
#         },
#         {
#             "position": [
#                 7,
#                 5
#             ],
#             "orientation": [
#                 1,
#                 0
#             ],
#             "held_object": {
#                 "id": 2,
#                 "name": "meat",
#                 "position": [
#                     7,
#                     5
#                 ],
#                 "state": null
#             }
#         }
#     ],
#     "objects": [
#         {
#             "id": 1,
#             "name": "hot_plate",
#             "position": [
#                 7,
#                 1
#             ],
#             "state": 2
#         }
#     ],
#     "order_list": [
#         "steak",
#         "steak"
#     ]
# }