import math
import time

from igibson import object_states
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots.manipulation_robot import IsGraspingState
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.environment.actions import ACTION_COMMANDS
from lsi_3d.environment.kitchen import Kitchen
from lsi_3d.environment.lsi_env import LsiEnv
from lsi_3d.environment.tracking_env import TrackingEnv
from lsi_3d.mdp.lsi_mdp import LsiMdp
from lsi_3d.mdp.state import AgentState, WorldState
from lsi_3d.utils.functions import find_nearby_open_spaces
from utils import grid_to_real_coord, quat2euler, real_to_grid_coord

STICKY_ITEMS = False


class VisionLimitEnv(LsiEnv):
    def __init__(self, mdp: LsiMdp, nav_env: iGibsonEnv, tracking_env: TrackingEnv, ig_human: iGibsonAgent,
                 ig_robot: iGibsonAgent, kitchen: Kitchen, recalc_res=None, avoid_radius=None, log_dict={}) -> None:
        super().__init__(mdp, nav_env, tracking_env, ig_human, ig_robot, kitchen, recalc_res, avoid_radius)

        self.robot_state = AgentState()
        self.human_state = AgentState()
        self.world_state = WorldState(orders=['steak', 'steak'])
        self.world_state.state_dict['obj_curr_idx'] = 0
        self.world_state.state_dict['objects'] = {}
        self.log_time = time.time()
        self.log_dict = log_dict

    def update_world(self):

        human_in_hand_assigned = False

        # pans
        pans_status = self.tracking_env.get_pan_status()
        self.world_state.state_dict['pot_states'] = {}
        self.world_state.state_dict['pot_states']['empty'] = []
        self.world_state.state_dict['pot_states']['ready'] = []
        self.world_state.state_dict['pot_states']['cooking'] = []
        for pan in pans_status:
            if len(pans_status[pan]) == 0:
                self.world_state.state_dict['pot_states']['empty'].append(pan)
                self.kitchen.stove.states[object_states.ToggledOn].set_value(False)
            elif len(pans_status[pan]) == self.kitchen.onions_for_soup:
                for item in pans_status[pan]:
                    self.kitchen.stove.states[object_states.ToggledOn].set_value(True)
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

        for obj, state in self.kitchen.overcooked_object_states.items():
            position = self.tracking_env.get_real_position(obj)
            x, y, z = position
            if z < 0.1 and x < 20 and y < 20:
                self.tracking_env.set_item_on_closest_counter(obj)

        for obj, state in self.kitchen.overcooked_object_states.items():
            if obj != self.kitchen.robot_stash_dish:
                state['position'] = self.to_overcooked_grid(self.tracking_env.get_position(obj))
            else:
                state['position'] = self.to_overcooked_grid(self.robot_state.ml_state[0:2])

        # start mapping from igibson to overcooked
        in_human_hand = self.tracking_env.obj_in_human_hand()
        self.kitchen.update_overcooked_robot_holding()
        in_robot_hand = self.kitchen.overcooked_robot_holding

        # update holding
        if len(self.tracking_env.obj_in_robot_hand()) > 0:
            self.robot_state.holding = self.tracking_env.obj_in_robot_hand()[0][0]
        else:
            # overcooked wont allow object in same location but not holding
            self.robot_state.holding = 'None'
            for obj, state in self.kitchen.overcooked_object_states.items():
                if state['position'] == self.to_overcooked_grid(self.robot_state.ml_state[0:2]):
                    # self.robot_state.holding = obj
                    in_robot_hand = obj

        if self.tracking_env.obj_in_human_hand() is not None:
            self.human_state.holding = self.tracking_env.get_human_holding()
        else:
            # overcooked wont allow object in same location but not holding
            self.human_state.holding = 'None'
            for obj, state in self.kitchen.overcooked_object_states.items():
                # grab previous location to check in middle of interact
                if state['position'] == self.to_overcooked_grid(self.human_state.ml_state[0:2]):
                    in_human_hand = obj

        for plate in self.kitchen.plates:
            for s, ps in self.tracking_env.get_sink_status(same_location=True).items():
                for p in ps:
                    # if plate is in sink and its state is none, change to 0
                    p_state = self.kitchen.overcooked_object_states[p]
                    if p_state['state'] is None:
                        self.kitchen.execute_action(ACTION_COMMANDS.DROP, p)
                    elif s in self.world_state.state_dict['sink_states']['ready'] and p_state['state'] < 2:
                        self.kitchen.heat_plate(p)

                if real_to_grid_coord(s.get_position()) == real_to_grid_coord(
                        plate.get_position()) and sink in self.kitchen.ready_sinks:
                    # if plate is in same location as sink and sink is in ready sinks then heat it
                    p_state = self.kitchen.overcooked_object_states[plate]
                    if p_state['state'] is None:
                        self.kitchen.execute_action(ACTION_COMMANDS.DROP, plate)
                    elif p_state['state'] < 2:
                        self.kitchen.heat_plate(p)

                    if in_human_hand == plate:
                        in_human_hand = None

        # if meat is now steak change id and object
        for object, state in self.kitchen.overcooked_object_states.items():
            # meat becomes steak human holding nothing and meat is on
            if state['name'] == 'meat' or state['name'] == 'steak':
                # if object is in same location as pan then make it a steak
                pans = self.tracking_env.get_pan_status().items()
                for p, objs in pans:
                    if self.tracking_env.get_position(p) == self.tracking_env.get_position(object):
                        # if p.states[object_states.Inside].get_value(object):
                        if state['name'] == 'meat':
                            self.kitchen.drop_meat(object)
                        if in_human_hand == object:
                            in_human_hand = None

        # if onion is now on chopping board change state and position
        for object, state in self.kitchen.overcooked_object_states.items():
            if state['name'] == 'onion' or state['name'] == 'garnish':
                boards = self.tracking_env.get_chopping_board_status().items()
                for b, objs in boards:
                    # for o in objs:
                    o_state = self.kitchen.overcooked_object_states[object]
                    if self.tracking_env.get_position(b) == self.tracking_env.get_position(object):
                        if o_state['state'] is None:
                            self.kitchen.execute_action(ACTION_COMMANDS.DROP, object)
                        elif object.current_index == 1 and o_state['state'] < 2:
                            self.kitchen.execute_action(ACTION_COMMANDS.CHOP, object, name='garnish', state=2)

                        if in_human_hand == object:
                            in_human_hand = None

        # if hot plate is in same location as the pan then set the overcooked position to be the same as the agent
        # overcooked expects hot plate to be in human hand
        for obj, st in self.kitchen.overcooked_object_states.items():
            for pan in self.kitchen.pans:

                if 'hot_plate' in st['name']:
                    # sets hot plate to in hand obj if in same position as pan
                    obj_position = real_to_grid_coord(obj.get_position())
                    if obj_position == real_to_grid_coord(pan.get_position()):
                        human_dist = math.dist(self.human_state.ml_state[0:2], obj_position)
                        robot_dist = math.dist(self.robot_state.ml_state[0:2], obj_position)
                        if human_dist < robot_dist:
                            in_human_hand = obj
                        else:
                            in_robot_hand = obj

        # need to also make plate with steak when at same position as cutting board to pass holding to overcooked
        for obj, st in self.kitchen.overcooked_object_states.items():
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

        # also need to modify in hand object to be steak or dish if inside hot plate
        for obj, st in self.kitchen.overcooked_object_states.copy().items():
            if st['name'] == 'hot_plate':
                is_dish = False

                onion = self.tracking_env.get_closest_chopped_onion(self.ig_human.object.get_position())
                plate = self.tracking_env.get_closest_plate(self.ig_human.object.get_position())

                human_created_dish = False

                if onion is not None and (onion.states[object_states.Inside].get_value(obj) or onion.states[
                    object_states.OnTop].get_value(obj)):
                    human_created_dish = True

                if (human_created_dish and obj != self.kitchen.robot_stash_dish) or (
                        obj == self.kitchen.robot_stash_dish and self.kitchen.robot_carrying_dish):  # or self.kitchen.robot_carrying_dish:
                    print('setting hot plate state to dish')
                    is_dish = True

                    if human_created_dish:
                        position = self.kitchen.overcooked_object_states[obj]['position']
                    if obj == self.kitchen.robot_stash_dish and self.kitchen.robot_carrying_dish:
                        position = self.to_overcooked_grid(self.robot_state.ml_state[0:2])

                    self.kitchen.overcooked_object_states[obj] = {
                        'id': self.kitchen.overcooked_max_id,
                        'name': 'dish',
                        'position': position,
                        'state': None
                    }

                    self.kitchen.overcooked_hot_plates_now_dish.append(obj)

                    self.kitchen.overcooked_max_id += 1
                    st['in_hot_plate'] = [onion]

                    for steak in self.kitchen.steaks:
                        if steak.states[object_states.Inside].get_value(obj):
                            # add tag saying steak is in hot plate
                            st['in_hot_plate'].append(steak)

                if obj == self.kitchen.robot_stash_steak_bowl:
                    st['position'] = self.to_overcooked_grid(self.robot_state.ml_state[0:2])
                if not is_dish:

                    for steak in self.kitchen.steaks:
                        # if steak.states[object_states.Inside].get_value(obj): in_same_pos = real_to_grid_coord(
                        # steak.get_position()) == real_to_grid_coord(obj.get_position())
                        in_same_pos = self.to_overcooked_grid(real_to_grid_coord(steak.get_position())) == st[
                            'position']
                        if steak.states[object_states.Inside].get_value(obj) or in_same_pos:
                            if obj == in_human_hand:
                                in_human_hand = steak

                                if STICKY_ITEMS:
                                    # For items floating above steak
                                    x, y, z = obj.get_position()
                                    steak.set_position([x, y, z + 0.03])

                            elif obj == in_robot_hand:
                                in_robot_hand = steak

                            # add tag saying steak is in hot plate
                            st['in_hot_plate'] = [steak]

        # but if now we have a dish, then that will always be in someones hand
        for obj, st in self.kitchen.overcooked_object_states.copy().items():
            if st['name'] == 'dish':
                if st['name'] == 'dish' and st['state'] != 'delivered':

                    if obj == self.kitchen.robot_stash_dish:
                        in_robot_hand = obj
                    else:
                        in_human_hand = obj

                    onion = self.tracking_env.get_closest_chopped_onion(obj.get_position())
                    st['in_hot_plate'] = [onion]
                    steak = self.tracking_env.get_closest_steak(obj.get_position())
                    st['in_hot_plate'].append(steak)

                    # if STICKY_ITEMS:
                    # For items floating above dish
                    if in_human_hand == obj:
                        current_idx = 1
                        x, y, z = obj.get_position()
                        for item in st['in_hot_plate']:
                            if item.name == 'green_onion_multiplexer':
                                for sub_obj in onion.objects:
                                    # if sub_obj.get_position()[2] < 0.1: sub_obj.states[
                                    # object_states.Inside].set_value(obj, True, use_ray_casting_method=True)
                                    sub_obj.set_position([x, y, z + 0.05 * current_idx])
                                    current_idx += 1
                            else:
                                # item.states[object_states.Inside].set_value(obj, True, use_ray_casting_method=True)
                                item.set_position([x, y, z + 0.05 * current_idx])
                                current_idx += 1

                if st['name'] == 'dish' and self.tracking_env.is_item_on_table(obj) and st['state'] != 'delivered':
                    st['state'] = 'delivered'
                    print('delivered dish!')
                    onion = self.tracking_env.get_closest_chopped_onion(obj.get_position())
                    st['in_hot_plate'] = [onion]

                    for i, sub_obj in enumerate(onion.objects):
                        sub_obj.set_position([50 + 3 * i, 50, 0.5])

                    self.kitchen.overcooked_object_states.pop(onion)
                    steak = self.tracking_env.get_closest_steak(obj.get_position())
                    self.kitchen.overcooked_object_states.pop(steak)
                    st['in_hot_plate'].append(steak)
                    if obj == in_human_hand:
                        in_human_hand = None
                    elif obj == in_robot_hand:
                        in_robot_hand = None
                elif st['name'] == 'dish' and self.tracking_env.is_item_on_table(obj) and st['state'] == 'delivered':
                    if obj == in_human_hand:
                        in_human_hand = None
                    elif obj == in_robot_hand:
                        in_robot_hand = None

        # keep track of objects and the order they were picked up as they now have an id in_human_hand =
        # self.kitchen.update_overcooked_human_holding(self.human_state.holding, self.tracking_env.obj_in_human_hand())

        self.human_state.state_dict['human_holding'] = {}
        if in_human_hand is not None and in_human_hand in self.kitchen.get_mobile_objects() and not self.tracking_env.in_start_location(
                in_human_hand):
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
            self.human_state.state_dict['human_holding']['position'] = self.to_overcooked_grid(
                self.human_state.ml_state[0:2])
            self.kitchen.overcooked_object_states[in_human_hand]['position'] = self.to_overcooked_grid(
                self.human_state.ml_state[0:2])

        # do same for robot
        self.robot_state.state_dict['robot_holding'] = {}
        if in_robot_hand is not None and in_robot_hand in self.kitchen.get_mobile_objects():
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
            self.robot_state.state_dict['robot_holding']['position'] = self.to_overcooked_grid(
                self.robot_state.ml_state[0:2])
            self.kitchen.overcooked_object_states[in_robot_hand]['position'] = self.to_overcooked_grid(
                self.robot_state.ml_state[0:2])

        if in_robot_hand == in_human_hand and in_robot_hand != None:
            print('same object being held')
            if in_robot_hand in [x[1] for x in self.kitchen.in_robot_hand]:
                in_human_hand = None
            else:
                in_human_hand = None

        # check that objects are not in holding list and object list
        unowned_objects = self.kitchen.overcooked_object_states.copy()
        for obj, st in self.kitchen.overcooked_object_states.items():
            if obj == in_robot_hand:
                unowned_objects.pop(obj)

            if obj == in_human_hand:
                unowned_objects.pop(obj)

        # also remove hot plates containing steak or garnish
        for obj, st in self.kitchen.overcooked_object_states.items():

            if 'in_hot_plate' in st.keys():
                if len(st['in_hot_plate']) > 0:
                    if obj in unowned_objects:
                        unowned_objects.pop(obj)
                    for object in st['in_hot_plate']:
                        if object in unowned_objects:
                            unowned_objects.pop(object)
                st.pop('in_hot_plate')

        for obj, st in self.kitchen.overcooked_object_states.items():
            if st['name'] == 'dish':
                if st['state'] == 'delivered' and obj in unowned_objects:
                    unowned_objects.pop(obj)

        self.world_state.state_dict['unowned_objects'] = unowned_objects

        self.world_state.state_dict['objects'] = self.kitchen.overcooked_object_states

        # check object states
        for onion in self.kitchen.onions:
            for agent in self.nav_env.robots:
                body_id = onion.get_body_ids()[0]
                grasping = agent.is_grasping_all_arms(body_id)
                holding_onion = IsGraspingState.TRUE in grasping
                if holding_onion:
                    break

            if onion.current_index == 1 and onion in unowned_objects and not holding_onion:
                if not onion.objects[0].states[object_states.OnTop].get_value(self.kitchen.chopping_boards[0]):
                    onion.objects[0].states[object_states.OnTop].set_value(self.kitchen.chopping_boards[0], True,
                                                                           use_ray_casting_method=True)

        self.log_state()
        self.log_dict_state()
        return

    def check_last_steak(self):
        # if self.log_dict[self.log_dict['i']-1]['low_level_logs']
        self.human_state.state_dict['human_holding']

    def log_dict_state(self):
        elapsed = time.time() - self.log_dict['event_start_time']

        if elapsed > 0.05:
            ll_dict = {}
            ll_dict['robot_ml'] = self.robot_state.ml_state
            ll_dict['human_ml'] = self.human_state.ml_state
            ll_dict['robot_ll'] = self.ig_robot.object.get_position().tolist()
            ll_dict['human_ll'] = self.ig_human.object.get_position().tolist()
            x, y, z, w = self.ig_robot.object.get_orientation().tolist()
            ll_dict['robot_ll_ori'] = quat2euler(x, y, z, w)
            x, y, z, w = self.ig_human.object.get_orientation().tolist()
            ll_dict['human_ll_ori'] = quat2euler(x, y, z, w)

            self.log_dict[self.log_dict['i']]['low_level_logs'].append(ll_dict.copy())
            self.log_dict['event_start_time'] = time.time()

    def log_state(self):
        elapsed = time.time() - self.log_time

        if elapsed > 1:
            self.log_time = time.time()
            filename = 'lsi_3d/logs/' + self.kitchen.kitchen_name + '_log.txt'
            f = open(filename, "a")
            s = ''
            s += 'Logging State at Time:\t' + str(self.log_time) + '\n'
            s += 'Robot ML State:\t' + str(self.robot_state.ml_state) + '\n'
            s += 'Human ML State:\t' + str(self.human_state.ml_state) + '\n'
            s += 'Robot LL State:\t' + str(self.ig_robot.object.get_position()) + '\n'
            s += 'Human LL State:\t' + str(self.ig_human.object.get_position()) + '\n'
            s += '\n'
            f.write(s)
            f.close()

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
            pans_sorted = sorted(self.kitchen.pans, key=lambda pan: (
                self.tracking_env.get_pan_enum_status(pan), math.dist(pan.get_position()[0:2], agent_location_real)))
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
        r, c = ml_state[0:2]

        # add one for overcooked border padding
        pos = c + 1, r + 1
        ori = ml_state[2]

        if ori == 'N':
            o = (0, -1)
        elif ori == 'E':
            o = (1, 0)
        elif ori == 'S':
            o = (0, 1)
        elif ori == 'W':
            o = (-1, 0)
        else:
            o = None
        return pos, o

    def to_overcooked_holding(self, holding):
        if holding == {}:
            return None
        else:
            if 'in_hot_plate' in holding.keys():
                holding.pop('in_hot_plate')
            return holding

    def to_overcooked_grid(self, loc):
        pos = loc
        r, c = pos
        x, y = c + 1, r + 1
        return (x, y)

    def all_objects_ids_dict(self):
        id_dict = {}

        for obj, st in self.kitchen.overcooked_object_states.items():
            id_dict[st['id']] = st

        return id_dict

    def to_overcooked_state(self):
        state_dict = {}
        h_p, h_o = self.to_overcooked_pos_or(self.human_state.ml_state)
        r_p, r_o = self.to_overcooked_pos_or(self.robot_state.ml_state)

        state_dict['players'] = [
            {
                'position': r_p,
                'orientation': r_o,
                'held_object': self.to_overcooked_holding(self.robot_state.state_dict['robot_holding'].copy())
            }, {
                'position': h_p,
                'orientation': h_o,
                'held_object': self.to_overcooked_holding(self.human_state.state_dict['human_holding'].copy())
            },
        ]
        # state_dict['objects'] = [v for k,v in self.world_state.state_dict['objects'].items()]
        state_dict['objects'] = [v.copy() for k, v in self.world_state.state_dict['unowned_objects'].items()]
        for st in state_dict['objects']:
            if 'in_hot_plate' in st.keys():
                del st['in_hot_plate']
        state_dict['order_list'] = self.world_state.orders.copy()

        return state_dict
