import math
import random
import time
from argparse import Namespace
from math import floor

import numpy as np
import pybullet as p

from igibson import object_states
from igibson.objects.multi_object_wrappers import ObjectMultiplexer
from igibson.robots.manipulation_robot import IsGraspingState
from lsi_3d.environment.object_config import (
    OBJECT_KEYS,
    OBJECT_ABBRS,
    OBJECT_ABBR_MAP,
    OBJECT_CONFIG,
    OBJECT_TRANSLATIONS
)
from lsi_3d.environment.objects import (
    Fridge,
    Onion,
    Steak,
    Plate,
    Stove,
    Pan,
    GreenOnion,
    Counter,
    OtherKitchenObject,
    OtherBowl,
    Knife,
    ChoppingBoard,
    VidaliaOnion
)
from lsi_3d.utils.constants import DIRE2POSDIFF
from utils import normalize_radians, real_to_grid_coord, to_overcooked_grid


class Kitchen:
    _DYNAMIC_OBJECTS = [OBJECT_KEYS.BOWL, OBJECT_KEYS.PAN, OBJECT_KEYS.VIDALIA_ONION, OBJECT_KEYS.STEAK,
                        OBJECT_KEYS.PLATE, OBJECT_KEYS.CHOPPING_BOARD, OBJECT_KEYS.KNIFE, OBJECT_KEYS.GREEN_ONION]

    def __init__(self, env, max_in_pan, rinse_time=5):
        self.env = env
        self.map = None
        self.kitchen_name = None
        self.HEIGHT = 13  # x 13
        self.WIDTH = 13  # y 13
        self.orientation_map = ""
        self.grid = ""
        self.bowls = []
        self.pans = []
        self.plates = []
        self.rinsing_sinks = {}
        self.ready_sinks = []
        self.hot_plates = []
        self.onions = []
        self.steaks = []
        self.knives = []
        self.meats = []
        self.counters = []
        self.table = None
        self.fridges = []
        self.chopping_boards = []
        self.chopping_counter = None
        self.sinks = []
        self.food_obj = []
        self.static_objs = {}
        self.in_robot_hand = []
        self.onions_for_soup = max_in_pan
        self.robot_carrying_dish = False
        self.robot_stash_dish = None

        self.stove = None

        self.robot_carrying_steak = False
        self.robot_stash_steak_bowl = None

        # tile location is a dictionary of item locations in the environment indexed by letter (eg F for fridge)
        self.tile_location = {}

        # key is plate object and value is state
        self.overcooked_object_states = {}
        self.overcooked_max_id = 0
        self.overcooked_obj_to_id = {}
        self.overcooked_robot_holding = ('None', None)
        self.overcooked_human_holding = ('None', None)
        self.stored_overcooked_object_states = {}

        self.overcooked_hot_plates_now_dish = []

    def drop_plate(self, plate):
        id = self.overcooked_max_id
        self.overcooked_max_id += 1
        self.overcooked_obj_to_id[plate] = id
        curr_state = self.overcooked_object_states[plate]['state']
        state = 0 if curr_state is None else curr_state + 1
        self.overcooked_object_states[plate] = {
            'id': id,
            'name': 'hot_plate',
            'position': to_overcooked_grid(real_to_grid_coord(plate.get_position())),
            'state': state
        }

        return

    def heat_plate(self, plate):
        curr_state = self.overcooked_object_states[plate]['state']
        state = 0 if curr_state is None else curr_state + 1
        self.overcooked_object_states[plate] = {
            'id': self.overcooked_obj_to_id[plate],
            'name': 'hot_plate',
            'position': to_overcooked_grid(real_to_grid_coord(plate.get_position())),
            'state': state
        }

        plate.states[object_states.Dusty].set_value(False)
        plate.states[object_states.Stained].set_value(False)

        return

    def drop_meat(self, obj):
        id = self.overcooked_max_id
        self.overcooked_max_id += 1
        self.overcooked_obj_to_id[obj] = id
        curr_state = self.overcooked_object_states[obj]['state']
        state = 1 if curr_state is None else curr_state + 1
        self.overcooked_object_states[obj] = {
            'id': id,
            'name': 'steak',
            'position': to_overcooked_grid(real_to_grid_coord(obj.get_position())),
            'state': ('steak', 1, 10)
        }

        self.stove.states[object_states.ToggledOn].set_value(True)

        return

    def drop_onion(self, obj):
        id = self.overcooked_max_id
        self.overcooked_max_id += 1
        self.overcooked_obj_to_id[obj] = id
        curr_state = self.overcooked_object_states[obj]['state']
        state = 0 if curr_state is None else curr_state + 1
        self.overcooked_object_states[obj] = {
            'id': id,
            'name': 'garnish',
            'position': to_overcooked_grid(real_to_grid_coord(obj.get_position())),
            'state': state
        }

        return

    def chop_onion(self, obj):
        curr_state = self.overcooked_object_states[obj]['state']
        state = 2
        self.overcooked_object_states[obj] = {
            'id': self.overcooked_obj_to_id[obj],
            'name': 'garnish',
            'position': to_overcooked_grid(real_to_grid_coord(obj.current_selection().objects[0].get_position())),
            'state': state
        }

        return

    def update_overcooked_human_holding(self, holding, obj):
        # need to delay human holding for overcooked 
        # overcooked human holding is tuple of (human_)
        if self.overcooked_human_holding[0] != holding:
            self.overcooked_human_holding = (holding, obj)

        return self.overcooked_human_holding[1]

    def update_overcooked_robot_holding(self):
        if len(self.in_robot_hand) == 1:
            self.overcooked_robot_holding = self.in_robot_hand[0][1]
        elif len(self.in_robot_hand) == 2:
            steak = steak = [x for x in self.in_robot_hand if 'steak' in x[1].name][0][1]
            self.overcooked_robot_holding = steak
        elif len(self.in_robot_hand) == 3:
            onion = [x for x in self.in_robot_hand if 'onion' in x[1].name][0][1]
            self.overcooked_robot_holding = onion
        else:
            self.overcooked_robot_holding = None

    def setup(self, filepath, order_list):
        self.kitchen_name = filepath.split('/')[1].split('.')[0]
        obj_x_y, orientation_map, grid = self.read_from_grid_text(filepath)
        self.map = orientation_map
        self.load_objects(obj_x_y, orientation_map, order_list)
        self.load_interact_objs()

    def load_interact_objs(self):
        self.interact_objs = {}
        for pan in self.pans:
            self.interact_objs[pan] = False

    def read_from_grid_text(self, filepath):
        object_locs = []
        sum_x, sum_y, count = 0, 0, 0  # for calculation of center mass (excluding table)
        orientation_map = {}
        grid_items = self.get_grid_objects(filepath)
        grid = [[OBJECT_ABBRS[OBJECT_KEYS.EMPTY]] * self.WIDTH for _ in range(self.HEIGHT)]

        for name, x, y in grid_items:
            object_locs.append((name, x, y))
            if grid[x][y] in [
                OBJECT_ABBRS[OBJECT_KEYS.EMPTY],
                OBJECT_ABBRS[OBJECT_KEYS.COUNTER]
            ] and name != OBJECT_KEYS.VIDALIA_ONION:
                grid[x][y] = OBJECT_ABBRS[name]
            if name == OBJECT_KEYS.TABLE_H:
                grid[x][y + 1] = OBJECT_ABBRS[name]
            elif name == OBJECT_KEYS.TABLE_V:
                grid[x + 1][y] = OBJECT_ABBRS[name]
            elif name == OBJECT_KEYS.PAN:
                grid[x][y] = OBJECT_ABBRS[name]
            else:
                sum_x += x
                sum_y += y
                count += 1

        count = max(count, 1)  # ensure count is at least 1 to avoid division by zero
        center_x, center_y = sum_x / count, sum_y / count

        for name, x, y in object_locs:
            if name == OBJECT_KEYS.TABLE_H:
                orientation_map[(name, x, y)] = (0., 0., np.pi / 2.)
            elif name == OBJECT_KEYS.TABLE_V:
                orientation_map[(name, x, y)] = (0., 0., 0.)
            else:
                ori = self.ori_filter(grid, x + 1, y) + \
                      self.ori_filter(grid, x - 1, y) - \
                      self.ori_filter(grid, x, y + 1) - \
                      self.ori_filter(grid, x, y - 1)
                orientation_map[(name, x, y)] = self.get_orientation(center_x, center_y, x, y, ori)

            self.tile_location[OBJECT_ABBRS[name]] = (x, y)

        self.orientation_map = orientation_map
        self.grid = grid
        return object_locs, orientation_map, grid

    def load_objects(self, object_poses, orientation_map, order_list):
        shift_l = 0.1
        mapping = {
            (0, 0, 3.1415926): (0, -shift_l),
            (0, 0, 0): (0, shift_l),
            (0, 0, 1.5707): (-shift_l, 0),
            (0, 0, -1.5707): (shift_l, 0),
        }

        obj_handlers = Namespace(
            import_obj=self.env.simulator.import_object,
            set_pos_orn=self.env.set_pos_orn_with_z_offset,
            change_pb_dynamics=p.changeDynamics,
        )

        for name, x, y in object_poses:
            obj = None
            orn = orientation_map[(name, x, y)]
            shift = OBJECT_TRANSLATIONS[name]
            if name == OBJECT_KEYS.COUNTER:
                x_shift, y_shift = mapping[orn]
                shift = (x_shift, y_shift, 0)
            elif name == OBJECT_KEYS.FRIDGE:
                x_shift, y_shift = mapping[orn]
                shift = (x_shift, y_shift, 0)

            pos = [x + shift[0] - 4.5, y + shift[1] - 4.5, 0 + shift[2]]

            if name == OBJECT_KEYS.FRIDGE:
                fridge = Fridge(**OBJECT_CONFIG[OBJECT_KEYS.COUNTER], pos=pos, orn=orn, obj_handlers=obj_handlers)
                fridge.load()
                obj = fridge.obj
                self.fridges.append(obj)

                if OBJECT_KEYS.ONION in order_list:
                    for _ in range(10):
                        onion = Onion(**OBJECT_CONFIG[OBJECT_KEYS.VIDALIA_ONION], obj_handlers=obj_handlers, mass=.001)
                        onion.load(obj)
                        self.onions.append(onion.obj)
                if OBJECT_KEYS.STEAK in order_list:
                    for _ in range(7):
                        steak = Steak(**OBJECT_CONFIG[OBJECT_KEYS.STEAK], obj_handlers=obj_handlers, mass=.001)
                        steak.load(obj)
                        self.meats.append(steak.obj)

            elif name == OBJECT_KEYS.PLATE:
                plate = Plate(
                    **OBJECT_CONFIG[OBJECT_KEYS.PLATE],
                    obj_handlers=obj_handlers,
                    pos=pos,
                    orn=orn,
                    dusty=True,
                    stained=True
                )
                plate.load()
                obj = plate.obj
                self.bowl_spawn_pos = pos
                self.plates.append(obj)
                self.bowls.append(obj)

                for idx in range(3):
                    other_plate = Plate(
                        **OBJECT_CONFIG[OBJECT_KEYS.PLATE],
                        obj_handlers=obj_handlers,
                        pos=[200 + 5 * idx, 200, 1],
                        orn=orn,
                        dusty=True,
                        stained=True
                    )
                    other_plate.load()
                    self.plates.append(other_plate.obj)

            elif name == OBJECT_KEYS.STOVE:
                stove = Stove(**OBJECT_CONFIG[OBJECT_KEYS.STOVE], obj_handlers=obj_handlers, pos=pos, orn=orn)
                stove.load()
                obj = stove.obj
                self.stove = obj

            elif name == OBJECT_KEYS.PAN:
                pan = Pan(
                    **OBJECT_CONFIG[OBJECT_KEYS.PAN],
                    obj_handlers=obj_handlers,
                    pos=self.translate_loc(self.get_rotated_basis(orn), tuple([x - 4.5, y - 4.5, 0]), shift),
                    orn=orn
                )
                pan.load()
                obj = pan.obj
                self.pans.append(obj)

            elif name == OBJECT_KEYS.GREEN_ONION:
                pos = [pos[0], pos[1], pos[2] + .05]
                green_onion = GreenOnion(
                    **OBJECT_CONFIG[OBJECT_KEYS.GREEN_ONION],
                    obj_handlers=obj_handlers,
                    pos=pos,
                    orn=orn,
                    mass=.001,
                    away_pos=[100, 100, -100]
                )
                green_onion.load()
                obj = green_onion.multiplexed_obj
                self.onions.append(obj)
                self.onion_spawn_pos = pos

                for j in range(2):
                    green_onion_extra = GreenOnion(
                        **OBJECT_CONFIG[OBJECT_KEYS.GREEN_ONION],
                        obj_handlers=obj_handlers,
                        pos=[200, 100, 1],
                        orn=orn,
                        mass=.001,
                        away_pos=[200 + 2 * j, 100, 1]
                    )
                    green_onion_extra.load()
                    self.onions.append(green_onion_extra.multiplexed_obj)

            elif OBJECT_KEYS.COUNTER in name:
                counter = Counter(**OBJECT_CONFIG[OBJECT_KEYS.COUNTER], obj_handlers=obj_handlers, pos=pos, orn=orn)
                counter.load()
                obj = counter.obj
                self.counters.append(obj)

            elif name == OBJECT_KEYS.BOWL:
                bowl = Plate(
                    **OBJECT_CONFIG[OBJECT_KEYS.BOWL],
                    obj_handlers=obj_handlers,
                    pos=pos,
                    orn=orn,
                    dusty=True,
                    stained=True,
                    mass=.01
                )
                obj = bowl.obj
                bowl.load()
                self.bowls.append(obj)

            elif name == OBJECT_KEYS.KNIFE:
                knife = Knife(**OBJECT_CONFIG[OBJECT_KEYS.KNIFE], obj_handlers=obj_handlers, pos=pos, orn=orn, mass=.01)
                knife.load()
                obj = knife.obj
                self.knives.append(obj)
                self.init_knife_pos = None

            elif name == OBJECT_KEYS.CHOPPING_BOARD:
                chopping_board = ChoppingBoard(
                    **OBJECT_CONFIG[OBJECT_KEYS.CHOPPING_BOARD],
                    obj_handlers=obj_handlers,
                    pos=pos,
                    orn=orn,
                    mass=100
                )
                chopping_board.load()
                obj = chopping_board.obj
                self.chopping_boards.append(obj)

            elif name == OBJECT_KEYS.VIDALIA_ONION:
                vidalia_onion = VidaliaOnion(
                    **OBJECT_CONFIG[OBJECT_KEYS.VIDALIA_ONION],
                    obj_handlers=obj_handlers,
                    pos=pos,
                    orn=orn,
                    mass=.001
                )
                vidalia_onion.load()
                obj = vidalia_onion.obj
                self.onions.append(obj)

            else:
                other = OtherKitchenObject(
                    **OBJECT_CONFIG[name],
                    obj_handlers=obj_handlers,
                    pos=pos,
                    orn=orn,
                    static=name not in self._DYNAMIC_OBJECTS
                )
                other.load()
                obj = other.obj
                if name == OBJECT_KEYS.SINK:
                    self.sinks.append(obj)
                if name == OBJECT_KEYS.TABLE_H or name == OBJECT_KEYS.TABLE_V:
                    self.table = obj

            if name not in self._DYNAMIC_OBJECTS:
                self.static_objs[obj] = (x, y)
            if name == OBJECT_KEYS.TABLE_H:
                self.static_objs[obj] = [(x, y), (x, y + 1)]
            if name == OBJECT_KEYS.TABLE_H:
                self.static_objs[obj] = [(x, y), (x + 1, y)]

        bowl = OtherBowl(**OBJECT_CONFIG[OBJECT_KEYS.LARGE_BOWL], obj_handlers=obj_handlers, away_pos=[300, 200, 1])
        bowl.load()
        self.large_bowl = bowl.obj

        bowl = OtherBowl(**OBJECT_CONFIG[OBJECT_KEYS.STEAK_BOWL], obj_handlers=obj_handlers, away_pos=[275, 275, 1])
        bowl.load()
        self.steak_bowl = bowl.obj

        # remove chopping board location from counters

        for counter in self.counters.copy():
            if real_to_grid_coord(counter.get_position()) == real_to_grid_coord(self.chopping_boards[0].get_position()):
                self.counters.remove(counter)

            # set onion counter
            if real_to_grid_coord(counter.get_position()) == self.get_green_onion_station()[0]:
                self.onion_counter = counter

            if real_to_grid_coord(counter.get_position()) == self.get_chopping_station()[0]:
                self.chopping_counter = counter

            if real_to_grid_coord(counter.get_position()) == self.where_grid_is('D')[0]:
                self.dish_station = counter

    def get_grid_objects(self, filepath):
        with open(filepath, 'r') as fh:
            grid_rows = fh.read().strip().split('\n')
            grid = [list(each) for each in grid_rows]
            self.HEIGHT = len(grid[0])
            self.WIDTH = len(grid[0])

            object_mapping = {
                OBJECT_ABBRS[OBJECT_KEYS.BOWL]: [OBJECT_KEYS.COUNTER],
                OBJECT_ABBRS[OBJECT_KEYS.PAN]: [OBJECT_KEYS.STOVE],
                OBJECT_ABBRS[OBJECT_KEYS.STOVE]: [OBJECT_KEYS.PAN],
                OBJECT_ABBRS[OBJECT_KEYS.CHOPPING_BOARD]: [OBJECT_KEYS.COUNTER, OBJECT_KEYS.KNIFE],
                OBJECT_ABBRS[OBJECT_KEYS.GREEN_ONION]: [OBJECT_KEYS.COUNTER],
                OBJECT_ABBRS[OBJECT_KEYS.PLATE]: [OBJECT_KEYS.COUNTER],
            }

            object_locs = []
            for row_idx in range(len(grid)):
                for col_idx in range(len(grid[row_idx])):
                    cell = grid[row_idx][col_idx]
                    if cell == OBJECT_ABBRS[OBJECT_KEYS.EMPTY]:  # ignore empty space
                        continue
                    if cell == OBJECT_ABBRS[OBJECT_KEYS.TABLE_V]:  # add table
                        if col_idx + 1 < self.WIDTH and grid[row_idx][col_idx + 1] == cell:  # check table orientation
                            grid[row_idx][col_idx + 1] = OBJECT_ABBRS[OBJECT_KEYS.EMPTY]
                            object_locs.append((OBJECT_KEYS.TABLE_H, row_idx, col_idx))
                        else:
                            grid[row_idx + 1][col_idx] = OBJECT_ABBRS[OBJECT_KEYS.EMPTY]
                            object_locs.append((OBJECT_KEYS.TABLE_V, row_idx, col_idx))
                    else:  # other objects
                        object_locs.extend(list(map(
                            lambda o: (o, row_idx, col_idx),
                            object_mapping.get(cell, [])
                        )))  # add related objects
                        object_locs.append((OBJECT_ABBR_MAP[cell], row_idx, col_idx))  # add current object

        return object_locs

    def sample_position(self, x, y, range):
        x_range = random.uniform(x - range, x + range)
        y_range = random.uniform(y - range, y + range)
        return x_range, y_range

    def ori_filter(self, grid, x, y):
        if not (x >= 0 and x < self.HEIGHT and y >= 0 and y < self.WIDTH):
            return 0

        if grid[x][y] == "X":
            return 0
        else:
            return 1

    def get_orientation(self, center_x, center_y, x, y, ori):
        '''
        if ori > 0 then it's facing left/right, otherwise it's facing up/down
        '''
        orientation = (0, 0, 0)
        if ori > 0:
            if center_y > y:
                orientation = (0, 0, 3.1415926)
            else:
                orientation = (0, 0, 0)
        else:
            if center_x > x:
                orientation = (0, 0, 1.5707)
            else:
                orientation = (0, 0, -1.5707)
        return orientation

    def get_rotated_basis(self, ori):
        _, _, z = ori
        z_theta = normalize_radians(z) - math.pi / 2
        regular_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotation_matrix = np.array([[math.cos(z_theta), -math.sin(z_theta), 0],
                                    [math.sin(z_theta),
                                     math.cos(z_theta), 0], [0, 0, 1]])
        rotated_basis = np.matmul(rotation_matrix, regular_basis)
        return rotated_basis

    def translate_loc(self, rotated_basis, loc, offset):
        offset_scaling = np.array([[offset[0], 0, 0], [0, offset[1], 0],
                                   [0, 0, offset[2]]])
        scaled_rotated_basis = np.matmul(rotated_basis, offset_scaling)
        translated_loc = np.matmul(scaled_rotated_basis,
                                   np.array([1, 1,
                                             1]).transpose()).transpose()
        translated_loc = translated_loc + np.array(loc)
        return translated_loc

    def where_grid_is(self, letter):
        indexes = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] == letter:
                    indexes.append((i, j))

        return indexes

    def get_immobile_objects(self):
        key_objects_list = [
            *self.pans
            , *self.fridges
            , *self.chopping_boards
            , *self.sinks
        ]

        return key_objects_list

    def get_mobile_objects(self):
        key_objects_list = [
            *self.bowls
            , *self.plates
            , *self.hot_plates
            , *self.onions
            , *self.steaks
            , *self.knives
            , *self.meats
        ]

        return key_objects_list

    def get_empty_squares(self):
        return self.where_grid_is('X')

    def get_onion_station(self):
        return self.where_grid_is('F')

    def get_green_onion_station(self):
        return self.where_grid_is('G')

    def get_chopping_station(self):
        return self.where_grid_is('K')

    def get_center(self):
        grid_center = (floor(len(self.grid) / 2), floor(len(self.grid[0]) / 2))
        return grid_center

    def nearest_empty_tile(self, loc):
        grid_center = (floor(len(self.grid) / 2), floor(len(self.grid[0]) / 2))
        open_spaces = []
        x, y = loc
        if self.grid[x][y] == 'X':
            return loc
        else:
            for dir, add in DIRE2POSDIFF.items():
                d_x, d_y = add
                n_x, n_y = (x + d_x, y + d_y)
                if n_x > len(self.grid) - 1 or n_y > len(self.grid) - 1:
                    continue
                if self.grid[n_x][n_y] == 'X':
                    open_spaces.append((n_x, n_y))

        # manhattan dist to center pick closest
        chosen_space = None
        chosen_dist = 1000000
        for space in open_spaces:
            gcx, gcy = grid_center
            sx, sy = space
            dist = abs(gcx - sx) + abs(gcy - sy)
            if dist < chosen_dist:
                chosen_space = space
                chosen_dist = dist

        return chosen_space

    def rinse_sink(self, sink):
        if sink not in self.rinsing_sinks:
            self.rinsing_sinks[sink] = time.time()

    def get_name(self, obj):
        # if obj in self.bowls:
        #     return 'bowl'
        if obj in self.plates:
            return 'plate'
        elif obj in self.hot_plates:
            return 'hot_plate'
        elif obj in self.onions:
            return 'onion'
        elif obj in self.steaks:
            return 'steak'
        elif obj in self.knives:
            return 'knife'
        elif obj in self.meats:
            return 'meat'
        else:
            return 'object not in any mobile list'

    def get_position(self, obj):
        if obj.name == 'green_onion_multiplexer':
            if obj.current_index == 0:
                return real_to_grid_coord(obj.get_position())
            else:
                return real_to_grid_coord(obj.current_selection().objects[0].get_position())
        else:
            return real_to_grid_coord(obj.get_position())

    def step(self, count=0):
        for onion in self.onions:
            holding_onion = False
            for agent in self.env.robots:
                body_id = onion.get_body_ids()[0]
                grasping = agent.is_grasping_all_arms(body_id)
                holding_onion = IsGraspingState.TRUE in grasping
                if holding_onion:
                    break
            if not holding_onion:
                if onion.current_index == 0 and onion.get_position()[
                    0] < 50 and onion not in self.overcooked_object_states.keys():
                    if not onion.states[object_states.OnTop].get_value(self.onion_counter, use_ray_casting_method=True):
                        onion.states[object_states.OnTop].set_value(self.onion_counter, True,
                                                                    use_ray_casting_method=True)

        for obj, state in self.overcooked_object_states.items():
            if state['name'] == 'hot_plate':
                state = self.overcooked_object_states[obj]['state']

                # TODO: wont work with multiple sinks
                sink = self.sinks[0]
                if state is not None and state < 2:

                    self.rinse_sink(sink)
                elif state == 2:
                    if sink not in self.ready_sinks:
                        self.ready_sinks.append(sink)

        for sink, time in self.rinsing_sinks.copy().items():

            if time > 5:
                self.rinsing_sinks.pop(sink)
                self.ready_sinks.append(sink)

        for meat in self.meats.copy():
            for pan in self.pans:
                if meat.states[object_states.Inside].get_value(pan):
                    self.steaks.append(meat)
                    self.meats.remove(meat)

        in_robot_hand_id = 0
        if self.robot_carrying_dish:

            if self.robot_stash_steak_bowl is not None:
                self.robot_carrying_steak = False
                self.steak_bowl.set_position([110, 110, 1])
                self.robot_stash_steak_bowl = None

            if self.robot_stash_dish is None:
                # stash items far away
                self.robot_carrying_dish = True

                offset_idx = 0

                plate = [x for x in self.in_robot_hand if 'plate' in x[1].name][0][1]
                steak = [x for x in self.in_robot_hand if 'steak' in x[1].name][0][1]
                onion = [x for x in self.in_robot_hand if 'onion' in x[1].name][0][1]

                stash_pos = [30, 30, 1]
                (x, y, z) = stash_pos
                new_pos = (x + (0.1 * offset_idx), y, z)

                self.robot_stash_dish = plate
                plate.set_position(new_pos)
                offset_idx += 1
                new_pos = (x + (0.5 * offset_idx), y, z)
                steak.set_position(new_pos)
                offset_idx += 1

                for sub_obj in onion.objects:
                    new_pos = (x + (0.5 * offset_idx), y, z)
                    sub_obj.set_position(new_pos)
                    offset_idx += 1

            b_x, b_y, b_z = self.env.robots[0].get_eef_position()
            self.large_bowl.set_position([b_x, b_y, b_z + 0.12])
        elif self.robot_carrying_steak:
            if self.robot_stash_steak_bowl is None:
                plate = [x for x in self.in_robot_hand if 'plate' in x[1].name][0][1]
                self.robot_stash_steak_bowl = plate
                plate.set_position([80, 80, 1])
            b_x, b_y, b_z = self.env.robots[0].get_eef_position()
            self.steak_bowl.set_position([b_x, b_y, b_z + 0.12])
            steak = [x for x in self.in_robot_hand if 'steak' in x[1].name][0][1]
            steak.set_position(self.env.robots[0].get_eef_position())
        else:
            if self.robot_stash_dish is not None:
                self.robot_carrying_dish = False
                self.large_bowl.set_position([130, 130, 1])
                self.robot_stash_dish = None
                self.in_robot_hand.clear()

            for obj in self.in_robot_hand:
                (x, y, z) = self.env.robots[0].get_eef_position()
                new_pos = (x, y, z + (0.05 * in_robot_hand_id))
                ig_obj = obj[-1]
                if type(ig_obj) == ObjectMultiplexer and ig_obj.current_index == 1:
                    for sub_obj in ig_obj.objects:
                        sub_obj.set_position(new_pos)
                else:
                    obj[-1].set_position(self.env.robots[0].get_eef_position())

                in_robot_hand_id += 1

        plate_in_dish_station = False
        for plat in self.plates:
            dish_station_pos = self.where_grid_is('D')[0]
            if real_to_grid_coord(plat.get_position()) == dish_station_pos:
                plate_in_dish_station = True
                break

        if not plate_in_dish_station:
            # get farthest plate
            plate = None
            for plat in self.plates:
                pos = plat.get_position()
                if pos[0] > 100:
                    plate = plat
            if plate is not None:
                plate.states[object_states.OnTop].set_value(self.dish_station, True, use_ray_casting_method=True)

        # if knife not in human hand then make sure on chopping counter
        for knife in self.knives:
            holding_knife = False
            onion_chopped = False
            for agent in self.env.robots:
                body_id = knife.get_body_ids()[0]
                grasping = agent.is_grasping_all_arms(body_id)
                holding_knife = IsGraspingState.TRUE in grasping
                if holding_knife:
                    break

            for onion in self.onions:
                if onion.current_index == 1:
                    onion_chopped = True

            if self.init_knife_pos is None and knife.states[object_states.OnTop].get_value(self.chopping_counter):
                self.init_knife_pos = knife.get_position()

            if self.init_knife_pos is not None and not holding_knife and not knife.states[
                object_states.OnTop].get_value(self.chopping_counter):
                knife.set_position(self.init_knife_pos)

        onion_in_onion_station = False
        for o in self.onions:
            onion_station_pos = self.where_grid_is('G')[0]
            if self.get_position(o) == onion_station_pos:
                onion_in_onion_station = True
                break

        if not onion_in_onion_station:
            # get farthest plate
            onion = None
            for o in self.onions:
                pos = self.get_position(o)
                if pos[0] > 100:
                    onion = o

            if onion is not None:
                self.env.set_pos_orn_with_z_offset(onion, self.onion_spawn_pos, [0, 0, 0])
                body_ids = onion.get_body_ids()
                p.changeDynamics(body_ids[0], -1, mass=0.001)
