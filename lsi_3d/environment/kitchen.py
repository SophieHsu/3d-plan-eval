import copy
from math import floor
import time
import numpy as np
import os
import igibson
import math
import random
from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.robots.manipulation_robot import IsGraspingState
from igibson.utils.assets_utils import get_ig_model_path
from igibson.object_states.heat_source_or_sink import HeatSourceOrSink
from igibson import object_states
from utils import grid_to_real_coord, quat2euler, normalize_radians, real_to_grid_coord, to_overcooked_grid

import pybullet as p

from lsi_3d.utils.constants import DIRE2POSDIFF
from igibson.objects.visual_marker import VisualMarker


class Kitchen():

    def __init__(self, env, max_in_pan, rinse_time=5):
        self.env = env
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
        self.overcooked_robot_holding = ('None',None)
        self.overcooked_human_holding = ('None',None)
        self.stored_overcooked_object_states = {}

        self.overcooked_hot_plates_now_dish = []

    # def pickup_meat(self, meat, agent_ml_state):
    #     self.overcooked_object_states[meat] = {
    #         "id": self.overcooked_max_id,
    #         "name": self.get_name(meat),
    #         "position": to_overcooked_grid(agent_ml_state[0:2]),
    #         "state": None
    #     }
    #     self.overcooked_max_id += 1

    def drop_plate(self, plate):
        id = self.overcooked_max_id
        self.overcooked_max_id +=1
        self.overcooked_obj_to_id[plate] = id
        curr_state = self.overcooked_object_states[plate]['state']
        state = 0 if curr_state is None else curr_state + 1
        self.overcooked_object_states[plate] = {
                'id': id,
                'name': 'hot_plate',
                'position': to_overcooked_grid(real_to_grid_coord(plate.get_position())),
                'state':state
            }
        
        return
        
    def heat_plate(self, plate):
        curr_state = self.overcooked_object_states[plate]['state']
        state = 0 if curr_state is None else curr_state + 1
        self.overcooked_object_states[plate] = {
                'id': self.overcooked_obj_to_id[plate],
                'name': 'hot_plate',
                'position': to_overcooked_grid(real_to_grid_coord(plate.get_position())),
                'state':state
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
                'state':('steak',1,10)
            }
        
        self.stove.states[object_states.ToggledOn].set_value(True)
        
        return
    
    def drop_onion(self, obj):
        id = self.overcooked_max_id
        self.overcooked_max_id +=1
        self.overcooked_obj_to_id[obj] = id
        curr_state = self.overcooked_object_states[obj]['state']
        state = 0 if curr_state is None else curr_state + 1
        self.overcooked_object_states[obj] = {
                'id': id,
                'name': 'garnish',
                'position': to_overcooked_grid(real_to_grid_coord(obj.get_position())),
                'state':state
            }
        
        return
    
    def chop_onion(self, obj):
        curr_state = self.overcooked_object_states[obj]['state']
        state = 2
        self.overcooked_object_states[obj] = {
                'id': self.overcooked_obj_to_id[obj],
                'name': 'garnish',
                'position': to_overcooked_grid(real_to_grid_coord(obj.current_selection().objects[0].get_position())),
                'state':state
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
        obj_x_y = []
        sum_x, sum_y, count = 0, 0, 0  # for calculation of center mass (excluding table)
        orientation_map = dict()

        raw_str = self.grid2raw(filepath)
        grid = [["X"] * self.WIDTH for _ in range(self.HEIGHT)]
        for line in raw_str.strip().split("\n"):
            if len(line) == 0:
                break
            name, x, y = line.split()
            x, y = int(x), int(y)
            obj_x_y.append((name, x, y))
            if (grid[x][y] == "X"
                    or grid[x][y] == "C") and name != "vidalia_onion":
                grid[x][y] = self.name2abbr(name)
            if name == "table_h":
                grid[x][y + 1] = self.name2abbr(name)
            elif name == "table_v":
                grid[x + 1][y] = self.name2abbr(name)
            elif name == 'pan':
                grid[x][y] = self.name2abbr(name)
            else:
                sum_x += x
                sum_y += y
                count += 1
        if count == 0:
            count = 1
        center_x, center_y = sum_x / count, sum_y / count

        for name, x, y in obj_x_y:
            if name == "table_h":
                orientation_map[(name, x, y)] = (0, 0, 1.5707)
            elif name == "table_v":
                orientation_map[(name, x, y)] = (0, 0, 0)
            else:
                ori = 0  # if ori > 0 then it's facing left/right, otherwise it's facing up/down
                ori += self.ori_filter(grid, x + 1, y) + self.ori_filter(
                    grid, x - 1, y)  # upper and lower neighbor
                ori -= self.ori_filter(grid, x, y + 1) + self.ori_filter(
                    grid, x, y - 1)  # left and right neighbor
                orientation_map[(name, x, y)] = self.get_orientation(
                    center_x, center_y, x, y, ori)

            # TODO: complete this
            if name == 'fridge':
                self.tile_location['F'] = (x, y)
            elif name == 'stove':
                self.tile_location['S'] = (x, y)
            elif 'pan' in name:
                self.tile_location['P'] = (x, y)
            elif name == 'bowl':
                self.tile_location['B'] = (x, y)
            elif name == 'plate':
                self.tile_location['D'] = (x, y)
            elif 'green_onion' in name:
                self.tile_location['G'] = (x, y)
            elif 'chopping_board' in name:
                self.tile_location['K'] = (x, y)
            elif 'sink' in name:
                self.tile_location['W'] = (x, y)
            elif name == 'table_h' or name == 'table_v':
                self.tile_location['T'] = (x, y)

        self.orientation_map = orientation_map
        self.grid = grid
        return obj_x_y, orientation_map, grid

    def name2abbr(self, name):

        name2abbr_map = {
            "counter": "C",
            "table_v": "T",
            "table_h": "T",
            "stove": "H",
            "bowl": "B",
            "pan": "P",
            "sink": "W",
            "fridge": "F",
            "broccoli": "F",
            "steak": "F",
            "green_onion": "G",
            "tray": "F",
            "apple": "F",
            "plate": "D",
            "scrub_brush": "W",
            "chopping_board": "K",
            "knife": "K"
        }

        return name2abbr_map[name]

    def load_objects(self, obj_x_y, orientation_map, order_list):
        name2path = {
            "counter":
            os.path.join(igibson.ig_dataset_path,
                         "objects/bottom_cabinet/46452/46452.urdf"),
            "table_h":
            os.path.join(igibson.ig_dataset_path,
                         "objects/breakfast_table/26670/26670.urdf"),
            #"table_h" : os.path.join(igibson.ig_dataset_path, "objects/breakfast_table/1b4e6f9dd22a8c628ef9d976af675b86/1b4e6f9dd22a8c628ef9d976af675b86.urdf"),
            "table_v":
            os.path.join(igibson.ig_dataset_path,
                         "objects/breakfast_table/26670/26670.urdf"),
            "stove":
            os.path.join(igibson.ig_dataset_path,
                         "objects/stove/101940/101940.urdf"),
            # "stove":
            # os.path.join(igibson.ig_dataset_path,
            #              "objects/stove/102019/102019.urdf"),
            "bowl":
            os.path.join(
                igibson.ig_dataset_path,
                "objects/bowl/a1393437aac09108d627bfab5d10d45d/a1393437aac09108d627bfab5d10d45d.urdf"
            ),
            "color_bowl":
            os.path.join(
                igibson.ig_dataset_path,
                "objects/bowl/56803af65db53307467ca2ad6571afff/56803af65db53307467ca2ad6571afff.urdf"
            ),
            "large_bowl":
            os.path.join(
                igibson.ig_dataset_path,
                "objects/bowl/5aad71b5e6cb3967674684c50f1db165/5aad71b5e6cb3967674684c50f1db165.urdf"
            ),
            "steak_bowl":
            os.path.join(
                igibson.ig_dataset_path,
                "objects/bowl/7d7bdea515818eb844638317e9e4ff18/7d7bdea515818eb844638317e9e4ff18.urdf"
            ),
            "pan":
            os.path.join(igibson.ig_dataset_path,
                         "objects/frying_pan/36_0/36_0.urdf"),
            "sink":
            os.path.join(igibson.ig_dataset_path,
                         "objects/sink/kitchen_sink/kitchen_sink.urdf"),
            "fridge":
            os.path.join(igibson.ig_dataset_path,
                         "objects/fridge/10373/10373.urdf"),
            "vidalia_onion":
            os.path.join(igibson.ig_dataset_path,
                         "objects/vidalia_onion/18_1/18_1.urdf"),
            "broccoli":
            os.path.join(igibson.ig_dataset_path,
                         "objects/broccoli/28_0/28_0.urdf"),
            "green_onion":
            os.path.join(
                igibson.ig_dataset_path,
                "objects/green_onion/green_onion_000/green_onion_000.urdf"),
            "steak":
            os.path.join(igibson.ig_dataset_path,
                         "objects/steak/steak_000/steak_000.urdf"),
            "tray":
            os.path.join(igibson.ig_dataset_path,
                         "objects/tray/tray_000/tray_000.urdf"),
            "apple":
            os.path.join(igibson.ig_dataset_path,
                         "objects/apple/00_0/00_0.urdf"),
            "plate":
            
            os.path.join(
                igibson.ig_dataset_path,
                "objects/bowl/a1393437aac09108d627bfab5d10d45d/a1393437aac09108d627bfab5d10d45d.urdf"
                # "objects/bowl/56803af65db53307467ca2ad6571afff/56803af65db53307467ca2ad6571afff.urdf"
            ),
            # os.path.join(igibson.ig_dataset_path,
            #              "objects/plate/plate_000/plate_000.urdf"),
            "scrub_brush":
            os.path.join(
                igibson.ig_dataset_path,
                "objects/scrub_brush/scrub_brush_000/scrub_brush_000.urdf"),
            "chopping_board":
            os.path.join(igibson.ig_dataset_path,
                         "objects/chopping_board/10_0/10_0.urdf"),
            "knife":
            os.path.join(igibson.ig_dataset_path,
                         "objects/carving_knife/14_1/14_1.urdf"),
            "onion": os.path.join(igibson.ig_dataset_path, "objects/vidalia_onion/17_0/17_0.urdf")
        }

        name2scale_map = {
            "counter": np.array([1.04, 0.97, 0.95]),
            "table_h": np.array([1.2, 1.2, 1.3]),
            "table_v": np.array([1.2, 1.2, 1.3]),
            "stove": np.array([1.1, 1, 1]),
            "bowl": np.array([0.8, 0.8, 0.8]),
            "pan": np.array([1.3, 1.3, 1]),
            "steak": np.array([0.1, 0.1, 0.1]),
            "broccoli": np.array([0.1, 0.1, 0.1]),
            "green_onion": np.array([0.1, 0.1, 0.1]),
            "tray": np.array([0.1, 0.1, 0.1]),
            "sink": np.array([1.2, 1.25, 1.25]),
            "fridge": np.array([1.5, 1.2, 1.2]),
            "vidalia_onion": np.array([2.0, 2.0, 2.0]),
            "apple": np.array([0.1, 0.1, 0.1]),
            "plate": np.array([0.8, 0.8, 0.8]), # np.array([0.01, 0.01, 0.01]),
            "scrub_brush": np.array([0.01, 0.01, 0.01]),
            "chopping_board": np.array([1.2, 1.2, 1.2]),
            "knife": np.array([1, 1, 1]),
            "onion": np.array([1.0, 1.0, 1.0]),
            "large_bowl": np.array([0.5, 0.5, 0.5]),
            "steak_bowl": np.array([0.5, 0.5, 0.5])
        }

        name2shift_map = {
            "counter": (0, 0, 0),
            "table_v": (0.5, 0, 0),
            "table_h": (0, 0.5, 0),
            "stove": (0, 0, 0),
            "bowl": (0, 0, 1.1),
            "pan": (0.23, 0.24, 1.24),  # shift height
            "sink": (0, 0, 0.1),
            "fridge": (0, 0, 0.2),
            "vidalia_onion": (0.15, -0.1, 1.3),
            "steak": (0, 0, 0.9), #(0.23, -0.1, 1.25),
            "tray": (0, 0, 0.9),
            "apple": (0, 0.2, 1.0),
            "broccoli": (0, 0.2, 0.6),
            "green_onion": (0, -0.2, 0.6),
            "green_onion": (0, 0, 1.25),
            "plate": (0, 0, 1.15),
            "scrub_brush": (0, 0, 1.3),
            "chopping_board": (0, 0, 1.2),
            "knife": (0, 0.3, 1.4),
            "onion": (0.15, -0.1, 0),
            "large_bowl": (0,0,0),
            "steak_bowl": (0,0,0)
        }

        # name2abl = {
        #     "counter": None,
        #     "table_v": None,
        #     "table_h": None,
        #     "stove": None,
        #     "bowl": {
        #         "dustyable": {},
        #         "stainable": {}
        #     },
        #     "pan": None,
        #     "sink": None,
        #     "fridge": {
        #         "coldSource": {
        #             "temperature": 7.0,
        #             "requires_inside": True,
        #         }
        #     }
        # }

        name2abl = {
            "counter": None,
            "table_v": None,
            "table_h": None,
            "stove": None,
            "bowl": {
                "dustyable": {},
                "stainable": {}
            },
            "pan": None,
            "sink": None,
            "fridge": {
                "coldSource": {
                    "temperature": 7.0,
                    "requires_inside": True,
                }
            },
            "steak": None,
            "tray": None,
            "apple": None,
            "broccoli": None,
            "green_onion": {
                "burnable": {},
                "freezable": {},
                "cookable": {},
                "sliceable": {
                    "slice_force": 0.0
                }
            },
            "plate1": {
                "dustyable": {},
                "stainable": {}
            },
            "plate2": {
                "dustyable": {},
                "stainable": {}
            },
            "scrub_brush": {
                "soakable": {},
                "cleaningTool": {}
            },
            "chopping_board": None,
            "knife": {
                "slicer": {}
            },
            "large_bowl": None,
            "steak_bowl": None
        }

        shift_l = 0.1
        mapping = {
            (0, 0, 3.1415926): (0, -shift_l),
            (0, 0, 0): (0, shift_l),
            (0, 0, 1.5707): (-shift_l, 0),
            (0, 0, -1.5707): (shift_l, 0),
        }

        for name, x, y in obj_x_y:
            obj = None
            orn = orientation_map[(name, x, y)]
            shift = name2shift_map[name]
            if name == "counter":
                x_shift, y_shift = mapping[orn]
                shift = (x_shift, y_shift, 0)
            elif name == "fridge":
                x_shift, y_shift = mapping[orn]
                shift = (x_shift, y_shift, 0)

            # pos = [x+shift[0]+0.5, y+shift[1]+0.5, 0+shift[2]]
            pos = [x + shift[0] - 4.5, y + shift[1] - 4.5, 0 + shift[2]]
            if name == "fridge":
                # obj = URDFObject(name2path[name], scale=name2scale_map[name]/1.15, model_path="/".join(name2path[name].split("/")[:-1]), category="fridge")
                obj = URDFObject(name2path["counter"], avg_obj_dims={'density': 10000},
                                 scale=name2scale_map["counter"] / 1.15,
                                 model_path="/".join(
                                     name2path["counter"].split("/")[:-1]),
                                 category="counter", fixed_base=True)
                
                self.env.simulator.import_object(obj)
                self.env.set_pos_orn_with_z_offset(obj, tuple(pos), orn)
                # obj.states[object_states.Open].set_value(True)

                self.fridges.append(obj)

                if 'onion' in order_list:
                    for _ in range(10):
                        onion = URDFObject(
                            name2path["vidalia_onion"],
                            scale=name2scale_map["vidalia_onion"] / 1.15,
                            model_path="/".join(
                                name2path["vidalia_onion"].split("/")[:-1]),
                            category="vidalia_onion")
                        self.env.simulator.import_object(onion)
                        onion.states[object_states.OnTop].set_value(obj, True, use_ray_casting_method=True)
                        # pos[0], pos[1] = self.sample_position(pos[0], pos[1], 0.1)
                        # pos[2] = 1.1
                        # self.env.set_pos_orn_with_z_offset(onion, pos)
                        self.onions.append(onion)
                        body_ids = onion.get_body_ids()
                        p.changeDynamics(body_ids[0], -1, mass=0.001)
                if 'steak' in order_list:
                    for _ in range(7):
                        steak = URDFObject(
                            name2path["steak"],
                            scale=name2scale_map["steak"] / 1.15,
                            model_path="/".join(
                                name2path["steak"].split("/")[:-1]),
                            category="steak")
                        self.env.simulator.import_object(steak)
                        steak.states[object_states.OnTop].set_value(obj, True, use_ray_casting_method=True)

                        self.meats.append(steak)
                        body_ids = steak.get_body_ids()
                        p.changeDynamics(body_ids[0], -1, mass=0.001)

            elif name == 'plate':
                obj = URDFObject(name2path[name],
                                 name=name,
                                 category=name,
                                 scale=name2scale_map[name] / 1.15,
                                 # avg_obj_dims={'density': 10000},
                                 model_path="/".join(
                                     name2path[name].split("/")[:-1]))
                self.env.simulator.import_object(obj)
                self.env.set_pos_orn_with_z_offset(obj, tuple(pos), orn)
                self.bowl_spawn_pos = pos

                for i in range(3):
                    obj2 = URDFObject(name2path[name],
                                    name=name,
                                    category=name,
                                    scale=name2scale_map[name] / 1.15,
                                    # avg_obj_dims={'density': 10000},
                                    model_path="/".join(
                                        name2path[name].split("/")[:-1]))
                    self.env.simulator.import_object(obj2)
                    self.env.set_pos_orn_with_z_offset(obj2, tuple([200 + 5*i,200, 1]), orn)
                    obj2.states[object_states.Dusty].set_value(True)
                    obj2.states[object_states.Stained].set_value(True)
                    self.plates.append(obj2)
            elif name == "stove":
                obj = URDFObject(name2path[name],
                                 scale=name2scale_map[name] / 1.15,
                                 model_path="/".join(
                                     name2path[name].split("/")[:-1]),
                                 category="stove")
                self.env.simulator.import_object(obj)
                self.env.set_pos_orn_with_z_offset(obj, tuple(pos), orn)
                obj.states[object_states.ToggledOn].set_value(False)
                self.stove = obj

            elif name == "pan":
                obj = URDFObject(name2path[name],
                                 name=name,
                                 category=name,
                                 scale=name2scale_map[name] / 1.15,
                                 model_path="/".join(
                                     name2path[name].split("/")[:-1]))
                rotated_basis = self.get_rotated_basis(orn)
                translated_pos = self.translate_loc(
                    rotated_basis, tuple([x - 4.5, y - 4.5, 0]), shift)
                self.env.simulator.import_object(obj)
                self.env.set_pos_orn_with_z_offset(obj, translated_pos, orn)
            elif name == "green_onion":
                # Create an URDF object of an apple, but doesn't load it in the simulator
                model_path = name2path[name]
                whole_obj = URDFObject(name2path[name],
                             name=name,
                             category=name,
                             scale=name2scale_map[name] / 1.15,
                             model_path="/".join(
                                 name2path[name].split("/")[:-1]),
                             abilities=name2abl[name])

                object_parts = []
                # Check the parts that compose the apple and create URDF objects of them
                for i, part in enumerate(whole_obj.metadata["object_parts"]):
                    part_category = part["category"]
                    part_model = part["model"]
                    # Scale the offset accordingly
                    part_pos = part["pos"] * whole_obj.scale
                    part_orn = part["orn"]
                    part_model_path = get_ig_model_path(part_category, part_model)
                    part_filename = os.path.join(part_model_path, part_model + ".urdf")
                    part_obj_name = whole_obj.name + "_part_{}".format(i)
                    part_obj = URDFObject(
                        part_filename,
                        name=part_obj_name,
                        category=part_category,
                        model_path=part_model_path,
                        scale=whole_obj.scale,
                    )
                    object_parts.append((part_obj, (part_pos, part_orn)))

                # Group the apple parts into a single grouped object
                grouped_parts_obj = ObjectGrouper(object_parts)

                # Create a multiplexed object: either the full apple, or the parts
                multiplexed_obj = ObjectMultiplexer(whole_obj.name + "_multiplexer", [whole_obj, grouped_parts_obj], 0)

                # Finally, load the multiplexed object
                self.env.simulator.import_object(multiplexed_obj)
                whole_obj.set_position([100, 100, -100])
                for i, (part_obj, _) in enumerate(object_parts):
                    part_obj.set_position([101 + i, 100, -100])

                # multiplexed_obj.set_position([0, 0, 0.72])
                # for i, (part_obj, _) in enumerate(object_parts):
                #     # self.env.set_pos_orn_with_z_offset(part_obj, tuple(pos), orn)
                #     new_pos = pos
                #     new_pos[2] += 0.07 * ((-1)**i)
                #     part_obj.set_position(new_pos)
                pos[2] += 0.05
                self.env.set_pos_orn_with_z_offset(whole_obj, tuple(pos), orn)
                #multiplexed_obj.states[object_states.OnTop].set_value(obj, True, use_ray_casting_method=True)
                self.onions.append(multiplexed_obj)
                body_ids = multiplexed_obj.get_body_ids()
                p.changeDynamics(body_ids[0], -1, mass=0.001)

                ####################################################
                self.onion_spawn_pos = pos

                for j in range(1):
                    whole_obj_2 = URDFObject(name2path[name],
                                name=name,
                                category=name,
                                scale=name2scale_map[name] / 1.15,
                                model_path="/".join(
                                    name2path[name].split("/")[:-1]),
                                abilities=name2abl[name])

                    object_parts = []
                    # Check the parts that compose the apple and create URDF objects of them
                    for i, part in enumerate(whole_obj_2.metadata["object_parts"]):
                        part_category = part["category"]
                        part_model = part["model"]
                        # Scale the offset accordingly
                        part_pos = part["pos"] * whole_obj_2.scale
                        part_orn = part["orn"]
                        part_model_path = get_ig_model_path(part_category, part_model)
                        part_filename = os.path.join(part_model_path, part_model + ".urdf")
                        part_obj_name = whole_obj_2.name + "_part_{}".format(i)
                        part_obj = URDFObject(
                            part_filename,
                            name=part_obj_name,
                            category=part_category,
                            model_path=part_model_path,
                            scale=whole_obj_2.scale,
                        )
                        object_parts.append((part_obj, (part_pos, part_orn)))

                    # Group the apple parts into a single grouped object
                    grouped_parts_obj_2 = ObjectGrouper(object_parts)

                    # Create a multiplexed object: either the full apple, or the parts
                    multiplexed_obj_2 = ObjectMultiplexer(whole_obj_2.name + "_multiplexer", [whole_obj_2, grouped_parts_obj_2], 0)

                    # Finally, load the multiplexed object
                    self.env.simulator.import_object(multiplexed_obj_2)
                    whole_obj_2.set_position([200 + 2*j, 100, 1])
                    for i, (part_obj, _) in enumerate(object_parts):
                        part_obj.set_position([201 + 2*j, 90, 1])

                    # multiplexed_obj.set_position([0, 0, 0.72])
                    # for i, (part_obj, _) in enumerate(object_parts):
                    #     # self.env.set_pos_orn_with_z_offset(part_obj, tuple(pos), orn)
                    #     new_pos = pos
                    #     new_pos[2] += 0.07 * ((-1)**i)
                    #     part_obj.set_position(new_pos)
                    pos[2] += 0.05
                    self.env.set_pos_orn_with_z_offset(whole_obj_2, tuple([200,100,1]), orn)
                    self.onions.append(multiplexed_obj_2)
                    body_ids = multiplexed_obj_2.get_body_ids()
                    p.changeDynamics(body_ids[0], -1, mass=0.001)

            #######################################################
                # whole_obj_3 = URDFObject(name2path[name],
                #             name=name,
                #             category=name,
                #             scale=name2scale_map[name] / 1.15,
                #             model_path="/".join(
                #                 name2path[name].split("/")[:-1]),
                #             abilities=name2abl[name])

                # object_parts = []
                # # Check the parts that compose the apple and create URDF objects of them
                # for i, part in enumerate(whole_obj_3.metadata["object_parts"]):
                #     part_category = part["category"]
                #     part_model = part["model"]
                #     # Scale the offset accordingly
                #     part_pos = part["pos"] * whole_obj_3.scale
                #     part_orn = part["orn"]
                #     part_model_path = get_ig_model_path(part_category, part_model)
                #     part_filename = os.path.join(part_model_path, part_model + ".urdf")
                #     part_obj_name = whole_obj_3.name + "_part_{}".format(i)
                #     part_obj = URDFObject(
                #         part_filename,
                #         name=part_obj_name,
                #         category=part_category,
                #         model_path=part_model_path,
                #         scale=whole_obj_3.scale,
                #     )
                #     object_parts.append((part_obj, (part_pos, part_orn)))

                # # Group the apple parts into a single grouped object
                # grouped_parts_obj_2 = ObjectGrouper(object_parts)

                # # Create a multiplexed object: either the full apple, or the parts
                # multiplexed_obj_2 = ObjectMultiplexer(whole_obj_3.name + "_multiplexer", [whole_obj_3, grouped_parts_obj_2], 0)

                # # Finally, load the multiplexed object
                # self.env.simulator.import_object(multiplexed_obj_2)
                # whole_obj_3.set_position([133, 1, 1])
                # for i, (part_obj, _) in enumerate(object_parts):
                #     part_obj.set_position([133, 10, 1])

                # # multiplexed_obj.set_position([0, 0, 0.72])
                # # for i, (part_obj, _) in enumerate(object_parts):
                # #     # self.env.set_pos_orn_with_z_offset(part_obj, tuple(pos), orn)
                # #     new_pos = pos
                # #     new_pos[2] += 0.07 * ((-1)**i)
                # #     part_obj.set_position(new_pos)
                # pos[2] += 0.05
                # self.env.set_pos_orn_with_z_offset(whole_obj_3, tuple([200,100,1]), orn)
                # self.onions.append(multiplexed_obj_2)
                # body_ids = multiplexed_obj_2.get_body_ids()
                # p.changeDynamics(body_ids[0], -1, mass=0.001)
            elif 'counter' in name:
                obj = URDFObject(name2path[name],
                                 name=name,
                                 category=name,
                                 scale=name2scale_map[name] / 1.15,
                                 avg_obj_dims={'density': 10000},
                                 model_path="/".join(
                                     name2path[name].split("/")[:-1]),
                                 abilities=name2abl[name])
                self.env.simulator.import_object(obj)
                self.env.set_pos_orn_with_z_offset(obj, tuple(pos), orn)
            else:
                obj = URDFObject(name2path[name],
                                 name=name,
                                 category=name,
                                 scale=name2scale_map[name] / 1.15,
                                 # avg_obj_dims={'density': 10000},
                                 model_path="/".join(
                                     name2path[name].split("/")[:-1]),
                                 abilities=name2abl[name])
                self.env.simulator.import_object(obj)
                self.env.set_pos_orn_with_z_offset(obj, tuple(pos), orn)

            if name not in ("bowl", "pan", "vidalia_onion", "steak", "plate", "chopping_board", "knife", "green_onion"):
                self.static_objs[obj] = (x, y)
            if name == "bowl" or name == "plate":
                obj.states[object_states.Dusty].set_value(True)
                obj.states[object_states.Stained].set_value(True)
                self.bowls.append(obj)
                body_ids = obj.get_body_ids()
                p.changeDynamics(body_ids[0], -1, mass=0.01)
            if name == "pan":
                self.pans.append(obj)

            if name == "vidalia_onion":
                self.onions.append(obj)
                body_ids = obj.get_body_ids()
                p.changeDynamics(body_ids[0], -1, mass=0.001)
            if name == "table_h":
                self.table = obj
                self.static_objs[obj] = [(x, y), (x,y+1)]
            if name == "table_v":
                self.table = obj
                self.static_objs[obj] = [(x, y), (x+1, y)]
            if name == "counter":
                self.counters.append(obj)
            if name == "chopping_board":
                self.chopping_boards.append(obj)
                body_ids = obj.get_body_ids()
                p.changeDynamics(body_ids[0], -1, mass=100)
            if name == "sink":
                self.sinks.append(obj)
            if name == "plate":
                self.plates.append(obj)
            if name == "knife":
                self.knives.append(obj)
                self.init_knife_pos = None
                body_ids = obj.get_body_ids()
                p.changeDynamics(body_ids[0], -1, mass=0.01)
        
        name = "large_bowl"
        large_bowl = URDFObject(name2path[name],
                                    name=name,
                                    category=name,
                                    scale=name2scale_map[name] / 1.15,
                                    # avg_obj_dims={'density': 10000},
                                    model_path="/".join(
                                        name2path[name].split("/")[:-1]))
        self.env.simulator.import_object(large_bowl)
        large_bowl.set_position([300,300,1])
        self.large_bowl = large_bowl

        name = "steak_bowl"
        steak_bowl = URDFObject(name2path[name],
                                    name=name,
                                    category=name,
                                    scale=name2scale_map[name] / 1.15,
                                    # avg_obj_dims={'density': 10000},
                                    model_path="/".join(
                                        name2path[name].split("/")[:-1]))
        self.env.simulator.import_object(steak_bowl)
        steak_bowl.set_position([275,275,1])
        self.steak_bowl = steak_bowl

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
            # for steak order
            # if name in ("apple", "green_onion", "broccoli", "steak"):
            #     self.food_obj.append(obj)
            #     obj.states[object_states.Temperature].get_value(),
            #     obj.states[object_states.Temperature].set_value(7)

            # if "plate" in name:
            #     if object_states.Dusty in obj.states:
            #         obj.states[object_states.Dusty].set_value(True)

            #     if object_states.Stained in obj.states:
            #         obj.states[object_states.Stained].set_value(True)
            # if name == "sink":
            #     obj.states[object_states.ToggledOn].set_value(False)
            # if name == "chopping_board":
            #     obj.states[object_states.OnTop].set_value(obj, self.static_objs[-1])

        try:
            for obj in self.static_objs.keys():
                p.changeDynamics(obj.get_body_ids()[0], -1,
                                 mass=800)  # mass=500
        except:
            print("****** Error *******")
            # pass

    def grid2raw(self, filepath):
        grid = open(filepath, "r").read().strip().split("\n")
        self.HEIGHT = len(grid[0])
        self.WIDTH = len(grid[0])
        grid = [list(each) for each in grid]
        abbr2name = {
            "X": "space",
            "H": "stove",
            "C": "counter",
            "B": "bowl",
            "P": "pan",
            "F": "fridge",
            "T": "table",
            "W": "sink",
            "K": "chopping_board",
            "G": "green_onion",
            "D": "plate"
        }
        return_str = ""
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if grid[x][y] == "X":
                    continue
                elif grid[x][y] == "T":
                    if y + 1 < self.WIDTH and grid[x][y + 1] == "T":
                        grid[x][y + 1] = "X"
                        return_str += "{} {} {}\n".format("table_h", x, y)
                    else:
                        grid[x + 1][y] = "X"
                        return_str += "{} {} {}\n".format("table_v", x, y)
                else:
                    name = abbr2name[grid[x][y]]
                    if name == "bowl":
                        return_str += "{} {} {}\n".format("counter", x, y)
                        # return_str += "{} {} {}\n".format("vidalia_onion", x, y)
                    if name == "pan":
                        return_str += "{} {} {}\n".format("stove", x, y)
                    if name == "stove":
                        return_str += "{} {} {}\n".format("pan", x, y)
                    if name == "chopping_board":
                        return_str += "{} {} {}\n".format("counter", x, y)
                        return_str += "{} {} {}\n".format("knife", x, y)
                    if name == "green_onion":
                        return_str += "{} {} {}\n".format("counter", x, y)
                    if name == "plate":
                        return_str += "{} {} {}\n".format("counter", x, y)

                    return_str += "{} {} {}\n".format(name, x, y)

        return return_str

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
                    indexes.append((i,j))

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
                if n_x > len(self.grid)-1 or n_y > len(self.grid)-1:
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
                if onion.current_index == 0 and onion.get_position()[0] < 50 and onion not in self.overcooked_object_states.keys():
                    if not onion.states[object_states.OnTop].get_value(self.onion_counter, use_ray_casting_method=True):
                        onion.states[object_states.OnTop].set_value(self.onion_counter, True, use_ray_casting_method=True)
                    # x,y = grid_to_real_coord(self.onion_counter.get_position())
                    # onion.set_position([x,y,1.1])

        # to check if onion is on chopping board
        # for onion in self.onions:
        #     holding_onion = False
        #     for agent in self.env.robots:
        #         body_id = onion.get_body_ids()[0]
        #         grasping = agent.is_grasping_all_arms(body_id)
        #         holding_onion = IsGraspingState.TRUE in grasping
        #         if holding_onion:
        #             break
        #     if not holding_onion:
        #         if onion.current_index == 1:
        #             if not onion.states[object_states.OnTop].get_value(self.chopping_boards[0]):
        #                 onion.objects[0].states[object_states.OnTop].set_value(self.chopping_boards[0], True, use_ray_casting_method=True)
        #             # x,y = grid_to_real_coord(self.onion_counter.get_position())
        #             # onion.set_position([x,y,1.1])
            
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

            # also check world
            # for plate in self.plates:
            #     if plate.states[object_states.OnTop].get_value(sink):
            #         self.rinse_sink(sink)

        for sink, time in self.rinsing_sinks.copy().items():
            # if time == None:
            #     time = time.time()
            
            if time > 5:
                self.rinsing_sinks.pop(sink)
                self.ready_sinks.append(sink)

        for meat in self.meats.copy():
            for pan in self.pans:
            # if meat.states[object_states.Cooked].get_value():
                if meat.states[object_states.Inside].get_value(pan):
                    self.steaks.append(meat)
                    self.meats.remove(meat)

        in_robot_hand_id = 0
        if self.robot_carrying_dish:

            if self.robot_stash_steak_bowl is not None:
                self.robot_carrying_steak = False
                self.steak_bowl.set_position([110,110,1])
                self.robot_stash_steak_bowl = None
                # self.in_robot_hand.clear()

            if self.robot_stash_dish is None:
                # stash items far away
                self.robot_carrying_dish = True
                
                offset_idx = 0

                plate = [x for x in self.in_robot_hand if 'plate' in x[1].name][0][1]
                steak = [x for x in self.in_robot_hand if 'steak' in x[1].name][0][1]
                onion = [x for x in self.in_robot_hand if 'onion' in x[1].name][0][1]

                stash_pos = [30,30,1]
                (x,y,z) = stash_pos
                new_pos = (x+(0.1*offset_idx),y,z)

                self.robot_stash_dish = plate
                plate.set_position(new_pos)
                offset_idx += 1
                new_pos = (x+(0.5*offset_idx),y,z)
                steak.set_position(new_pos)
                offset_idx += 1
                
                for sub_obj in onion.objects:
                    new_pos = (x+(0.5*offset_idx),y,z)
                    sub_obj.set_position(new_pos)
                    offset_idx += 1

            b_x, b_y, b_z = self.env.robots[0].get_eef_position()
            self.large_bowl.set_position([b_x, b_y, b_z + 0.12])
        elif self.robot_carrying_steak:
            if self.robot_stash_steak_bowl == None:
                plate = [x for x in self.in_robot_hand if 'plate' in x[1].name][0][1]
                self.robot_stash_steak_bowl = plate
                plate.set_position([80,80,1])
            b_x, b_y, b_z = self.env.robots[0].get_eef_position()
            self.steak_bowl.set_position([b_x, b_y, b_z + 0.12])
            steak = [x for x in self.in_robot_hand if 'steak' in x[1].name][0][1]
            steak.set_position(self.env.robots[0].get_eef_position())
        else:
            if self.robot_stash_dish is not None:
                self.robot_carrying_dish = False
                self.large_bowl.set_position([130,130,1])
                self.robot_stash_dish = None
                self.in_robot_hand.clear()
            
            for obj in self.in_robot_hand:
                (x,y,z) = self.env.robots[0].get_eef_position()
                new_pos = (x,y,z+(0.05*in_robot_hand_id))
                ig_obj = obj[-1]
                if type(ig_obj) == ObjectMultiplexer and ig_obj.current_index == 1:
                    for sub_obj in ig_obj.objects:
                        sub_obj.set_position(new_pos)
                else:
                    obj[-1].set_position(self.env.robots[0].get_eef_position())

            # 
            # obj[-1].set_position((x,y,z+(0.12*in_robot_hand_id)))
            
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
            # self.env.set_pos_orn_with_z_offset(plate, self.bowl_spawn_pos, [0,0,0])
            if plate is not None:
                plate.states[object_states.OnTop].set_value(self.dish_station, True, use_ray_casting_method=True)
            # body_ids = plate.get_body_ids()
            # p.changeDynamics(body_ids[0], -1, mass=0.001)

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

            if self.init_knife_pos is not None and not holding_knife and not knife.states[object_states.OnTop].get_value(self.chopping_counter):
                knife.set_position(self.init_knife_pos)
                # knife.states[object_states.OnTop].set_value(self.chopping_counter, True, use_ray_casting_method=True)
        
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
                self.env.set_pos_orn_with_z_offset(onion, self.onion_spawn_pos, [0,0,0])
                body_ids = onion.get_body_ids()
                p.changeDynamics(body_ids[0], -1, mass=0.001)