from math import floor
import numpy as np
import os
import igibson
from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.utils.assets_utils import get_ig_model_path
from igibson.object_states.heat_source_or_sink import HeatSourceOrSink
from igibson import object_states
import pybullet as p

from lsi_3d.utils.constants import DIRE2POSDIFF


class Kitchen():

    def __init__(self, env):
        self.env = env
        self.HEIGHT = 8  # x
        self.WIDTH = 8  # y
        self.orientation_map = ""
        self.grid = ""
        self.bowls = []
        self.pans = []
        self.onions = []
        self.table = None
        self.food_obj = []

        # tile location is a dictionary of item locations in the environment indexed by letter (eg F for fridge)
        self.tile_location = {}

    def setup(self, filepath):
        obj_x_y, orientation_map, grid = self.read_from_grid_text(filepath)
        self.map = orientation_map
        self.load_objects(obj_x_y, orientation_map)

    def read_from_grid_text(self, filepath):
        obj_x_y = []
        grid = [["X"] * self.WIDTH for _ in range(self.HEIGHT)]
        sum_x, sum_y, count = 0, 0, 0  # for calculation of center mass (excluding table)
        orientation_map = dict()

        raw_str = self.grid2raw(filepath)
        for line in raw_str.strip().split("\n"):
            if len(line) == 0:
                break
            name, x, y = line.split()
            x, y = int(x), int(y)
            obj_x_y.append((name, x, y))
            if (grid[x][y] == "X" or grid[x][y] == "C") and name != "vidalia_onion":
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
                self.tile_location['F'] = (x,y)
            elif name == 'stove':
                self.tile_location['S'] = (x,y)
            elif name == 'bowl':
                self.tile_location['B'] = (x,y)
            elif name == 'table_h':
                self.tile_location['T'] = (x,y)

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
            "fridge": "F"
        }

        return name2abbr_map[name]

    def load_objects(self, obj_x_y, orientation_map):
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
                         "objects/stove/102019/102019.urdf"),
            "bowl":
            os.path.join(
                igibson.ig_dataset_path,
                "objects/bowl/a1393437aac09108d627bfab5d10d45d/a1393437aac09108d627bfab5d10d45d.urdf"
            ),
            "pan":
            os.path.join(igibson.ig_dataset_path,
                         "objects/frying_pan/36_0/36_0.urdf"),
            "sink":
            os.path.join(igibson.ig_dataset_path,
                         "objects/sink/kitchen_sink/kitchen_sink.urdf"),
            "fridge":
            os.path.join(igibson.ig_dataset_path,
                         "objects/fridge/11712/11712.urdf"),
            "vidalia_onion": os.path.join(igibson.ig_dataset_path, "objects/vidalia_onion/17_0/17_0.urdf")
        }

        name2scale_map = {
            "counter": np.array([1.04, 0.97, 0.95]),
            "table_h": np.array([1.2, 1.2, 1.3]),
            "table_v": np.array([1.2, 1.2, 1.3]),
            "stove": np.array([0.88, 1.1, 0.95]),
            "bowl": np.array([0.8, 0.8, 1]),
            "pan": np.array([1, 1, 1]),
            "tray": np.array([0.1, 0.1, 0.1]),
            "sink": np.array([1.2, 1.25, 1.25]),
            "fridge": np.array([1.5, 1.2, 1.2]),
            "vidalia_onion": np.array([2.0, 2.0, 2.0])
        }

        name2shift_map = {
            "counter": (0, 0, 0),
            "table_v": (0.5, 0, 0),
            "table_h": (0, 0.5, 0),
            "stove": (0, 0, 0),
            "bowl": (0, 0, 1.2),
            "pan": (0.15, -0.1, 1.24),  # shift height
            "sink": (0, 0, 0.1),
            "fridge": (0, 0, 0.2),
            "vidalia_onion": (0.15, -0.1, 1.3)
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

        shift_l = 0.1
        mapping = {
            (0, 0, 3.1415926): (0, -shift_l),
            (0, 0, 0): (0, shift_l),
            (0, 0, 1.5707): (-shift_l, 0),
            (0, 0, -1.5707): (shift_l, 0),
        }

        static_objs = []
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
                obj = URDFObject(name2path[name], scale=name2scale_map[name]/1.15, model_path="/".join(name2path[name].split("/")[:-1]), category="refrigerator")
                self.env.simulator.import_object(obj)
                self.env.set_pos_orn_with_z_offset(obj, tuple(pos), orn)
                # obj.states[object_states.Open].set_value(True)
                for i in range(20):
                    onion = URDFObject(name2path["vidalia_onion"], scale=name2scale_map["vidalia_onion"]/1.15, model_path="/".join(name2path["vidalia_onion"].split("/")[:-1]), category="vidalia_onion")
                    self.env.simulator.import_object(onion)
                    onion.states[object_states.Inside].set_value(obj, True, use_ray_casting_method=True)
                    self.onions.append(onion)
            elif name == "stove":
                obj = URDFObject(name2path[name], scale=name2scale_map[name]/1.15, model_path="/".join(name2path[name].split("/")[:-1]), category="stove")
                self.env.simulator.import_object(obj)
                self.env.set_pos_orn_with_z_offset(obj, tuple(pos), orn)
                obj.states[object_states.ToggledOn].set_value(True)
            else:
                obj = URDFObject(name2path[name],
                             name=name,
                             category=name,
                             scale=name2scale_map[name] / 1.15,
                             model_path="/".join(
                                 name2path[name].split("/")[:-1]))
                self.env.simulator.import_object(obj)
                self.env.set_pos_orn_with_z_offset(obj, tuple(pos), orn)

            if name not in ("bowl", "pan", "vidalia_onion"):
                static_objs.append(obj)
            if name == "bowl":
                self.bowls.append(obj)
            if name == "pan":
                self.pans.append(obj)
            if name == "vidalia_onion":
                self.onions.append(obj)
            if name == "table_h" or name == "table_v":
                self.table = obj

        try:
            for obj in static_objs:
                p.changeDynamics(obj.get_body_ids()[0], -1,
                                 mass=800)  # mass=500
        except:
            pass

    def grid2raw(self, filepath):
        grid = open(filepath, "r").read().strip().split("\n")
        grid = [list(each) for each in grid]
        abbr2name = {
            "X": "space",
            "H": "stove",
            "C": "counter",
            "B": "bowl",
            "P": "pan",
            "F": "fridge",
            "T": "table",
            "W": "sink"
        }
        return_str = ""
        for x in range(self.HEIGHT):
            for y in range(self.WIDTH):
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
                        return_str += "{} {} {}\n".format("vidalia_onion", x, y)
                    if name == "pan":
                        return_str += "{} {} {}\n".format("stove", x, y)
                        return_str += "{} {} {}\n".format("vidalia_onion", x, y)

                    return_str += "{} {} {}\n".format(name, x, y)
        return return_str

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

    def get_center(self):
        grid_center = (floor(len(self.grid)/2),floor(len(self.grid[0])/2))
        return grid_center
    
    def nearest_empty_tile(self, loc):
        grid_center = (floor(len(self.grid)/2),floor(len(self.grid[0])/2))
        open_spaces = []
        x,y = loc
        if self.grid[x][y] == 'X':
            return loc
        else:
            for dir,add in DIRE2POSDIFF.items():
                d_x,d_y = add
                n_x,n_y = (x+d_x, y+d_y)
                if self.grid[n_x][n_y] == 'X':
                    open_spaces.append((n_x,n_y))

        # manhattan dist to center pick closest
        chosen_space = None
        chosen_dist = 1000000
        for space in open_spaces:
            gcx, gcy = grid_center
            sx,sy = space
            dist = abs(gcx-sx)+abs(gcy-sy)
            if dist < chosen_dist:
                chosen_space = space
                chosen_dist = dist

        return chosen_space

    def step(self, count=0):
        for obj in self.food_obj:
            # try:
            #     print(
            #         "%s. Temperature: %.2f. Frozen: %r. Cooked: %r. Burnt: %r."
            #         % (
            #             obj.name,
            #             obj.states[object_states.Temperature].get_value(),
            #             obj.states[object_states.Frozen].get_value(),
            #             obj.states[object_states.Cooked].get_value(),
            #             obj.states[object_states.Burnt].get_value(),
            #         ))
            # except:
            #     pass

            if obj.name == "green_onion_multiplexer" and count > 80:
                print("Slicing the green onion and moving the parts")
                # Slice the apple and set the object parts away
                part_pos = obj.get_position()
                obj.states[object_states.Sliced].set_value(True)

                # Check that the multiplexed changed to the group of parts
                assert isinstance(obj.current_selection(), ObjectGrouper)
                self.food_obj.remove(obj)

                # Move the parts
                for i, part_obj in enumerate(obj.objects):
                    new_pos = part_pos.copy()
                    new_pos[1] += 0.05 * ((-1)**i)
                    part_obj.set_position(new_pos)
                    self.food_obj.append(part_obj)

