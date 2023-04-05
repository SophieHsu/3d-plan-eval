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


class Kitchen():

    def __init__(self, env):
        self.env = env
        self.HEIGHT = 8  # x
        self.WIDTH = 8  # y
        self.orientation_map = ""
        self.grid = ""
        self.bowlpans = []
        self.food_obj = []
        self.bowls = []
        self.pans = []
        self.onions = []

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
            if grid[x][y] == "X" or grid[x][y] == "C":
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
            "steak": "P",
            "green_onion": "K",
            "tray": "F",
            "apple": "F",
            "plate1": "W",
            "plate2": "W",
            "scrub_brush": "W",
            "chopping_board": "K",
            "knife": "K"
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
            "plate1":
            os.path.join(igibson.ig_dataset_path,
                         "objects/plate/plate_000/plate_000.urdf"),
            "plate2":
            os.path.join(igibson.ig_dataset_path,
                         "objects/plate/plate_000/plate_000.urdf"),
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
            "stove": np.array([0.88, 1.1, 0.95]),
            "bowl": np.array([0.8, 0.8, 1]),
            "pan": np.array([1, 1, 1]),
            "steak": np.array([0.1, 0.1, 0.1]),
            "broccoli": np.array([0.1, 0.1, 0.1]),
            "green_onion": np.array([0.1, 0.1, 0.1]),
            "tray": np.array([0.1, 0.1, 0.1]),
            "sink": np.array([1.2, 1.25, 1.25]),
            "fridge": np.array([1.5, 1.2, 1.2]),
            "apple": np.array([0.1, 0.1, 0.1]),
            "plate1": np.array([0.01, 0.01, 0.01]),
            "plate2": np.array([0.01, 0.01, 0.01]),
            "scrub_brush": np.array([0.01, 0.01, 0.01]),
            "chopping_board": np.array([1.2, 1.2, 1.2]),
            "knife": np.array([1, 1, 1]),
            "onion": np.array([1.0, 1.0, 1.0])
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
            "steak": (0.23, -0.1, 1.25),
            "tray": (0, 0, 0.9),
            "apple": (0, 0.2, 1.0),
            "broccoli": (0, 0.2, 0.6),
            # "green_onion": (0, -0.2, 0.6),
            "green_onion": (0, 0, 1.25),
            "plate1": (0, 0, 1.2),
            "plate2": (0, 0, 1.0),
            "scrub_brush": (0, 0, 1.3),
            "chopping_board": (0, 0, 1.2),
            "knife": (0.4, 0, 1.2),
            "onion": (0.15, -0.1, 0)
        }

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
            }
        }

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
            if name == "green_onion":
                obj = URDFObject(name2path[name],
                             name=name,
                             category=name,
                             scale=name2scale_map[name] / 1.15,
                             model_path="/".join(
                                 name2path[name].split("/")[:-1]),
                             abilities=name2abl[name])
                object_parts = []
                # Check the parts that compose the apple and create URDF objects of them
                for i, part in enumerate(obj.metadata["object_parts"]):
                    part_category = part["category"]
                    part_model = part["model"]
                    # Scale the offset accordingly
                    part_pos = part["pos"] * obj.scale
                    part_orn = part["orn"]
                    part_model_path = get_ig_model_path(
                        part_category, part_model)
                    part_filename = os.path.join(part_model_path,
                                                 part_model + ".urdf")
                    part_obj_name = obj.name + "_part_{}".format(i)
                    part_obj = URDFObject(
                        part_filename,
                        name=part_obj_name,
                        category=part_category,
                        model_path=part_model_path,
                        scale=obj.scale,
                    )
                    object_parts.append((part_obj, (part_pos, part_orn)))

                # Group the apple parts into a single grouped object
                grouped_parts_obj = ObjectGrouper(object_parts)

                # Create a multiplexed object: either the full apple, or the parts
                multiplexed_obj = ObjectMultiplexer(obj.name + "_multiplexer",
                                                    [obj, grouped_parts_obj],
                                                    0)
                self.env.simulator.import_object(multiplexed_obj)
                obj.set_position([100, 100, -100])
                for i, (part_obj, _) in enumerate(object_parts):
                    part_obj.set_position([101 + i, 100, -100])

                obj = multiplexed_obj
            elif name == "bowl":
                obj = URDFObject(name2path[name], scale=name2scale_map[name]/1.15, model_path="/".join(name2path[name].split("/")[:-1]))
                self.env.simulator.import_object(obj)
                self.bowls.append(obj)

                onion = URDFObject(name2path["onion"], scale=name2scale_map["onion"]/1.15, model_path="/".join(name2path["onion"].split("/")[:-1]), category="vidalia_onion")
                self.env.simulator.import_object(onion)
                orn = (0, 0, 0)
                shift = name2shift_map["onion"]
                pos = [x+shift[0]-4.5, y+shift[1]-4.5, 2.0+shift[2]]
                self.env.set_pos_orn_with_z_offset(onion, tuple(pos), orn)
                self.onions.append(onion)
            elif name == "pan":
                obj = URDFObject(name2path[name], scale=name2scale_map[name]/1.15, model_path="/".join(name2path[name].split("/")[:-1]))
                self.env.simulator.import_object(obj)
                self.pans.append(obj)
                
                onion = URDFObject(name2path["onion"], scale=name2scale_map["onion"]/1.15, model_path="/".join(name2path["onion"].split("/")[:-1]), category="vidalia_onion")
                self.env.simulator.import_object(onion)
                orn = (0, 0, 0)
                shift = name2shift_map["onion"]
                pos = [x+shift[0]-4.5, y+shift[1]-4.5, 2.0+shift[2]]
                self.env.set_pos_orn_with_z_offset(onion, tuple(pos), orn)
                self.onions.append(onion)
            elif name == "stove":
                obj = URDFObject(name2path[name], scale=name2scale_map[name]/1.15, model_path="/".join(name2path[name].split("/")[:-1]), category="stove")
                self.env.simulator.import_object(obj)
                obj.states[object_states.ToggledOn].set_value(True)
                # print(obj.states[object_states.ToggledOn].set_value(True))
            else:
                obj = URDFObject(name2path[name],
                             name=name,
                             category=name,
                             scale=name2scale_map[name] / 1.15,
                             model_path="/".join(
                                 name2path[name].split("/")[:-1]),
                             abilities=name2abl[name])
                self.env.simulator.import_object(obj)

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
            self.env.set_pos_orn_with_z_offset(obj, tuple(pos), orn)

            if name not in ("bowl", "pan", "green_onion", "broccoli", "steak",
                            "tray", "plate", "chopping_board", "knife"):
                static_objs.append(obj)
            else:
                self.bowlpans.append((obj, pos))

            if name in ("apple", "green_onion", "broccoli", "steak"):
                self.food_obj.append(obj)
                obj.states[object_states.Temperature].get_value(),
                obj.states[object_states.Temperature].set_value(7)
            if "plate" in name:
                if object_states.Dusty in obj.states:
                    obj.states[object_states.Dusty].set_value(True)

                if object_states.Stained in obj.states:
                    obj.states[object_states.Stained].set_value(True)
            if name == "sink":
                obj.states[object_states.ToggledOn].set_value(False)
            # if name == "chopping_board":
            #     obj.states[object_states.OnTop].set_value(obj, static_objs[-1])

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
            "W": "sink",
            "K": "chopping_board",
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
                        #pass
                        return_str += "{} {} {}\n".format("counter", x, y)
                    if name == "pan":
                        return_str += "{} {} {}\n".format("stove", x, y)
                        return_str += "{} {} {}\n".format("steak", x, y)
                    if name == "fridge":
                        return_str += "{} {} {}\n".format("apple", x, y)
                        return_str += "{} {} {}\n".format("tray", x, y)
                        # return_str += "{} {} {}\n".format("green_onion", x, y)
                        return_str += "{} {} {}\n".format("broccoli", x, y)

                    if name == "sink":
                        return_str += "{} {} {}\n".format("plate1", x, y)
                        return_str += "{} {} {}\n".format("plate2", x, y)
                        return_str += "{} {} {}\n".format("scrub_brush", x, y)

                    if name == "chopping_board":
                        return_str += "{} {} {}\n".format("counter", x, y)
                        return_str += "{} {} {}\n".format("green_onion", x, y)
                        return_str += "{} {} {}\n".format("knife", x, y)

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

    def step(self, count=0):
        for obj in self.food_obj:
            try:
                print(
                    "%s. Temperature: %.2f. Frozen: %r. Cooked: %r. Burnt: %r."
                    % (
                        obj.name,
                        obj.states[object_states.Temperature].get_value(),
                        obj.states[object_states.Frozen].get_value(),
                        obj.states[object_states.Cooked].get_value(),
                        obj.states[object_states.Burnt].get_value(),
                    ))
            except:
                pass

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

        print()