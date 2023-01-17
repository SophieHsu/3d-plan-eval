import numpy as np
import os
import igibson
from igibson.objects.articulated_object import URDFObject
import pybullet as p

class Kitchen():
    def __init__(self, env):
        self.env = env
        self.HEIGHT = 8 # x
        self.WIDTH = 8 # y
        self.orientation_map = ""
        self.grid = ""
        self.bowlpans = []

    def setup(self, filepath):
        obj_x_y, orientation_map, grid = self.read_from_grid_text(filepath)
        self.map = orientation_map
        self.load_objects(obj_x_y, orientation_map)

    def read_from_grid_text(self, filepath):
        obj_x_y = []
        grid = [["X"]*self.WIDTH for _ in range(self.HEIGHT)]
        sum_x, sum_y, count = 0, 0, 0 # for calculation of center mass (excluding table)
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
                grid[x][y+1] = self.name2abbr(name)
            elif name == "table_v":
                grid[x+1][y] = self.name2abbr(name)
            elif name == 'pan':
                grid[x][y] = self.name2abbr(name)
            else:
                sum_x += x
                sum_y += y
                count += 1
        if count == 0:
            count = 1
        center_x, center_y = sum_x/count, sum_y/count

        for name, x, y in obj_x_y:
            if name == "table_h":
                orientation_map[(name, x, y)] = (0, 0, 1.5707)
            elif name == "table_v":
                orientation_map[(name, x, y)] = (0, 0, 0)
            else:
                ori = 0 # if ori > 0 then it's facing left/right, otherwise it's facing up/down
                ori += self.ori_filter(grid, x+1, y) + self.ori_filter(grid, x-1, y) # upper and lower neighbor
                ori -= self.ori_filter(grid, x, y+1) + self.ori_filter(grid, x, y-1) # left and right neighbor
                orientation_map[(name, x, y)] = self.get_orientation(center_x, center_y, x, y, ori)

        self.orientation_map = orientation_map
        self.grid = grid
        return obj_x_y, orientation_map, grid

    def load_objects(self, obj_x_y, orientation_map):
        name2path = {
            "counter" : os.path.join(igibson.ig_dataset_path, "objects/bottom_cabinet/46452/46452.urdf"),
            "table_h" : os.path.join(igibson.ig_dataset_path, "objects/breakfast_table/26670/26670.urdf"),
            #"table_h" : os.path.join(igibson.ig_dataset_path, "objects/breakfast_table/1b4e6f9dd22a8c628ef9d976af675b86/1b4e6f9dd22a8c628ef9d976af675b86.urdf"),
            "table_v" : os.path.join(igibson.ig_dataset_path, "objects/breakfast_table/26670/26670.urdf"),
            "stove" : os.path.join(igibson.ig_dataset_path, "objects/stove/102019/102019.urdf"),
            "bowl" : os.path.join(igibson.ig_dataset_path, "objects/bowl/a1393437aac09108d627bfab5d10d45d/a1393437aac09108d627bfab5d10d45d.urdf"),
            "pan" : os.path.join(igibson.ig_dataset_path, "objects/frying_pan/36_0/36_0.urdf"),
            "sink" : os.path.join(igibson.ig_dataset_path, "objects/sink/kitchen_sink/kitchen_sink.urdf"),
            "fridge": os.path.join(igibson.ig_dataset_path, "objects/fridge/11712/11712.urdf")
        }

        name2scale_map = {
            "counter": np.array([1.04, 0.97, 0.95]),
            "table_h": np.array([1.2, 1.2, 1.3]),
            "table_v": np.array([1.2, 1.2, 1.3]),
            "stove": np.array([0.88, 1.1, 0.95]),
            "bowl": np.array([0.8, 0.8, 1]),
            "pan": np.array([1, 1, 1]),
            "sink": np.array([1.2, 1.25, 1.25]),
            "fridge": np.array([1.5, 1.2, 1.2]),
        }

        name2shift_map = {
            "counter": (0, 0, 0),
            "table_v": (0.5, 0, 0),
            "table_h": (0, 0.5, 0),
            "stove": (0, 0, 0),
            "bowl": (0, 0, 1.2),
            "pan": (0, 0, 1.24),
            "sink": (0, 0, 0.1),
            "fridge": (0, 0, 0.2),
        }

        shift_l = 0.1
        mapping = {
            (0, 0, 3.1415926): (0, -shift_l),
            (0, 0, 0): (0, shift_l),
            (0, 0, 1.5707): (-shift_l, 0),
            (0, 0, -1.5707): (shift_l, 0),
        }

        objs = []
        for name, x, y in obj_x_y:
            obj = URDFObject(name2path[name], scale=name2scale_map[name]/1.15, model_path="/".join(name2path[name].split("/")[:-1]))
            self.env.simulator.import_object(obj)

            orn = orientation_map[(name, x, y)]
            shift = name2shift_map[name]
            if name == "counter":
                x_shift, y_shift = mapping[orn]
                shift = (x_shift, y_shift, 0)
            # pos = [x+shift[0]+0.5, y+shift[1]+0.5, 0+shift[2]]
            pos = [x+shift[0]-4.5, y+shift[1]-4.5, 0+shift[2]]
            self.env.set_pos_orn_with_z_offset(obj, tuple(pos), orn) 

            if name not in ("bowl", "pan"):
                objs.append(obj)
            else:
                self.bowlpans.append((obj, pos))
        try:
            for obj in objs:
                p.changeDynamics(obj.get_body_ids()[0],-1,mass=500)
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
        }
        return_str = ""
        for x in range(self.HEIGHT):
            for y in range(self.WIDTH):
                if grid[x][y] == "X":
                    continue
                elif grid[x][y] == "T":
                    if y+1 < self.WIDTH and grid[x][y+1] == "T":
                        grid[x][y+1] = "X"
                        return_str += "{} {} {}\n".format("table_h", x, y)
                    else:
                        grid[x+1][y] = "X"
                        return_str += "{} {} {}\n".format("table_v", x, y)
                else:
                    name = abbr2name[grid[x][y]]
                    if name == "bowl":
                        #pass
                        return_str += "{} {} {}\n".format("counter", x, y)
                    if name == "pan":
                        return_str += "{} {} {}\n".format("stove", x, y)
                    return_str += "{} {} {}\n".format(name, x, y)
        return return_str

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
        }

        return name2abbr_map[name]

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
        orientation = (0,0,0)
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