import os
import sys

import numpy as np

import igibson
from igibson.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import get_scene_path
import cv2
from numpngw import write_apng

from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

HEIGHT = 8 # x
WIDTH = 8 # y


def load_obj(renderer, name, x, y, orientation=(0, 0, 0)):
    path = name2path(name)
    scale = name2scale(name)
    shift = name2shift(name)

    position = [x+shift[0], y+shift[1], 0+shift[2]] 

    obj_files = os.listdir(path)
    obj_ids = []
    for obj_file in obj_files:
        if(obj_file[-3:] != "obj"):
            continue
        obj_file_path = path + "/" + obj_file
        obj_id = renderer.load_object(obj_file_path, scale=scale, transform_orn=orientation, transform_pos=position)
        obj_ids += obj_id
    render(renderer, obj_ids)

def render(renderer, obj_ids):
    for obj_id in obj_ids:
        renderer.add_instance(obj_id)

def get_image(renderer, img_name):
    frames = renderer.render(modes=("rgb","normal","3d"))
    frames = cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)
    frames *= 255
    cv2.imwrite("/project/lsi_3d/kitchen_layouts_rendered/"+img_name + ".png", frames)
    

def name2path(name):
    name2path_map = {
        "counter": os.path.join(igibson.ig_dataset_path, "objects/bottom_cabinet/46452/shape/visual"),
        "table_h": os.path.join(igibson.ig_dataset_path, "objects/breakfast_table/26670/shape/visual"),
        "table_v": os.path.join(igibson.ig_dataset_path, "objects/breakfast_table/26670/shape/visual"),
        "stove": os.path.join(igibson.ig_dataset_path, "objects/stove/102019/shape/visual"),
        "bowl": os.path.join(igibson.ig_dataset_path, "objects/bowl/a1393437aac09108d627bfab5d10d45d/shape/visual"),
        "pan": os.path.join(igibson.ig_dataset_path, "objects/frying_pan/36_0/shape/visual"),
        "sink": os.path.join(igibson.ig_dataset_path, "objects/sink/kitchen_sink/shape/visual"),
        "fridge": os.path.join(igibson.ig_dataset_path, "objects/fridge/11712/shape/visual"),
    }
    return name2path_map[name]

def name2scale(name):
    name2scale_map = {
        "counter": np.array([0.95, 0.95, 0.95]),
        "table_h": np.array([1.2, 1.2, 1.3]),
        "table_v": np.array([1.2, 1.2, 1.3]),
        "stove": np.array([0.9, 1, 0.95]),
        "bowl": np.array([0.8, 0.8, 1]),
        "pan": np.array([1, 1, 1]),
        "sink": np.array([1, 1, 1.5]),
        "fridge": np.array([1.45, 1.2, 1.2]),
    }
    return name2scale_map[name]

def name2shift(name):
    name2shift_map = {
        "counter": (0, 0, 0),
        "table_v": (0.5, 0, 0),
        "table_h": (0, 0.5, 0),
        "stove": (0, 0, 0),
        "bowl": (0, 0, 0.7),
        "pan": (0, 0, 0.7),
        "sink": (0, 0, 0.1),
        "fridge": (0, 0, 0.2),
    }
    return name2shift_map[name]

def name2abbr(name):
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

def pprint_grid(grid):
    grid_str = ""
    for x in range(HEIGHT):
        for y in range(WIDTH):
            grid_str += grid[x][y]
        grid_str += "\n"
    grid_str = grid_str[:-1]
    return grid_str

def ori_filter(grid, x, y):
    if not (x >= 0 and x < HEIGHT and y >= 0 and y < WIDTH):
        return 0

    if grid[x][y] == "X":
        return 0
    else:
        return 1

def get_orientation(center_x, center_y, x, y, ori):
    '''
    if ori > 0 then it's facing left/right, otherwise it's facing up/down

    orientation = (0,0,0) # left
    orientation = (0,0,360,1) # right
    orientation = (0,0,-1,1) # up
    orientation = (0,0,1) # down
    '''
    orientation = (0,0,0)
    if ori > 0:
        if center_y > y:
            orientation = (0,0,360,1) # right
        else:
            orientation = (0,0,0) # left
    else:
        if center_x > x:
            orientation = (0,0,1) 
        else:
            orientation = (0,0,-1,1)
    return orientation

def grid2raw(filepath):
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
    for x in range(HEIGHT):
        for y in range(WIDTH):
            if grid[x][y] == "X":
                continue
            elif grid[x][y] == "T":
                if y+1 < WIDTH and grid[x][y+1] == "T":
                    grid[x][y+1] = "X"
                    return_str += "{} {} {}\n".format("table_h", x, y)
                else:
                    grid[x+1][y] = "X"
                    return_str += "{} {} {}\n".format("table_v", x, y)
            else:
                name = abbr2name[grid[x][y]]
                return_str += "{} {} {}\n".format(name, x, y)
                if name == "bowl":
                    return_str += "{} {} {}\n".format("counter", x, y)
                if name == "pan":
                    return_str += "{} {} {}\n".format("stove", x, y)
    return return_str

def main():
    renderer_settings = MeshRendererSettings(
        #env_texture_filename=None,#os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr"),
        #env_texture_filename2=None,#os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr"),
        env_texture_filename3=os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr"),
    )
    read_option = "raw" # raw / grid
    layout_path = "/project/lsi_3d/kitchen_layouts/"
    if read_option == "grid":
        layout_path = "/project/lsi_3d/kitchen_layouts_grid_text/"

    for filename in os.listdir(layout_path):
        filename = filename.split(".")[0] # remove .txt

        # renderer & camera setting
        renderer = MeshRenderer(width=760, height=760, rendering_settings=renderer_settings)
        camera_pose = np.array([HEIGHT/2, WIDTH/2-0.5, 5.5])
        view_direction = np.array([HEIGHT/2, WIDTH/2-0.5, 0])
        renderer.set_camera(camera_pose, view_direction, [-3600, 10, 0])
        renderer.set_fov(90)

        obj_x_y = []
        grid = [["X"]*WIDTH for _ in range(HEIGHT)]
        sum_x, sum_y, count = 0, 0, 0 # for calculation of center mass (excluding table)
        orientation_map = dict()

        raw_str = open(layout_path+filename+".txt", "r").read() if read_option == "raw" else grid2raw(layout_path+filename+".txt")
        for line in raw_str.strip().split("\n"):
            name, x, y = line.split()
            x, y = int(x), int(y)
            obj_x_y.append((name, x, y))
            if grid[x][y] == "X" or grid[x][y] == "C":
                grid[x][y] = name2abbr(name)
            if name == "table_h":
                grid[x][y+1] = name2abbr(name)
            elif name == "table_v":
                grid[x+1][y] = name2abbr(name)
            else:
                sum_x += x
                sum_y += y
                count += 1

        if(count == 0):
            continue
        center_x, center_y = sum_x/count, sum_y/count

        for name, x, y in obj_x_y:
            if name == "table_h":
                orientation_map[(name, x, y)] = (0, 0, 1)
            elif name == "table_v":
                orientation_map[(name, x, y)] = (0, 0, 0)
            else:
                ori = 0 # if ori > 0 then it's facing left/right, otherwise it's facing up/down
                ori += ori_filter(grid, x+1, y) + ori_filter(grid, x-1, y) # upper and lower neighbor
                ori -= ori_filter(grid, x, y+1) + ori_filter(grid, x, y-1) # left and right neighbor
                orientation_map[(name, x, y)] = get_orientation(center_x, center_y, x, y, ori)

        for name, x, y in obj_x_y:
            load_obj(renderer, name, x, y, orientation_map[(name, x, y)])
        if read_option == "raw":
            open("/project/lsi_3d/kitchen_layouts_grid_text/"+filename+".txt", "w").write(pprint_grid(grid))
        else:
            open("/project/lsi_3d/kitchen_layouts/"+filename+".txt", "w").write(raw_str)
        #load_obj(renderer, "counter", 0, 0)
        #load_obj(renderer, "counter", 8, 10)
        #load_obj(renderer, "counter", 8, 0)
        #load_obj(renderer, "counter", 0, 10)
        get_image(renderer, filename)


if __name__ == "__main__":
    main()