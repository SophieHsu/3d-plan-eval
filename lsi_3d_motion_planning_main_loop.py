import argparse
import os

import numpy as np

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
import pybullet as p
from time import time
from time import sleep
from igibson.objects.ycb_object import YCBObject
from igibson.objects.articulated_object import URDFObject, URDFObject
from render_layouts import HEIGHT, WIDTH, name2abbr
from igibson.robots.behavior_robot import BehaviorRobot
import math
from multiprocessing import Process
import logging
from igibson.utils.utils import quatToXYZW
from transforms3d.euler import euler2quat

TARGET_ORNS = {
    "S": 0,
    "E": 1.5707,
    "N": 3.1415926,
    "W": -1.5707,
    None: -1
}

DIRE2POSDIFF = {
    "E": (0, 1),
    "W": (0, -1),
    "S": (1, 0),
    "N": (-1, 0)
}

ONE_STEP = 0.02

def read_from_grid_text(filepath):
    obj_x_y = []
    grid = [["X"]*WIDTH for _ in range(HEIGHT)]
    sum_x, sum_y, count = 0, 0, 0 # for calculation of center mass (excluding table)
    orientation_map = dict()

    raw_str = grid2raw(filepath)
    print(raw_str.strip().split("\n"))
    for line in raw_str.strip().split("\n"):
        if len(line) == 0:
            break
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
            ori += ori_filter(grid, x+1, y) + ori_filter(grid, x-1, y) # upper and lower neighbor
            ori -= ori_filter(grid, x, y+1) + ori_filter(grid, x, y-1) # left and right neighbor
            orientation_map[(name, x, y)] = get_orientation(center_x, center_y, x, y, ori)
    return obj_x_y, orientation_map, grid

def ori_filter(grid, x, y):
    if not (x >= 0 and x < HEIGHT and y >= 0 and y < WIDTH):
        return 0

    if grid[x][y] == "X":
        return 0
    else:
        return 1

def name2shift(name):
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
    return name2shift_map[name]

def name2scale(name):
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
    return name2scale_map[name]

def get_counter_shift(orn):
    shift_l = 0.1
    mapping = {
        (0, 0, 3.1415926): (0, -shift_l),
        (0, 0, 0): (0, shift_l),
        (0, 0, 1.5707): (-shift_l, 0),
        (0, 0, -1.5707): (shift_l, 0),
    }
    return mapping[orn]

def get_orientation(center_x, center_y, x, y, ori):
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
                if name == "bowl":
                    #pass
                    return_str += "{} {} {}\n".format("counter", x, y)
                if name == "pan":
                    return_str += "{} {} {}\n".format("stove", x, y)
                return_str += "{} {} {}\n".format(name, x, y)
    return return_str


class MainLoopWrappedAgent:
    def __init__(self, robot, direction, path, name, target_x=None, target_y=None, target_direction=None):
        self.object = robot
        self.direction = direction
        self.path = path
        self.target_x = target_x
        self.target_y = target_y
        self.target_direction = target_direction
        self.action_index = 0
        self.name = name
    
    def update(self, target_x, target_y, direction, target_direction):
        self.target_x = target_x
        self.target_y = target_y
        self.direction = direction
        self.target_direction = target_direction

    def get_action(self):
        if self.action_index >= len(self.path):
            return None
        current_action = self.path[self.action_index]
        ready_for_next_action = False
        x, y, z = self.object.get_position()
        #print(self.name, current_action, self.get_current_orn_z(), self.target_direction, turn_distance(self.get_current_orn_z(), TARGET_ORNS[self.target_direction]))
        if self.target_x == None:
            ready_for_next_action = True
            x, y, z = self.object.get_position()
            self.target_x = x
            self.target_y = y
        elif current_action == "F" and forward_distance(x, y, self.target_x, self.target_y, self.direction) < ONE_STEP*1.5:
            self.action_index += 1
            ready_for_next_action = True
        elif current_action in "NWES" and turn_distance(self.get_current_orn_z(), TARGET_ORNS[self.target_direction]) < ONE_STEP*1.5:
            self.action_index += 1
            self.direction = current_action
            ready_for_next_action = True
        
        if self.action_index >= len(self.path):
            return None
        
        if ready_for_next_action:
            next_action = self.path[self.action_index]
            if next_action == "F":
                diff_x, diff_y = DIRE2POSDIFF[self.direction]
                self.target_x += diff_x
                self.target_y += diff_y
            elif next_action in "NWES":
                self.target_direction = next_action
        return self.path[self.action_index]
            
    def get_current_orn_z(self):
        x, y, z, w = self.object.get_orientation()
        x, y, z = quat2euler(x, y, z, w)
        return z

def turn_distance(cur_orn_z, target_orn_z):
    return abs(cur_orn_z - target_orn_z)

def forward_distance(cur_x, cur_y, target_x, target_y, direction):
    if direction in "NS":
        return abs(cur_x-target_x)
    else:
        return abs(cur_y-target_y)

def agent_move_one_step(env, agent, action):
    if action == None:
        if agent.name == "robot":
            action = np.zeros(env.action_space.shape)
            agent.object.apply_action(action)
        return
    if action in "NWES":
        agent_turn_one_step(env, agent, action)
    elif action == "F":
        agent_forward_one_step(env, agent)
    else:
        pass

def agent_forward_one_step(env, agent):
    if agent.name == "human":
        x,y,z = agent.object.get_position()
        if agent.direction == "N":
            agent.object.set_position_orientation([x-ONE_STEP,y,z], agent.object.get_orientation())
        elif agent.direction == "S":
            agent.object.set_position_orientation([x+ONE_STEP,y,z], agent.object.get_orientation())
        elif agent.direction == "E":
            agent.object.set_position_orientation([x,y+ONE_STEP,z], agent.object.get_orientation())
        elif agent.direction == "W":
            agent.object.set_position_orientation([x,y-ONE_STEP,z], agent.object.get_orientation())
    else:
        action = np.zeros(env.action_space.shape)
        action[0] = 0.15
        action[1] = 0
        start_x, start_y = agent.object.get_position()[:2]

        cur_x, cur_y = agent.object.get_position()[:2]
        distance_to_target = forward_distance(cur_x, cur_y, agent.target_x, agent.target_y, agent.direction)

        if distance_to_target < 0.2:
            action[0] /= 2
        elif distance_to_target < 0.1:
            action[0] /= 4
        elif distance_to_target < 0.05:
            action[0] /= 8
        agent.object.apply_action(action)


def agent_turn_one_step(env, agent, action):
    if agent.name == "human":
        x, y, z, w = agent.object.get_orientation()
        x, y, z = quat2euler(x, y, z, w)
        #print("turn z:", z, action)
        target_orn_z = TARGET_ORNS[agent.target_direction]
        
        pos = z - target_orn_z
        neg = target_orn_z - z
        if pos < 0:
            pos += 3.1415926*2
        elif neg < 0:
            neg += 3.1415926*2
        if pos < neg:
            z -= ONE_STEP
        else:
            z += ONE_STEP
        agent.object.set_position_orientation(agent.object.get_position(), quatToXYZW(euler2quat(x, y, z), "wxyz"))
    else:
        x, y, z, w = agent.object.get_orientation()
        x, y, z = quat2euler(x, y, z, w)
        cur_orn_z = z
        target_orn_z = TARGET_ORNS[agent.target_direction]
        action = np.zeros(env.action_space.shape)
        action[0] = 0
        if(cur_orn_z < target_orn_z):
            action[1] = -0.2
        else:
            action[1] = 0.2
        #print((cur_orn_z-target_orn_z) / (action[1]/action[1]), action[1], cur_orn_z, target_orn_z)
        if ((cur_orn_z-target_orn_z) / (action[1]/abs(action[1]))) > 4: # > 3.14
            action[1] = -action[1] 
        if abs(target_orn_z - cur_orn_z) < 0.5:
            action[1] /= 2
        elif abs(target_orn_z - cur_orn_z) < 0.2:
            action[1] /= 4
        agent.object.apply_action(action)

def quat2euler(x, y, z, w):
        """
        https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
        """

        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def main_loop(env, human, robot, bowlpans):
    print('Press enter to start...')
    input()
    while(True):
        agent_move_one_step(env, human, human.get_action())
        agent_move_one_step(env, robot, robot.get_action())
        #print(human.get_action(), robot.get_action())
        for obj, pos in bowlpans:
            obj.set_position(pos)

        env.simulator.step()


def robot_hand_up(env, bowlpans):
    action = np.zeros(env.action_space.shape[0])
    action[7] = 0.5
    action[8] = 0.5
    for i in range(70):
        for obj, pos in bowlpans:
            obj.set_position(pos)
        env.step(action)


def load_objects(env, obj_x_y, orientation_map, robot_x, robot_y, human):
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
    objs = []
    bowlpans = []
    for name, x, y in obj_x_y:
        obj = URDFObject(name2path[name], scale=name2scale(name)/1.15, model_path="/".join(name2path[name].split("/")[:-1]))
        env.simulator.import_object(obj)

        orn = orientation_map[(name, x, y)]
        shift = name2shift(name)
        if name == "counter":
            x_shift, y_shift = get_counter_shift(orn)
            shift = (x_shift, y_shift, 0)
        pos = [x+shift[0]-4.5, y+shift[1]-4.5, 0+shift[2]]
        env.set_pos_orn_with_z_offset(obj, tuple(pos), orn) 

        if name not in ("bowl", "pan"):
            objs.append(obj)
        else:
            bowlpans.append((obj, pos))
    print("import done, landing")
    try:
        print("landing objects")
        robot_hand_up(env, bowlpans)
        for obj in objs:
            p.changeDynamics(obj.get_body_ids()[0],-1,mass=500)
        
    except:
        pass
    print("loading done")
    return bowlpans
     
def run_example(args):
    obj_x_y, orientation_map, grid = read_from_grid_text("demo_layout.txt")
    
    robot_x, robot_y = 2, 5
    human_x, human_y = 20,20#1, 2
    nav_env = iGibsonEnv(
        config_file=args.config, mode=args.mode, action_timestep=1.0 / 12, physics_timestep=1.0 / 12, use_pb_gui=True
    ) 
    print("**************loading objects***************")
    human = BehaviorRobot()
    nav_env.simulator.import_object(human)
    nav_env.set_pos_orn_with_z_offset(human, [human_x-4.5, human_y-4.5, 0], [0, 0, 0])
    nav_env.set_pos_orn_with_z_offset(nav_env.robots[0], [robot_x-4.5, robot_y-4.5, 0], [0, 0, 0])
    bowlpans = load_objects(nav_env, obj_x_y, orientation_map, robot_x, robot_y, human)
    motion_planner = MotionPlanningWrapper(nav_env)
    print("**************loading done***************")
    
    human = MainLoopWrappedAgent(human, "S", ["W", "F", "F", "F", "F", "S", "F", "E", "F", "N", "F"], "human")
    robot = MainLoopWrappedAgent(nav_env.robots[0], "S", ["W", "F", "F", "F", "F", "S", "F", "E", "F", "N", "F"], "robot")
    
    main_loop(nav_env, human, robot, bowlpans)
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="igibson/configs/fetch_motion_planning_3d_lsi.yaml",#os.path.join(igibson.example_config_path, "fetch_motion_planning_3d_lsi.yaml"),
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "headless_tensor", "gui_non_interactive", "gui_interactive"],
        default="gui_interactive",
        help="which mode for simulation (default: gui_interactive)",
    )

    args = parser.parse_args()
    run_example(args)
