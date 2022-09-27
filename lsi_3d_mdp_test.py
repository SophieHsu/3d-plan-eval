import argparse
import os
from tokenize import String
import toml

import numpy as np
from agent import Agent, FixedMediumPlan

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
import pybullet as p
from time import time
from time import sleep
from igibson.objects.ycb_object import YCBObject
from igibson.objects.articulated_object import URDFObject, URDFObject
from lsi_3d.agents.fixed_policy_human_agent import FixedPolicyAgent
from lsi_3d.agents.hl_mdp_planning_agent import HlMdpPlanningAgent
from lsi_3d.mdp.lsi_env import LsiEnv
from lsi_3d.planners.high_level_mdp import HighLevelMdpPlanner
from lsi_3d.planners.mid_level_motion import AStarMotionPlanner
from lsi_3d.utils.enums import ExecutingState, MLAction
from render_layouts import HEIGHT, WIDTH, name2abbr
from igibson.robots.behavior_robot import BehaviorRobot
import math
from multiprocessing import Process
import logging
from igibson.utils.utils import quatToXYZW
from transforms3d.euler import euler2quat
from agent import FixedMediumPlan
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.config.reader import read_in_lsi_config
from lsi_3d.mdp.lsi_mdp import LsiMdp

# TARGET_ORNS = {
#     "S": 0,
#     "E": 1.5707,
#     "N": 3.1415926,
#     "W": -1.5707,
#     None: -1
# }

# DIRE2POSDIFF = {
#     "E": (0, 1),
#     "W": (0, -1),
#     "S": (1, 0),
#     "N": (-1, 0)
# }

# ONE_STEP = 0.02

def read_from_grid_text(grid_str):
    obj_x_y = []
    grid = [["X"]*WIDTH for _ in range(HEIGHT)]
    sum_x, sum_y, count = 0, 0, 0 # for calculation of center mass (excluding table)
    orientation_map = dict()

    raw_str = grid2raw(grid_str)
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
        elif name == 'pan':
            grid[x][y] = name2abbr(name)
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
        "pan": (0, 0, 1.2),
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

def grid2raw(grid_str):
    """
    Reads in a grid string and converts to a 2d array
    """
    grid = grid_str.strip().split("\n")
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

def get_human_sub_path(path, current_index):
    sub_path = path[current_index:(len(path)-1)]
    ml_path = []
    for loc in sub_path:
        ml_path.append(loc[1])

    return ml_path

def main_loop(mdp, env:LsiEnv, ig_human:iGibsonAgent, ig_robot:iGibsonAgent, hl_robot_agent:HlMdpPlanningAgent, hl_human_agent:FixedPolicyAgent, bowlpans):
    print('Press enter to start...')
    input()
    
    # start = ig_human.start + ig_robot.start
    # end = ((3,0),(5,3))
    #plan = run_astar_two_agent(grid, start, end)
    #human_plan = ['pickup_onion', 'drop_onion', 'pickup_onion', 'drop_onion', 'pickup_dish', 'deliver_soup']
    

    while True:
        if env.human_state.executing_state == ExecutingState.NO_ML_PATH:
            next_human_hl_state,human_plan, human_goal = hl_human_agent.action(env.human_state)
            human = FixedMediumPlan(human_plan)
            env.human_state.executing_state = ExecutingState.EXEC_SUB_PATH
            a_h = human.action()
            ig_human.prepare_for_next_action(a_h)

        if env.robot_state.executing_state == ExecutingState.NO_ML_PATH:
            next_robot_hl_state, plan = hl_robot_agent.prepare_optimal_path(env.robot_state)
            robot = FixedMediumPlan(plan)
            env.robot_state.executing_state = ExecutingState.CALC_SUB_PATH

        if env.robot_state.executing_state == ExecutingState.CALC_SUB_PATH:
            human_sub_path = get_human_sub_path(human_plan, human.i)
            next_robot_ml_goal, plan = hl_robot_agent.action(env.robot_state, human_sub_path, human_goal)
            env.robot_state.executing_state = ExecutingState.EXEC_SUB_PATH
            a_r = robot.action()
            ig_robot.prepare_for_next_action(a_r)

        if env.human_state.executing_state == ExecutingState.EXEC_SUB_PATH:
            if a_h == MLAction.STAY or a_h == MLAction.INTERACT:
                env.human_state.executing_state = ExecutingState.NO_ML_PATH
            else:
                ig_human.agent_move_one_step(env.nav_env, a_h)
                if ig_human.action_completed(a_h):
                    # human.action() gets next FNESW medium level action to take
                    a_h = human.action()
                    ig_human.prepare_for_next_action(a_h)

        if env.robot_state.executing_state == ExecutingState.EXEC_SUB_PATH:
            if a_r == MLAction.STAY or a_r == MLAction.INTERACT:
                env.robot_state.executing_state = ExecutingState.NO_ML_PATH
            else:
                ig_robot.agent_move_one_step(env.nav_env, a_r)

                if ig_robot.action_completed(a_r):
                    a_r = robot.action()
                    ig_robot.prepare_for_next_action(a_r)

        if human.i % 3 == 2:
            env.robot_state.executing_state = ExecutingState.CALC_SUB_PATH

        for obj, pos in bowlpans:
                obj.set_position(pos)
        env.nav_env.simulator.step()
        
        if env.robot_state.executing_state == ExecutingState.NO_ML_PATH:
            env.update_robot_world_state(next_human_hl_state)
        if env.human_state.executing_state == ExecutingState.NO_ML_PATH:
            env.update_human_world_state(next_robot_hl_state)





def robot_hand_up(env, bowlpans):
    # action = np.zeros(env.action_space.shape[0])
    action = [
                0.0,
                0.0,  # wheels
                0.385,  # trunk
                0.0,
                0.0,  # head
                1.1707963267948966,
                1.4707963267948965,
                -0.4,
                1.6707963267948966,
                0.0,
                1.5707963267948966,
                0.0,  # arm
                0.05,
                0.05,  # gripper
            ]
    env.robots[0].set_joint_positions(action)
    for i in range(70):
        for obj, pos in bowlpans:
            obj.set_position(pos)
        env.step()


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
    exp_config, alg_config, map_config, agent_configs = read_in_lsi_config('two_agent_mdp.tml')
    #layout_config = toml.load('lsi_3d/lsi_config/layout/demo_layout.tml')
    obj_x_y, orientation_map, grid = read_from_grid_text(map_config['layout'])

    human_start = (human_x, human_y) = (agent_configs[0]['start_x'], agent_configs[0]['start_y'])
    robot_start = (robot_x, robot_y) = (agent_configs[1]['start_x'], agent_configs[1]['start_y'])

    mdp = LsiMdp.from_config(map_config, agent_configs, exp_config, grid)

    order_list = exp_config['order_list']
    hlp = HighLevelMdpPlanner(mdp)
    mlp = AStarMotionPlanner(grid)
    hlp.compute_mdp_policy(order_list)
    robot_agent = HlMdpPlanningAgent(hlp, mlp)
    human_agent = FixedPolicyAgent(hlp,mlp)

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
    
    human = iGibsonAgent(human, human_start, MLAction.SOUTH, "human")
    robot = iGibsonAgent(nav_env.robots[0], robot_start, MLAction.SOUTH, "robot")

    lsi_env = LsiEnv(mdp, nav_env, human, robot)
    
    main_loop(mdp, lsi_env, human, robot, robot_agent, human_agent, bowlpans)


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
