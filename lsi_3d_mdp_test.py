import argparse
import os
from tokenize import String
import toml

import numpy as np
from agent import Agent, FixedMediumPlan, FixedMediumSubPlan

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
from lsi_3d.planners.greedy_human_planner import HLGreedyHumanPlanner
from lsi_3d.planners.high_level_mdp import HighLevelMdpPlanner
from lsi_3d.planners.hl_human_aware_mdp import HLHumanAwareMDPPlanner
from lsi_3d.planners.hl_human_planner import HLHumanPlanner
from lsi_3d.planners.mid_level_motion import AStarMotionPlanner
from lsi_3d.utils.enums import Mode
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
from lsi_3d.utils.enums import Mode
from lsi_3d.utils.functions import grid_transition
from igibson.external.pybullet_tools.utils import (
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    joints_from_names,
    set_joint_positions,
)

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

def get_human_sub_path(path, current_index, human_ml_state):
    sub_path = []
    if current_index == 1 and human_ml_state != path[current_index][0]:
        # in starting state
        return path
    # if human_ml_state == path[current_index][0]:
    #     sub_path = path[current_index:(len(path))]
    # elif human_ml_state == path[current_index-1][0] and current_index>=1:
    #     sub_path = path[current_index-1:(len(path))]
    # elif human_ml_state == path[current_index+1][0]:
    #     sub_path = path[current_index+1:(len(path))]

    for idx, state in enumerate(path):
        if state[0] == human_ml_state:
            return path[idx+1:len(path)]
    
    return sub_path

def main_loop(
    mdp, env:LsiEnv,
    ig_human:iGibsonAgent,
    ig_robot:iGibsonAgent,
    hl_robot_agent:HlMdpPlanningAgent,
    hl_human_agent:FixedPolicyAgent,
    bowlpans,
    recalc_res,
    avoid_radius):
    """_summary_

    Args:
        mdp (_type_): _description_
        env (LsiEnv): _description_
        ig_human (iGibsonAgent): _description_
        ig_robot (iGibsonAgent): _description_
        hl_robot_agent (HlMdpPlanningAgent): _description_
        hl_human_agent (FixedPolicyAgent): _description_
        bowlpans (_type_): _description_
        recalc_res (int): defines number of steps before the robot recalculates its path
    """

    # print('Press enter to start...')
    # input()
    
    while True:
        init_action = np.zeros(env.nav_env.action_space.shape)
        ig_robot.object.apply_action(init_action)

        if env.human_state.mode == Mode.CALC_HL_PATH:
            '''
            human gets high level action and plans path to it. when human finishes path
            it will re-enter this state
            '''
            next_human_hl_state, human_plan, human_goal, human_action_object_pair = hl_human_agent.action(env.world_state, env.human_state)
            #human_plan.append('I')
            human = FixedMediumPlan(human_plan)
            env.human_state.mode = Mode.EXEC_ML_PATH
            pos_h, a_h = human.action()
            ig_human.prepare_for_next_action(a_h)

        if env.robot_state.mode == Mode.CALC_HL_PATH:
            '''
            robot gets high level action and translates into mid-level path
            when robot completes this path, it returns to this state
            '''
            next_robot_hl_state, robot_goal, robot_action_object_pair = hl_robot_agent.action(env.world_state, env.robot_state, env.human_state.holding)
            optimal_plan = hl_robot_agent.optimal_motion_plan(env.robot_state, robot_goal)
            optimal_plan_goal = optimal_plan[len(optimal_plan)-1]
            env.robot_state.mode = Mode.CALC_SUB_PATH
            robot = FixedMediumSubPlan(optimal_plan, recalc_res)
            next_robot_goal = robot.next_goal()

        if env.robot_state.mode == Mode.CALC_SUB_PATH:
            '''
            robot executes mid-level path in stages to avoid collision. collisions occur
            in real-world scenarios because the agents are not operating in lock-step time.
            (i.e. the human may move faster/slower than the robot and vice-versa)

            the way this works is an optimal path to the mid-level goal is computed using a-star
            without considering human path, next intermediate goals along mid-level path are
            set and a path-avoidance a-star computes an optimal path which avoids the humans current
            mid-level path. This path begins executing

            Because the human's plan may change at any time, a parameter of the main loop is the 
            recalculation resolution, which defines how many steps the robot takes in the world
            before recalculating its sub path.
            '''
            env.update_joint_ml_state()
            if env.robot_state.ml_state == next_robot_goal:
                next_robot_goal = robot.next_goal()
                
            human_sub_path = get_human_sub_path(human_plan, (human.i), env.human_state.ml_state)
            plan = hl_robot_agent.avoidance_motion_plan((env.human_state.ml_state, env.robot_state.ml_state), next_robot_goal, human_sub_path, human_goal, radius=1)

            if plan == [] and optimal_plan_goal[0] == env.robot_state.ml_state:
                # could not find path to goal, so idle 1 step and then recalculate
                plan.append((next_robot_goal,'I'))
            elif plan == []:
                # if this is final subpath on optimal plan, the append interact at the end
                plan.append((env.robot_state.ml_state, 'D'))


            robot_plan = FixedMediumPlan(plan)
            env.robot_state.mode = Mode.EXEC_ML_PATH
            a_r = None
            # a_r = robot_plan.action()
            # ig_robot.prepare_for_next_action(a_r)

        elif env.robot_state.mode == Mode.EXEC_ML_PATH:

            ig_robot.agent_move_one_step(env.nav_env, a_r)
            # env.update_joint_ml_state()
            
            if ig_robot.action_completed(a_r) or a_r == 'D':
                # if :
                #     env.robot_state.executing_state = ExecutingState.CALC_SUB_PATH

                if robot_plan.i == len(robot_plan.plan) or robot_plan.i == 1: #recalc_res: #or a_r == MLAction.STAY:
                    env.robot_state.mode = Mode.CALC_SUB_PATH
                else:
                    pos_r, a_r = robot_plan.action()
                    env.update_joint_ml_state()
                    ig_robot.prepare_for_next_action(a_r)

                    reset_arm_position(ig_robot)

                    if a_r == 'D' and env.robot_state.mode != Mode.IDLE:
                        env.robot_state.mode = Mode.IDLE

                    if a_r == 'I': #and env.robot_state == robot_goal:
                        env.robot_state.mode = Mode.INTERACT

                    if env.human_state.mode == Mode.IDLE:
                        env.human_state.mode = Mode.EXEC_ML_PATH

            
        if env.human_state.mode == Mode.EXEC_ML_PATH:
            if human.i == len(human.plan) or a_h == 'I': #or a_h == MLAction.STAY:
                env.human_state.mode = Mode.INTERACT
            else:
                # if robot is in a goal state and humans next state is also this state,
                    # then idle until robot moves
                    # next_robot_goal == env.robot_state.ml_state and 
                if grid_transition(a_h, env.human_state.ml_state)[0:2] != env.robot_state.ml_state[0:2]:
                    ig_human.agent_move_one_step(env.nav_env, a_h)
                elif env.robot_state.mode == Mode.IDLE:
                    env.robot_state.mode = Mode.CALC_SUB_PATH

                    # if next step means human crashes into robot, add a delay to the plan
                    #delay_step = human_sub_path[0]
                    #delay_step = (delay_step[0], 'D')
                    #human_sub_path.insert(0,delay_step)

                if ig_human.action_completed(a_h):
                    # human.action() gets next FNESW medium level action to take
                    pos_h, a_h = human.action()
                    env.update_joint_ml_state()
                    ig_human.prepare_for_next_action(a_h)

                    if env.robot_state.mode == Mode.IDLE:
                        env.robot_state.mode = Mode.EXEC_ML_PATH

                    

                

        for obj, pos in bowlpans:
                obj.set_position(pos)
        env.nav_env.simulator.step()

        if env.robot_state.mode == Mode.INTERACT:
            env.update_robot_hl_state(next_robot_hl_state, robot_action_object_pair)
            env.robot_state.mode = Mode.CALC_HL_PATH
            #env.human_state.executing_state = ExecutingState.CALC_HL_PATH
        if env.human_state.mode == Mode.INTERACT:
            env.update_human_hl_state(next_human_hl_state, human_action_object_pair)
            #env.robot_state.executing_state = ExecutingState.CALC_HL_PATH
            env.human_state.mode = Mode.CALC_HL_PATH

def reset_arm_position(ig_robot):
    arm_joints_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ]

    arm_default_joint_positions = (
        -1.414019864768982,
        1.5178184935241699,
        0.8189625336474915,
        2.200358942909668,
        2.9631312579803466,
        -1.2862852996643066,
        0.0008453550418615341,
    )

    body_ids = ig_robot.object.get_body_ids()
    assert len(body_ids) == 1, "Fetch robot is expected to be single-body."
    robot_id = body_ids[0]
    arm_joint_ids = joints_from_names(robot_id, arm_joints_names)

    set_joint_positions(robot_id, arm_joint_ids, arm_default_joint_positions)


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
    recalc_res = exp_config['recalculation_resolution']
    
    mlp = AStarMotionPlanner(grid)
    hhlp = HLGreedyHumanPlanner(mdp, mlp)
    #hhlp = HLHumanPlanner(mdp, mlp)
    hlp = HLHumanAwareMDPPlanner(mdp, hhlp)
    #hlp = HighLevelMdpPlanner(mdp)
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
    
    human = iGibsonAgent(human, human_start, 'S', "human")
    robot = iGibsonAgent(nav_env.robots[0], robot_start, 'S', "robot")

    lsi_env = LsiEnv(mdp, nav_env, human, robot)
    
    main_loop(mdp, lsi_env, human, robot, robot_agent, human_agent, bowlpans, recalc_res,1)

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
