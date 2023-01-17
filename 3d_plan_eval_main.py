# Human
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.robots.behavior_robot import BehaviorRobot
import logging
from igibson.utils.utils import quatToXYZW, parse_config
from transforms3d.euler import euler2quat
import pybullet as p

from lsi_3d.planners.a_star_planner import AStarPlanner
from lsi_3d.motion_controllers.motion_controller_robot import MotionControllerRobot
from lsi_3d.motion_controllers.motion_controller_human import MotionControllerHuman
from human_wrapper import HumanWrapper
from utils import real_to_grid_coord, grid_to_real_coord

from kitchen import Kitchen

# Robot
import argparse
import os
from tokenize import String
import toml

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

def follow_robot_view_top(robot):
    x, y, z = robot.get_position()
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[x, y, 2.0])

def human_setup(igibson_env, kitchen, configs):
    exp_configs, map_configs = configs
    #config_file = "igibson/configs/fetch_motion_planning_3d_lsi.yaml"
    #kitchen_layout = r"C:\Users\icaro\3d_lsi_2\kitchen_layouts_grid_text\kitchen1_alt.txt"


    #robot_x, robot_y = 5.5, 3.5
    #robot_end = (7.5,3.5)
    #human_x, human_y = 2.5, 2.5
    #human_end = (6.5,3.5)

    #env = iGibsonEnv(
    #    config_file=config_file, mode="headless", action_timestep=1.0 / 30.0, physics_timestep=1.0 / 120.0, use_pb_gui=True
    #)

    # kitchen = Kitchen(env)
    # kitchen.setup(kitchen_layout)
    # _, _, occupancy_grid = kitchen.read_from_grid_text(kitchen_layout)

    # robot = env.robots[0]
    #config = configs[4]
    # human = BehaviorRobot(**config["human"])
    human = BehaviorRobot()
    igibson_env.simulator.import_object(human)
    # nav_env.simulator.switch_main_vr_robot(human)
    #igibson_env.set_pos_orn_with_z_offset(human, [human_x, human_y, 0], [0, 0, 0])
    #igibson_env.set_pos_orn_with_z_offset(robot, [robot_x, robot_y, 0], [0, 0, 0])
    igibson_env.set_pos_orn_with_z_offset(human, [exp_configs["human_start_x"], exp_configs["human_start_y"], 0], [0, 0, 0])
    a_star_planner = AStarPlanner(igibson_env)
    motion_controller = MotionControllerHuman()
    human_wrapper = HumanWrapper(human, a_star_planner, motion_controller, kitchen.grid)

    return human_wrapper

# motion_controller_robot = MotionControllerRobot(robot, a_star_planner, occupancy_grid)

def robot_setup(igibson_env, kitchen, configs, human):
    # Human
    # a_star_planner = AStarPlanner(env)
    # motion_controller = MotionControllerHuman()
    # human_wrapper = HumanWrapper(human, robot, a_star_planner, motion_controller, occupancy_grid)

    # Robot
    # exp_config, alg_config, map_config, agent_configs = read_in_lsi_config('two_agent_mdp.tml')
    #layout_config = toml.load('lsi_3d/lsi_config/layout/demo_layout.tml')
    # obj_x_y, orientation_map, grid = read_from_grid_text(map_config['layout'])

    # = (agent_configs[0]['start_x'], agent_configs[0]['start_y'])
    # robot_start = (robot_x, robot_y) = (agent_configs[1]['start_x'], agent_configs[1]['start_y'])
    exp_config, map_config = configs
    robot_start = (exp_config["robot_start_x"], exp_config["robot_start_y"])
    human_start = (exp_config["human_start_x"], exp_config["human_start_y"]) 

    mdp = LsiMdp.from_config(map_config, exp_config, kitchen.grid)

    order_list = exp_config['order_list']
    recalc_res = exp_config['recalculation_resolution']
    
    mlp = AStarMotionPlanner(kitchen.grid)
    hhlp = HLGreedyHumanPlanner(mdp, mlp)
    #hhlp = HLHumanPlanner(mdp, mlp)
    #hlp = HLHumanAwareMDPPlanner(mdp, hhlp)
    hlp = HighLevelMdpPlanner(mdp)
    hlp.compute_mdp_policy(order_list)

    # nav_env = iGibsonEnv(
    #     config_file=args.config, mode=args.mode, action_timestep=1.0 / 12, physics_timestep=1.0 / 12, use_pb_gui=True
    # )
    # print("**************loading objects***************")
    # human = BehaviorRobot()
    # nav_env.simulator.import_object(human)
    h_x,h_y = human_start
    r_x,r_y = robot_start
    igibson_env.set_pos_orn_with_z_offset(igibson_env.robots[1], [h_x-4.5, h_y-4.5, 0], [0, 0, 0])
    igibson_env.set_pos_orn_with_z_offset(igibson_env.robots[0], [r_x-4.5, r_y-4.5, 0], [0, 0, 0])
    # bowlpans = load_objects(nav_env, obj_x_y, orientation_map, robot_x, robot_y, human)
    # # motion_planner = MotionPlanningWrapper(nav_env)
    # print("**************loading done***************")    
    human = iGibsonAgent(human.human, human_start, 'S', "human")
    robot = iGibsonAgent(igibson_env.robots[0], robot_start, 'S', "robot")



    env = LsiEnv(mdp, igibson_env, human, robot, kitchen)

    human_agent = FixedPolicyAgent(hlp,mlp)
    robot_agent = HlMdpPlanningAgent(hlp, mlp, human_agent, env, human, robot)

    #main_loop(mdp, lsi_env, human, robot, robot_agent, human_agent, bowlpans, recalc_res,1)
    #main_loop(mdp, lsi_env, human, robot, robot_agent, human_agent, recalc_res,1)
    # return mdp, lsi_env, human, robot, robot_agent, human_agent, recalc_res,1
    return robot_agent

def environment_setup():
    exp_config, map_config = read_in_lsi_config('two_agent_mdp.tml')
    configs = read_in_lsi_config('two_agent_mdp.tml')

    

    igibson_env = iGibsonEnv(
        config_file=exp_config['ig_config_file'], mode=exp_config['ig_mode'], action_timestep=1.0 / 12, physics_timestep=1.0 / 12, use_pb_gui=True
    )

    kitchen = Kitchen(igibson_env)
    kitchen.setup(map_config["layout"])
    print(map_config['layout'])
    _, _, occupancy_grid = kitchen.read_from_grid_text(map_config["layout"])

    return igibson_env, kitchen, configs

def main():
    igibson_env, kitchen, configs = environment_setup()
    human = human_setup(igibson_env, kitchen, configs)
    robot_agent = robot_setup(igibson_env, kitchen, configs, human)
    main_loop(robot_agent)

def main_loop(robot_agent):
    while True:
        # human.step(human_end, 1.57)
        robot_agent.step()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()