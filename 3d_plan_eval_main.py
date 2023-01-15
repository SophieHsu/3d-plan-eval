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

    # human_start = (human_x, human_y) = (agent_configs[0]['start_x'], agent_configs[0]['start_y'])
    # robot_start = (robot_x, robot_y) = (agent_configs[1]['start_x'], agent_configs[1]['start_y'])
    exp_config, map_config = configs
    robot_start = exp_config["robot_start_x"], exp_config["robot_start_y"]

    mdp = LsiMdp.from_config(map_config, exp_config, kitchen.grid)

    order_list = exp_config['order_list']
    recalc_res = exp_config['recalculation_resolution']
    
    mlp = AStarMotionPlanner(kitchen.grid)
    hhlp = HLGreedyHumanPlanner(mdp, mlp)
    #hhlp = HLHumanPlanner(mdp, mlp)
    #hlp = HLHumanAwareMDPPlanner(mdp, hhlp)
    hlp = HighLevelMdpPlanner(mdp)
    hlp.compute_mdp_policy(order_list)
    robot_agent = HlMdpPlanningAgent(hlp, mlp)
    human_agent = FixedPolicyAgent(hlp,mlp)

    # nav_env = iGibsonEnv(
    #     config_file=args.config, mode=args.mode, action_timestep=1.0 / 12, physics_timestep=1.0 / 12, use_pb_gui=True
    # )
    # print("**************loading objects***************")
    # human = BehaviorRobot()
    # nav_env.simulator.import_object(human)
    # nav_env.set_pos_orn_with_z_offset(human, [human_x-4.5, human_y-4.5, 0], [0, 0, 0])
    # nav_env.set_pos_orn_with_z_offset(nav_env.robots[0], [robot_x-4.5, robot_y-4.5, 0], [0, 0, 0])
    # bowlpans = load_objects(nav_env, obj_x_y, orientation_map, robot_x, robot_y, human)
    # # motion_planner = MotionPlanningWrapper(nav_env)
    # print("**************loading done***************")
    
    # human = iGibsonAgent(human, human_start, 'S', "human")
    robot = iGibsonAgent(igibson_env.robots[0], robot_start, 'S', "robot")

    lsi_env = LsiEnv(mdp, igibson_env, human, robot)

    main_loop(mdp, lsi_env, human, robot, robot_agent, human_agent, bowlpans, recalc_res,1)

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
            next_robot_hl_state, robot_goal, robot_action_object_pair = hl_robot_agent.action(env.world_state, env.robot_state, env.human_state)
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

def environment_setup():
    exp_config, map_config = read_in_lsi_config('two_agent_mdp.tml')
    configs = read_in_lsi_config('two_agent_mdp.tml')
    #parse_config(exp_config['ig_config_file'])
    
    #layout_config = toml.load('lsi_3d/lsi_config/layout/demo_layout.tml')
    #obj_x_y, orientation_map, grid = read_from_grid_text(map_config['layout'])

    

    igibson_env = iGibsonEnv(
        config_file=exp_config['ig_config_file'], mode=exp_config['ig_mode'], action_timestep=1.0 / 12, physics_timestep=1.0 / 12, use_pb_gui=True
    )

    kitchen = Kitchen(igibson_env)
    kitchen.setup(map_config["layout"])
    _, _, occupancy_grid = kitchen.read_from_grid_text(map_config["layout"])

    return igibson_env, kitchen, configs

def main():
    igibson_env, kitchen, configs = environment_setup()
    human = human_setup(igibson_env, kitchen, configs)
    robot_setup(igibson_env, kitchen, configs, human)
    main_loop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()