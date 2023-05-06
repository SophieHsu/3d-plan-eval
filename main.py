# Human
import argparse
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
import logging
from igibson.utils.utils import quatToXYZW, parse_config
from transforms3d.euler import euler2quat
import pybullet as p

from lsi_3d.planners.a_star_planner import AStarPlanner
from lsi_3d.motion_controllers.motion_controller_human import MotionControllerHuman
from lsi_3d.agents.human_agent import HumanAgent
from lsi_3d.planners.hl_qmdp_planner import HumanSubtaskQMDPPlanner
from tracking_env import TrackingEnv
from utils import real_to_grid_coord, grid_to_real_coord

from kitchen import Kitchen
from igibson import object_states

# Robot
from tokenize import String

from igibson.envs.igibson_env import iGibsonEnv
import pybullet as p
from lsi_3d.agents.fixed_policy_human_agent import FixedPolicyAgent
from lsi_3d.agents.hl_mdp_planning_agent import HlMdpPlanningAgent
from lsi_3d.agents.hl_qmdp_agent import HlQmdpPlanningAgent
from lsi_3d.mdp.lsi_env import LsiEnv
from lsi_3d.planners.high_level_mdp import HighLevelMdpPlanner
from lsi_3d.planners.hl_human_aware_mdp import HLHumanAwareMDPPlanner
from lsi_3d.planners.hl_human_planner import HLHumanPlanner
from lsi_3d.planners.mid_level_motion import AStarMotionPlanner
from igibson.robots.behavior_robot import BehaviorRobot
import logging
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.config.reader import read_in_lsi_config
from lsi_3d.mdp.lsi_mdp import LsiMdp

from tracking_env import TrackingEnv

def setup(igibson_env, kitchen, configs, args):
    exp_config, map_config = configs
    order_list = exp_config['order_list']
    human_start = (exp_config["human_start_x"], exp_config["human_start_y"])
    robot_start = (exp_config["robot_start_x"], exp_config["robot_start_y"])
    config = parse_config(exp_config['ig_config_file'])
    human_bot = BehaviorRobot(**config["human"])
    human_vr = True if args.mode == 'vr' else False

    # human_sim = BehaviorRobot()
    # igibson_env.simulator.import_object(human_sim)
    # igibson_env.set_pos_orn_with_z_offset(human_sim, [exp_config["human_start_x"], exp_config["human_start_y"], 0], [0, 0, 0])

    mdp = LsiMdp.from_config(map_config, exp_config, kitchen.grid)
    hlp = HighLevelMdpPlanner(mdp)
    hlp.compute_mdp_policy(order_list)

    human = iGibsonAgent(human_bot, human_start, 'S', "human")

    robot = iGibsonAgent(igibson_env.robots[0], robot_start, 'S', "robot")
    tracking_env = TrackingEnv(igibson_env, kitchen, igibson_env.robots[0], human_bot)
    env = LsiEnv(mdp, igibson_env, tracking_env, human, robot, kitchen)
    
    #robot = iGibsonAgent(igibson_env.robots[0], robot_start, 'S', "robot")
    #env = LsiEnv(mdp, igibson_env, human, robot, kitchen)

    #######################################################################################
    igibson_env.simulator.import_object(human_bot)
    igibson_env.set_pos_orn_with_z_offset(
        human_bot, [human_start[0], human_start[1], 0.6], [0, 0, 0])
    a_star_planner = AStarPlanner(igibson_env)
    motion_controller = MotionControllerHuman()
    human_agent = HumanAgent(human_bot, a_star_planner, motion_controller,
                             kitchen.grid, hlp, env, tracking_env, human_vr)

    #######################################################################################

    mlp = AStarMotionPlanner(kitchen)
    # hhlp = HLGreedyHumanPlanner(mdp, mlp)

    planner_config = 1
    if planner_config == 1:
        hhlp = HLHumanPlanner(mdp, mlp, False)
        robot_hlp = HLHumanAwareMDPPlanner(mdp, hhlp)
        robot_hlp.compute_mdp_policy(order_list)

        human_sim_agent = FixedPolicyAgent(robot_hlp, mlp)
        robot_agent = HlMdpPlanningAgent(robot_hlp, mlp, human_sim_agent, env,
                                     robot)
    elif planner_config == 2:
        tracking_env = TrackingEnv(env, kitchen, robot, human)
        robot_hlp = HumanSubtaskQMDPPlanner(mdp, mlp, tracking_env)
        # mdp_planner = planners.HumanSubtaskQMDPPlanner.from_pickle_or_compute(scenario_1_mdp, NO_COUNTERS_PARAMS, force_compute_all=True)
        # mdp_planner = planners.HumanAwareMediumMDPPlanner.from_pickle_or_compute(scenario_1_mdp, NO_COUNTERS_PARAMS, hmlp, force_compute_all=True)

        # greedy = True is same as Human Aware MDP
        # ai_agent = agent.MediumQMdpPlanningAgent(mdp_planner, greedy=False, auto_unstuck=True)

        #hlp = HighLevelMdpPlanner(mdp)
        robot_hlp.compute_mdp(order_list)
        robot_hlp.post_mdp_setup()
        human_sim_agent = FixedPolicyAgent(robot_hlp, mlp)
        robot_agent = HlQmdpPlanningAgent(robot_hlp, mlp, human_sim_agent, env,
                                     robot)

    # TODO: Get rid of 4.5 offset
    h_x, h_y = human_start
    r_x, r_y = robot_start
    igibson_env.set_pos_orn_with_z_offset(igibson_env.robots[1],
                                          [h_x - 4.5, h_y - 4.5, 0.8], [0, 0, 0])
    igibson_env.set_pos_orn_with_z_offset(igibson_env.robots[0],
                                          [r_x - 4.5, r_y - 4.5, 0], [0, 0, 0])
    
    # igibson_env.set_pos_orn_with_z_offset(igibson_env.robots[0],
    #                                       [r_x, r_y, 0], [0, 0, 0])

    # human_sim = iGibsonAgent(human_sim, human_start, 'S', "human_sim")

    # human_agent = HumanAgent(human_bot, a_star_planner, motion_controller,
                            #  kitchen.grid, hlp, env, igibson_env)

    return robot_agent, human_agent, human_bot

def environment_setup(args, headless=None):
    exp_config, map_config = read_in_lsi_config('two_agent_mdp.tml')
    configs = read_in_lsi_config('two_agent_mdp.tml')

    igibson_env = iGibsonEnv(
        config_file=exp_config['ig_config_file'],
        mode=args.mode,
        action_timestep=1.0 / 30,
        physics_timestep=1.0 / 120,  # 1.0 / 30,
        use_pb_gui=True)

    # if not headless:
    #     # Set a better viewing direction
    #     igibson_env.simulator.viewer.initial_pos = [-0.3, -0.3, 1.1]
    #     igibson_env.simulator.viewer.initial_view_direction = [0.7, 0.6, -0.4]
    #     igibson_env.simulator.viewer.reset_viewer()

    kitchen = Kitchen(igibson_env)
    kitchen.setup(map_config["layout"])
    # print(map_config['layout'])
    _, _, occupancy_grid = kitchen.read_from_grid_text(map_config["layout"])

    return igibson_env, kitchen, configs

c_pos = [0, 0, 1.5]

def top_camera_view():
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-90, cameraTargetPosition=[4, 4, 10.0])
     
def follow_entity_view(entity):
    x, y, z = entity.get_position()
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=-90, cameraPitch=-30, cameraTargetPosition=[x, y, 1.5])

def follow_entity_view_top(entity):
    x, y, z = entity.get_position()
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[x, y, 2.0])

# def control_camera():
#     p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=0, cameraTargetPosition=c_pos)

#     keys = p.getKeyboardEvents()
#     #Keys to change camera
#     if keys.get(102):  #F
#         c_pos[1]+=1
#     if keys.get(99):   #C
#         c_pos[1]-=1
#     if keys.get(122):  #Z
#         c_pos[0]-=1
#     if keys.get(120):  #X
#         c_pos[0]+=1

def main(args):
    igibson_env, kitchen, configs = environment_setup(args)
    robot_agent, human_agent, human_bot = setup(igibson_env, kitchen, configs, args)
    human_agent.set_robot(igibson_env.robots[0])
    # human_agent.change_state()
    main_loop(igibson_env, robot_agent, human_agent, kitchen, human_bot)

def main_loop(igibson_env, robot_agent, human_agent, kitchen, human_bot):
    count = 0
    while True:
        follow_entity_view(human_bot)
        human_agent.step()
        robot_agent.step()
        kitchen.step(count)
        igibson_env.simulator.step()
        count += 1

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        "-m",
        choices=[
            "headless", "headless_tensor", "gui_non_interactive",
            "gui_interactive", "vr"
        ],
        default="gui_interactive",
        help="which mode for simulation (default: gui_interactive)",
    )

    args = parser.parse_args()
    main(args)


# def main():
#     config_file = "igibson/configs/fetch_motion_planning_3d_lsi.yaml"
#     kitchen_layout = "./kitchen_layouts_grid_text/kitchen.txt"
#     # Simple test:
#     robot_x, robot_y = 0, 0
#     # robot_end = (0, 0)
#     human_x, human_y = -1.5, 0.5
#     # human_end = (2, 1)
    
#     env = iGibsonEnv(
#         config_file=config_file, mode="headless", action_timestep=1.0 / 30.0, physics_timestep=1.0 / 120.0, use_pb_gui=True
#     )

#     kitchen = Kitchen(env)
#     kitchen.setup(kitchen_layout)
#     _, _, occupancy_grid = kitchen.read_from_grid_text(kitchen_layout)

#     robot = env.robots[0]
#     # robot.tuck()
#     config = parse_config(config_file)
#     human = BehaviorRobot(**config["human"])
#     env.simulator.import_object(human)
#     # nav_env.simulator.switch_main_vr_robot(human)
#     env.set_pos_orn_with_z_offset(human, [human_x, human_y, 0.6], [0, 0, 1.57])
#     env.set_pos_orn_with_z_offset(robot, [robot_x, robot_y, 0], [0, 0, 0])
    
#     a_star_planner = AStarPlanner(env)
#     motion_controller = MotionControllerHuman()
#     human_agent = HumanAgent(human, a_star_planner, motion_controller, occupancy_grid, None, None, env, vr=False)
#     human_agent.set_robot(robot)

#     tracking_env = TrackingEnv(env, kitchen, robot, human)

#     # motion_controller_robot = MotionControllerRobot(robot, a_star_planner, occupancy_grid)
#     # top_camera_view()
#     done = False
#     while(True):
#         # follow_entity_view(human)
#         # if done:
#         #     human_agent.drop([-1.5, 1.5, 1.2])
#         # else:
#         #     done = human_agent.pick([-1.5, 1.5, 1.2])
#         # motion_controller_robot.step(robot_end, 1.57)

#         human_agent.step()
#         # if tracking_env.obj_in_human_hand() is not None:
#         #     print(tracking_env.obj_in_human_hand().name)

#         # test = tracking_env.get_pan_status()
#         # key = list(test.keys())[0]
#         # print(tracking_env.is_pan_cooked(key))

#         # test = tracking_env.get_pan_status()
#         # key = list(test.keys())[0]
#         # print(len(test[key]))

#         # test = tracking_env.get_bowl_status()
#         # key = list(test.keys())[0]
#         # print(len(test[key]))

#         env.simulator.step()

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     main()
