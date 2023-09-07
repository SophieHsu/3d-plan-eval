# Human
import argparse
import random
import time
from igibson.envs.igibson_env import iGibsonEnv
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.stadium_scene import StadiumScene
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
import logging
from igibson.utils.utils import quatToXYZW, parse_config
from transforms3d.euler import euler2quat
import pybullet as p
from lsi_3d.agents.vision_limit_human import VisionLimitHumanAgent
from lsi_3d.environment.vision_limit_env import VisionLimitEnv

from lsi_3d.planners.a_star_planner import AStarPlanner
from lsi_3d.motion_controllers.motion_controller_human import MotionControllerHuman
from lsi_3d.agents.human_agent import HumanAgent
from lsi_3d.planners.hl_qmdp_planner import HumanSubtaskQMDPPlanner
from lsi_3d.environment.tracking_env import TrackingEnv
from utils import real_to_grid_coord, grid_to_real_coord

from lsi_3d.environment.kitchen import Kitchen
from igibson import object_states

# Robot
from tokenize import String

from igibson.envs.igibson_env import iGibsonEnv
import pybullet as p
from lsi_3d.agents.fixed_policy_human_agent import FixedPolicyAgent
from lsi_3d.agents.hl_mdp_planning_agent import HlMdpPlanningAgent
from lsi_3d.agents.hl_qmdp_agent import HlQmdpPlanningAgent
from lsi_3d.environment.lsi_env import LsiEnv
from lsi_3d.planners.high_level_mdp import HighLevelMdpPlanner
from lsi_3d.planners.hl_human_aware_mdp import HLHumanAwareMDPPlanner
from lsi_3d.planners.hl_human_planner import HLHumanPlanner
from lsi_3d.planners.mid_level_motion import AStarMotionPlanner
from igibson.robots.behavior_robot import BehaviorRobot
import logging
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.config.reader import read_in_lsi_config
from lsi_3d.mdp.lsi_mdp import LsiMdp

TIME_LIMIT_FAILURE = 900 # 15 mins

def set_start_locations(args, map_config, exp_config, igibson_env, kitchen):
    #TODO consolidate config files
    
    
    # if using script select random start
    if args.kitchen != 'none':
        kitchen.read_from_grid_text(args.kitchen)
        # kitchen.read_from_grid_text(map_config["layout"])
        open_squares = kitchen.get_empty_squares()
        robot_start = random.choice(open_squares)

        open_squares.remove(robot_start)
        human_start = random.choice(open_squares)
    else:
        kitchen.read_from_grid_text(exp_config["layout"])
        robot_start = (exp_config["robot_start_x"], exp_config["robot_start_y"])
        human_start = (exp_config["human_start_x"], exp_config["human_start_y"])
        
    r_x, r_y = robot_start
    igibson_env.set_pos_orn_with_z_offset(igibson_env.robots[0],
                                          [r_x - 4.5, r_y - 4.5, 0], [0, 0, 0])
    return robot_start, human_start

def print_grid(array):
    grid_str = ""
    for row in array:
        for element in row:
            grid_str+= element + ' '
        grid_str += '\n'
    
    return grid_str

def setup_log(kitchen, start_locations):
    filename = 'lsi_3d/test_logs/' + kitchen.kitchen_name + '_log.txt'
    f = open(filename, 'w')
    f.write(f"Start Locations (robot, human): {start_locations}\n")
    f.write("Kitchen Layout:\n")
    f.write(print_grid(kitchen.grid))
    f.close()
    return True

def setup(args):

    if args.config != 'none':
        exp_config, map_config  = read_in_lsi_config(args.config)
    else:
        # exp_config, map_config = read_in_lsi_config('steak.tml')
        exp_config, map_config = read_in_lsi_config('two_agent_mdp.tml')

    igibson_env = iGibsonEnv(
        config_file=exp_config['ig_config_file'],
        mode=args.mode,
        # action_timestep=1.0 / 15,
        # physics_timestep=1.0 / 30,  #1.0 / 30,
        action_timestep=1.0 / 30,
        physics_timestep=1.0 / 120,  #1.0 / 30,
        use_pb_gui=True)

    # if not headless:
    #     # Set a better viewing direction
    #     igibson_env.simulator.viewer.initial_pos = [-0.3, -0.3, 1.1]
    #     igibson_env.simulator.viewer.initial_view_direction = [0.7, 0.6, -0.4]
    #     igibson_env.simulator.viewer.reset_viewer()

    # scene = EmptyScene()
    # igibson_env.simulator.import_scene(scene)

    kitchen = Kitchen(igibson_env)
    igibson_env.simulator.scene.floor_plane_rgba = [.5,.5,.5,1]

    robot_start, human_start = set_start_locations(args, map_config, exp_config, igibson_env, kitchen)

    if args.kitchen != 'none':
        kitchen.setup(args.kitchen)
    else:
        kitchen.setup(exp_config["layout"], exp_config["order_list"])

    print(exp_config['layout'])
    # _, _, occupancy_grid = kitchen.read_from_grid_text(map_config["layout"])

    order_list = exp_config['order_list']
    
    config = parse_config(exp_config['ig_config_file'])
    human_bot = BehaviorRobot(**config["human"])
    human_vr = True if args.mode == 'vr' else False

    # human_sim = BehaviorRobot()
    # igibson_env.simulator.import_object(human_sim)
    # igibson_env.set_pos_orn_with_z_offset(human_sim, [exp_config["human_start_x"], exp_config["human_start_y"], 0], [0, 0, 0])
    r_x,r_y = robot_start
    h_x,h_y = human_start
    start_locations = ((r_x,r_y,'S'), (h_x,h_y,'S'))
    mdp = LsiMdp.from_config(start_locations, exp_config, kitchen.grid)
    hlp = HighLevelMdpPlanner(mdp)
    hlp.compute_mdp_policy(order_list)
    setup_log(kitchen, start_locations)

    human = iGibsonAgent(human_bot, human_start, 'S', "human")

    robot = iGibsonAgent(igibson_env.robots[0], robot_start, 'S', "robot")

    tracking_env = TrackingEnv(igibson_env, kitchen, robot, human_bot)
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

    planner_config = 2
    if planner_config == 1:
        hhlp = HLHumanPlanner(mdp, mlp, False)
        robot_hlp = HLHumanAwareMDPPlanner(mdp, hhlp)
        robot_hlp.compute_mdp_policy(order_list)

        human_sim_agent = FixedPolicyAgent(robot_hlp, mlp, mdp.num_items_for_soup)
        robot_agent = HlMdpPlanningAgent(robot_hlp, mlp, human_sim_agent, env,
                                         robot)
    elif planner_config == 2:
        # tracking_env.step()
        robot_hlp = HumanSubtaskQMDPPlanner(mdp, mlp)
        # mdp_planner = planners.HumanSubtaskQMDPPlanner.from_pickle_or_compute(scenario_1_mdp, NO_COUNTERS_PARAMS, force_compute_all=True)
        # mdp_planner = planners.HumanAwareMediumMDPPlanner.from_pickle_or_compute(scenario_1_mdp, NO_COUNTERS_PARAMS, hmlp, force_compute_all=True)

        # greedy = True is same as Human Aware MDP
        # ai_agent = agent.MediumQMdpPlanningAgent(mdp_planner, greedy=False, auto_unstuck=True)

        #hlp = HighLevelMdpPlanner(mdp)
        robot_hlp.compute_mdp(order_list)
        robot_hlp.post_mdp_setup()
        human_sim_agent = FixedPolicyAgent(robot_hlp, mlp, mdp.num_items_for_soup)
        robot_agent = HlQmdpPlanningAgent(robot_hlp, mlp, human_sim_agent, env,
                                          robot)
    elif planner_config == 3:
        env = VisionLimitEnv(mdp, igibson_env, tracking_env, human, robot, kitchen)
        human_agent = VisionLimitHumanAgent(human_bot, a_star_planner, motion_controller,
                             kitchen.grid, hlp, env, tracking_env, human_vr)

        robot_hlp = HumanSubtaskQMDPPlanner(mdp, mlp)
        robot_hlp.compute_mdp(order_list)
        robot_hlp.post_mdp_setup()
        human_sim_agent = FixedPolicyAgent(robot_hlp, mlp, mdp.num_items_for_soup)
        robot_agent = HlQmdpPlanningAgent(robot_hlp, mlp, human_sim_agent, env,
                                          robot)

    # TODO: Get rid of 4.5 offset
    h_x, h_y = human_start
    igibson_env.set_pos_orn_with_z_offset(igibson_env.robots[1],
                                          [h_x - 4.5, h_y - 4.5, 0.8],
                                          [0, 0, 0])

    # human_sim = iGibsonAgent(human_sim, human_start, 'S', "human_sim")

    # human_agent = HumanAgent(human_bot, a_star_planner, motion_controller,
    #                          kitchen.grid, hlp, env, tracking_env)

    return robot_agent, human_agent, env, igibson_env, kitchen


def environment_setup(args, headless=None):
    exp_config, map_config = read_in_lsi_config('two_agent_mdp.tml')
    configs = read_in_lsi_config('two_agent_mdp.tml')

    igibson_env = iGibsonEnv(
        config_file=exp_config['ig_config_file'],
        mode=args.mode,
        # action_timestep=1.0 / 15,
        # physics_timestep=1.0 / 30,  #1.0 / 30,
        action_timestep=1.0 / 30,
        physics_timestep=1.0 / 120,  #1.0 / 30,
        use_pb_gui=True)

    # if not headless:
    #     # Set a better viewing direction
    #     igibson_env.simulator.viewer.initial_pos = [-0.3, -0.3, 1.1]
    #     igibson_env.simulator.viewer.initial_view_direction = [0.7, 0.6, -0.4]
    #     igibson_env.simulator.viewer.reset_viewer()

    kitchen = Kitchen(igibson_env)

    if args.kitchen != 'none':
        kitchen.setup(args.kitchen)
    else:
        kitchen.setup(map_config["layout"])

    print(map_config['layout'])
    _, _, occupancy_grid = kitchen.read_from_grid_text(map_config["layout"])

    return igibson_env, kitchen, configs


def main(args):
    # igibson_env, kitchen, configs = environment_setup(args)
    robot_agent, human_agent, lsi_env, igibson_env, kitchen = setup(args)
    human_agent.set_robot(igibson_env.robots[0])
    main_loop(igibson_env, robot_agent, human_agent, kitchen, lsi_env)

def check_completion(lsi_env, start_time, kitchen):
    if lsi_env.world_state.orders == []:
        print("Orders Completed")

        filename = 'lsi_3d/test_logs/' + kitchen.kitchen_name + '_log.txt'
        f = open(filename, 'a')
        f.write("success")
        f.close()
        return True
    
    

    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT_FAILURE:
        filename = 'lsi_3d/test_logs/' + kitchen.kitchen_name + '_log.txt'
        f = open(filename, 'a')
        f.write("failure by timeout")
        f.close()
        print('timed out')
        return True
    
    return False


def main_loop(igibson_env, robot_agent, human_agent, kitchen, env:LsiEnv):
    start_time = time.time()
    count = 0
    while True:
        env.update_world()
        human_agent.step()
        # robot_agent.step()
        kitchen.step(count)
        igibson_env.simulator.step()
        count += 1

        if check_completion(env, start_time, kitchen):
            break 

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
        default="headless",
        help="which mode for simulation (default: gui_interactive)",
    )

    parser.add_argument(
        "--kitchen",
        "-k",
        default="none",
        help="filepath of kitchen layout",
    )

    parser.add_argument(
        "--config",
        "-c",
        default="none",
        help="name of config file",
    )

    args = parser.parse_args()
    print(args.kitchen)
    main(args)

exit()