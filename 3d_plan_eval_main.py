# Human
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.robots.behavior_robot import BehaviorRobot
import logging
from igibson.utils.utils import quatToXYZW, parse_config
from transforms3d.euler import euler2quat
import pybullet as p

from lsi_3d.planners.a_star_planner import AStarPlanner
from lsi_3d.motion_controllers.motion_controller_human import MotionControllerHuman
from lsi_3d.agents.human_agent import HumanAgent
from utils import real_to_grid_coord, grid_to_real_coord

from kitchen import Kitchen

# Robot
from tokenize import String

from igibson.envs.igibson_env import iGibsonEnv
import pybullet as p
from lsi_3d.agents.fixed_policy_human_agent import FixedPolicyAgent
from lsi_3d.agents.hl_mdp_planning_agent import HlMdpPlanningAgent
from lsi_3d.mdp.lsi_env import LsiEnv
from lsi_3d.planners.greedy_human_planner import HLGreedyHumanPlanner
from lsi_3d.planners.high_level_mdp import HighLevelMdpPlanner
from lsi_3d.planners.hl_human_aware_mdp import HLHumanAwareMDPPlanner
from lsi_3d.planners.hl_human_planner import HLHumanPlanner
from lsi_3d.planners.mid_level_motion import AStarMotionPlanner
from igibson.robots.behavior_robot import BehaviorRobot
import logging
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.config.reader import read_in_lsi_config
from lsi_3d.mdp.lsi_mdp import LsiMdp

def human_setup(igibson_env, kitchen, configs):
    exp_configs, map_configs = configs

    human = BehaviorRobot()
    igibson_env.simulator.import_object(human)
    igibson_env.set_pos_orn_with_z_offset(human, [exp_configs["human_start_x"], exp_configs["human_start_y"], 0], [0, 0, 0])
    a_star_planner = AStarPlanner(igibson_env)
    motion_controller = MotionControllerHuman()
    human_agent = HumanAgent(human, a_star_planner, motion_controller, kitchen.grid)

    return human_agent

def robot_setup(igibson_env, kitchen, configs):

    exp_config, map_config = configs
    robot_start = (exp_config["robot_start_x"], exp_config["robot_start_y"])
    human_start = (exp_config["human_start_x"], exp_config["human_start_y"]) 

    human = BehaviorRobot()
    igibson_env.simulator.import_object(human)
    igibson_env.set_pos_orn_with_z_offset(human, [exp_config["human_start_x"], exp_config["human_start_y"], 0], [0, 0, 0])

    mdp = LsiMdp.from_config(map_config, exp_config, kitchen.grid)

    order_list = exp_config['order_list']
    recalc_res = exp_config['recalculation_resolution']
    
    mlp = AStarMotionPlanner(kitchen.grid)
    # hhlp = HLGreedyHumanPlanner(mdp, mlp)
    hhlp = HLHumanPlanner(mdp, mlp)
    hlp = HLHumanAwareMDPPlanner(mdp, hhlp)
    #hlp = HighLevelMdpPlanner(mdp)
    hlp.compute_mdp_policy(order_list)

    # TODO: Get rid of 4.5 offset
    h_x,h_y = human_start
    r_x,r_y = robot_start
    igibson_env.set_pos_orn_with_z_offset(igibson_env.robots[1], [h_x-4.5, h_y-4.5, 0], [0, 0, 0])
    igibson_env.set_pos_orn_with_z_offset(igibson_env.robots[0], [r_x-4.5, r_y-4.5, 0], [0, 0, 0])

    human = iGibsonAgent(human, human_start, 'S', "human_sim")
    robot = iGibsonAgent(igibson_env.robots[0], robot_start, 'S', "robot")
    env = LsiEnv(mdp, igibson_env, human, robot, kitchen)

    human_agent = FixedPolicyAgent(hlp,mlp)
    robot_agent = HlMdpPlanningAgent(hlp, mlp, human_agent, env, human, robot)

    return robot_agent

def environment_setup():
    exp_config, map_config = read_in_lsi_config('two_agent_mdp.tml')
    configs = read_in_lsi_config('two_agent_mdp.tml')

    igibson_env = iGibsonEnv(
        config_file=exp_config['ig_config_file'], mode=exp_config['ig_mode'], action_timestep=1.0 / 15, physics_timestep=1.0 / 30, use_pb_gui=False
    )

    kitchen = Kitchen(igibson_env)
    kitchen.setup(map_config["layout"])
    print(map_config['layout'])
    _, _, occupancy_grid = kitchen.read_from_grid_text(map_config["layout"])

    return igibson_env, kitchen, configs

def main():
    igibson_env, kitchen, configs = environment_setup()
    robot_agent = robot_setup(igibson_env, kitchen, configs)
    # human_agent = human_setup(igibson_env, kitchen, configs)
    #human_agent.set_robot(igibson_env.robots[0])
    human_agent = None
    main_loop(igibson_env, robot_agent, human_agent)

def main_loop(igibson_env, robot_agent, human_agent):
    while True:
        # human_agent.step()
        robot_agent.step()
        igibson_env.simulator.step()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()