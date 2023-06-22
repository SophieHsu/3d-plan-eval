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

from tracking_env import TrackingEnv
import numpy as np
import time

from igibson.utils.grasp_planning_utils import get_grasp_poses_for_object_sticky

def main():
    config_file = "igibson/configs/fetch_motion_planning_3d_lsi.yaml"
    kitchen_layout = "./kitchen_layouts_grid_text/kitchen.txt"
    # Simple test:
    robot_x, robot_y = 0, 0
    # robot_end = (0, 0)
    human_x, human_y = -1.5, 0.5
    # human_end = (2, 1)
    
    env = iGibsonEnv(
        config_file=config_file, mode="headless", action_timestep=1.0 / 30.0, physics_timestep=1.0 / 120.0, use_pb_gui=True
    )

    kitchen = Kitchen(env)
    kitchen.setup(kitchen_layout)
    _, _, occupancy_grid = kitchen.read_from_grid_text(kitchen_layout)

    robot = env.robots[0]
    # robot.tuck()
    config = parse_config(config_file)
    human = BehaviorRobot(**config["human"])
    env.simulator.import_object(human)
    # nav_env.simulator.switch_main_vr_robot(human)
    env.set_pos_orn_with_z_offset(human, [human_x, human_y, 0.6], [0, 0, 1.57])
    env.set_pos_orn_with_z_offset(robot, [robot_x, robot_y, 0], [0, 0, 0])
    
    tracking_env = TrackingEnv(env, kitchen, robot, human)
    
    a_star_planner = AStarPlanner(env)
    motion_controller = MotionControllerHuman()
    human_agent = HumanAgent(human, a_star_planner, motion_controller, occupancy_grid, None, None, tracking_env, vr=False)
    human_agent.set_robot(robot)

    # motion_controller_robot = MotionControllerRobot(robot, a_star_planner, occupancy_grid)
    # top_camera_view()
    done = False
    # time.sleep(5)
    print(kitchen.onions[0])
    val = get_grasp_poses_for_object_sticky(env.robots[0], kitchen.onions[0])
    while(True):
        # follow_entity_view(human)
        # action = np.zeros((28,))
        # action[0] = 0.1

        # human.apply_action(action)

        # if tracking_env.obj_in_human_hand() is not None:
        #     print(tracking_env.obj_in_human_hand().name)
        
        # human_agent.step()
        # test = tracking_env.get_pan_status()
        # key = list(test.keys())[0]
        # print(tracking_env.is_pan_cooked(key))

        # print(tracking_env.test())

        # test = tracking_env.get_pan_status()
        # key = list(test.keys())[0]
        # print(len(test[key]))

        # test = tracking_env.get_bowl_status()
        # key = list(test.keys())[0]
        # print(len(test[key]))

        env.simulator.step()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()