from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.robots.behavior_robot import BehaviorRobot
import logging
from igibson.utils.utils import quatToXYZW, parse_config
from transforms3d.euler import euler2quat

# from main_loop_wrapped_agent import MainLoopWrappedAgent

# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config

from kitchen import Kitchen
     
def main():
    config_file = "igibson/configs/fetch_motion_planning_3d_lsi.yaml"
    kitchen_layout = r"C:\Users\icaro\3d_lsi_2\kitchen_layouts_grid_text\kitchen1_alt.txt"
    robot_x, robot_y = 2, 5
    human_x, human_y = 3, 3

    print("**************loading objects***************")
    env = iGibsonEnv(
        config_file=config_file, mode="headless", action_timestep=1.0 / 30.0, physics_timestep=1.0 / 120.0, use_pb_gui=True
    )

    kitchen = Kitchen(env)
    kitchen.setup(kitchen_layout)
    config = parse_config(config_file)
    human = BehaviorRobot(**config["human"])
    env.simulator.import_object(human)
    # nav_env.simulator.switch_main_vr_robot(human)
    env.set_pos_orn_with_z_offset(human, [human_x-4.5, human_y-4.5, 0], [0, 0, 0])
    env.set_pos_orn_with_z_offset(env.robots[0], [robot_x-4.7, robot_y-4.5, 0], [0, 0, 0])
    
    # motion_planner = MotionPlanningWrapper(nav_env)
    print("**************loading done***************")
    
    # human = MainLoopWrappedAgent(human, "S", ["W", "F", "F", "F", "F", "S", "F", "E", "F", "N", "F"], "human")
    # robot = MainLoopWrappedAgent(nav_env.robots[0], "S", ["W", "F", "F", "F", "F", "S", "F", "E", "F", "N", "F"], "robot")

    while(True):
        # agent_move_one_step(env, human, human.get_action())

        # actionStep = nav_env.simulator.gen_vr_robot_action()
        # human.apply_action(actionStep)

        # robot.agent_move_one_step(nav_env)
        env.simulator.step()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
