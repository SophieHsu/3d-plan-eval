from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.robots.behavior_robot import BehaviorRobot
import logging
from igibson.utils.utils import quatToXYZW, parse_config
from transforms3d.euler import euler2quat
import pybullet as p

from a_star_planner import AStarPlanner
from motion_controller import MotionController

# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR

from kitchen import Kitchen

def top_camera_view():
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-90, cameraTargetPosition=[4, 4, 10.0])
     
def follow_robot_view(robot):
    x, y, z = robot.get_position()
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=-90, cameraPitch=-30, cameraTargetPosition=[x, y, 1.5])

def main():
    config_file = "igibson/configs/fetch_motion_planning_3d_lsi.yaml"
    kitchen_layout = r"C:\Users\icaro\3d_lsi_2\kitchen_layouts_grid_text\kitchen1_alt.txt"
    robot_x, robot_y = 0.5, 0.5
    human_x, human_y = 3, 3

    env = iGibsonEnv(
        config_file=config_file, mode="headless", action_timestep=1.0 / 30.0, physics_timestep=1.0 / 120.0, use_pb_gui=True
    )

    kitchen = Kitchen(env)
    kitchen.setup(kitchen_layout)
    _, _, occupancy_grid = kitchen.read_from_grid_text(kitchen_layout)

    robot = env.robots[0]
    robot.tuck()
    # config = parse_config(config_file)
    # human = BehaviorRobot(**config["human"])
    # env.simulator.import_object(human)
    # # nav_env.simulator.switch_main_vr_robot(human)
    # env.set_pos_orn_with_z_offset(human, [human_x-4.5, human_y-4.5, 0], [0, 0, 0])
    env.set_pos_orn_with_z_offset(env.robots[0], [robot_x, robot_y, 0], [0, 0, 90])

    end = (5, 3)
    aStarPlanner = AStarPlanner(env, occupancy_grid)
    motionController = MotionController(robot, aStarPlanner)
    while(True):
        follow_robot_view(robot)
        # actionStep = nav_env.simulator.gen_vr_robot_action()
        # human.apply_action(actionStep)
        motionController.step(end)
        env.simulator.step()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
