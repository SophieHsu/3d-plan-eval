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

# HDR files for PBR rendering
from igibson.simulator_vr import SimulatorVR

from kitchen import Kitchen

c_pos = [0, 0, 1.5]

def top_camera_view():
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-90, cameraTargetPosition=[4, 4, 10.0])
     
def follow_robot_view(robot):
    x, y, z = robot.get_position()
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=-90, cameraPitch=-30, cameraTargetPosition=[x, y, 1.5])

def follow_robot_view_top(robot):
    x, y, z = robot.get_position()
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


def main():
    config_file = "igibson/configs/fetch_motion_planning_3d_lsi.yaml"
    kitchen_layout = r"C:\Users\icaro\3d_lsi_2\kitchen_layouts_grid_text\kitchen1_alt.txt"
    # Simple test:
    # robot_x, robot_y = 0, 0
    # robot_end = (7,4)
    # human_x, human_y = 3, 2
    # human_end = (5, 2)
    
    # Failing case:
    robot_x, robot_y = 5.5, 3.5
    robot_end = (7.5,3.5)
    human_x, human_y = 2.5, 2.5
    human_end = (6.5,3.5)

    # robot_x, robot_y = 5.5, 3.5
    # robot_end = (7.5, 6.5)
    # human_x, human_y = 2.5, 2.5
    # human_end = (0.0, 0.0)

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
    env.set_pos_orn_with_z_offset(human, [human_x, human_y, 0], [0, 0, 0])
    env.set_pos_orn_with_z_offset(robot, [robot_x, robot_y, 0], [0, 0, 0])
    
    a_star_planner = AStarPlanner(env)
    motion_controller = MotionControllerHuman()
    human_wrapper = HumanWrapper(human, robot, a_star_planner, motion_controller, occupancy_grid)

    motion_controller_robot = MotionControllerRobot(robot, a_star_planner, occupancy_grid)
    # top_camera_view()
    while(True):
        follow_robot_view_top(human)
        # actionStep = nav_env.simulator.gen_vr_robot_action()
        # human.apply_action(actionStep)
        # prop = human.get_proprioception()
        human_wrapper.step(human_end, 1.57)
        motion_controller_robot.step(robot_end, 1.57)
        env.simulator.step()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
