# Human
import argparse
import random
import time
import sys
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
from lsi_3d.agents.vision_limit_robot import VisionLimitRobotAgent
from lsi_3d.environment.vision_limit_env import VisionLimitEnv

from lsi_3d.planners.a_star_planner import AStarPlanner
from lsi_3d.motion_controllers.motion_controller_human import MotionControllerHuman
from lsi_3d.agents.human_agent import HumanAgent
from lsi_3d.planners.hl_qmdp_planner import HumanSubtaskQMDPPlanner
from lsi_3d.environment.tracking_env import TrackingEnv
from lsi_3d.planners.steak_human_subtask_qmdp_planner import SteakHumanSubtaskQMDPPlanner
from lsi_3d.planners.steak_knowledge_base_planner import SteakKnowledgeBasePlanner
from lsi_3d.planners.steak_mdp_planner import SteakMediumLevelMDPPlanner
from utils import real_to_grid_coord, grid_to_real_coord

from lsi_3d.environment.kitchen import Kitchen
from igibson import object_states

# Robot
from tokenize import String

from igibson.envs.igibson_env import iGibsonEnv
import pybullet as p
from lsi_3d.agents.fixed_policy_human_agent import FixedPolicyAgent, SteakFixedPolicyHumanAgent
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

TIME_LIMIT_FAILURE = 1000000  # 900 # 15 mins


def set_start_locations(args, map_config, exp_config, igibson_env, kitchen):
    # TODO consolidate config files

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
            grid_str += element + ' '
        grid_str += '\n'

    return grid_str


def setup_log(kitchen, start_locations):
    filename = 'lsi_3d/logs/' + kitchen.kitchen_name + '_log.txt'
    f = open(filename, 'w')
    f.write(f"Start Locations (robot, human): {start_locations}\n")
    f.write("Kitchen Layout:\n")
    f.write(print_grid(kitchen.grid))
    f.close()
    return True


SKIP_NUMBER = 30


class Runner:
    def _setup(self):
        exp_config, map_config = read_in_lsi_config('steak.tml')

        igibson_env = iGibsonEnv(
            config_file=exp_config['ig_config_file'],
            mode=ARGS.mode,
            action_timestep=1.0 / 30,
            physics_timestep=1.0 / 120,
            use_pb_gui=True)

        kitchen = Kitchen(igibson_env, exp_config['max_in_pan'])
        igibson_env.simulator.scene.floor_plane_rgba = [.5, .5, .5, 1]

        robot_start, human_start = set_start_locations(ARGS, map_config, exp_config, igibson_env, kitchen)

        if ARGS.kitchen != 'none':
            kitchen.setup(ARGS.kitchen)
        else:
            kitchen.setup(exp_config["layout"], exp_config["order_list"])

        order_list = exp_config['order_list']

        config = parse_config(exp_config['ig_config_file'])
        human_bot = BehaviorRobot(**config["human"])
        human_vr = True if ARGS.mode == 'vr' else False

        r_x, r_y = robot_start
        h_x, h_y = human_start
        start_locations = ((r_x, r_y, 'S'), (h_x, h_y, 'S'))
        mdp = LsiMdp.from_config(start_locations, exp_config, kitchen.grid)
        hlp = HighLevelMdpPlanner(mdp)
        hlp.compute_mdp_policy(order_list)
        setup_log(kitchen, start_locations)

        human = iGibsonAgent(human_bot, human_start, 'S', "human")

        robot = iGibsonAgent(igibson_env.robots[0], robot_start, 'S', "robot")

        tracking_env = TrackingEnv(igibson_env, kitchen, robot, human_bot)
        env = LsiEnv(mdp, igibson_env, tracking_env, human, robot, kitchen)
        igibson_env.simulator.import_object(human_bot)
        igibson_env.set_pos_orn_with_z_offset(
            human_bot, [human_start[0], human_start[1], 0.6], [0, 0, 0])
        a_star_planner = AStarPlanner(igibson_env)
        motion_controller = MotionControllerHuman()
        human_agent = HumanAgent(human_bot, a_star_planner, motion_controller,
                                 kitchen.grid, hlp, env, tracking_env, human_vr)

        mlp = AStarMotionPlanner(kitchen)

        planner_config = 3
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
            robot_hlp.compute_mdp(order_list)
            robot_hlp.post_mdp_setup()
            human_sim_agent = FixedPolicyAgent(robot_hlp, mlp, mdp.num_items_for_soup)
            robot_agent = HlQmdpPlanningAgent(robot_hlp, mlp, human_sim_agent, env,
                                              robot)
        elif planner_config == 3:

            log_dict = {'i': 0, 'event_start_time': time.time()}
            log_dict['log_id'] = str(random.randint(0, 100000))
            log_dict[log_dict['i']] = {}
            log_dict[log_dict['i']]['low_level_logs'] = []
            log_dict['layout'] = kitchen.grid

            env = VisionLimitEnv(mdp, igibson_env, tracking_env, human, robot, kitchen, log_dict=log_dict)
            human_agent = VisionLimitHumanAgent(human_bot, a_star_planner, motion_controller,
                                                kitchen.grid, hlp, env, tracking_env, human_vr)
            robot_hlp = HumanSubtaskQMDPPlanner(mdp, mlp)
            robot_hlp.compute_mdp(filename='hi')

            human_sim_agent = SteakFixedPolicyHumanAgent(env, human_agent)
            robot_agent = VisionLimitRobotAgent(robot_hlp, mlp, human_sim_agent, env,
                                                robot, log_dict=log_dict)

        # TODO: Get rid of 4.5 offset
        h_x, h_y = human_start
        igibson_env.set_pos_orn_with_z_offset(igibson_env.robots[1],
                                              [h_x - 4.5, h_y - 4.5, 0.8],
                                              [0, 0, 0])

        self._robot_agent = robot_agent
        self._human_agent = human_agent
        self._env = env
        self._igibson_env = igibson_env
        self._kitchen = kitchen

    def _check_completion(self, start_time):
        if not self._env.world_state.orders:
            with open('lsi_3d/logs/{}_log.txt'.format(self._kitchen.kitchen_name), 'a') as fh:
                fh.write('success')
            print('orders completed')
            return True

        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT_FAILURE:
            with open('lsi_3d/test_logs/{}_log.txt'.format(self._kitchen.kitchen_name), 'a') as fh:
                fh.write('failure by timeout')
            print('timed out')
            return True

        return False

    def _run_loop(self):
        start_time = time.time()
        ctr = 0

        input('press a key to begin game...')

        while self._check_completion(start_time):
            self._env.update_world()

            self._human_agent.step()

            if not ARGS.practice:
                self._robot_agent.step()

            self._kitchen.step(ctr)
            self._igibson_env.simulator.step()
            ctr += 1

    def run(self):
        self._setup()
        self._human_agent.set_robot(self._igibson_env.robots[0])
        self._run_loop()


def main():
    runner = Runner()
    runner.run()


def set_args(parser):
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
        type='str',
        action='store',
        default="steak.tml",
        dest='config',
        help="name of config file",
    )

    parser.add_argument(
        "--practice",
        "-p",
        default=False,
        help="name of config file",
    )


def get_args():
    parser = argparse.ArgumentParser()
    set_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    # get args
    ARGS = get_args()

    # set logging level
    logging.basicConfig(level=logging.DEBUG)

    # launch
    main()

    # exit without errors
    sys.exit(0)
