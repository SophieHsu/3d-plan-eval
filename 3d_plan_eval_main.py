import argparse
import functools
import logging
import random
import sys
import time

from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.utils.utils import parse_config

from lsi_3d.agents.fixed_policy_human_agent import FixedPolicyAgent, SteakFixedPolicyHumanAgent
from lsi_3d.agents.hl_mdp_planning_agent import HlMdpPlanningAgent
from lsi_3d.agents.hl_qmdp_agent import HlQmdpPlanningAgent
from lsi_3d.agents.human_agent import HumanAgent
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.agents.vision_limit_human import VisionLimitHumanAgent
from lsi_3d.agents.vision_limit_robot import VisionLimitRobotAgent
from lsi_3d.config.reader import read_in_lsi_config
from lsi_3d.environment.kitchen import Kitchen
from lsi_3d.environment.lsi_env import LsiEnv
from lsi_3d.environment.tracking_env import TrackingEnv
from lsi_3d.environment.vision_limit_env import VisionLimitEnv
from lsi_3d.mdp.lsi_mdp import LsiMdp
from lsi_3d.motion_controllers.motion_controller_human import MotionControllerHuman
from lsi_3d.planners.a_star_planner import AStarPlanner
from lsi_3d.planners.high_level_mdp import HighLevelMdpPlanner
from lsi_3d.planners.hl_human_aware_mdp import HLHumanAwareMDPPlanner
from lsi_3d.planners.hl_human_planner import HLHumanPlanner
from lsi_3d.planners.hl_qmdp_planner import HumanSubtaskQMDPPlanner
from lsi_3d.planners.mid_level_motion import AStarMotionPlanner


class Runner:
    _TIME_LIMIT_FAILURE = 1000000
    _IGIBSON_ACTION_TIMESTEP = 1. / 30.
    _IGIBSON_PHYSICS_TIMESTEP = 1. / 120.

    def __init__(self):
        self._robot_agent = None
        self._human_agent = None
        self._env = None
        self._igibson_env = None
        self._kitchen = None

    def _init_logfile(self, start_locations):
        with open('lsi_3d/logs{}_log.txt'.format(self._kitchen.kitchen_name), 'w') as fh:
            fh.write('Start Locations (robot, human): {}\n'.format(start_locations))

            grid_str = functools.reduce(
                lambda acc, row: acc + '{}\n'.format(functools.reduce(
                    lambda row_acc, element: row_acc + '{} '.format(element),
                    row,
                    ''
                )),
                self._kitchen.grid,
                ''
            )
            fh.write('Kitchen Layout:\n{}'.format(grid_str))

        return True

    def _set_start_locations(self, exp_config):
        # TODO consolidate config files

        if ARGS.kitchen != 'none':  # if using script select random start
            self._kitchen.read_from_grid_text(ARGS.kitchen)
            open_squares = self._kitchen.get_empty_squares()
            robot_start = random.choice(open_squares)
            open_squares.remove(robot_start)
            human_start = random.choice(open_squares)
        else:
            self._kitchen.read_from_grid_text(exp_config["layout"])
            robot_start = (exp_config["robot_start_x"], exp_config["robot_start_y"])
            human_start = (exp_config["human_start_x"], exp_config["human_start_y"])

        r_x, r_y = robot_start
        self._igibson_env.set_pos_orn_with_z_offset(self._igibson_env.robots[0], [r_x - 4.5, r_y - 4.5, 0], [0, 0, 0])
        return robot_start, human_start

    def _setup(self):
        exp_config, map_config = read_in_lsi_config('steak.tml')

        self._igibson_env = iGibsonEnv(
            config_file=exp_config['ig_config_file'],
            mode=ARGS.mode,
            action_timestep=self._IGIBSON_ACTION_TIMESTEP,
            physics_timestep=self._IGIBSON_PHYSICS_TIMESTEP,
            use_pb_gui=True
        )

        self._kitchen = Kitchen(self._igibson_env, exp_config['max_in_pan'])
        self._igibson_env.simulator.scene.floor_plane_rgba = [.5, .5, .5, 1]

        robot_start, human_start = self._set_start_locations(exp_config)

        if ARGS.kitchen != 'none':
            self._kitchen.setup(ARGS.kitchen)
        else:
            self._kitchen.setup(exp_config["layout"], exp_config["order_list"])

        order_list = exp_config['order_list']

        config = parse_config(exp_config['ig_config_file'])
        human_bot = BehaviorRobot(**config["human"])
        human_vr = True if ARGS.mode == 'vr' else False

        r_x, r_y = robot_start
        h_x, h_y = human_start
        start_locations = ((r_x, r_y, 'S'), (h_x, h_y, 'S'))
        mdp = LsiMdp.from_config(start_locations, exp_config, self._kitchen.grid)
        hlp = HighLevelMdpPlanner(mdp)
        hlp.compute_mdp_policy(order_list)
        self._init_logfile(start_locations)

        human = iGibsonAgent(human_bot, human_start, 'S', "human")

        robot = iGibsonAgent(self._igibson_env.robots[0], robot_start, 'S', "robot")

        tracking_env = TrackingEnv(self._igibson_env, self._kitchen, robot, human_bot)
        env = LsiEnv(mdp, self._igibson_env, tracking_env, human, robot, self._kitchen)
        self._igibson_env.simulator.import_object(human_bot)
        self._igibson_env.set_pos_orn_with_z_offset(
            human_bot, [human_start[0], human_start[1], 0.6], [0, 0, 0])
        a_star_planner = AStarPlanner(self._igibson_env)
        motion_controller = MotionControllerHuman()
        human_agent = HumanAgent(human_bot, a_star_planner, motion_controller,
                                 self._kitchen.grid, hlp, env, tracking_env, human_vr)

        mlp = AStarMotionPlanner(self._kitchen)

        log_dict = {'i': 0, 'event_start_time': time.time()}
        log_dict['log_id'] = str(random.randint(0, 100000))
        log_dict[log_dict['i']] = {}
        log_dict[log_dict['i']]['low_level_logs'] = []
        log_dict['layout'] = self._kitchen.grid

        env = VisionLimitEnv(mdp, self._igibson_env, tracking_env, human, robot, self._kitchen, log_dict=log_dict)
        human_agent = VisionLimitHumanAgent(human_bot, a_star_planner, motion_controller,
                                            self._kitchen.grid, hlp, env, tracking_env, human_vr)
        robot_hlp = HumanSubtaskQMDPPlanner(mdp, mlp)
        robot_hlp.compute_mdp(filename='hi')

        human_sim_agent = SteakFixedPolicyHumanAgent(env, human_agent)
        robot_agent = VisionLimitRobotAgent(robot_hlp, mlp, human_sim_agent, env, robot, log_dict=log_dict)

        # TODO: Get rid of 4.5 offset
        h_x, h_y = human_start
        self._igibson_env.set_pos_orn_with_z_offset(self._igibson_env.robots[1],
                                                    [h_x - 4.5, h_y - 4.5, 0.8],
                                                    [0, 0, 0])

        self._robot_agent = robot_agent
        self._human_agent = human_agent
        self._env = env

    def _check_completion(self, start_time):
        if not self._env.world_state.orders:
            with open('lsi_3d/logs/{}_log.txt'.format(self._kitchen.kitchen_name), 'a') as fh:
                fh.write('success')
            print('orders completed')
            return True

        elapsed = time.time() - start_time
        if elapsed > self._TIME_LIMIT_FAILURE:
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
    Runner().run()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', '-m', type=str, dest='mode', action='store',
                        choices=['headless', 'headless_tensor', 'gui_non_interactive', 'gui_interactive', 'vr'],
                        default='headless', help='Specifies the mode for simulation (default: gui_interactive)')
    parser.add_argument('--kitchen', '-k', type=str, dest='kitchen', action='store',
                        default='none', help='Filepath of the kitchen layout')
    parser.add_argument('--config', '-c', type=str, dest='config', action='store',
                        default='steak.tml', help='Name of the config file')
    parser.add_argument('--practice', '-p', action='store_true', dest='practice',
                        help='Flag indicating whether it is a practice session')

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
