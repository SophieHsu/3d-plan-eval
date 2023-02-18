import copy
import math
import numpy as np
from utils import quat2euler, real_to_grid_coord, grid_to_real_coord, normalize_radians
from agent import Agent
from lsi_3d.mdp.lsi_env import LsiEnv
from lsi_3d.mdp.hl_state import AgentState, WorldState
import time


class HumanAgent():

    def __init__(self,
                 human,
                 planner,
                 motion_controller,
                 occupancy_grid,
                 hlp,
                 lsi_env,
                 igibson_env,
                 vr=False,
                 insight_threshold=5):
        self.human = human
        self.robot = None
        self.planner = planner
        self.motion_controller = motion_controller
        self.occupancy_grid = copy.deepcopy(occupancy_grid)
        self.hlp = hlp
        self.lsi_env = lsi_env
        self.vision_range = math.pi
        self.igibson_env = igibson_env
        self.vr = vr
        self.insight_threshold = insight_threshold

    def set_robot(self, robot):
        self.robot = robot

    def step(self):
        if self.vr:
            self.igibson_env.simulator.switch_main_vr_robot(self.human)
            actionStep = self.igibson_env.simulator.gen_vr_robot_action()
            self.human.apply_action(actionStep)
        else:
            x, y, z = self.human.get_position()
            end, next_hl_state, action_object = self.get_next_goal()
            end, ori = self.transform_end_location(end)
            arrived = self._step(end, ori)
            self.lsi_env.update_joint_ml_state()
            if arrived:
                self.lsi_env.update_human_hl_state(next_hl_state,
                                                   action_object)
                # time.sleep(5)

    def _step(self, end, final_ori):
        self.update_occupancy_grid()
        x, y, z = self.human.get_position()
        path = self.planner.find_path((x, y), end, self.occupancy_grid)
        return self.motion_controller.step(self.human, self.robot, final_ori,
                                           path)

    def pick(self, loc):
        self.motion_controller.pick(self.human, loc)

    def drop(self):
        pass

    def open(self):
        pass

    def get_next_goal(self):
        agent_state = self.lsi_env.human_state
        world_state = self.lsi_env.world_state
        action, object = 'stay', agent_state.holding
        if agent_state.holding == 'None':
            if world_state.in_pot == 2 and self.lsi_env.robot_state.holding == 'onion':
                action, object = ('pickup', 'dish')
                next_hl_state = f'dish_{world_state.in_pot}'
            elif world_state.in_pot == 3 and self.lsi_env.robot_state.holding != 'dish':
                action, object = ('pickup', 'dish')
                next_hl_state = f'dish_{world_state.in_pot}'
            else:
                action, object = ('pickup', 'onion')
                next_hl_state = f'onion_{world_state.in_pot}'
                agent_state.next_holding = 'onion'
        elif agent_state.holding == 'onion':
            action, object = ('drop', 'onion')
            next_hl_state = f'None_{world_state.in_pot+1}'
            agent_state.next_holding = 'None'
        elif world_state.in_pot <= 3 and agent_state.holding == 'None':
            action, object = ('pickup', 'dish')
            next_hl_state = f'dish_{world_state.in_pot}'
            agent_state.next_holding = 'dish'
        elif agent_state.holding == 'dish' and world_state.in_pot == 3:
            action, object = ('pickup', 'soup')
            # world_state.in_pot = 0
            next_hl_state = f'soup_{world_state.in_pot}'
            agent_state.next_holding = 'soup'
        elif agent_state.holding == 'soup':
            action, object = ('deliver', 'soup')
            next_hl_state = f'None_{world_state.in_pot}'
            agent_state.next_holding = 'None'

        for order in world_state.orders:
            next_hl_state += f'_{order}'

        possible_motion_goals = self.hlp.map_action_to_location(
            world_state, agent_state, (action, object))
        goal = possible_motion_goals[0]
        return goal, next_hl_state, (action, object)

    def transform_end_location(self, loc):
        objects = self.igibson_env.simulator.scene.get_objects()
        end_continuous = grid_to_real_coord(loc)
        selected_object = None
        for o in objects:
            pos = o.get_position()
            if self.is_at_location(pos, end_continuous, 0.2):
                selected_object = o
        pos, ori = selected_object.get_position_orientation()
        ori = quat2euler(ori[0], ori[1], ori[2], ori[3])[2]
        # absolute transformation
        ori = normalize_radians(ori - 1.57)
        pos[0] = pos[0] + math.cos(ori)
        pos[1] = pos[1] + math.sin(ori)
        return pos[0:2], normalize_radians(ori + math.pi)

    def is_at_location(self, loc, dest, tolerance):
        if (dest[0] - tolerance) < loc[0] < (dest[0] + tolerance) and (
                dest[1] - tolerance) < loc[1] < (dest[1] + tolerance):
            return True
        else:
            return False

    def update_occupancy_grid(self):
        # if self.is_observable(self.robot):
        x, y, z = self.robot.get_position()
        loc = real_to_grid_coord((x, y))
        for i in range(len(self.occupancy_grid)):
            for j in range(len(self.occupancy_grid[0])):
                if self.occupancy_grid[i][j] == 'R':
                    self.occupancy_grid[i][j] = 'X'
        self.occupancy_grid[loc[0]][loc[1]] = 'R'

    def is_observed(self, object, track):
        '''
        An object is defined to be observed if it is in sight for a period of time.
        track: an array containing True/False. The length should be the same as the insight threshold.
        '''
        track.pop(0)
        track.append(self.is_observable(object))

        return all(track), track

    def is_observable(self, object):
        object_x, object_y, _ = object.get_position()

        human_x, human_y, _ = self.human.get_position()
        qx, qy, qz, qw = self.human.get_orientation()
        _, _, theta = quat2euler(qx, qy, qz, qw)

        slope_x = object_x - human_x
        slope_y = object_y - human_y

        # calculate angle between human and object
        ori_vec_x = math.cos(theta)
        ori_vec_y = math.sin(theta)

        ori_vec = [ori_vec_x, ori_vec_y]
        vec = [slope_x, slope_y]

        ori_vec = ori_vec / np.linalg.norm(ori_vec)
        vec = vec / np.linalg.norm(vec)
        dot_product = np.dot(ori_vec, vec)
        angle = np.arccos(dot_product)

        # calculate gridspaces line of sight intersects with
        line_of_sight = False
        block_flag = False
        grid_spaces = set()
        slope_x = object_x - human_x
        slope_y = object_y - human_y
        dx = 0 if slope_x == 0 else slope_x / (abs(slope_x) * 10)
        dy = slope_y / (abs(slope_y) *
                        10) if slope_x == 0 else slope_y / (abs(slope_x) * 10)
        current_x = human_x
        current_y = human_y
        object_coord = real_to_grid_coord((object_x, object_y))
        # print(object_x, object_y)
        while True:
            grid_coord = real_to_grid_coord((current_x, current_y))
            # print(current_x, current_y)
            if math.dist([current_x, current_y], [object_x, object_y]) < 0.2:
                break
            else:
                grid_spaces.add(grid_coord)
            current_x += dx
            current_y += dy
            # print(current_x, current_y)

        for grid in grid_spaces:
            if self.occupancy_grid[grid[0]][
                    grid[1]] != 'X' and self.occupancy_grid[grid[0]][
                        grid[1]] != 'R':
                block_flag = True

        line_of_sight = not block_flag

        return line_of_sight and angle <= self.vision_range / 2

    def get_position(self):
        return self.human.get_position()
