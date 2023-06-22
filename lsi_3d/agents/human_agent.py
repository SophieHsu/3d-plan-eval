import copy
import math
import numpy as np
from utils import quat2euler, real_to_grid_coord, grid_to_real_coord, normalize_radians
from agent import Agent
from lsi_3d.mdp.lsi_env import LsiEnv
from lsi_3d.mdp.hl_state import AgentState, WorldState
import time
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.objects.articulated_object import URDFObject
from igibson.objects.visual_marker import VisualMarker
import pybullet as p


class HumanAgent():

    def __init__(self,
                 human,
                 planner,
                 motion_controller,
                 occupancy_grid,
                 hlp,
                 lsi_env,
                 tracking_env,
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
        self.igibson_env = tracking_env.env
        self.tracking_env = tracking_env
        self.vr = vr
        self.insight_threshold = insight_threshold
        self.prev_end = None
        self.object_position = None
        self.arrived = False
        self.step_index = 0
        self.avoiding_robot = False
        self.next_hl_state = None
        self.action_object = None

    def change_state(self):
        # pass
        self.lsi_env.update_human_hl_state("onion_0_onion_onion",
                                           ("pickup", "onion"))
        self.lsi_env.update_human_hl_state("None_1_onion_onion",
                                           ("drop", "onion"))
        self.lsi_env.update_human_hl_state("onion_1_onion_onion",
                                           ("pickup", "onion"))
        self.lsi_env.update_human_hl_state("None_2_onion_onion",
                                           ("drop", "onion"))
        # self.lsi_env.update_human_hl_state("onion_2_onion_onion", ("pickup", "onion"))
        # self.lsi_env.update_human_hl_state("None_3_onion_onion", ("drop", "onion"))
        # self.lsi_env.update_human_hl_state("dish_3_onion_onion", ("pickup", "dish"))
        # self.lsi_env.update_human_hl_state("soup_3_onion_onion", ("pickup", "soup"))
        self.lsi_env.update_joint_ml_state()

    def set_robot(self, robot):
        self.robot = robot

    def step(self):
        if self.vr:
            self.igibson_env.simulator.switch_main_vr_robot(self.human)
            actionStep = self.igibson_env.simulator.gen_vr_robot_action()
            self.human.apply_action(actionStep)
        else:
            x, y, z = self.human.get_position()

            if self.arrived == False:
                end = self.get_next_goal()

                if end != None:
                    end, ori = self.transform_end_location(end)
                    self.interacting = False
                    self.arrived = self._step(end, ori)
            else:
                self._arrival_step()
                self.interacting = True
            self.lsi_env.update_joint_ml_state()

    def _step(self, end, final_ori):
        self.update_occupancy_grid()
        x, y, z = self.human.get_position()
        robot_x, robot_y, _ = self.robot.get_position()
        if math.dist([x, y], [robot_x, robot_y]) < 0.8 or self.avoiding_robot:
            self.avoiding_robot = True
            end = self.loc_to_avoid_robot()
        if math.dist([x, y], [robot_x, robot_y]) > 1.2:
            self.avoiding_robot = False
        path = self.planner.find_path((x, y), end, self.occupancy_grid)

        round_prev_end = [round(p, 3) for p in self.prev_end
                          ] if self.prev_end is not None else self.prev_end
        round_end = [round(e, 3) for e in end] if end is not None else end
        is_new_end = True if round_prev_end != round_end else False
        self.prev_end = end
        return self.motion_controller.step(self.human, self.robot, final_ori,
                                           path, is_new_end)

    def _arrival_step(self):
        next_hl_state = self.next_hl_state
        action_object = self.action_object
        action = self.action_object[0]
        object = self.action_object[1]
        if action == "pickup" and object == "onion":
            if self.object_position is None:
                self.object_position = self.tracking_env.get_closest_onion(
                ).get_position()
            # marker_2 = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06)
            # self.igibson_env.simulator.import_object(marker_2)
            # marker_2.set_position(self.object_position)
            done = self.pick(self.object_position, [0, 0, 0.05])
            if done:
                self.completed_goal(next_hl_state, action_object)
        elif action == "drop" and object == "onion":
            if self.object_position is None:
                self.object_position = self.tracking_env.get_closest_pan(
                ).get_position()
            done = self.drop(self.object_position, [0, -0.1, 0.25])
            if done:
                self.completed_goal(next_hl_state, action_object)
        elif action == "pickup" and object == "dish":
            if self.object_position is None:
                self.object_position = self.tracking_env.get_closest_bowl(
                ).get_position()
            done = self.pick(self.object_position, [0, -0.25, 0.1])
            if done:
                self.completed_goal(next_hl_state, action_object)
        elif action == "deliver" and object == "soup":
            done = self.drop(self.human.get_position(), [0, 0.5, 0.2])
            if done:
                self.completed_goal(next_hl_state, action_object)
        elif (action == "pickup" and object == "soup") or self.step_index >= 1:
            if self.object_position is None:
                pan = self.tracking_env.get_closest_pan()
                self.object_position = pan.get_position()
            if self.step_index == 0:
                done = self.drop(self.object_position, [-0.4, -0.25, 0.3])
                if done:
                    self.step_index = self.step_index + 1
                    self.object_position = self.tracking_env.get_closest_onion(
                        on_pan=True).get_position()
            elif self.step_index == 1:
                done = self.pick(self.object_position, [0, -0.05, 0.05])
                if done:
                    self.step_index = self.step_index + 1
                    self.object_position = self.tracking_env.get_closest_bowl(
                    ).get_position()
            elif self.step_index == 2:
                done = self.drop(self.object_position, [0, -0.1, 0.3])
                if done:
                    num_item_in_bowl = len(self.tracking_env.get_bowl_status()[self.tracking_env.get_closest_bowl()])
                    if num_item_in_bowl < self.tracking_env.kitchen.onions_for_soup:
                        self.step_index = 1
                        self.object_position = self.tracking_env.get_closest_onion(on_pan=True
                    ).get_position()
                    else:
                        self.step_index = 3
                        self.object_position = self.tracking_env.get_closest_bowl(
                    ).get_position()
            # elif self.step_index == 3:
            #     done = self.pick(self.object_position, [0, 0, 0.05])
            #     if done:
            #         self.step_index = self.step_index + 1
            #         self.object_position = self.tracking_env.get_closest_bowl(
            #         ).get_position()
            # elif self.step_index == 4:
            #     done = self.drop(self.object_position, [0, -0.1, 0.3])
            #     if done:
            #         self.step_index = self.step_index + 1
            #         self.object_position = self.tracking_env.get_closest_onion(
            #             on_pan=True).get_position()
            # elif self.step_index == 5:
            #     done = self.pick(self.object_position, [0, 0, 0.05])
            #     if done:
            #         self.step_index = self.step_index + 1
            #         self.object_position = self.tracking_env.get_closest_bowl(
            #         ).get_position()
            # elif self.step_index == 6:
            #     done = self.drop(self.object_position, [0, -0.1, 0.3])
            #     if done:
            #         self.step_index = self.step_index + 1
            #         self.object_position = self.tracking_env.get_closest_bowl(
            #         ).get_position()
            elif self.step_index == 3:
                done = self.pick(self.object_position, [0, -0.3, 0.1])
                if done:
                    self.step_index = self.step_index + 1
            else:
                self.completed_goal(next_hl_state, action_object)
        # print(next_hl_state, action_object)

    def pick(self, loc, offset=[0, 0, 0]):
        return self.motion_controller.pick(self.human, loc, offset)

    def drop(self, loc, offset=[0, 0, 0]):
        return self.motion_controller.drop(self.human, loc, offset)

    def completed_goal(self, next_hl_state, action_object):
        self.step_index = 0
        self.lsi_env.human_state.update_hl_state(next_hl_state,
                                                 self.lsi_env.world_state)
        self.object_position = None
        self.arrived = False

    def get_next_goal(self):
        agent_state = self.lsi_env.human_state
        world_state = self.lsi_env.world_state
        action, object = 'stay', agent_state.holding
        if agent_state.holding == 'None':
            if world_state.in_pot == self.tracking_env.kitchen.onions_for_soup - 1 and self.lsi_env.robot_state.holding == 'onion':
                action, object = ('pickup', 'dish')
                next_hl_state = f'dish_{world_state.in_pot}'
            elif world_state.in_pot >= self.tracking_env.kitchen.onions_for_soup and self.lsi_env.robot_state.holding != 'dish':
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
        elif agent_state.holding == 'dish' and (world_state.in_pot >= self.lsi_env.mdp.num_items_for_soup or self.interacting == True):
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

        possible_motion_goals = self.lsi_env.map_action_to_location(
            (action, object))
        goal = possible_motion_goals
        self.next_hl_state = next_hl_state
        self.action_object = (action, object)
        return goal

    def transform_end_location(self, loc):
        # objects = self.igibson_env.simulator.scene.get_objects()
        objects = self.tracking_env.kitchen.static_objs
        selected_object = None
        for o in objects.keys():
            if type(o) != URDFObject:
                continue

            if type(objects[o]) == list:
                for o_loc in objects[o]:
                    if o_loc == loc: selected_object = o
                        # pos = grid_to_real_coord(loc)
            elif objects[o] == loc:
                selected_object = o

        pos = list(grid_to_real_coord(loc))
        _, ori = selected_object.get_position_orientation()
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

    def loc_to_avoid_robot(self):
        human_x, human_y, _ = self.human.get_position()
        human_pos_grid = real_to_grid_coord([human_x, human_y])
        robot_x, robot_y, _ = self.robot.get_position()

        # relative_neighbor_locs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        relative_neighbor_locs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        farthest_neighbor = None
        dist = -1
        for n in relative_neighbor_locs:
            x_neighbor = human_pos_grid[0] + n[0]
            y_neighbor = human_pos_grid[1] + n[1]

            if 0 <= x_neighbor < 8 and 0 <= y_neighbor < 8 and self.occupancy_grid[
                    x_neighbor][y_neighbor] == "X":
                neighbor_continuous = grid_to_real_coord(
                    [x_neighbor, y_neighbor])
                neighbor_dist = math.dist(
                    [neighbor_continuous[0], neighbor_continuous[1]],
                    [robot_x, robot_y])
                neighbor = grid_to_real_coord([x_neighbor, y_neighbor])
                path = self.planner.find_path((human_x, human_y), neighbor,
                                              self.occupancy_grid)
                if neighbor_dist > dist and len(path) > 0:
                    farthest_neighbor = neighbor
                    dist = neighbor_dist
        return farthest_neighbor

    # def is_observed(self, object, track):
    #     '''
    #     An object is defined to be observed if it is in sight for a period of time.
    #     track: an array containing True/False. The length should be the same as the insight threshold.
    #     '''
    #     track.pop(0)
    #     track.append(self.is_observable(object))

    #     return all(track), track

    # def is_observable(self, object):
    #     object_x, object_y, _ = object.get_position()

    #     human_x, human_y, _ = self.human.get_position()
    #     qx, qy, qz, qw = self.human.get_orientation()
    #     _, _, theta = quat2euler(qx, qy, qz, qw)

    #     slope_x = object_x - human_x
    #     slope_y = object_y - human_y

    #     # calculate angle between human and object
    #     ori_vec_x = math.cos(theta)
    #     ori_vec_y = math.sin(theta)

    #     ori_vec = [ori_vec_x, ori_vec_y]
    #     vec = [slope_x, slope_y]

    #     ori_vec = ori_vec / np.linalg.norm(ori_vec)
    #     vec = vec / np.linalg.norm(vec)
    #     dot_product = np.dot(ori_vec, vec)
    #     angle = np.arccos(dot_product)

    #     # calculate gridspaces line of sight intersects with
    #     line_of_sight = False
    #     block_flag = False
    #     grid_spaces = set()
    #     slope_x = object_x - human_x
    #     slope_y = object_y - human_y
    #     dx = 0 if slope_x == 0 else slope_x / (abs(slope_x) * 10)
    #     dy = slope_y / (abs(slope_y) *
    #                     10) if slope_x == 0 else slope_y / (abs(slope_x) * 10)
    #     current_x = human_x
    #     current_y = human_y
    #     object_coord = real_to_grid_coord((object_x, object_y))
    #     # print(object_x, object_y)
    #     while True:
    #         grid_coord = real_to_grid_coord((current_x, current_y))
    #         # print(current_x, current_y)
    #         if math.dist([current_x, current_y], [object_x, object_y]) < 0.2:
    #             break
    #         else:
    #             grid_spaces.add(grid_coord)
    #         current_x += dx
    #         current_y += dy
    #         # print(current_x, current_y)

    #     for grid in grid_spaces:
    #         if self.occupancy_grid[grid[0]][
    #                 grid[1]] != 'X' and self.occupancy_grid[grid[0]][
    #                     grid[1]] != 'R':
    #             block_flag = True

    #     line_of_sight = not block_flag

    #     return line_of_sight and angle <= self.vision_range / 2

    def get_position(self):
        return self.human.get_position()
