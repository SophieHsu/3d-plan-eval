"""
Module interfaces between lsi directives and low level iGibson robot behavior
"""
from re import A
import numpy as np
import math
import logging
from igibson.utils.utils import quatToXYZW
from transforms3d.euler import euler2quat
from lsi_3d.utils.constants import DIRE2POSDIFF, TARGET_ORNS
from lsi_3d.utils.functions import quat2euler
from scipy.spatial.transform import Rotation
from igibson.external.pybullet_tools.utils import (
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    joints_from_names,
    set_joint_positions,
)
from utils import grid_to_real_coord, quat2euler, normalize_radians, real_to_grid_coord
from numpy.linalg import inv
import pybullet as p
from igibson.utils.utils import l2_distance, parse_config, restoreState

ONE_STEP = 0.02


class iGibsonAgent:
    '''
    Controls low level agent agent actions acting as an interface between medium level turn and forward actions
    and continuous iGibson environment
    '''

    def __init__(self,
                 robot,
                 start,
                 direction,
                 name,
                 target_x=None,
                 target_y=None,
                 target_direction=None):
        self.object = robot
        self.direction = direction
        self.start = start + (direction, )
        # self.path = path
        self.target_x = target_x
        self.target_y = target_y
        self.target_direction = target_direction
        self.action_index = 0
        self.name = name
        self.interact_step_index = 0
        self.arrived_hand_step = 0
        self.object_position = None
        self.grasping_delay = 10
        self.counters = [0, 0]
        self.prev_gripper_action = []
        self.interact_obj = None
        if self.name == 'robot':
            self.arm_init()

        self.arm_eef_to_obj_dict = {
            'onion': [0, 0.0, 0.05],
            'dish': [0, -0.25, 0.1],
            'pan': [0, -0.1, 0.25],
            'soup': [0, 0.5, 0.2]
        }

    def update(self, target_x, target_y, direction, target_direction):
        self.target_x = target_x
        self.target_y = target_y
        self.direction = direction
        self.target_direction = target_direction

    def action_completed(self, current_action):
        if current_action == None:
            return True
        #if self.action_index >= len(self.path):
        #    return None
        #current_action = self.path[self.action_index]
        ready_for_next_action = False
        x, y, z = self.object.get_position()
        #print(self.name, current_action, self.get_current_orn_z(), self.target_direction, turn_distance(self.get_current_orn_z(), TARGET_ORNS[self.target_direction]))
        # if self.target_x == None:
        #     ready_for_next_action = True
        #     x, y, z = self.object.get_position()
        #     self.target_x = x
        #     self.target_y = y
        if current_action == 'F' and self.forward_distance(
                x, y, self.target_x, self.target_y,
                self.direction) < ONE_STEP * 1.5:
            #self.action_index += 1
            ready_for_next_action = True
        elif current_action in 'NESW' and self.turn_distance(
                self.get_current_orn_z(),
                TARGET_ORNS[self.target_direction]) < ONE_STEP * 1.5:
            #self.action_index += 1
            self.direction = current_action
            ready_for_next_action = True
        elif current_action == 'I':
            if self.interact_step_index == -1:
                self.interact_step_index = 0
                return True
            else:
                return False
        elif current_action == None:  # None when first action
            return True

        # if self.action_index >= len(self.path):
        #     return None

        # if ready_for_next_action:
        #     next_action = self.path[self.action_index]
        #     if next_action == "F":
        #         diff_x, diff_y = DIRE2POSDIFF[self.direction]
        #         self.target_x += diff_x
        #         self.target_y += diff_y
        #     elif next_action in "NWES":
        #         self.target_direction = next_action
        return ready_for_next_action

    def prepare_for_next_action(self, current_pos, next_action):
        r, c, f = current_pos
        x, y = grid_to_real_coord((r, c))
        if self.target_x == None or self.target_y == None:
            # x, y, z = self.object.get_position()

            self.target_x = x
            self.target_y = y

        if next_action == 'F':
            diff_x, diff_y = DIRE2POSDIFF[self.direction]
            self.target_x = diff_x + x
            self.target_y = diff_y + y
        elif next_action in 'NESW':
            self.target_direction = next_action
            x, y, z = self.object.get_position()
            self.target_x = x
            self.target_y = y
        elif next_action == 'STAY' or next_action == 'D':
            return

    def get_current_orn_z(self):
        x, y, z, w = self.object.get_orientation()
        x, y, z = quat2euler(x, y, z, w)
        return z

    def turn_distance(self, cur_orn_z, target_orn_z):
        return abs(cur_orn_z - target_orn_z)

    def forward_distance(self, cur_x, cur_y, target_x, target_y, direction):
        if direction in 'NS':
            return abs(cur_x - target_x)
        else:
            return abs(cur_y - target_y)

    def agent_move_one_step(self, env, action):
        #if action == None or action == MLAction.IDLE:
        #if self.name == "robot":
        #action = np.zeros(env.action_space.shape)
        #action = np.full(env.action_space.shape, 0.000000000000000000001)

        # action = env.action_space
        # action[0] = 0
        # action[1] = 0
        #self.object.apply_action(action)
        #return
        if action == None:
            return

        if action in 'NESW':
            self.agent_turn_one_step(env, action)
        elif action == 'F':
            cur_x, cur_y = self.object.get_position()[:2]
            # cur_x, cur_y = real_to_grid_coord((cur_x,cur_y))
            goal_angle = math.atan2((self.target_y - cur_y),
                                    (self.target_x - cur_x))
            current_heading = self.get_current_orn_z()
            angle_delta = self.calc_angle_distance(goal_angle, current_heading)

            if angle_delta > 0.2:
                while angle_delta > 0.1:
                    self.turn_toward(env, goal_angle, angle_delta, cur_x,
                                     cur_y)
                    current_heading = self.get_current_orn_z()
                    angle_delta = self.calc_angle_distance(
                        goal_angle, current_heading)
            else:
                self.agent_forward_one_step(env)
        else:
            pass

    def agent_forward_one_step(self, env):
        if self.name == "human_sim":
            x, y, z = self.object.get_position()
            if self.direction == 'N':
                self.object.set_position_orientation(
                    [x - ONE_STEP, y, z], self.object.get_orientation())
            elif self.direction == 'S':
                self.object.set_position_orientation(
                    [x + ONE_STEP, y, z], self.object.get_orientation())
            elif self.direction == 'E':
                self.object.set_position_orientation(
                    [x, y + ONE_STEP, z], self.object.get_orientation())
            elif self.direction == 'W':
                self.object.set_position_orientation(
                    [x, y - ONE_STEP, z], self.object.get_orientation())
        else:
            action = np.zeros(env.action_space.shape)
            action[0] = 0.2
            action[1] = 0
            start_x, start_y = self.object.get_position()[:2]

            cur_x, cur_y = self.object.get_position()[:2]
            distance_to_target = self.forward_distance(cur_x, cur_y,
                                                       self.target_x,
                                                       self.target_y,
                                                       self.direction)

            if distance_to_target < 0.3:
                action[0] *= 0.8
            elif distance_to_target < 0.1:
                action[0] *= 0.7
            # elif distance_to_target < 0.05:
            #     action[0] /= 8
            self.object.apply_action(action)

    def calc_angle_distance(self, a1, a2):

        if a1 < 0: a1 += 6.28319
        if a2 < 0: a2 += 6.28319

        d = abs(a2 - a1)

        if d > 3.14159:
            d = 6.28319 - d

        return d

    def turn_toward(self, env, goal_angle, angle_delta, cur_x, cur_y):
        cur_orn_z = self.get_current_orn_z()
        target_orn_z = goal_angle
        action = np.zeros(env.action_space.shape)
        action[0] = 0

        distance_to_target = self.forward_distance(cur_x, cur_y, self.target_x,
                                                   self.target_y,
                                                   self.direction)

        # Decides to turn right or left
        if (cur_orn_z < target_orn_z):
            action[1] = -angle_delta
        else:
            action[1] = angle_delta
        #print((cur_orn_z-target_orn_z) / (action[1]/action[1]), action[1], cur_orn_z, target_orn_z)
        if ((cur_orn_z - target_orn_z) /
            (action[1] / abs(action[1]))) > 4:  # > 3.14
            action[1] = -action[1]
        if abs(target_orn_z - cur_orn_z) < 0.15:
            action[1] /= 2
        elif abs(target_orn_z - cur_orn_z) < 0.08:
            action[1] /= 4
        self.object.apply_action(action)
        env.simulator.step()

    def agent_turn_one_step(self, env, action):
        if self.name == "human_sim":
            x, y, z, w = self.object.get_orientation()
            x, y, z = quat2euler(x, y, z, w)
            #print("turn z:", z, action)
            target_orn_z = TARGET_ORNS[self.target_direction]

            pos = z - target_orn_z
            neg = target_orn_z - z
            if pos < 0:
                pos += 3.1415926 * 2
            elif neg < 0:
                neg += 3.1415926 * 2
            if pos < neg:
                z -= ONE_STEP
            else:
                z += ONE_STEP
            self.object.set_position_orientation(
                self.object.get_position(),
                quatToXYZW(euler2quat(x, y, z), "wxyz"))
        else:
            x, y, z, w = self.object.get_orientation()
            x, y, z = quat2euler(x, y, z, w)
            cur_orn_z = z
            target_orn_z = TARGET_ORNS[self.target_direction]
            action = np.zeros(env.action_space.shape)
            action[0] = 0
            if (cur_orn_z < target_orn_z):
                action[1] = -0.2
            else:
                action[1] = 0.2
            #print((cur_orn_z-target_orn_z) / (action[1]/action[1]), action[1], cur_orn_z, target_orn_z)
            if ((cur_orn_z - target_orn_z) /
                (action[1] / abs(action[1]))) > 4:  # > 3.14
                action[1] = -action[1]
            if abs(target_orn_z - cur_orn_z) < 0.5:
                action[1] /= 2
            elif abs(target_orn_z - cur_orn_z) < 0.2:
                action[1] /= 4
            self.object.apply_action(action)

    def interact_ll_control(self,
                            action_object,
                            tracking_env,
                            num_item_needed_in_dish=1):
        HEIGHT_OFFSET = 0.3
        action = action_object[0]
        object = action_object[1]
        if action == "pickup" and object == "onion":
            if self.object_position is None:
                # self.object_position = tracking_env.get_closest_onion(
                #     agent_pos=self.object.get_eef_position()).get_position()
                # marker_2 = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06)
                # self.igibson_env.simulator.import_object(marker_2)
                # marker_2.set_position(self.object_position)
                self.object_position = self.object.get_position().copy()
            done, in_hand = self.pick(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_onion(
                    agent_pos=self.object.get_eef_position()),
                name='onion',
                offset=[0, 0.5, 1.3])  # offset=[0, 0, 0.05 + HEIGHT_OFFSET])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None
        elif action == "drop" and object == "onion":
            # if self.object_position is None:
            self.object_position = tracking_env.get_closest_pan(
                agent_pos=self.object.get_position()).get_position()
            # self.object_position = self.object.get_eef_position()
            done, in_hand = self.drop(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_pan(
                    agent_pos=self.object.get_eef_position()),
                name='onion',
                offset=[0, 0, 0.25 + HEIGHT_OFFSET])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None

        # elif action == "drop" and object == "dish":
        #     # if self.object_position is None:
        #     self.object_position = tracking_env.get_closest_pan(
        #         agent_pos=self.object.get_eef_position()).get_position()
        #     # self.object_position = self.object.get_eef_position()
        #     done, in_hand = self.drop(
        #         self.object_position,
        #         tracking_env,
        #         tracking_env.get_closest_pan(
        #             agent_pos=self.object.get_eef_position()),
        #         name='dish',
        #         offset=[-0.4, -0.25, 0.3 + HEIGHT_OFFSET])
        #     if done or in_hand:
        #         self.interact_step_index = -1
        #         self.object_position = None

        elif action == "drop" and object == "dish":
            if self.object_position is None:
                self.object_position = self.object.get_position()
            done, in_hand = self.drop(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_bowl(
                    agent_pos=self.object.get_eef_position()),
                name='dish',
                offset=[0, 1, 1.2 + HEIGHT_OFFSET])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None

        elif action == "pickup" and object == "dish":
            # if self.object_position is None:
            self.object_position = tracking_env.get_closest_bowl(
                agent_pos=self.object.get_eef_position()).get_position()
            done, in_hand = self.pick(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_bowl(
                    agent_pos=self.object.get_eef_position()),
                name='dish',
                offset=[0, -0.25, 0.1 + HEIGHT_OFFSET])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None
        elif action == "deliver" and object == "soup":
            if self.object_position is None:
                self.object_position = self.object.get_position()
            done, in_hand = self.drop(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_bowl(
                    agent_pos=self.object.get_eef_position()),
                name='soup',
                offset=[0, 1, 1.2 + HEIGHT_OFFSET])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None
        elif action == "pickup" and object == "soup":

            if self.object_position is None:
                pan = tracking_env.get_closest_pan(
                    agent_pos=self.object.get_position())
                tracking_env.kitchen.interact_objs[pan] = True
                self.interact_obj = pan
                self.object_position = pan.get_position()
            if self.interact_step_index == 0:
                done, in_hand = self.drop(
                    self.object_position,
                    tracking_env,
                    tracking_env.get_closest_pan(
                        agent_pos=self.object.get_position()),
                    name='soup',
                    offset=[-0.6, 0, 0.2])
                if done or in_hand:
                    self.interact_step_index = 1
                    self.object_position = tracking_env.get_closest_onion(
                        agent_pos=self.object.get_position(),
                        on_pan=True).get_position()
                    self.item_in_bowl = 0

            # start pick and drop items from pan/pot to bowl/dish/plate
            elif self.interact_step_index == 1:
                done, in_hand = self.pick(
                    self.object_position,
                    tracking_env,
                    tracking_env.get_closest_onion(
                        agent_pos=self.object.get_position()),
                    name='soup',
                    offset=[0, -0.05, 0.05 + HEIGHT_OFFSET])
                if done or in_hand:
                    self.interact_step_index = self.interact_step_index = 2
                    self.object_position = tracking_env.get_closest_bowl(
                        agent_pos=self.object.get_position()).get_position()
            elif self.interact_step_index == 2:
                done, in_hand = self.drop(
                    self.object_position,
                    tracking_env,
                    tracking_env.get_closest_bowl(
                        agent_pos=self.object.get_position()),
                    name='soup',
                    offset=[0, -0.1, HEIGHT_OFFSET])
                if done or in_hand:
                    self.item_in_bowl += 1
                    if self.item_in_bowl < num_item_needed_in_dish:
                        self.interact_step_index = self.interact_step_index = 1
                        self.object_position = tracking_env.get_closest_onion(
                            agent_pos=self.object.get_position(),
                            on_pan=True).get_position()
                    else:
                        self.interact_step_index = self.interact_step_index = 3

            # elif self.interact_step_index == 3:
            #     done, in_hand = self.pick(
            #         self.object_position,
            #         tracking_env,
            #         tracking_env.get_closest_onion(
            #             agent_pos=self.object.get_eef_position()),
            #         name='soup',
            #         offset=[0, 0, 0.05 + HEIGHT_OFFSET])
            #     if done or in_hand:
            #         self.interact_step_index = self.interact_step_index + 1
            #         self.object_position = tracking_env.get_closest_bowl(
            #             agent_pos=self.object.get_eef_position()).get_position(
            #             )
            # elif self.interact_step_index == 4:
            #     done, in_hand = self.drop(
            #         self.object_position,
            #         tracking_env,
            #         tracking_env.get_closest_bowl(
            #             agent_pos=self.object.get_eef_position()),
            #         name='soup',
            #         offset=[0, -0.1, 0.3 + HEIGHT_OFFSET])
            #     if done or in_hand:
            #         self.interact_step_index = self.interact_step_index + 1
            #         self.object_position = tracking_env.get_closest_onion(
            #             agent_pos=self.object.get_eef_position(),
            #             on_pan=True).get_position()
            # elif self.interact_step_index == 5:
            #     done, in_hand = self.pick(
            #         self.object_position,
            #         tracking_env,
            #         tracking_env.get_closest_onion(
            #             agent_pos=self.object.get_eef_position(),
            #             on_pan=True,
            #             name='soup',
            #         ),
            #         offset=[0, 0, 0.05 + HEIGHT_OFFSET])
            #     if done or in_hand:
            #         self.interact_step_index = self.interact_step_index + 1
            #         self.object_position = tracking_env.get_closest_bowl(
            #             agent_pos=self.object.get_eef_position()).get_position(
            #             )
            # elif self.interact_step_index == 6:
            #     done, in_hand = self.drop(
            #         self.object_position,
            #         tracking_env,
            #         tracking_env.get_closest_bowl(
            #             agent_pos=self.object.get_eef_position()),
            #         name='soup',
            #         offset=[0, -0.1, 0.3 + HEIGHT_OFFSET])
            #     if done or in_hand:
            #         self.interact_step_index = self.interact_step_index + 1
            #         self.object_position = tracking_env.get_closest_bowl(
            #             agent_pos=self.object.get_eef_position()).get_position(
            #             )
            elif self.interact_step_index == 3:  #7:
                done, in_hand = self.pick(
                    self.object_position,
                    tracking_env,
                    tracking_env.get_closest_bowl(
                        agent_pos=self.object.get_eef_position()),
                    name='soup',
                    offset=[0, -0.3, 0.1 + HEIGHT_OFFSET])
                if done or in_hand:
                    self.interact_step_index = self.interact_step_index + 1
            else:
                self.interact_step_index = -1
                self.object_position = None
                tracking_env.kitchen.interact_objs[self.interact_obj] = False

    def arm_init(self):
        body_ids = self.object.get_body_ids()
        assert len(body_ids) == 1, "Fetch robot is expected to be single-body."
        self.robot_id = body_ids[0]

        self.arm_default_joint_positions = (
            0.10322468280792236,
            -1.414019864768982,
            1.5178184935241699,
            0.8189625336474915,
            2.200358942909668,
            2.9631312579803466,
            -1.2862852996643066,
            0.0008453550418615341,
        )

        self.robot_default_joint_positions = (
            [0.0, 0.0] + [self.arm_default_joint_positions[0]] + [0.0, 0.0] +
            list(self.arm_default_joint_positions[1:]) + [0.01, 0.01])

        self.robot_joint_names = [
            "r_wheel_joint",
            "l_wheel_joint",
            "torso_lift_joint",
            "head_pan_joint",
            "head_tilt_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "r_gripper_finger_joint",
            "l_gripper_finger_joint",
        ]
        self.arm_joints_names = [
            "torso_lift_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

        # Indices of the joints of the arm in the vectors returned by IK and motion planning (excluding wheels, head, fingers)
        self.robot_arm_indices = [
            self.robot_joint_names.index(arm_joint_name)
            for arm_joint_name in self.arm_joints_names
        ]

        # PyBullet ids of the joints corresponding to the joints of the arm
        self.arm_joint_ids = joints_from_names(self.robot_id,
                                               self.arm_joints_names)
        self.all_joint_ids = joints_from_names(self.robot_id,
                                               self.robot_joint_names)

        self.arm_max_limits = get_max_limits(self.robot_id, self.all_joint_ids)
        # for i, v in enumerate(
        #         self.object._default_controller_config['arm_0']
        #     ['InverseKinematicsController']['control_limits']['position'][1]):
        #     if v < 1000:
        #         self.arm_max_limits[i] = v

        self.arm_min_limits = get_min_limits(self.robot_id, self.all_joint_ids)
        # for i, v in enumerate(
        #         self.object._default_controller_config['arm_0']
        #     ['InverseKinematicsController']['control_limits']['position'][0]):
        #     if v > -1000:
        #         self.arm_min_limits[i] = v

        self.arm_min_limits[2] = 0.35
        self.arm_rest_position = self.robot_default_joint_positions
        self.arm_joint_range = list(
            np.array(self.arm_max_limits) - np.array(self.arm_min_limits))
        self.arm_joint_range = [item + 1 for item in self.arm_joint_range]
        self.arm_joint_damping = [0.1 for _ in self.arm_joint_range]

    def pick(self,
             loc,
             tracking_env,
             target_obj,
             name,
             offset=[0, 0, 0],
             obj_name=None):
        if obj_name is not None:
            offset = self.arm_eef_to_obj_dict[obj_name]
        translated_loc = self.translate_loc(loc, offset)
        gripper_action = [-0.01, -0.01]
        return self.move_hand(translated_loc, tracking_env, target_obj, name,
                              gripper_action)

    def drop(self,
             loc,
             tracking_env,
             target_obj,
             name=None,
             offset=[0, 0, 0],
             obj_name=None):
        # if obj_name is not None:
        #     offset = self.arm_eef_to_obj_dict[obj_name]
        translated_loc = self.translate_loc(loc, offset)
        gripper_action = [0.01, 0.01]
        return self.move_hand(translated_loc, tracking_env, None, name,
                              gripper_action)

    def move_hand(self, loc, tracking_env, target_obj, name, gripper_action):
        position = self.object.get_eef_position()
        if gripper_action != self.prev_gripper_action:
            # loc = self.reset_hand_position
            self.prev_gripper_action = gripper_action
            self.arrived_hand_step = 0

        x_diff = loc[0] - position[0]
        y_diff = loc[1] - position[1]
        z_diff = loc[2] - position[2]
        norm_val = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

        x_diff = x_diff / (norm_val * 10)
        y_diff = y_diff / (norm_val * 10)
        z_diff = z_diff / (norm_val * 10)

        rotated_basis = self.get_rotated_basis()
        diffs = np.array([x_diff, y_diff, z_diff])
        diff_new_basis = np.matmul(inv(rotated_basis), diffs)

        x_diff = diff_new_basis[0]
        y_diff = diff_new_basis[1]
        z_diff = diff_new_basis[2]

        in_hand = False

        if self.arrived_hand_step == 0:
            self.reset_hand_position = position
            # self.original_right_hand_orientation = human._parts["right_hand"].get_orientation()
            # ori = self.original_right_hand_orientation
            # print(quat2euler(ori[0], ori[1], ori[2], ori[3]))
            self.arrived_hand_step = 1
        # Go to location
        elif self.arrived_hand_step == 1:
            distance = .3  #if mid_action[26] == -1 else .01
            # if not (abs(loc - position) - distance <= 0).all():
            # print(math.dist(loc, position))
            if math.dist(loc, position) > distance:
                # action = self.action(0, 0, x_diff, y_diff, z_diff, 0)
                if not self.robot_arm_move(position, x_diff, y_diff, z_diff):
                    self.arrived_hand_step = 2
            else:
                self.arrived_hand_step = 2
                joint_pos = self.accurate_calculate_inverse_kinematics(
                    self.robot_id,
                    self.object.eef_links[self.object.default_arm].link_id,
                    position, 0.05, 100)
                if joint_pos is not None and len(joint_pos) > 0:
                    print("Solution found. Setting new arm configuration.")
                    set_joint_positions(self.robot_id, self.arm_joint_ids,
                                        joint_pos)
            # print(math.dist(loc, position))
            # print("reach")
        # Middle action (grasp or drop)
        elif self.arrived_hand_step == 2:
            # self.robot_arm_move(position, x_diff, y_diff, z_diff)
            gripper_new_pos = [
                self.object.get_joint_states()['r_gripper_finger_joint'][0] +
                gripper_action[0],
                self.object.get_joint_states()['l_gripper_finger_joint'][0] +
                gripper_action[1]
            ]
            if min(gripper_new_pos) < 0.01:
                gripper_new_pos = [0.01, 0.01]
            if max(gripper_new_pos) > 0.05:
                gripper_new_pos = [0.05, 0.05]
            set_joint_positions(self.robot_id, self.all_joint_ids[-2:],
                                gripper_new_pos)
            if self.counters[1] > self.grasping_delay:
                if target_obj is not None:
                    if target_obj.name == 'bowl':
                        for i in tracking_env.items_in_bowl(target_obj):
                            tracking_env.set_in_robot_hand(name, i)
                    tracking_env.set_in_robot_hand(name, target_obj)
                    in_hand = True
                else:
                    for i, item in enumerate(
                            tracking_env.kitchen.in_robot_hand[::-1]):
                        tracking_env.remove_in_robot_hand(item,
                                                          pos=loc,
                                                          counter=i)
                self.arrived_hand_step = 3
                self.counters[1] = 0
            self.counters[1] += 1
            # print("mid")
        # Return to location
        elif self.arrived_hand_step == 3:
            # in_hand = True
            distance = 0.52
            done = False
            # if not (abs(loc - position) - distance <= 0).all():
            if math.dist(loc, position) > distance and not (
                    abs(loc - position) - 0.3 <= 0).all():
                # action = self.action(0, 0, x_diff, y_diff, z_diff, 0)
                # human.apply_action(action)
                if not self.robot_arm_move(position, x_diff, y_diff, z_diff):
                    done = True
                    self.arrived_hand_step = 0
            else:
                done = True
                self.arrived_hand_step = 0

            if done:
                # orientation = self.object.get_orientation()
                # _, _, z = quat2euler(orientation[0], orientation[1],
                #                      orientation[2], orientation[3])
                # z_theta = normalize_radians(z) - math.pi / 2
                # self.reset_right_hand_orientation[2] = z_theta
                # self.original_right_hand_orientation = human._parts[
                # "right_hand"].get_orientation()

                # set_joint_positions(self.robot_id, self.arm_joint_ids, loc)
                # human._parts["right_hand"].set_position(loc)
                # human._parts["right_hand"].set_orientation(
                #     p.getQuaternionFromEuler(
                #         self.reset_right_hand_orientation))
                joint_pos = self.accurate_calculate_inverse_kinematics(
                    self.robot_id,
                    self.object.eef_links[self.object.default_arm].link_id,
                    position, 0.05, 100)
                if joint_pos is not None and len(joint_pos) > 0:
                    print("Solution found. Setting new arm configuration.")
                    set_joint_positions(self.robot_id, self.arm_joint_ids,
                                        joint_pos)
                self.arrived_hand_step = 0
                return True, in_hand
        return False, in_hand

    def get_rotated_basis(self):
        orientation = self.object.get_orientation()
        x, y, z = quat2euler(orientation[0], orientation[1], orientation[2],
                             orientation[3])
        z_theta = normalize_radians(z) - math.pi / 2
        regular_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotation_matrix = np.array([[math.cos(z_theta), -math.sin(z_theta), 0],
                                    [math.sin(z_theta),
                                     math.cos(z_theta), 0], [0, 0, 1]])
        rotated_basis = np.matmul(rotation_matrix, regular_basis)
        return rotated_basis

    def translate_loc(self, targe_loc, offset):
        rotated_basis = self.get_rotated_basis()
        offset_scaling = np.array([[offset[0], 0, 0], [0, offset[1], 0],
                                   [0, 0, offset[2]]])
        scaled_rotated_basis = np.matmul(rotated_basis, offset_scaling)
        translated_loc = np.matmul(scaled_rotated_basis,
                                   np.array([1, 1,
                                             1]).transpose()).transpose()
        translated_loc = translated_loc + np.array(targe_loc)
        return translated_loc

    def robot_arm_move(self, ori_pos, x_diff, y_diff, z_diff):
        target_pos = [
            ori_pos[0] + x_diff, ori_pos[1] + y_diff, ori_pos[2] + z_diff
        ]
        threshold = 0.03  #0.05
        max_iter = 100

        joint_pos = self.accurate_calculate_inverse_kinematics(
            self.robot_id,
            self.object.eef_links[self.object.default_arm].link_id, target_pos,
            threshold, max_iter)
        if joint_pos is not None and len(joint_pos) > 0:
            print("Solution found. Setting new arm configuration.")
            set_joint_positions(self.robot_id, self.arm_joint_ids, joint_pos)
        else:
            print("EE position not reachable.")
            return False
        # self.object.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
        self.object.keep_still()
        # igibson_env.simulator.step()
        return True

    def accurate_calculate_inverse_kinematics(self,
                                              robot_id,
                                              eef_link_id,
                                              target_pos,
                                              threshold,
                                              max_iter,
                                              init_pos=None):
        print("IK solution to end effector position {}".format(target_pos))
        # Save initial robot pose
        state_id = p.saveState()

        max_attempts = 5
        solution_found = False
        joint_poses = None
        print(target_pos)
        for attempt in range(1, max_attempts + 1):
            print("Attempt {} of {}".format(attempt, max_attempts))
            # # Get a random robot pose to start the IK solver iterative process
            # # We attempt from max_attempt different initial random poses
            # sample_fn = get_sample_fn(robot_id, arm_joint_ids)
            # sample = np.array(sample_fn())
            # # Set the pose of the robot there
            # set_joint_positions(robot_id, arm_joint_ids, sample)

            it = 0
            # Query IK, set the pose to the solution, check if it is good enough repeat if not
            while it < max_iter:

                joint_poses = p.calculateInverseKinematics(
                    robot_id,
                    eef_link_id,
                    target_pos,
                    lowerLimits=self.arm_min_limits,
                    upperLimits=self.arm_max_limits,
                    jointRanges=self.arm_joint_range,
                    restPoses=self.arm_rest_position,
                    jointDamping=self.arm_joint_damping,
                )
                joint_poses = np.array(joint_poses)[self.robot_arm_indices]

                set_joint_positions(robot_id, self.arm_joint_ids, joint_poses)

                dist = l2_distance(self.object.get_eef_position(), target_pos)
                if dist < threshold:
                    solution_found = True
                    break
                logging.debug("Dist: " + str(dist))
                it += 1

            if solution_found:
                print("Solution found at iter: " + str(it) + ", residual: " +
                      str(dist))
                break
            else:
                print("Attempt failed. Retry")
                joint_poses = None

        restoreState(state_id)
        p.removeState(state_id)
        return joint_poses