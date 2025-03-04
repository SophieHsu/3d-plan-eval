"""
Module interfaces between lsi directives and low level iGibson robot behavior
"""
import logging
import math

import numpy as np
import pybullet as p
from numpy.linalg import inv
from transforms3d.euler import euler2quat

from igibson import object_states
from igibson.external.pybullet_tools.utils import (
    get_max_limits,
    get_min_limits,
    joints_from_names,
    set_joint_positions,
)
from src.environment.actions import ACTION_COMMANDS
from igibson.utils.utils import l2_distance, restoreState
from igibson.utils.utils import quatToXYZW
from src.utils.constants import DIRE2POSDIFF, TARGET_ORNS
from src.utils.functions import quat2euler
from src.utils.helpers import grid_to_real_coord, quat2euler, normalize_radians

ONE_STEP = 0.02


class iGibsonAgent:
    """
    Controls low level agent agent actions acting as an interface between medium level turn and forward actions
    and continuous iGibson environment
    """

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
        self.start = start + (direction,)
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
        if current_action is None:
            return True

        ready_for_next_action = False
        x, y, z = self.object.get_position()

        if current_action == 'F' and \
                self.forward_distance(x, y, self.target_x, self.target_y, self.direction) < ONE_STEP * 1.5:
            ready_for_next_action = True
        elif current_action in 'NESW' and \
                self.turn_distance(self.get_current_orn_z(), TARGET_ORNS[self.target_direction]) < ONE_STEP * 5:
            self.direction = current_action
            ready_for_next_action = True
        elif current_action == 'I':
            if self.interact_step_index == -1:
                self.interact_step_index = 0
                return True
            else:
                return False
        elif current_action is None:  # None when first action
            return True

        return ready_for_next_action

    def prepare_for_next_action(self, current_pos, next_action):
        r, c, f = current_pos
        x, y = grid_to_real_coord((r, c))
        if self.target_x is None or self.target_y is None:

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
        if action is None:
            return

        if action in 'NESW':
            self.agent_turn_one_step(env, action)
        elif action == 'F':
            self.agent_forward_one_step(env)

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
            distance_to_target = self.forward_distance(cur_x, cur_y, self.target_x, self.target_y, self.direction)

            if distance_to_target < 0.3:
                action[0] *= 0.8
            elif distance_to_target < 0.1:
                action[0] *= 0.7
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
        if cur_orn_z < target_orn_z:
            action[1] = -angle_delta
        else:
            action[1] = angle_delta

        if ((cur_orn_z - target_orn_z) / (action[1] / abs(action[1]))) > 4:
            action[1] = -action[1]
        if abs(target_orn_z - cur_orn_z) < 0.15:
            action[1] /= 2
        elif abs(target_orn_z - cur_orn_z) < 0.08:
            action[1] /= 4
        self.object.apply_action(action)
        env.simulator.step()

    def agent_set_pos_orn(self, x, y, dir):
        x, y, z, w = self.object.get_orientation()
        x, y, z = quat2euler(x, y, z, w)
        target_orn_z = TARGET_ORNS[dir]

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
            [x, y, 0.6],
            quatToXYZW(euler2quat(x, y, z), "wxyz"))

    def agent_turn_one_step(self, env, action):
        if self.name == "human_sim":
            x, y, z, w = self.object.get_orientation()
            x, y, z = quat2euler(x, y, z, w)
            target_orn_z = TARGET_ORNS[self.target_direction]

            pos = z - target_orn_z
            neg = target_orn_z - z
            if pos < 0:
                pos += 3.1415926 * 2
            elif neg < 0:
                neg += 3.1415926 * 2
            if pos < neg:
                z -= ONE_STEP * 4
            else:
                z += ONE_STEP * 4
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
            if cur_orn_z < target_orn_z:
                action[1] = -0.2
            else:
                action[1] = 0.2

            if ((cur_orn_z - target_orn_z) / (action[1] / abs(action[1]))) > 4:  # > 3.14
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
        if action_object == ('pickup', 'plate'):
            if self.object_position is None:
                self.object_position = self.object.get_position().copy()

            # check if on counter
            done, in_hand = self.pick(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_plate(
                    agent_pos=self.object.get_eef_position()),
                name='plate',
                offset=[0, 0.5, 1.3])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None
        if action_object == ('pickup', 'hot_plate'):
            if self.object_position is None:
                self.object_position = self.object.get_position().copy()

            # check if on counter
            done, in_hand = self.pick(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_plate_in_sink(
                    agent_pos=self.object.get_eef_position()),
                name='hot_plate',
                offset=[0, 0.5, 1.3])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None
        if action == "pickup" and object == "meat":
            if self.object_position is None:
                self.object_position = self.object.get_position().copy()

            # check if on counter
            done, in_hand = self.pick(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_meat(
                    agent_pos=self.object.get_eef_position()),
                name='meat',
                offset=[0, 0.5, 1.3])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None

            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None
        if action == "pickup" and object == "onion":
            if self.object_position is None:
                self.object_position = self.object.get_position().copy()

            # check if on counter
            done, in_hand = self.pick(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_green_onion(
                    agent_pos=self.object.get_eef_position(), chopped=False),
                name='onion',
                offset=[0, 0.5, 1.3])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None
        elif action == "drop" and object == "onion":
            self.object_position = tracking_env.get_closest_chopping_board(
                agent_pos=self.object.get_position()).get_position()
            done, in_hand = self.drop(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_green_onion(
                    agent_pos=self.object.get_eef_position()),
                name='onion',
                offset=[0, 0, HEIGHT_OFFSET])

            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None
        elif action == "drop" and object == "meat":
            self.object_position = tracking_env.get_closest_pan(agent_pos=self.object.get_position()).get_position()
            done, in_hand = self.drop(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_meat(
                    agent_pos=self.object.get_eef_position()),
                name='meat',
                offset=[0, 0, HEIGHT_OFFSET])

            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None
        elif action == "drop" and object == "item":
            self.target_object = tracking_env.get_closest_counter(agent_pos=self.object.get_position())
            self.object_position = self.target_object.get_position()
            done, in_hand = self.drop(
                self.object_position,
                tracking_env,
                self.target_object,
                name=None,
                offset=[0, 0, 1.1])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None
        elif action == 'chop' and object == 'onion':
            self.target_object = tracking_env.get_closest_green_onion(agent_pos=self.object.get_position())
            self.target_object.states[object_states.Sliced].set_value(True)
            self.interact_step_index = -1
            self.object_position = None
        elif action == 'pickup' and object == 'garnish':
            onion = tracking_env.get_closest_chopped_onion(agent_pos=self.object.get_position())

            offset = 1
            x, y, z = self.object.get_eef_position()
            for obj in onion.objects:
                obj.set_position([x, y, z + offset * 0.1])
                offset += 1

            tracking_env.set_in_robot_hand('onion', onion)
            self.interact_step_index = -1
            self.object_position = None
            tracking_env.kitchen.robot_carrying_dish = True
            tracking_env.kitchen.robot_carrying_steak = False

            self.target_object = tracking_env.get_closest_chopped_onion(
                agent_pos=self.object.get_eef_position())
            self.object_position = tracking_env.get_real_position(self.target_object)
            done, in_hand = self.pick(
                self.object_position,
                tracking_env,
                self.target_object,
                name='onion',
                offset=[0, 0.5, 1.3])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None

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

        elif action == "drop" and object == "plate":
            self.object_position = tracking_env.get_closest_sink(agent_pos=self.object.get_position()).get_position()
            done, in_hand = self.drop(
                self.object_position,
                tracking_env,
                tracking_env.get_closest_sink(
                    agent_pos=self.object.get_position()),
                name='onion',
                offset=[0, 0, 0.7])
            if done or in_hand:
                self.interact_step_index = -1
                self.object_position = None
        elif action_object == ('heat', 'plate'):
            plate = tracking_env.get_closest_plate_in_sink(self.object.get_position())
            if plate is not None:
                # immediately done
                self.interact_step_index = -1
                tracking_env.kitchen.execute_action(ACTION_COMMANDS.CLEAN, plate, name='hot_plate')
                self.object_position = None

        elif action == "pickup" and object == "dish":
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
        elif action == "deliver" and (object == "soup" or object == "dish"):
            agent_location = self.object.get_position()
            locs = [grid_to_real_coord(x) for x in tracking_env.kitchen.where_grid_is('T')]
            plate = tracking_env.kitchen.robot_stash_dish
            closest = sorted(locs, key=lambda location: math.dist(location[0:2], agent_location[0:2]))[0]
            onion = tracking_env.get_closest_chopped_onion(tracking_env.kitchen.robot_stash_dish.get_position())
            steak = tracking_env.get_closest_steak(tracking_env.kitchen.robot_stash_dish.get_position())

            tracking_env.kitchen.overcooked_object_states[plate]['state'] = 'delivered'
            try:
                tracking_env.kitchen.overcooked_object_states.pop(onion)
                tracking_env.kitchen.overcooked_object_states.pop(steak)
            except:
                print('')
            tracking_env.kitchen.robot_stash_dish.set_position([closest[0], closest[1], 1.15])
            offset = 1

            x, y, z = tracking_env.kitchen.robot_stash_dish.get_position()
            steak.set_position([x, y, z + offset * 0.05])
            offset += 1

            for on in onion.objects:
                offset += 1

            tracking_env.kitchen.robot_carrying_dish = False

            self.interact_step_index = -1
            self.object_position = None

        elif action == "pickup" and object == "steak":
            steak = tracking_env.get_closest_steak(agent_pos=self.object.get_position())
            tracking_env.set_in_robot_hand('steak', steak)
            tracking_env.kitchen.robot_carrying_steak = True

            self.interact_step_index = -1
            self.object_position = None

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

        # Indices of the joints of the arm in the vectors returned by IK and motion planning (excluding wheels, head,
        # fingers)
        self.robot_arm_indices = [
            self.robot_joint_names.index(arm_joint_name)
            for arm_joint_name in self.arm_joints_names
        ]

        # PyBullet ids of the joints corresponding to the joints of the arm
        self.arm_joint_ids = joints_from_names(self.robot_id, self.arm_joints_names)
        self.all_joint_ids = joints_from_names(self.robot_id, self.robot_joint_names)
        self.arm_max_limits = get_max_limits(self.robot_id, self.all_joint_ids)
        self.arm_min_limits = get_min_limits(self.robot_id, self.all_joint_ids)
        self.arm_min_limits[2] = 0.35
        self.arm_rest_position = self.robot_default_joint_positions
        self.arm_joint_range = list(np.array(self.arm_max_limits) - np.array(self.arm_min_limits))
        self.arm_joint_range = [item + 1 for item in self.arm_joint_range]
        self.arm_joint_damping = [0.1 for _ in self.arm_joint_range]

    def pick(self, loc, tracking_env, target_obj, name, offset=[0, 0, 0], obj_name=None):
        if obj_name is not None:
            offset = self.arm_eef_to_obj_dict[obj_name]
        translated_loc = self.translate_loc(loc, offset)
        gripper_action = [-0.01, -0.01]
        return self.move_hand(translated_loc, tracking_env, target_obj, name, gripper_action)

    def drop(self, loc, tracking_env, target_obj, name=None, offset=[0, 0, 0], obj_name=None):
        translated_loc = self.translate_loc(loc, offset)
        gripper_action = [0.01, 0.01]
        return self.move_hand(translated_loc, tracking_env, None, name,
                              gripper_action)

    def move_hand(self, loc, tracking_env, target_obj, name, gripper_action):
        position = self.object.get_eef_position()
        if gripper_action != self.prev_gripper_action:
            self.prev_gripper_action = gripper_action
            self.arrived_hand_step = 0

        x_diff = loc[0] - position[0]
        y_diff = loc[1] - position[1]
        z_diff = loc[2] - position[2]
        norm_val = math.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)

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
            self.arrived_hand_step = 1
        # Go to location
        elif self.arrived_hand_step == 1:
            distance = .3
            if math.dist(loc, position) > distance:
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
                    set_joint_positions(self.robot_id, self.arm_joint_ids, joint_pos)
        elif self.arrived_hand_step == 2:
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

            set_joint_positions(self.robot_id, self.all_joint_ids[-2:], gripper_new_pos)

            if self.counters[1] > self.grasping_delay:
                if target_obj is not None:
                    if target_obj.name == 'bowl' or target_obj.name == 'plate':
                        for i in tracking_env.items_in_bowl(target_obj):
                            tracking_env.set_in_robot_hand(name, i)
                    tracking_env.set_in_robot_hand(name, target_obj)
                    in_hand = True
                else:
                    for i, item in enumerate(tracking_env.kitchen.in_robot_hand[::-1]):
                        tracking_env.remove_in_robot_hand(item, pos=loc, counter=i)
                self.arrived_hand_step = 3
                self.counters[1] = 0
            self.counters[1] += 1
        # Return to location
        elif self.arrived_hand_step == 3:
            distance = 0.52
            done = False
            if math.dist(loc, position) > distance and not (
                    abs(loc - position) - 0.3 <= 0).all():
                if not self.robot_arm_move(position, x_diff, y_diff, z_diff):
                    done = True
                    self.arrived_hand_step = 0
            else:
                done = True
                self.arrived_hand_step = 0

            if done:
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
        threshold = 0.03
        max_iter = 10

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
        self.object.keep_still()
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
