import math
from utils import quat2euler, normalize_radians
import numpy as np
from numpy.linalg import inv
import pybullet as p
from igibson import object_states

class MotionControllerHuman():

    def __init__(self):
        self.MAX_LIN_VEL = 13.3
        self.MAX_ANG_VEL = 12.28
        self.dt = 0.05
        self.num_dt_to_predict = 100
        self.rotate = False
        self.rotate_angle = 0.0
        self.arrived_hand_step = 0
        self.reset_hand_position = None
        self.reset_right_hand_orientation = [-2.908, 0.229, 0]
        self.counters = [0, 0]
        self.grasping_delay = 40

    def step(self, human, robot, final_ori, path, is_new_end):
        x, y, z = human.get_position()
        qx, qy, qz, qw = human.get_orientation()
        _, _, theta = quat2euler(qx, qy, qz, qw)
        theta = normalize_radians(theta)
        vel = None
        arrived = False
        if len(path) > 0:
            end = path[-1]
            if len(path) > 1:
                next_loc = path[1]
            else:
                next_loc = end

            action = np.zeros((28,))
            # If in grid square where the end location is
            if math.dist([x,y], end) < 0.05:
                vel = self.find_velocities_rotate(theta, final_ori)
                if vel[0] == 0 and vel[1] == 0:
                    arrived = True
            else:
                possible_vels = self.get_possible_velocities()
                vel = self.find_best_velocities(x, y, theta, possible_vels, next_loc, robot, human)

            # for new end goal, just rotate
            if is_new_end:
                self.rotate = True
                x_diff = next_loc[0] - x
                y_diff = next_loc[1] - y
                self.rotate_angle = math.atan2(y_diff, x_diff)

            if self.rotate:
                vel = self.find_velocities_rotate(theta, self.rotate_angle)
                if vel[0] == 0 and vel[1] == 0:
                    self.rotate = False

            action = self.action(vel[0]/self.MAX_LIN_VEL, vel[1]/self.MAX_ANG_VEL, 0, 0, 0, 0)
            human.apply_action(action)

        return arrived

    # def pick(self, human, loc):
    #     if self.counters[0] < 50:
    #         forward_action = self.action(0.01, 0, 0, 0, 0, 0)
    #         human.apply_action(forward_action)
    #         self.counters[0] += 1
    #         return False
    #     else:
    #         pick_action = self.action(0, 0, 0, 0, 0, 1.0)
    #         done = self.move_hand(human, loc, pick_action)
    #         self.counters[0] = 0 if done else self.counters[0]
    #         return done

    # def drop(self, human, loc):
    #     if self.counters[0] < 50:
    #         forward_action = self.action(0.01, 0, 0, 0, 0, 0)
    #         human.apply_action(forward_action)
    #         self.counters[0] += 1
    #         return False
    #     else:
    #         pick_action = self.action(0, 0, 0, 0, 0, -1.0)
    #         done = self.move_hand(human, loc, pick_action)
    #         self.counters[0] = 0 if done else self.counters[0]
    #         return done
       
    def pick(self, human, loc, offset=[0, 0, 0]):
        translated_loc = self.translate_loc(human, loc, offset)
        pick_action = self.action(0, 0, 0, 0, 0, 1.0)
        return self.move_hand(human, translated_loc, pick_action)
    
    def drop(self, human, loc, offset=[0, 0, 0]):
        translated_loc = self.translate_loc(human, loc, offset)
        pick_action = self.action(0, 0, 0, 0, 0, -1.0)
        return self.move_hand(human, translated_loc, pick_action)
    
    def move_hand(self, human, loc, mid_action):
        right_hand = human._parts["right_hand"]
        position = right_hand.get_position()
        loc = self.reset_hand_position if self.arrived_hand_step == 3 else loc

        x_diff = loc[0] - position[0]
        y_diff = loc[1] - position[1]
        z_diff = loc[2] - position[2]
        norm_val =  math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

        x_diff = x_diff / (norm_val * 100)
        y_diff = y_diff / (norm_val * 100)
        z_diff = z_diff / (norm_val * 100)

        rotated_basis = self.get_rotated_basis(human)
        diffs = np.array([x_diff, y_diff, z_diff])
        diff_new_basis = np.matmul(inv(rotated_basis), diffs)

        x_diff = diff_new_basis[0]
        y_diff = diff_new_basis[1]
        z_diff = diff_new_basis[2]

        if self.arrived_hand_step == 0:
            self.reset_hand_position = position
            # self.original_right_hand_orientation = human._parts["right_hand"].get_orientation()
            # ori = self.original_right_hand_orientation
            # print(quat2euler(ori[0], ori[1], ori[2], ori[3]))
            self.arrived_hand_step = 1
        # Go to location
        elif self.arrived_hand_step == 1:
            distance = .01 if mid_action[26] == -1 else .01
            if math.dist(loc, position) > distance:
                action = self.action(0, 0, x_diff, y_diff, z_diff, 0)
                human.apply_action(action)
            else:
                self.arrived_hand_step = 2
            # print(math.dist(loc, position))
            # print("reach")
        # Middle action
        elif self.arrived_hand_step == 2:
            human.apply_action(mid_action)
            if self.counters[1] > self.grasping_delay:
                self.arrived_hand_step = 3
                self.counters[1] = 0
            self.counters[1] += 1
            # print("mid")
        # Return to location
        elif self.arrived_hand_step == 3:
            if math.dist(loc, position) > 0.01:
                action = self.action(0, 0, x_diff, y_diff, z_diff, 0)
                human.apply_action(action)
            else:
                orientation = human.get_orientation()
                x, y, z = quat2euler(orientation[0], orientation[1], orientation[2], orientation[3])
                z_theta = normalize_radians(z) - math.pi/2
                self.reset_right_hand_orientation[2] = z_theta
                self.original_right_hand_orientation = human._parts["right_hand"].get_orientation()

                human._parts["right_hand"].set_position(loc)
                human._parts["right_hand"].set_orientation(p.getQuaternionFromEuler(self.reset_right_hand_orientation))

                self.arrived_hand_step = 0
                return True
        return False

    def action(self, forward, turn, right_hand_x, right_hand_y, right_hand_z, right_hand_grip):
        # 0 - Linear velocity
        # 5 - Angular velocity
        # 20 - Right hand y velocity
        # 21 - Right hand x velocity, so reversed in this function (left - positive, right - negative)
        # 22 - Right hand z velocity
        action = np.zeros((28,))
        action[0] = forward
        action[5] = turn
        action[20] = right_hand_y
        action[21] = -right_hand_x
        action[22] = right_hand_z
        action[26] = right_hand_grip
        return action

    def find_velocities_rotate(self, theta, final_ori):
        theta_difference = self.angle_difference(theta, final_ori)
        if theta_difference < -0.1:
            vel = (0, -self.MAX_ANG_VEL/15)
        elif theta_difference > 0.1:
            vel = (0, self.MAX_ANG_VEL/15)
        else:
            vel = (0, 0)
        return vel

    def find_best_velocities(self, x, y, theta, possible_vels, destination, robot, human):
        robot_x, robot_y, robot_z = robot.get_position()
        selected_vel = (0, 0)

        path = None
        min_dist = 100000
        # print("************")
        id = 0
        for vel in possible_vels:
            positions = self.predict_path(x, y, theta, vel[0], vel[1])
            # self.markers[id].set_position([positions[-1][0], positions[-1][1], 0.5])
            # p.addUserDebugLine([x, y, 1.0], [positions[-1][0], positions[-1][1], 1.0], lifeTime=0.2)
            for idx, pos in enumerate(positions):
                # p.addUserDebugLine([pos[0], pos[1], 1.0], [pos[0], pos[1] + 0.01, 1.0], lifeTime=20.0)
                pos_no_theta = [pos[0], pos[1]]
                dist = math.dist(pos_no_theta, destination)
                # robot_dist = math.dist(pos_no_theta, [robot_x, robot_y])
                # robot_in_FOV_human = robot.get_body_ids()[0] in human.states[object_states.ObjectsInFOVOfRobot].get_value()
                # if robot_dist < 0.5 and robot_in_FOV_human:
                #     break
                if dist < min_dist:
                    path = positions[0:idx]
                    min_dist = dist
                    selected_vel = vel
            id += 1

        # for i in range(0, len(path), 3):
        #     p.addUserDebugLine([path[i][0], path[i][1], 1.0], [path[i][0], path[i][1] + 0.01, 1.0], [1,0,0], lifeTime=1.0)
        if selected_vel[0] != 0:
            divisor = abs(selected_vel[0]) * 3
            selected_vel = (selected_vel[0]/divisor, selected_vel[1]/divisor)
        return selected_vel

    def predict_path(self, x, y, theta, vl, va):
        positions = []
        for step in range(self.num_dt_to_predict):
            x_predict = None
            y_predict = None
            theta_predict = None
            t = self.dt * step
            if (va == 0):
                x_predict = x + vl * t * math.cos(theta)
                y_predict = y + vl * t * math.sin(theta)
                theta_predict = theta
            else:
                x_predict = vl/va * (math.sin(theta + va * t) - math.sin(theta)) + x
                y_predict = vl/va * (-math.cos(theta + va * t) + math.cos(theta)) + y
                theta_predict = theta + va * t
            positions.append((x_predict, y_predict, theta_predict))

        return positions

    def get_possible_velocities(self):
        possible_vels = []
        # Velocities that are limited by acceleration and min/max velocities
        vl_possible_vels = np.linspace(0, self.MAX_LIN_VEL/20, 7)
        va_possible_vels = np.linspace(-self.MAX_ANG_VEL/20, self.MAX_ANG_VEL/20, 7)

        for vl in vl_possible_vels:
            for va in va_possible_vels:
                possible_vels.append((vl, va))

        return possible_vels

    def angle_difference(self, theta, optimal_heading):
        optimal_heading = normalize_radians(optimal_heading)
        theta = normalize_radians(theta)
        theta_1 = abs(optimal_heading - theta)
        theta_2 = 2 * math.pi - theta_1
        min_theta = min(theta_1, theta_2)
        temp = normalize_radians(theta + min_theta)
        if temp > optimal_heading - 0.05 and temp < optimal_heading + 0.05:
            return min_theta
        else:
            return -min_theta
        
    def get_rotated_basis(self, human):
        orientation = human.get_orientation()
        x, y, z = quat2euler(orientation[0], orientation[1], orientation[2], orientation[3])
        z_theta = normalize_radians(z) - math.pi/2
        regular_basis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
        rotation_matrix = np.array([
            [math.cos(z_theta), -math.sin(z_theta), 0],
            [math.sin(z_theta), math.cos(z_theta), 0],
            [0, 0, 1]])
        rotated_basis = np.matmul(rotation_matrix, regular_basis)
        return rotated_basis
    
    def translate_loc(self, human, loc, offset):
        rotated_basis = self.get_rotated_basis(human)
        offset_scaling = np.array([
            [offset[0], 0, 0],
            [0, offset[1], 0],
            [0, 0, offset[2]]    
        ])
        scaled_rotated_basis = np.matmul(rotated_basis, offset_scaling)
        translated_loc = np.matmul(scaled_rotated_basis, np.array([1, 1, 1]).transpose()).transpose()
        translated_loc = translated_loc + np.array(loc)
        return translated_loc