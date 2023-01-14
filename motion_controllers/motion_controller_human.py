import math
from utils import quat2euler
import numpy as np
from igibson.objects.visual_marker import VisualMarker
import pybullet as p

class MotionControllerHuman():

    def __init__(self):
        self.MAX_LIN_VEL = 13.3
        self.MAX_ANG_VEL = 12.28
        self.dt = 0.05
        self.num_dt_to_predict = 100

    def step(self, human, robot, final_ori, path):
        x, y, z = human.get_position()
        qx, qy, qz, qw = human.get_orientation()
        _, _, theta = quat2euler(qx, qy, qz, qw)
        theta = self.normalize_radians(theta)
        if len(path) > 0:
            end = path[-1]
            if len(path) > 1:
                next_loc = path[1]
            else:
                next_loc = end

            action = np.zeros((28,))
            # If in grid square where the end location is
            if math.dist([x,y], end) < 0.05:
                theta_difference = self.angle_difference(theta, final_ori)
                if theta_difference < -0.2:
                    vel = (0, -self.MAX_ANG_VEL/15)
                elif theta_difference > 0.2:
                    vel = (0, self.MAX_ANG_VEL/15)
                else:
                    vel = (0, 0)
            else:
                possible_vels = self.get_possible_velocities()
                vel = self.find_best_velocities(x, y, theta, possible_vels, next_loc, robot)

            action[0] = vel[0]/self.MAX_LIN_VEL
            action[5] = vel[1]/self.MAX_ANG_VEL
            human.apply_action(action)

    def find_best_velocities(self, x, y, theta, possible_vels, destination, robot):
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
                robot_dist = math.dist(pos_no_theta, [robot_x, robot_y])
                if robot_dist < 0.5:
                    break
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

    def normalize_radians(self, rad):
        # Convert radians to value between 0 and 2 * pi
        rad = rad % (2 * math.pi)
        if rad < 0:
            rad = rad + 2 * math.pi
        return rad

    def angle_difference(self, theta, optimal_heading):
        optimal_heading = self.normalize_radians(optimal_heading)
        theta = self.normalize_radians(theta)
        theta_1 = abs(optimal_heading - theta)
        theta_2 = 2 * math.pi - theta_1
        min_theta = min(theta_1, theta_2)
        temp = self.normalize_radians(theta + min_theta)
        if temp > optimal_heading - 0.05 and temp < optimal_heading + 0.05:
            return min_theta
        else:
            return -min_theta