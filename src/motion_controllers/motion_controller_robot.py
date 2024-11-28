import math

import numpy as np

from src.utils.helpers import quat2euler


class MotionControllerRobot:

    def __init__(self, robot, planner, occupancy_grid):
        self.robot = robot
        self.planner = planner
        self.occupancy_grid = occupancy_grid
        self.WHEEL_RADIUS = 0.0613
        self.WHEEL_AXLE_LENGTH = 0.372
        self.MIN_WHEEL_VELOCITY = -10.4
        self.MAX_WHEEL_VELOCITY = -1 * self.MIN_WHEEL_VELOCITY
        self.MAX_LIN_VEL = 1.06662
        self.MAX_ANG_VEL = 5.7345161
        self.MAX_ACCELERATION = 400
        self.dt = 0.05
        self.num_dt_to_predict = 100

    def step(self, end, final_ori):
        x, y, z = self.robot.get_position()
        qx, qy, qz, qw = self.robot.get_orientation()
        _, _, theta = quat2euler(qx, qy, qz, qw)
        theta = self.normalize_radians(theta)
        path = self.planner.find_path((x, y), end, self.occupancy_grid)
        if len(path) > 1:
            next_loc = path[1]
        else:
            next_loc = end

        action = np.zeros((11,))
        # If in grid square where the end location is
        if math.dist([x,y], end) < 0.05:
            theta_difference = self.angle_difference(theta, final_ori)
            if theta_difference < -0.2:
                vel = (self.MIN_WHEEL_VELOCITY/5, self.MAX_WHEEL_VELOCITY/5)
            elif theta_difference > 0.2:
                vel = (self.MAX_WHEEL_VELOCITY/5, self.MIN_WHEEL_VELOCITY/5)
            else:
                vel = (0, 0)
        else:
            possible_vels = self.get_possible_velocities()
            vel = self.find_best_velocities(x, y, theta, possible_vels, next_loc)

        action_l, action_a = self.calculate_action_from_wheel_vel(vel[0], vel[1]) 
        action[0] = action_l
        action[1] = action_a
        self.robot.apply_action(action)

    def find_best_velocities(self, x, y, theta, possible_vels, destination):
        selected_vel = None

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

                if dist < min_dist:
                    path = positions[0:idx]
                    min_dist = dist
                    selected_vel = vel

                elif dist == min_dist:
                    if abs(vel[0]) + abs(vel[1]) > abs(selected_vel[0]) + abs(selected_vel[1]):
                        path = positions[0:idx]
                        selected_vel = vel
            id += 1

        return selected_vel

    def predict_path(self, x, y, theta, vl, vr):
        vl = vl * self.WHEEL_RADIUS
        vr = vr * self.WHEEL_RADIUS
        positions = []
        for step in range(self.num_dt_to_predict):
            x_predict = None
            y_predict = None
            theta_predict = None
            t = self.dt * step
            if round(vl, 3) == round(vr, 3):
                x_predict = x + vl * t * math.cos(theta)
                y_predict = y + vl * t * math.sin(theta)
                theta_predict = theta
            else:
                R = self.WHEEL_AXLE_LENGTH / 2.0 * (vr + vl) / (vr - vl)
                delta_theta = (vr - vl) * t / self.WHEEL_AXLE_LENGTH
                # Need to review this
                x_predict = x + R * (math.sin(delta_theta + theta) - math.sin(theta))
                y_predict = y - R * (math.cos(delta_theta + theta) - math.cos(theta))
                theta_predict = theta + delta_theta
            positions.append((x_predict, y_predict, theta_predict))
        return positions

    def get_possible_velocities(self):
        possible_vels = []
        # Velocities that are limited by acceleration and min/max velocities
        vl_possible_vels = np.linspace(self.MIN_WHEEL_VELOCITY, self.MAX_WHEEL_VELOCITY, 7)
        vr_possible_vels = np.linspace(self.MIN_WHEEL_VELOCITY, self.MAX_WHEEL_VELOCITY, 7)

        for vl in vl_possible_vels:
            for vr in vr_possible_vels:
                if vl + vr > 0:
                    possible_vels.append((vl, vr))

        return possible_vels
    
    def calculate_action_from_wheel_vel(self, vl, vr):
        a = (vr - vl) * self.WHEEL_RADIUS / (-2 * self.WHEEL_AXLE_LENGTH / 2)
        l = self.WHEEL_RADIUS * (vl + vr) / 2
        action_a = a / self.MAX_ANG_VEL
        action_l = l / self.MAX_LIN_VEL
        return action_l, action_a

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
        if self.normalize_radians(theta + min_theta) == optimal_heading:
            return -min_theta
        else:
            return min_theta
