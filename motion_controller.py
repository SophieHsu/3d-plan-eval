import math
from utils import quat2euler
import numpy as np

class MotionController():

    def __init__(self, robot, planner):
        self.robot = robot
        self.planner = planner

    def step(self, end):
        x, y, z = self.robot.get_position()
        path = self.planner.find_path((x, y), end)
        next_loc = path[1]
        
        qx, qy, qz, qw = self.robot.get_orientation()
        _, _, theta = quat2euler(qx, qy, qz, qw)
        theta_difference = self.angle_difference(theta, x, y, next_loc[0], next_loc[1])
        print(theta_difference)
        if abs(theta_difference) > 0.2:
            if theta_difference < 0:
                self.turn_left()
            else:
                self.turn_right()
        else:
            self.forward()

    def turn_right(self):
        action = np.zeros((11,))
        action[1] = 0.05
        self.robot.apply_action(action)

    def turn_left(self):
        action = np.zeros((11,))
        action[1] = -0.05
        self.robot.apply_action(action)

    def forward(self):
        action = np.zeros((11,))
        action[0] = 0.05
        self.robot.apply_action(action)
        
    def angle_difference(self, theta, x, y, x_dest, y_dest):
        x_dist = x_dest - x
        y_dist = y_dest - y
        optimal_heading =  np.arctan2(y_dist, x_dist)
        optimal_heading = self.normalize_radians(optimal_heading)
        theta = self.normalize_radians(theta)
        theta_1 = abs(optimal_heading - theta)
        theta_2 = 2 * math.pi - theta_1
        min_theta = min(theta_1, theta_2)
        if self.normalize_radians(theta + min_theta) == optimal_heading:
            return -min_theta
        else:
            return min_theta

    def normalize_radians(self, rad):
        # Convert radians to value between 0 and 2 * pi
        rad = rad % (2 * math.pi)
        if rad < 0:
            rad = rad + 2 * math.pi
        return rad
        