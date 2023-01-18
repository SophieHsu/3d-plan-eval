import copy
import math
import numpy as np
from utils import quat2euler, real_to_grid_coord


class HumanAgent():

    def __init__(self, human, planner, motion_controller, occupancy_grid):
        self.human = human
        self.robot = None
        self.planner = planner
        self.motion_controller = motion_controller
        self.occupancy_grid = copy.deepcopy(occupancy_grid)
        self.vision_range = math.pi

    def set_robot(self, robot):
        self.robot = robot

    def step(self):
        # High level planner determines where end point is. Right now this is hardcoded
        end = (2, -1)
        final_ori = 1.57
        self._step(end, final_ori)

    def _step(self, end, final_ori):
        self.update_occupancy_grid()
        x, y, z = self.human.get_position()
        path = self.planner.find_path((x,y), end, self.occupancy_grid)
        self.motion_controller.step(self.human, self.robot, final_ori, path)

    def update_occupancy_grid(self):
        if self.is_observable(self.robot):
            x, y, z = self.robot.get_position()
            loc = real_to_grid_coord((x,y))
            for i in range(len(self.occupancy_grid)):
                for j in range(len(self.occupancy_grid[0])):
                    if self.occupancy_grid[i][j] == 'R':
                        self.occupancy_grid[i][j] = 'X'
            self.occupancy_grid[loc[0]][loc[1]] = 'R'

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
        line_of_sight = True
        grid_spaces = set()
        slope_x = object_x - human_x
        slope_y = object_y - human_y
        dx = 0 if slope_x == 0 else slope_x / (abs(slope_x) * 10)
        dy = slope_y / (abs(slope_y) * 10) if slope_x == 0 else slope_y / (abs(slope_x) * 10)
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
            if self.occupancy_grid[grid[0]][grid[1]] != 'X' and self.occupancy_grid[grid[0]][grid[1]] != 'R':
                line_of_sight = False
        
        return line_of_sight and angle <= self.vision_range/2
            
    
    def get_position(self):
        return self.human.get_position()
        