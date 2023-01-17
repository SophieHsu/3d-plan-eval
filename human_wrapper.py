import copy
from utils import real_to_grid_coord
import numpy as np


class HumanWrapper():

    def __init__(self, human, planner, motion_controller, occupancy_grid):
        self.human = human
        #self.robot = robot
        self.planner = planner
        self.motion_controller = motion_controller
        self.occupancy_grid = copy.deepcopy(occupancy_grid)

    def step():
        highlevel
        self._step()

    def _step(self, end, final_ori):
        self.update_occupancy_grid()
        x, y, z = self.human.get_position()
        path = self.planner.find_path((x,y), end, self.occupancy_grid)
        self.motion_controller.step(self.human, self.robot, final_ori, path)

    def update_occupancy_grid(self):
        x, y, z = self.robot.get_position()
        loc = real_to_grid_coord((x,y))
        old_loc = None
        for i in range(len(self.occupancy_grid)):
            for j in range(len(self.occupancy_grid[0])):
                if self.occupancy_grid[i][j] == 'R':
                    old_loc = (i, j)
                    self.occupancy_grid[i][j] = 'X'
        self.occupancy_grid[loc[0]][loc[1]] = 'R'
        # print(loc, old_loc)
        if old_loc != loc:
            print(np.matrix(self.occupancy_grid))
            print("----------------------------")
    
    def get_position():
        return self.human.get_position()
        