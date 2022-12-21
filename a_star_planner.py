import copy
import math
from utils import quat2euler
import numpy as np
from igibson.objects.visual_marker import VisualMarker
import pybullet as p

class AStarPlanner():

    def __init__(self, robot, env, occupancy_grid):
        self.robot = robot
        self.env = env
        self.occupancy_grid = copy.deepcopy(occupancy_grid)
        self.markers = []
        self.initialize_markers()

    def initialize_markers(self):
        for i in range (30):
            marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06)
            self.env.simulator.import_object(marker)
            marker.set_position([0, 0, 1])
            self.markers.append(marker)

    def step(self, end):
        x, y, z = self.robot.get_position()
        x_grid, y_grid = self.real_to_grid_coord(x, y)
        path = self.find_path((x_grid, y_grid), end)
        next_loc = path[1]
        x_next, y_next = self.grid_to_real_coord(next_loc[0], next_loc[1])
        qx, qy, qz, qw = self.robot.get_orientation()
        _, _, theta = quat2euler(qx, qy, qz, qw)
        theta_difference = self.angle_difference(theta, x, y, x_next, y_next)
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

    def real_to_grid_coord(self, x, y):
        return math.floor(x), math.floor(y)

    def grid_to_real_coord(self, x, y):
        return x + 0.5, y + 0.5

    # Run A Star algorithm
    def find_path(self, start, end):
        open = []
        closed = []
        start_node = AStarNode(loc=start, g_cost=0)
        end_node = AStarNode(end)
        self.set_f_cost(start_node, end_node)
        open.append(start_node)
        while True:
            current = min(open, key=lambda node: node.f_cost)
            open.remove(current)
            closed.append(current)

            if current == end_node:
                end_node = current
                break
            
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor in closed:
                    continue

                self.set_f_cost(neighbor, end_node)
                if neighbor in open:
                    idx = open.index(neighbor)
                    neighbor_in_open = open[idx]
                    if neighbor_in_open.f_cost > neighbor.f_cost:
                        open[idx] = neighbor
                else:
                    open.append(neighbor)

        path = [end_node.loc]
        while end_node.parent is not None:
            end_node = end_node.parent
            path.insert(0, end_node.loc)
        
        for i in range(len(self.markers)):
            if i < len(path):
                loc = path[i]
                grid_x, grid_y = self.grid_to_real_coord(loc[0], loc[1])
                marker = self.markers[i]
                marker.set_position([grid_x, grid_y, 1])
            else:
                marker = self.markers[i]
                marker.set_position([0, 0, 1])

        return path

    def get_neighbors(self, node):
        loc = node.loc
        x = loc[0]
        y = loc[1]
        # neighbor_locs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        neighbor_locs = [ (-1, 0), (0, -1), (0, 1), (1, 0)]
        neighbors = []
        for n in neighbor_locs:
            x_neighbor = x + n[0]
            y_neighbor = y + n[1]

            if 0 <= x_neighbor < 8 and 0 <= y_neighbor < 8:
                if self.occupancy_grid[x_neighbor][y_neighbor] == 'X':
                    g_cost = None
                    if abs(x) == 1 and abs(y) == 1:
                        g_cost = node.g_cost + math.sqrt(2)
                    else:
                        g_cost = node.g_cost + 1
                    neighbor_node = AStarNode((x_neighbor, y_neighbor), node, g_cost)
                    neighbors.append(neighbor_node)
        return neighbors

    def set_f_cost(self, node, end_node):
        start = node.loc
        end = end_node.loc

        dx = abs(start[0] - end[0])
        dy = abs(start[1] - end[1])
        heuristic = dx + dy + (math.sqrt(2) - 2) * min(dx, dy)
        node.f_cost = node.g_cost + heuristic

class AStarNode():

    def __init__(self, loc, parent=None, g_cost=None, f_cost=None):
        self.loc = loc
        self.parent = parent
        self.g_cost = g_cost
        self.f_cost = f_cost

    def __eq__(self, other):
        return self.loc == other.loc

        