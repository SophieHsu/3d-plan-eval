import copy
import math
from utils import quat2euler
import numpy as np
from igibson.objects.visual_marker import VisualMarker
import pybullet as p

class AStarPlanner():

    def __init__(self, env, occupancy_grid, robot):
        self.env = env
        self.occupancy_grid = copy.deepcopy(occupancy_grid)
        self.robot = robot
        self.markers = []
        self.initialize_markers()

    def initialize_markers(self):
        for i in range (30):
            marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06)
            self.env.simulator.import_object(marker)
            marker.set_position([0, 0, 1])
            self.markers.append(marker)

    # Run A Star algorithm
    def find_path(self, start, end):
        self.update_grid()
        start = self.real_to_grid_coord(start)
        end = self.real_to_grid_coord(end)

        open = []
        closed = []
        start_node = AStarNode(loc=start, g_cost=0)
        end_node = AStarNode(end)
        self.set_f_cost(start_node, end_node)
        open.append(start_node)

        while True:
            # No path case
            if len(open) == 0:
                return []
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

        path = [self.grid_to_real_coord(end_node.loc)]
        while end_node.parent is not None:
            end_node = end_node.parent
            path.insert(0, self.grid_to_real_coord(end_node.loc))
        
        for i in range(len(self.markers)):
            if i < len(path):
                loc = path[i]
                marker = self.markers[i]
                marker.set_position([loc[0], loc[1], 1])
            else:
                marker = self.markers[i]
                marker.set_position([0, 0, 1])

        return path

    def update_grid(self):
        x, y, z = self.robot.get_position()
        loc = self.real_to_grid_coord((x,y))
        neighbor_locs = self.get_neighbor_locs(loc)
        for n in neighbor_locs:
            if self.occupancy_grid[n[0]][n[1]] == 'R':
                self.occupancy_grid[n[0]][n[1]] = 'X'
        self.occupancy_grid[loc[0]][loc[1]] = 'R'

    def get_neighbor_locs(self, loc):
        # relative_neighbor_locs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        relative_neighbor_locs = [ (-1, 0), (0, -1), (0, 1), (1, 0)]
        neighbor_locs = []
        for n in relative_neighbor_locs:
            x_neighbor = loc[0] + n[0]
            y_neighbor = loc[1] + n[1]

            if 0 <= x_neighbor < 8 and 0 <= y_neighbor < 8:
                neighbor_locs.append((x_neighbor, y_neighbor))
        return neighbor_locs

    def get_neighbors(self, node):
        loc = node.loc
        x = loc[0]
        y = loc[1]
        neighbor_locs = self.get_neighbor_locs(loc)
        neighbors = []
        for n in neighbor_locs:
            x_neighbor = n[0]
            y_neighbor = n[1]
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

    def real_to_grid_coord(self, coord):
        return (math.floor(coord[0]), math.floor(coord[1]))

    def grid_to_real_coord(self, coord):
        return (coord[0] + 0.5, coord[1] + 0.5)

class AStarNode():

    def __init__(self, loc, parent=None, g_cost=None, f_cost=None):
        self.loc = loc
        self.parent = parent
        self.g_cost = g_cost
        self.f_cost = f_cost

    def __eq__(self, other):
        return self.loc == other.loc

        