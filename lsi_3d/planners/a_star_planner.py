import copy
import math
from utils import quat2euler, real_to_grid_coord, grid_to_real_coord
import numpy as np
from igibson.objects.visual_marker import VisualMarker
import pybullet as p

class AStarPlanner():

    def __init__(self, env):
        self.env = env
        self.markers = []
        # self.initialize_markers()

    def initialize_markers(self):
        for i in range (30):
            marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06)
            self.env.simulator.import_object(marker)
            marker.set_position([0, 0, 1])
            self.markers.append(marker)

    # Run A Star algorithm
    def find_path(self, start, end, occupancy_grid):
        start = real_to_grid_coord(start)
        end_grid = real_to_grid_coord(end)
        print(end_grid)
        end_grid_item = occupancy_grid[end_grid[0]][end_grid[1]]
        occupancy_grid[end_grid[0]][end_grid[1]] = "X"

        open = []
        closed = []
        start_node = AStarNode(loc=start, g_cost=0)
        end_node = AStarNode(end_grid)
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
            
            neighbors = self.get_neighbors(current, occupancy_grid)
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

        path = [end]
        while end_node.parent is not None:
            end_node = end_node.parent
            path.insert(0, grid_to_real_coord(end_node.loc))
        
        # self.draw_path(path)
        occupancy_grid[end_grid[0]][end_grid[1]] = end_grid_item
        return path

    def draw_path(self, path):
        for i in range(len(self.markers)):
            if i < len(path):
                loc = path[i]
                marker = self.markers[i]
                marker.set_position([loc[0], loc[1], 1])
            else:
                marker = self.markers[i]
                marker.set_position([0, 0, 1])

    def get_neighbor_locs(self, loc):
        relative_neighbor_locs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        # relative_neighbor_locs = [ (-1, 0), (0, -1), (0, 1), (1, 0)]
        neighbor_locs = []
        for n in relative_neighbor_locs:
            x_neighbor = loc[0] + n[0]
            y_neighbor = loc[1] + n[1]

            if 0 <= x_neighbor < 8 and 0 <= y_neighbor < 8:
                neighbor_locs.append((x_neighbor, y_neighbor))
        return neighbor_locs

    def get_neighbors(self, node, occupancy_grid):
        loc = node.loc
        x = loc[0]
        y = loc[1]
        neighbor_locs = self.get_neighbor_locs(loc)
        neighbors = []
        for n in neighbor_locs:
            x_neighbor = n[0]
            y_neighbor = n[1]
            if self.is_valid_location(loc, (x_neighbor, y_neighbor), occupancy_grid):
            # if self.occupancy_grid[x_neighbor][y_neighbor] == 'X':
                g_cost = None
                if abs(x) == 1 and abs(y) == 1:
                    g_cost = node.g_cost + math.sqrt(2)
                else:
                    g_cost = node.g_cost + 1
                neighbor_node = AStarNode((x_neighbor, y_neighbor), node, g_cost)
                neighbors.append(neighbor_node)
        return neighbors

    def is_valid_location(self, loc, neighbor_loc, occupancy_grid):
        x_diff = neighbor_loc[0] - loc[0]
        y_diff = neighbor_loc[1] - loc[1]
        neighbor_1 = (loc[0] + x_diff, loc[1])
        neighbor_2 = (loc[0], loc[1] + y_diff)
        if occupancy_grid[neighbor_loc[0]][neighbor_loc[1]] == 'X' and occupancy_grid[neighbor_1[0]][neighbor_1[1]] == 'X' and occupancy_grid[neighbor_2[0]][neighbor_2[1]] == 'X':
            return True
        return False

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

        