from src.environment.kitchen import Kitchen
from src.planners.two_agent_astar import run_astar_two_agent, single_agent_astar


class AStarMotionPlanner(object):
    def __init__(self, kitchen: Kitchen) -> None:
        self.map = kitchen.grid
        self.precompute_dists(kitchen)
        self.dist_between = {}
        # TODO: Add Cache

    def precompute_dists(self, kitchen: Kitchen):
        self.dist_between = {}

        for s_t, s_l in kitchen.tile_location.items():
            for e_t, e_l in kitchen.tile_location.items():
                if s_t is not e_t:
                    s_l = kitchen.nearest_empty_tile(s_l)
                    path = self.compute_motion_plan([s_l], [e_l])
                    self.dist_between[f'{s_t}_{e_t}'] = len(path)

        # also calculate distances to/from center
        for s_t, s_l in kitchen.tile_location.items():
            s_l = kitchen.nearest_empty_tile(s_l)
            center_loc = kitchen.get_center()
            path = self.compute_motion_plan([s_l], [center_loc])
            self.dist_between[f'{s_t}_C'] = len(path)

        self.dist_between['Drop'] = 0

        return

    def compute_motion_plan(self, starts, goals, avoid_path=None, radius=None):
        print(f'Computing plan from {starts} to {goals}')

        if len(starts) == 1:
            return self.compute_single_agent_astar_path(starts[0], goals[0])
        return run_astar_two_agent(self.map, starts, goals, avoid_path, radius)[1]

    def compute_single_agent_astar_path(self, start, goal, end_facing=None):
        if len(start) == 2:
            x, y = start
            start = x, y, 'N'

        if len(goal) == 3:
            x, y, f = goal
            goal = x, y
            end_facing = f

        r1, c1, d1 = start
        return single_agent_astar(self.map, (r1, c1, d1), goal, end_facing=end_facing)

    def min_cost_to_feature(self, start, feature_locations):
        """Finds shortest distance to all features

        Args:
            start (_type_): _description_
            feature_locations (_type_): _description_
        """
        min_dist = None
        min_feature_loc = None

        for location in feature_locations:
            path = self.compute_single_agent_astar_path(start, location)
            dist = len(path) + 1
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_feature_loc = location

        return min_dist, min_feature_loc
