import math
from lsi_3d.environment.kitchen import Kitchen
from lsi_3d.planners.two_agent_astar import run_astar_two_agent, single_agent_astar


class AStarMotionPlanner(object):
    def __init__(self, kitchen:Kitchen) -> None:
        self.map = kitchen.grid
        self.precompute_dists(kitchen)
        # TODO: Add Cache

    def precompute_dists(self, kitchen:Kitchen):
        self.dist_between = {}

        for s_t,s_l in kitchen.tile_location.items():
            for e_t,e_l in kitchen.tile_location.items():
                if s_t is not e_t:
                    s_l = kitchen.nearest_empty_tile(s_l)
                    path = self.compute_motion_plan([s_l],[e_l])
                    self.dist_between[f'{s_t}_{e_t}'] = len(path)

        # also calculate distances to/from center
        for s_t,s_l in kitchen.tile_location.items():
            s_l = kitchen.nearest_empty_tile(s_l)
            center_loc = kitchen.get_center()
            path = self.compute_motion_plan([s_l],[center_loc])
            self.dist_between[f'{s_t}_C'] = len(path)

        self.dist_between['Drop'] = 0

        return



    def compute_motion_plan(self, starts, goals, avoid_path = None, radius = None):
        
        print(f'Computing plan from {starts} to {goals}')

        # if not isinstance(starts, list) and not len(starts) > 1:
        #     starts = [starts]

        # if not isinstance(goals, list):
        #     goals = [goals]

        if len(starts) == 1:
            return self.compute_single_agent_astar_path(starts[0], goals[0])
        # elif avoid_path == []:
        #     # path = self.compute_single_agent_astar_path(starts[1], goals[1][0:2], end_facing=goals[1][2])
        #     # if len(path) > 0:
        #     #     path.append((path[-1][0], 'I'))
        #     # else:
        #     #     path.append((None, 'I'))
        #     # return path
        #     return run_astar_two_agent(self.map, starts, goals, avoid_path, radius)[1]
            
        # r1,c1,d1 = starts[0]
        # r2,c2,d2 = starts[1]
        # #starts = (r1,c1,MLA.to_string(d1),r2,c2,MLA.to_string(d2))
        # starts = (r1,c1,d1,r2,c2,d2)

        #goals = convert_mla_state_to_string(goals)

        return run_astar_two_agent(self.map, starts, goals, avoid_path, radius)[1]

    def compute_single_agent_astar_path(self, start, goal, end_facing=None):
        if len(start) == 2:
            x,y = start
            start = x,y,'N'

        r1,c1,d1 = start
        return single_agent_astar(self.map, (r1,c1,d1), goal, end_facing=end_facing)

    def min_cost_to_feature(self, start, feature_locations):
        """Finds shortest distance to all features

        Args:
            start (_type_): _description_
            feature_locations (_type_): _description_
        """
        min = None
        min_feature_loc = None

        for location in feature_locations:
            path = self.compute_single_agent_astar_path(start, location)
            dist = len(path) + 1
            if min == None or dist < min:
                min = dist
                min_feature_loc = location

        return (min, min_feature_loc)

    