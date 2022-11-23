from lsi_3d.planners.two_agent_astar import run_astar_two_agent, single_agent_astar


class AStarMotionPlanner(object):
    def __init__(self, map) -> None:
        self.map = map

    def compute_motion_plan(self, starts, goals, avoid_path = None, radius = None):
        print(f'Computing plan from {starts} to {goals}')
        r1,c1,d1 = starts[0]
        r2,c2,d2 = starts[1]
        #starts = (r1,c1,MLA.to_string(d1),r2,c2,MLA.to_string(d2))
        starts = (r1,c1,d1,r2,c2,d2)

        #goals = convert_mla_state_to_string(goals)

        return run_astar_two_agent(self.map, starts, goals, avoid_path, radius)

    def compute_single_agent_astar_path(self, start, goal):
        r1,c1,d1 = start
        return single_agent_astar(self.map, (r1,c1,d1), goal)

    