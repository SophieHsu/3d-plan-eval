from lsi_3d.planners.two_agent_astar import run_astar_two_agent


class AStarMotionPlanner(object):
    def __init__(self, map) -> None:
        self.map = map

    def compute_motion_plan(self, starts, goals):
        return run_astar_two_agent(self.map, starts, goals)