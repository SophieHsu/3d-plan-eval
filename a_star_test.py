from lsi_3d.planners.two_agent_astar import astar_avoid_path_forward_radius
from lsi_3d.config.reader import read_in_lsi_config

if __name__ == "__main__":
    grid = [['X', 'X', 'C', 'C', 'P', 'C', 'C', 'C'],
            ['B', 'X', 'X', 'X', 'X', 'X', 'X', 'C'],
            ['C', 'X', 'X', 'X', 'X', 'X', 'X', 'W'],
            ['C', 'X', 'X', 'X', 'C', 'C', 'C', 'C'],
            ['F', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
            ['C', 'X', 'T', 'X', 'X', 'X', 'X', 'X'],
            ['C', 'C', 'T', 'C', 'C', 'X', 'X', 'X'],
            ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']]

    #grid = [list(line) for line in open(filepath, "r").read().strip().split("\n")]
    start_state = (0, 0, "S", 2, 5, "W")
    #astar_avoid_path_forward_radius(grid,start_state,4,0,4,0,['S','F','E','F','S','F','F','F','W'],1)
    astar_avoid_path_forward_radius(grid,(2, 2, 'S', 2, 3, 'E'), 4, 0, 1, 4, ['F', 'F', 'F', 'W', 'F', 'I'],0, 'N')