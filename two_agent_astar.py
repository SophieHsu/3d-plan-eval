from heapq import heappush, heappop
# state: x1, y1, f1, x2, y2, f2 (x, y, facing_direction)

def copy_grid(grid):
    return [each[:] for each in grid]

def pp_grid(grid):
    print("\n".join(["".join(each) for each in grid]))

def crossing(old_state, new_state):
    cx1, cy1, cf1, cx2, cy2, cf2 = old_state
    nx1, ny1, nf1, nx2, ny2, nf2 = new_state
    return (nx1, ny1) == (cx2, cy2) and (nx2, ny2) == (cx1, cy1)

# Kitchen A-star
# add facing
# terminate early / decrease cost of idling
def transition(x, y, f, a):
    name2dire = {
        "E": (0, 1),
        "W": (0, -1),
        "S": (1, 0),
        "N": (-1, 0),
    }
    if a in "EWSN":
        f = a
    elif a == "F":
        dx, dy = name2dire[f]
        x, y = x+dx, y+dy
    return x, y, f

def heuristic(ex1, ey1, ex2, ey2, state):
    x1, y1, f1, x2, y2, f2 = state
    return (abs(x1-ex1)+abs(y1-ey1)+abs(x2-ex2)+abs(y2-ey2))

def valid_one(grid, x, y):
    return 0 <= x and x < len(grid) and 0 <= y and y < len(grid[0]) and grid[x][y] == "X"

def valid(grid, nx1, ny1, nx2, ny2):
    return valid_one(grid, nx1, ny1) and valid_one(grid, nx2, ny2)

def cost_func(grid, ex, ey, x, y, f, action):
    name2dire = {
        "E": (0, 1),
        "W": (0, -1),
        "S": (1, 0),
        "N": (-1, 0),
    }
    dx, dy = name2dire[f]
    if end_achieved(grid, x, y, ex, ey) and action == "I" and (x+dx, y+dy) == (ex, ey):
        return 0
    else:
        return 1

def end_achieved(grid, x, y, ex, ey):
    if(grid[ex][ey] == "X"):
        return (x, y) == (ex, ey)
    else:
        dx = abs(x-ex)
        dy = abs(y-ey)
        return dx + dy <= 1

def two_agent_astar(grid, start_state, ex1, ey1, ex2, ey2):
    actions = ["E", "W", "S", "N", "I", "F"] 
    visited = set()
    visited.add(start_state)
    path_prev = dict()
    queue = []
    heappush(queue, (0, 0, start_state)) # f, g, state, f=h+g
    f_values = dict()
    f_values[start_state] = 0
    break_all = False
    last_state = None
    while queue and not break_all:
        _, cur_g, cur_state = heappop(queue)
        cx1, cy1, cf1, cx2, cy2, cf2 = cur_state
        if end_achieved(grid, cx1, cy1, ex1, ey1) and end_achieved(grid, cx2, cy2, ex2, ey2):
            last_state = cur_state
            break_all = True
            break
        for action1 in actions:
            if break_all:
                break
            nx1, ny1, nf1 = transition(cx1, cy1, cf1, action1)
            cost1 = cost_func(grid, ex1, ey1, nx1, ny1, nf1, action1)
            for action2 in actions:
                nx2, ny2, nf2 = transition(cx2, cy2, cf2, action2)
                new_state = (nx1, ny1, nf1, nx2, ny2, nf2)
                cost2 = cost_func(grid, ex2, ey2, nx2, ny2, nf2, action2)
                #print(action1, action2, new_state)
                if valid(grid, nx1, ny1, nx2, ny2) and (nx1, ny1) != (nx2, ny2) and not crossing(cur_state, new_state):
                    new_h = heuristic(ex1, ey1, ex2, ey2, new_state)
                    if new_state not in visited or cur_g+cost1+cost2+new_h < f_values[new_state]:
                        heappush(queue, (cur_g+cost1+cost2+new_h, cur_g+cost1+cost2, new_state))
                        f_values[new_state] = cur_g+cost1+cost2+new_h
                        visited.add(new_state)
                        path_prev[new_state] = (cur_state, action1, action2)
                        
                if end_achieved(grid, nx1, ny1, ex1, ey1) and end_achieved(grid, nx2, ny2, ex2, ey2) and (nx1, ny1) != (nx2, ny2) and not crossing(cur_state, new_state):
                    path_prev[new_state] = (cur_state, action1, action2)
                    heappush(queue, (cur_g+cost1+cost2+0, cur_g+cost1+cost2, new_state)) # heuristic=0

    path = []
    if not break_all:
        return path
    px1, py1, pf1, px2, py2, pf2 = last_state
    sx1, sy1, sf1, sx2, sy2, sf2 = start_state
    while (px1, py1, pf1, px2, py2, pf2) != (sx1, sy1, sf1, sx2, sy2, sf2):
        p_state, command1, command2 = path_prev[(px1, py1, pf1, px2, py2, pf2)]
        path.append(((px1, py1, pf1, px2, py2, pf2), command1, command2))
        px1, py1, pf1, px2, py2, pf2 = p_state

    return path[::-1]

def run_astar_two_agent(layout, start, end):
    """Runs astar algorithm for two agent

    Args:
        layout (2d array): represents the layout of the environment
        start (tuple): tuple with start, end, and direction for two agents
        end (tuple): tuple with start, end for each agent

    Returns:
        plan as two lists consisting of actions the agent should take in
        consecutive order
    """
    # TODO: change args to be just start and end
    a_1_end, a_2_end = end
    path = two_agent_astar(layout, start, a_1_end[0], a_1_end[1], a_2_end[0], a_2_end[1])
    z = zip(*path)
    a_1_path = [l[1] for l in path]
    a_2_path = [l[2] for l in path]
    return (a_1_path, a_2_path)

if __name__ == "__main__":
    filepath = "kitchen_layouts_grid_text/empty.txt"
    grid = [list(line) for line in open(filepath, "r").read().strip().split("\n")]
    start_state = (0, 0, "S", 2, 0, "S")
    path = two_agent_astar(grid, start_state, 2,0,0,0)

    #path = two_agent_astar(layout, (), human_end[0], human_end[1], robot_end[0], robot_end[1]
    #print(5, 2, 6, 3)
    c_grid = copy_grid(grid)
    px1, py1, pf1, px2, py2, pf2 = start_state
    count = 0
    old_state = None
    p_state = start_state
    for each in path:
        c_grid = copy_grid(grid)
        c_grid[px1][py1] = "!"
        c_grid[px2][py2] = "?"
        old_state = p_state
        p_state, command1, command2 = each
        print(count)
        pp_grid(c_grid)
        print(old_state, command1, command2)
        print("-"*20)
        count+=1
        px1, py1, pf1, px2, py2, pf2 = p_state
    print(count)
    c_grid = copy_grid(grid)
    c_grid[px1][py1] = "!"
    c_grid[px2][py2] = "?"
    pp_grid(c_grid)
    print(p_state)
    #print(command1, command2)