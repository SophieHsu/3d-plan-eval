from heapq import heappush, heappop
from lsi_3d.utils.constants import DIRE2POSDIFF, FORWARD_RADIUS_POS
# state: x1, y1, f1, x2, y2, f2 (x, y, facing_direction)

def copy_grid(grid):
    return [each[:] for each in grid]

def pp_grid(grid):
    print("\n".join(["".join(each) for each in grid]))

def crossing(old_state, new_state):
    cx1, cy1, cf1, cx2, cy2, cf2 = old_state
    nx1, ny1, nf1, nx2, ny2, nf2 = new_state
    return (nx1, ny1) == (cx2, cy2) and (nx2, ny2) == (cx1, cy1)

def crossing_future(old_state, new_state, future_states):
    cx1, cy1, cf1, cx2, cy2, cf2 = old_state
    nx1, ny1, nf1, nx2, ny2, nf2 = new_state
    immediate_cross =  (nx1, ny1) == (cx2, cy2) and (nx2, ny2) == (cx1, cy1)
    
    future_cross = False
    for f_state in future_states:
        fx, fy, ff = f_state
        if (fx, fy) == (cx2, cy2) and (fx, fy) == (cx1, cy1):
            future_cross = True

    return immediate_cross and future_cross

def manhattan(a, b):
    return max(abs(val1-val2) for val1, val2 in zip(a,b))

def get_states_in_forward_radius(state, radius):
    y,x,f = state
    xmin_mul, xmax_mul, ymin_mul, ymax_mul = FORWARD_RADIUS_POS[f]
    x_min = x + radius * xmin_mul
    x_max = x + radius * xmax_mul
    y_min = y + radius * ymin_mul
    y_max = y + radius * ymax_mul

    states_within_radius = []
    for i in range(y_min, y_max+1):
        for j in range(x_min, x_max+1):
            #if manhattan((y,x),(i,j)) < radius:
            states_within_radius.append((i,j))

    return states_within_radius
    

def crossing_radius(old_state, new_state, f_radius):
    cx1, cy1, cf1, cx2, cy2, cf2 = old_state
    nx1, ny1, nf1, nx2, ny2, nf2 = new_state
    immediate_cross =  (nx1, ny1) == (cx2, cy2) and (nx2, ny2) == (cx1, cy1)

    avoid_cur_state = (cx1, cy1, cf1)

    future_cross = False

    # if the agent 2 (robot) doesnt move then return immediate cross
    if cx2 == nx2 and cy2 == ny2:
        return immediate_cross

    # check if the robot is travelling into a square in front of the human
    for f_state in get_states_in_forward_radius(avoid_cur_state, f_radius):
        fx, fy = f_state
        if (fx, fy) == (nx2, ny2):
            future_cross = True

    return immediate_cross or future_cross

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

def heuristic_one(ex1, ey1, state):
    x1, y1, f1 = state
    return (abs(x1-ex1)+abs(y1-ey1))

def valid_one(grid, x, y):
    return 0 <= x and x < len(grid) and 0 <= y and y < len(grid[0]) and grid[x][y] == "X"

def valid(grid, nx1, ny1, nx2, ny2):
    return valid_one(grid, nx1, ny1) and valid_one(grid, nx2, ny2)

def cost_func(grid, ex, ey, x, y, f, action, delay_cost = None):
    name2dire = {
        "E": (0, 1),
        "W": (0, -1),
        "S": (1, 0),
        "N": (-1, 0),
    }
    dx, dy = name2dire[f]
    if end_achieved(grid, x, y, ex, ey, f) and action == "I" and (x+dx, y+dy) == (ex, ey):
        return 0
    elif action == 'D':
        if delay_cost is not None:
            return delay_cost
        else:
            return 0.5
    else:
        return 1

def cost_func_radius(grid, ex, ey, x, y, f, action, radius = None, avoid_state = None):
    name2dire = {
        "E": (0, 1),
        "W": (0, -1),
        "S": (1, 0),
        "N": (-1, 0),
    }
    dx, dy = name2dire[f]
    if end_achieved(grid, x, y, ex, ey, f) and action == "I" and (x+dx, y+dy) == (ex, ey):
        return 0
    elif action == 'D':
        return 0.5
    else:
        if radius != None:
            cost = 1
            ax,ay,af = avoid_state
            nx,ny,nf = transition(x,y,f,action)
            for f_state in get_states_in_forward_radius(avoid_state, radius):
                fx, fy = f_state
                if (fx, fy) == (nx, ny):
                    cost = 3.5
            
            return cost
                
        else:
            return 1

def end_achieved(grid, x, y, ex, ey, f, ef=None):
    if(grid[ex][ey] == "X"):
        if ef != None:
            return (x, y, f) == (ex, ey, ef)
        else:
            return (x, y) == (ex, ey)
    else:
        dx = abs(x-ex)
        dy = abs(y-ey)

        name2dire = {
            "E": (0, 1),
            "W": (0, -1),
            "S": (1, 0),
            "N": (-1, 0),
        }

        offset = name2dire[f]
        return (dx + dy <= 1) and (offset == ((ex-x),(ey-y)))

def end_achieved_same_goal_check(grid, end1, end2, goal1, goal2,gf2):
    if goal1 == goal2:
        return agent1_end_achieved_agent2_within_r(grid, end1, end2, goal1, goal2, gf2, 1)
    else:
        x1,y1,f1 = end1
        x2,y2,f2 = end2
        gx1, gy1, *_ = goal1
        gx2, gy2, *_ = goal2
        # args = [x1, y1, gx1, gy1, f1, x2, y2, gx2, gy2, f2, gf2]
        # print(f'{args}')
        return end_achieved(grid,x1,y1,gx1,gy1,f1) and end_achieved(grid,x2,y2,gx2,gy2,f2,gf2)

def agent1_end_achieved_agent2_within_r(grid, end1, end2, goal1, goal2, gf2, radius):
    x1,y1,f1 = end1
    x2,y2,f2 = end2
    gx1, gy1 = goal1
    gx2, gy2 = goal2

    end_achieved_1 = end_achieved(grid,x1,y1,gx1,gy1,f1)
    
    manattan_2 = manhattan((x2,y2),(gx2,gy2))

    return end_achieved_1 and manattan_2 <= radius



def single_agent_astar(grid, start_state, end_state):
    ex1, ey1 = end_state
    if grid[ex1][ey1] == 'X':
        print('Warning: End goal is open space so agent may spin in place')
    
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
        cx1, cy1, cf1 = cur_state
        if end_achieved(grid, cx1, cy1, ex1, ey1, cf1):
            last_state = cur_state
            break_all = True
            break
        for action1 in actions:
            if break_all:
                break
            nx1, ny1, nf1 = transition(cx1, cy1, cf1, action1)
            cost1 = cost_func(grid, ex1, ey1, nx1, ny1, nf1, action1)
            
                
            new_state = (nx1, ny1, nf1)
            #print(action1, action2, new_state)
            if valid_one(grid, nx1, ny1):
                new_h = heuristic_one(ex1, ey1, new_state)
                if new_state not in visited or cur_g+cost1+new_h < f_values[new_state]:
                    heappush(queue, (cur_g+cost1+new_h, cur_g+cost1, new_state))
                    f_values[new_state] = cur_g+cost1+new_h
                    visited.add(new_state)
                    path_prev[new_state] = (cur_state, action1)
                    
            # placed this inside conditional (valid) becuase invalid position was getting
            # pushed onto heap
            if valid_one(grid, nx1, ny1) and end_achieved(grid, nx1, ny1, ex1, ey1, cf1):
                path_prev[new_state] = (cur_state, action1)
                heappush(queue, (cur_g+cost1+0, cur_g+cost1, new_state)) # heuristic=0

    path = []
    if not break_all:
        return path
    px1, py1, pf1 = last_state
    sx1, sy1, sf1 = start_state
    while (px1, py1, pf1) != (sx1, sy1, sf1):
        p_state, command1 = path_prev[(px1, py1, pf1)]
        path.append(((px1, py1, pf1), command1))
        px1, py1, pf1 = p_state

    return path[::-1]

def two_agent_astar(grid, start_state, ex1, ey1, ex2, ey2):

    if grid[ex1][ey1] == 'X' or grid[ex2][ey2] == 'X':
        print('Warning: End goal is open space so agent may spin in place')
    
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
        if end_achieved(grid, cx1, cy1, ex1, ey1, cf1) and end_achieved(grid, cx2, cy2, ex2, ey2, cf2):
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
                        
                # placed this inside conditional (valid) becuase invalid position was getting
                # pushed onto heap
                if valid(grid, nx1, ny1, nx2, ny2) and end_achieved(grid, nx1, ny1, ex1, ey1, cf1) and end_achieved(grid, nx2, ny2, ex2, ey2, cf2) and (nx1, ny1) != (nx2, ny2) and not crossing(cur_state, new_state):
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

def astar_avoid_path(grid, start_state, ex1, ey1, ex2, ey2, avoid_path):
    avoid_path_queue = avoid_path
    if grid[ex1][ey1] == 'X' or grid[ex2][ey2] == 'X':
        print('Warning: End goal is open space so agent may spin in place')
    
    actions = ["E", "W", "S", "N", "I", "F"]
    visited = set()
    visited.add(start_state)
    path_prev = dict()
    queue = []
    heappush(queue, (0, 0, start_state, 0)) # f, g, state, f=h+g, avoid_path_t_step
    f_values = dict()
    f_values[start_state] = 0
    break_all = False
    last_state = None
    while queue and not break_all:
        _, cur_g, cur_state, avoid_path_t_step = heappop(queue)
        cx1, cy1, cf1, cx2, cy2, cf2 = cur_state
        if end_achieved(grid, cx1, cy1, ex1, ey1, cf1) and end_achieved(grid, cx2, cy2, ex2, ey2, cf2):
            last_state = cur_state
            break_all = True
            break

        avoid_actions = []
        if len(avoid_path_queue) > avoid_path_t_step:
            avoid_actions.append(avoid_path[avoid_path_t_step])
        else:
            avoid_actions = ['I']
        # avoid_path_t_step+=1

        for action1 in avoid_actions:
            if break_all:
                break
            nx1, ny1, nf1 = transition(cx1, cy1, cf1, action1)

            # can remove for fixed path
            cost1 = cost_func(grid, ex1, ey1, nx1, ny1, nf1, action1)

            for action2 in actions:
                nx2, ny2, nf2 = transition(cx2, cy2, cf2, action2)
                new_state = (nx1, ny1, nf1, nx2, ny2, nf2)
                cost2 = cost_func(grid, ex2, ey2, nx2, ny2, nf2, action2)
                #print(action1, action2, new_state)
                if valid(grid, nx1, ny1, nx2, ny2) and (nx1, ny1) != (nx2, ny2) and not crossing(cur_state, new_state):
                    new_h = heuristic(ex1, ey1, ex2, ey2, new_state)
                    if new_state not in visited or cur_g+cost1+cost2+new_h < f_values[new_state]:
                        heappush(queue, (cur_g+cost1+cost2+new_h, cur_g+cost1+cost2, new_state, avoid_path_t_step+1))
                        f_values[new_state] = cur_g+cost1+cost2+new_h
                        visited.add(new_state)
                        path_prev[new_state] = (cur_state, action1, action2)
                        
                # placed this inside conditional (valid) becuase invalid position was getting
                # pushed onto heap
                if valid(grid, nx1, ny1, nx2, ny2) and end_achieved(grid, nx1, ny1, ex1, ey1, cf1) and end_achieved(grid, nx2, ny2, ex2, ey2, cf2) and (nx1, ny1) != (nx2, ny2) and not crossing(cur_state, new_state):
                    path_prev[new_state] = (cur_state, action1, action2)
                    heappush(queue, (cur_g+cost1+cost2+0, cur_g+cost1+cost2, new_state, avoid_path_t_step+1)) # heuristic=0

    path = []
    if not break_all:
        return path
    px1, py1, pf1, px2, py2, pf2 = last_state
    sx1, sy1, sf1, sx2, sy2, sf2 = start_state
    while (px1, py1, pf1, px2, py2, pf2) != (sx1, sy1, sf1, sx2, sy2, sf2):
        p_state, command1, command2 = path_prev[(px1, py1, pf1, px2, py2, pf2)]
        path.append(((px1, py1, pf1, px2, py2, pf2), command1, command2))
        px1, py1, pf1, px2, py2, pf2 = p_state

    path = path[::-1]
    # cur_avoid_pos = avoid_start
    # prev_avoid_pos = avoid_start
    # # avoid conflicts
    # for i in range(len(path)):
    #     (px,py,_,_,_),_,_ = path[i]

    #     px_a, py_a = avoid_path[i]

    #     if px == px_a and py == py_a:
    #         path.insert(MLAction.WAIT)

    #     prev_avoid_pos = cur_avoid_pos
    #     cur_avoid_pos = transition(cur_avoid_pos)

    return path

def astar_avoid_path_forward_radius(grid, start_state, ex1, ey1, ex2, ey2, avoid_path, f_radius, ef2 = None):
    avoid_path_queue = avoid_path
    # if grid[ex1][ey1] == 'X' or grid[ex2][ey2] == 'X':
    #    print('Warning: End goal is open space so agent may spin in place')
    
    actions = ["E", "W", "S", "N", "I", "F", "D"]
    visited = set()
    #visited.add(start_state)
    path_prev = dict()
    queue = []
    heappush(queue, (0, 0, start_state, 0)) # f, g, state, f=h+g, avoid_path_t_stepp
    f_values = dict()
    f_values[start_state] = 0
    break_all = False
    last_state = None
    while queue and not break_all:
        _, cur_g, cur_state, avoid_path_t_step = heappop(queue)
        cx1, cy1, cf1, cx2, cy2, cf2 = cur_state

        #if end_achieved(grid, cx1, cy1, ex1, ey1, cf1) and end_achieved(grid, cx2, cy2, ex2, ey2, cf2, ef2):
        # args = [cx1, cy1, ex1, ey1, cf1, cx2, cy2, ex2, ey2, cf2, ef2]
        # print(f'{args}')
        if end_achieved_same_goal_check(grid, (cx1,cy1,cf1), (cx2,cy2,cf2), (ex1, ey1), (ex2, ey2), ef2) or end_achieved(grid, cx2, cy2, ex2, ey2, cf2, ef2):
            last_state = cur_state
            break_all = True
            break
        
        # Add human actions at current time step for robot to avoid
        avoid_actions = []
        # avoid_actions.append('D')
        if len(avoid_path_queue) > avoid_path_t_step:
            avoid_actions.append(avoid_path[avoid_path_t_step])
        
        if avoid_actions == []:
            avoid_actions.append('I')

        for action1 in avoid_actions:
            if break_all:
                break
            nx1, ny1, nf1 = transition(cx1, cy1, cf1, action1)

            # can remove for fixed path
            cost1 = cost_func(grid, ex1, ey1, nx1, ny1, nf1, action1, 5)

            for action2 in actions:
                nx2, ny2, nf2 = transition(cx2, cy2, cf2, action2)
                new_state = (nx1, ny1, nf1, nx2, ny2, nf2)
                cost2 = cost_func_radius(grid, ex2, ey2, nx2, ny2, nf2, action2, f_radius, (nx1,nx1,nf1))
                #print(action1, action2, new_state)
                # if crossing_radius(cur_state, new_state, f_radius):
                #     asdf = 4
                # if (ex2, ey2, ef2) == (nx2, ny2, nf2):
                #     asdf = 2
                if valid(grid, nx1, ny1, nx2, ny2) and (nx1, ny1) != (nx2, ny2) and not crossing(cur_state, new_state):
                    new_h = heuristic(ex1, ey1, ex2, ey2, new_state)
                    if new_state not in visited or cur_g+cost1+cost2+new_h < f_values[new_state]:
                        heappush(queue, (cur_g+cost1+cost2+new_h, cur_g+cost1+cost2, new_state, avoid_path_t_step+1))
                        f_values[new_state] = cur_g+cost1+cost2+new_h
                        visited.add(new_state)
                        path_prev[new_state] = (cur_state, action1, action2)
                # else:
                #     asdf = 5
                        
                # placed this inside conditional (valid) becuase invalid position was getting
                # pushed onto heap
                if valid(grid, nx1, ny1, nx2, ny2) and end_achieved(grid, nx1, ny1, ex1, ey1, cf1) and end_achieved(grid, nx2, ny2, ex2, ey2, cf2, ef2) and (nx1, ny1) != (nx2, ny2) and not crossing(cur_state, new_state):
                    path_prev[new_state] = (cur_state, action1, action2)
                    heappush(queue, (cur_g+cost1+cost2+0, cur_g+cost1+cost2, new_state, avoid_path_t_step+1)) # heuristic=0

    path = []
    if not break_all:
        return path
    px1, py1, pf1, px2, py2, pf2 = last_state
    sx1, sy1, sf1, sx2, sy2, sf2 = start_state
    while (px1, py1, pf1, px2, py2, pf2) != (sx1, sy1, sf1, sx2, sy2, sf2):
        p_state, command1, command2 = path_prev[(px1, py1, pf1, px2, py2, pf2)]
        #path.append(((px1, py1, pf1, px2, py2, pf2), command1, command2))
        path.append((((px1, py1, pf1),command1), ((px2, py2, pf2), command2)))
        px1, py1, pf1, px2, py2, pf2 = p_state

    path = path[::-1]

    return path

def astar_avoid_path_with_lookahead(grid, start_state, ex1, ey1, ex2, ey2, avoid_path, lookahead_distance):
    avoid_path_queue = avoid_path
    if grid[ex1][ey1] == 'X' or grid[ex2][ey2] == 'X':
        print('Warning: End goal is open space so agent may spin in place')
    
    actions = ["E", "W", "S", "N", "I", "F"]
    visited = set()
    visited.add(start_state)
    path_prev = dict()
    queue = []
    heappush(queue, (0, 0, start_state, 0)) # f, g, state, f=h+g, avoid_path_t_step
    f_values = dict()
    f_values[start_state] = 0
    break_all = False
    last_state = None
    while queue and not break_all:
        _, cur_g, cur_state, avoid_path_t_step = heappop(queue)
        cx1, cy1, cf1, cx2, cy2, cf2 = cur_state
        if end_achieved(grid, cx1, cy1, ex1, ey1, cf1) and end_achieved(grid, cx2, cy2, ex2, ey2, cf2):
            last_state = cur_state
            break_all = True
            break

        avoid_actions = []
        if len(avoid_path_queue) > avoid_path_t_step:
            i = avoid_path_t_step
            action_sequence = []
            while i < lookahead_distance and i < len(avoid_path_queue):
                action_sequence.append(avoid_path[i])
                i += 1

            #avoid_actions.append(avoid_path[avoid_path_t_step])
            if len(action_sequence) > 0:
                avoid_actions.append(action_sequence)
        #else:
        avoid_actions.append(['I'])
        # avoid_path_t_step+=1

        for action1 in avoid_actions:
            if break_all:
                break
            
            #nx1, ny1, nf1 = transition(cx1, cy1, cf1, action1)

            future_states = []
            for a in action1:
                nx1, ny1, nf1 = transition(cx1, cy1, cf1, a)
                future_states.append((nx1, ny1, nf1))

            
            nx1, nx2, nf1 = future_states[0]

            # can remove for fixed path
            cost1 = cost_func(grid, ex1, ey1, nx1, ny1, nf1, action1[0])

            for action2 in actions:
                # action 2 is now a list of next steps

                nx2, ny2, nf2 = transition(cx2, cy2, cf2, action2)
                new_state = (nx1, ny1, nf1, nx2, ny2, nf2)
                cost2 = cost_func(grid, ex2, ey2, nx2, ny2, nf2, action2)
                #print(action1, action2, new_state)
                if valid(grid, nx1, ny1, nx2, ny2) and (nx1, ny1) != (nx2, ny2) and not crossing_future(cur_state, new_state, future_states):
                    new_h = heuristic(ex1, ey1, ex2, ey2, new_state)
                    if new_state not in visited or cur_g+cost1+cost2+new_h < f_values[new_state]:
                        heappush(queue, (cur_g+cost1+cost2+new_h, cur_g+cost1+cost2, new_state, avoid_path_t_step+1))
                        f_values[new_state] = cur_g+cost1+cost2+new_h
                        visited.add(new_state)
                        path_prev[new_state] = (cur_state, action1, action2)
                        
                # placed this inside conditional (valid) becuase invalid position was getting
                # pushed onto heap
                if valid(grid, nx1, ny1, nx2, ny2) and end_achieved(grid, nx1, ny1, ex1, ey1, cf1) and end_achieved(grid, nx2, ny2, ex2, ey2, cf2) and (nx1, ny1) != (nx2, ny2) and not crossing(cur_state, new_state):
                    path_prev[new_state] = (cur_state, action1, action2)
                    heappush(queue, (cur_g+cost1+cost2+0, cur_g+cost1+cost2, new_state, avoid_path_t_step+1)) # heuristic=0

    path = []
    if not break_all:
        return path
    px1, py1, pf1, px2, py2, pf2 = last_state
    sx1, sy1, sf1, sx2, sy2, sf2 = start_state
    while (px1, py1, pf1, px2, py2, pf2) != (sx1, sy1, sf1, sx2, sy2, sf2):
        p_state, command1, command2 = path_prev[(px1, py1, pf1, px2, py2, pf2)]
        path.append(((px1, py1, pf1, px2, py2, pf2), command1, command2))
        px1, py1, pf1, px2, py2, pf2 = p_state

    path = path[::-1]
    # cur_avoid_pos = avoid_start

    return path

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
    print(f'Robot path computed is: {a_2_path}')
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

def run_astar_two_agent(layout, start, end, avoid_path = None, radius = None):
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

    if avoid_path == None:
        path = two_agent_astar(layout, start, a_1_end[0], a_1_end[1], a_2_end[0], a_2_end[1])
    else:
        path = astar_avoid_path_forward_radius(layout, start, a_1_end[0], a_1_end[1], a_2_end[0], a_2_end[1], avoid_path, radius)# , a_2_end[2])
        #path = astar_avoid_path_with_lookahead(layout, start, a_1_end[0], a_1_end[1], a_2_end[0], a_2_end[1], avoid_path, 2)
    
    z = zip(*path)
    a_1_path = [l[0] for l in path]
    a_2_path = [l[1] for l in path]
    print(f'Robot path computed is: {a_2_path}')
    return (a_1_path, a_2_path)

if __name__ == "__main__":
    filepath = "kitchen_layouts_grid_text/empty.txt"
    grid = [list(line) for line in open(filepath, "r").read().strip().split("\n")]
    start_state = (0, 0, "S", 2, 5, "W")
    astar_avoid_path_forward_radius(grid,start_state,4,0,4,0,['S','F','E','F','S','F','F','F','W'],1,ef2='S')
    #path = astar_avoid_path(grid, start_state, 2,0,0,0, [])

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