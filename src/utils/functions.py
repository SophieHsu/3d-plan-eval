import math
import numpy as np
from numpy import diff
from src.utils.constants import TARGET_ORNS, DIRE2POSDIFF, FORWARD_RADIUS_POS

def orn_to_cardinal(angle):

    # converts an angle to whatever cardinal direction angle is closest to
    rad_45 = 0.785
    if angle >= rad_45 and angle < 3*rad_45:
        return 'E'
    if (angle >= 3*rad_45 and angle < math.pi) or (angle >= -math.pi and angle < -3*rad_45):
        return 'N'
    if angle >= -3*rad_45 and angle < -rad_45:
        return 'W'
    if (angle >= -rad_45 and angle < 0) or (angle < rad_45 and angle >= 0):
        return 'S'
    # diff = np.inf
    # direction = None

    # for k,v in TARGET_ORNS.items():
    #     if angle < 0: angle += 6.28319
    #     if v < 0: v += 6.28319

    #     d = abs(angle-v)
    #     if d < diff:
    #         direction = k
    #         diff = d

    #return direction

def norm_orn_to_cardinal(angle):

    # converts an angle to whatever cardinal direction angle is closest to
    rad_45 = 0.785
    if angle >= rad_45 and angle < 3*rad_45:
        return 'E'
    if (angle >= 3*rad_45 and angle < 5*rad_45):
        return 'N'
    if angle >= 5*rad_45 and angle < 7*rad_45:
        return 'W'
    if (angle >= 7*rad_45 and angle < 2*math.pi) or (angle < rad_45 and angle >= 0):
        return 'S'
    
def norm_cardinal_to_orn(dir):
    rad_45 = 0.785
    if dir is 'E':
        return 2*rad_45
    if dir is 'N':
        return 4*rad_45
    if dir is 'W':
        return 6*rad_45
    if dir is 'S':
        return 0

def quat2euler(x, y, z, w):
    """
    https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/

    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
  
    return roll_x, pitch_y, yaw_z # in radians

def grid_transition(action, state):
    # (s,a) -> (s')
    x,y,f = state
    a = action
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

def find_nearby_open_space(grid, loc):
    locR, locC = loc
    if grid[locR][locC] == 'X':
        return (*loc, 'S')
    for dir, diff in DIRE2POSDIFF.items():
        diffR, diffC = diff
        newLocR = locR + diffR
        newLocC = locC + diffC
        if 0 <= newLocR < len(grid) and 0 <= newLocC < len(grid[0]) and grid[newLocR][newLocC] == 'X':
            return (newLocR, newLocC, dir)
    return None

def opposite_dir(dir):
    if dir is 'E':
        return 'W'
    if dir is 'W':
        return 'E'
    if dir is 'N':
        return 'S'
    if dir is 'S':
        return 'N'

def find_nearby_open_spaces(grid, loc):
    open_spaces = []
    locR, locC = loc
    if grid[locR][locC] == 'X':
        return [(*loc, 'S')]
    for dir, diff in DIRE2POSDIFF.items():
        diffR, diffC = diff
        newLocR = locR + diffR
        newLocC = locC + diffC
        if 0 <= newLocR < len(grid) and 0 <= newLocC < len(grid[0]) and grid[newLocR][newLocC] == 'X':
            open_spaces.append((newLocR, newLocC, opposite_dir(dir)))
    return open_spaces

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
