import math
import numpy as np
from numpy import diff
from lsi_3d.utils.constants import TARGET_ORNS
from lsi_3d.utils.enums import MLAction

def orn_to_cardinal(angle):

    # converts an angle to whatever cardinal direction angle is closest to
    rad_45 = 0.785
    if angle >= rad_45 and angle < 3*rad_45:
        return MLAction.EAST
    if (angle >= 3*rad_45 and angle < math.pi) or (angle >= -math.pi and angle < -3*rad_45):
        return MLAction.NORTH
    if angle >= -3*rad_45 and angle < -rad_45:
        return MLAction.WEST
    if (angle >= -rad_45 and angle < 0) or (angle < rad_45 and angle >= 0):
        return MLAction.SOUTH
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