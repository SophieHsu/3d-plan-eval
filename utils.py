import math

def quat2euler(x, y, z, w):
        """
        https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
        """

        """
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

def real_to_grid_coord(coord):
        # return (math.floor(coord[0]), math.floor(coord[1]))
        return (math.floor(coord[0] + 5), math.floor(coord[1] + 5))

def grid_to_real_coord(coord):
        # return (coord[0] + 0.5, coord[1] + 0.5)
        return (coord[0] - 4.5, coord[1] - 4.5)

def to_overcooked_grid(loc):
        pos = loc
        r,c = pos
        x,y = c+1,r+1
        return (x,y)

def normalize_radians(rad):
        # Convert radians to value between 0 and 2 * pi
        rad = rad % (2 * math.pi)
        if rad < 0:
            rad = rad + 2 * math.pi
        return rad