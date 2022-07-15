"""
Module interfaces between lsi directives and low level iGibson robot behavior
"""
import numpy as np
import math
from igibson.utils.utils import quatToXYZW
from transforms3d.euler import euler2quat

TARGET_ORNS = {
    "S": 0,
    "E": 1.5707,
    "N": 3.1415926,
    "W": -1.5707,
    None: -1
}

DIRE2POSDIFF = {
    "E": (0, 1),
    "W": (0, -1),
    "S": (1, 0),
    "N": (-1, 0)
}

ONE_STEP = 0.02

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

class iGibsonAgent:
    '''
    Controls low level agent agent actions acting as an interface between medium level turn and forward actions
    and continuous iGibson environment
    '''
    def __init__(self, robot, start, direction, name, target_x=None, target_y=None, target_direction=None):
        self.object = robot
        self.direction = direction
        self.start = start + (direction,)
        #self.path = path
        self.target_x = target_x
        self.target_y = target_y
        self.target_direction = target_direction
        self.action_index = 0
        self.name = name

    def update(self, target_x, target_y, direction, target_direction):
        self.target_x = target_x
        self.target_y = target_y
        self.direction = direction
        self.target_direction = target_direction

    def action_completed(self, current_action):
        #if self.action_index >= len(self.path):
        #    return None
        #current_action = self.path[self.action_index]
        ready_for_next_action = False
        x, y, z = self.object.get_position()
        #print(self.name, current_action, self.get_current_orn_z(), self.target_direction, turn_distance(self.get_current_orn_z(), TARGET_ORNS[self.target_direction]))
        # if self.target_x == None:
        #     ready_for_next_action = True
        #     x, y, z = self.object.get_position()
        #     self.target_x = x
        #     self.target_y = y
        if current_action == "F" and self.forward_distance(x, y, self.target_x, self.target_y, self.direction) < ONE_STEP*1.5:
            #self.action_index += 1
            ready_for_next_action = True
        elif current_action in "NWES" and self.turn_distance(self.get_current_orn_z(), TARGET_ORNS[self.target_direction]) < ONE_STEP*1.5:
            #self.action_index += 1
            self.direction = current_action
            ready_for_next_action = True
        
        # if self.action_index >= len(self.path):
        #     return None
        
        # if ready_for_next_action:
        #     next_action = self.path[self.action_index]
        #     if next_action == "F":
        #         diff_x, diff_y = DIRE2POSDIFF[self.direction]
        #         self.target_x += diff_x
        #         self.target_y += diff_y
        #     elif next_action in "NWES":
        #         self.target_direction = next_action
        return ready_for_next_action

    def prepare_for_next_action(self, next_action):

        if self.target_x == None or self.target_y == None:
            x, y, z = self.object.get_position()
            self.target_x = x
            self.target_y = y
        
        if next_action == "F":
            diff_x, diff_y = DIRE2POSDIFF[self.direction]
            self.target_x += diff_x
            self.target_y += diff_y
        elif next_action in "NWES":
            self.target_direction = next_action
            x, y, z = self.object.get_position()
            self.target_x = x
            self.target_y = y

    def get_current_orn_z(self):
        x, y, z, w = self.object.get_orientation()
        x, y, z = quat2euler(x, y, z, w)
        return z

    def turn_distance(self, cur_orn_z, target_orn_z):
        return abs(cur_orn_z - target_orn_z)

    def forward_distance(self, cur_x, cur_y, target_x, target_y, direction):
        if direction in "NS":
            return abs(cur_x-target_x)
        else:
            return abs(cur_y-target_y)

    def agent_move_one_step(self, env, action):
        if action == None:
            if self.name == "robot":
                action = np.zeros(env.action_space.shape)
                self.object.apply_action(action)
            return
        if action in "NWES":
            self.agent_turn_one_step(env, action)
        elif action == "F":
            self.agent_forward_one_step(env)
        else:
            pass

    def agent_forward_one_step(self, env):
        if self.name == "human":
            x,y,z = self.object.get_position()
            if self.direction == "N":
                self.object.set_position_orientation([x-ONE_STEP,y,z], self.object.get_orientation())
            elif self.direction == "S":
                self.object.set_position_orientation([x+ONE_STEP,y,z], self.object.get_orientation())
            elif self.direction == "E":
                self.object.set_position_orientation([x,y+ONE_STEP,z], self.object.get_orientation())
            elif self.direction == "W":
                self.object.set_position_orientation([x,y-ONE_STEP,z], self.object.get_orientation())
        else:
            action = np.zeros(env.action_space.shape)
            action[0] = 0.15
            action[1] = 0
            start_x, start_y = self.object.get_position()[:2]

            cur_x, cur_y = self.object.get_position()[:2]
            distance_to_target = self.forward_distance(cur_x, cur_y, self.target_x, self.target_y, self.direction)

            if distance_to_target < 0.2:
                action[0] /= 2
            elif distance_to_target < 0.1:
                action[0] /= 4
            elif distance_to_target < 0.05:
                action[0] /= 8
            self.object.apply_action(action)


    def agent_turn_one_step(self, env, action):
        if self.name == "human":
            x, y, z, w = self.object.get_orientation()
            x, y, z = quat2euler(x, y, z, w)
            #print("turn z:", z, action)
            target_orn_z = TARGET_ORNS[self.target_direction]
            
            pos = z - target_orn_z
            neg = target_orn_z - z
            if pos < 0:
                pos += 3.1415926*2
            elif neg < 0:
                neg += 3.1415926*2
            if pos < neg:
                z -= ONE_STEP
            else:
                z += ONE_STEP
            self.object.set_position_orientation(self.object.get_position(), quatToXYZW(euler2quat(x, y, z), "wxyz"))
        else:
            x, y, z, w = self.object.get_orientation()
            x, y, z = quat2euler(x, y, z, w)
            cur_orn_z = z
            target_orn_z = TARGET_ORNS[self.target_direction]
            action = np.zeros(env.action_space.shape)
            action[0] = 0
            if(cur_orn_z < target_orn_z):
                action[1] = -0.2
            else:
                action[1] = 0.2
            #print((cur_orn_z-target_orn_z) / (action[1]/action[1]), action[1], cur_orn_z, target_orn_z)
            if ((cur_orn_z-target_orn_z) / (action[1]/abs(action[1]))) > 4: # > 3.14
                action[1] = -action[1] 
            if abs(target_orn_z - cur_orn_z) < 0.5:
                action[1] /= 2
            elif abs(target_orn_z - cur_orn_z) < 0.2:
                action[1] /= 4
            self.object.apply_action(action)