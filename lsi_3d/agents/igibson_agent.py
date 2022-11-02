"""
Module interfaces between lsi directives and low level iGibson robot behavior
"""
from re import A
import numpy as np
import math
from igibson.utils.utils import quatToXYZW
from transforms3d.euler import euler2quat
from lsi_3d.utils.enums import MLAction
from lsi_3d.utils.constants import DIRE2POSDIFF, TARGET_ORNS
from lsi_3d.utils.functions import quat2euler
from scipy.spatial.transform import Rotation

ONE_STEP = 0.02

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
        if current_action == MLAction.FORWARD and self.forward_distance(x, y, self.target_x, self.target_y, self.direction) < ONE_STEP*1.5:
            #self.action_index += 1
            ready_for_next_action = True
        elif current_action in MLAction.directions() and self.turn_distance(self.get_current_orn_z(), TARGET_ORNS[self.target_direction]) < ONE_STEP*1.5:
            #self.action_index += 1
            self.direction = current_action
            ready_for_next_action = True
        elif current_action == None: # None when first action
            return True

        
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
        
        if next_action == MLAction.FORWARD:
            diff_x, diff_y = DIRE2POSDIFF[self.direction]
            self.target_x += diff_x
            self.target_y += diff_y
        elif next_action in MLAction.directions():
            self.target_direction = next_action
            x, y, z = self.object.get_position()
            self.target_x = x
            self.target_y = y
        elif next_action == MLAction.STAY or next_action == MLAction.IDLE:
            return

    def get_current_orn_z(self):
        x, y, z, w = self.object.get_orientation()
        x, y, z = quat2euler(x, y, z, w)
        return z

    def turn_distance(self, cur_orn_z, target_orn_z):
        return abs(cur_orn_z - target_orn_z)

    def forward_distance(self, cur_x, cur_y, target_x, target_y, direction):
        if direction in [MLAction.NORTH, MLAction.SOUTH]:
            return abs(cur_x-target_x)
        else:
            return abs(cur_y-target_y)

    def agent_move_one_step(self, env, action):
        #if action == None or action == MLAction.IDLE:
            #if self.name == "robot":
                #action = np.zeros(env.action_space.shape)
                #action = np.full(env.action_space.shape, 0.000000000000000000001)

                # action = env.action_space
                # action[0] = 0
                # action[1] = 0
                #self.object.apply_action(action)
            #return
        
        if action in MLAction.directions():
            self.agent_turn_one_step(env, action)
        elif action == MLAction.FORWARD:
            cur_x, cur_y = self.object.get_position()[:2]
            goal_angle = math.atan2((self.target_y - cur_y), (self.target_x- cur_x))
            current_heading = self.get_current_orn_z()
            angle_delta = self.calc_angle_distance(goal_angle, current_heading)

            if angle_delta > 0.1:
                self.turn_toward(env, goal_angle, angle_delta, cur_x, cur_y)

            else:
                self.agent_forward_one_step(env)
        else:
            pass

    def agent_forward_one_step(self, env):
        if self.name == "human":
            x,y,z = self.object.get_position()
            if self.direction == MLAction.NORTH:
                self.object.set_position_orientation([x-ONE_STEP,y,z], self.object.get_orientation())
            elif self.direction == MLAction.SOUTH:
                self.object.set_position_orientation([x+ONE_STEP,y,z], self.object.get_orientation())
            elif self.direction == MLAction.EAST:
                self.object.set_position_orientation([x,y+ONE_STEP,z], self.object.get_orientation())
            elif self.direction == MLAction.WEST:
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

    def calc_angle_distance(self, a1, a2):

        if a1 < 0: a1 += 6.28319
        if a2 < 0: a2 += 6.28319

        d = abs(a2-a1)

        if d > 3.14159:
            d = 6.28319-d

        return d

    def turn_toward(self, env, goal_angle, angle_delta, cur_x, cur_y):
        cur_orn_z = self.get_current_orn_z()
        target_orn_z = goal_angle
        action = np.zeros(env.action_space.shape)
        action[0] = 0

        distance_to_target = self.forward_distance(cur_x, cur_y, self.target_x, self.target_y, self.direction)

        # Decides to turn right or left
        if(cur_orn_z < target_orn_z):
            action[1] = -angle_delta
        else:
            action[1] = angle_delta
        #print((cur_orn_z-target_orn_z) / (action[1]/action[1]), action[1], cur_orn_z, target_orn_z)
        if ((cur_orn_z-target_orn_z) / (action[1]/abs(action[1]))) > 4: # > 3.14
            action[1] = -action[1] 
        if abs(target_orn_z - cur_orn_z) < 0.5:
            action[1] /= 2
        elif abs(target_orn_z - cur_orn_z) < 0.2:
            action[1] /= 4
        self.object.apply_action(action)
        #env.simulator.step()


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