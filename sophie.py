'''
Human logic with vision based observation. This human model will mainly be designed as a state machine.
'''

import numpy as np
import math, random

class World(object):
    """docstring for World"""
    def __init__(self, timer, width, height):
        self.timer = timer
        self.width = width
        self.height = height
        self.timer_loc = [5,5]

    def update_timer(self, inc):
        self.timer += inc

class RandHuman(object):
    """docstring for RandHuman"""
    def __init__(self, pos, world):
        self.pos = pos
        self.world = world
    
    def update_pos(self):
        self.pos[0] = random.randint(0, self.world.width)
        self.pos[1] = random.randint(0, self.world.height)

class ImmersiveHuman(object):
    """docstring for ImmersiveHuman
    pos (2d array): [x,y]
    ori (int): north 0, east 1, south 2, west 3
    """
    def __init__(self, pos, ori, world, other_agent, angle=120):
        self.ori = ori
        self.pos = pos
        self.eye_sight = angle
        self.state = self.init_states(world, other_agent)


    def init_states(self, world_state, other_agent):
        ''' 
        Assume that the human contains the info on the world and the position of the other moving agent.
        State info includes counter on the timer and the x, y position of the other agent.
        '''
        return [self.pos, self.ori, 0, 0, 0]

    def get_states(self):
        ''' 
        Assume that the human contains the info on the world and the position of the other moving agent.
        State info includes counter on the timer and the x, y position of the other agent.
        '''
        return self.state

    def get_vision_bound(self):
        # get the two points first by assuming facing north
        vision_width = np.radians(30)
        right_pt = self.pos + np.array([math.cos(vision_width), math.sin(vision_width)])
        left_pt = self.pos + np.array([-math.cos(vision_width), math.sin(vision_width)])
        
        # right_pt = np.matmul(R,right_pt-self.pos)+self.pos
        # left_pt = np.matmul(R,left_pt-self.pos)+self.pos

        return right_pt, left_pt


    def in_bound(self, loc, right_pt, left_pt):
        '''
        Use cross product to see if the point is on the left or right side of the vision bound edges.
        '''
        right_in_bound = False
        left_in_bound = False
        thresh = 1e-9

        # angle based on the agent's facing
        theta = None
        if int(self.ori) == 0: # north
            theta = np.radians(0)
        elif int(self.ori) == 1: # east
            theta = np.radians(-270)
        elif int(self.ori) == 2: # south
            theta = np.radians(180)
        elif int(self.ori) == 3: # west
            theta = np.radians(-90)
        
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        rot_loc = np.matmul(R,np.array(loc)-self.pos)+self.pos

        # check right bound
        right_val = ((right_pt[0] - self.pos[0])*(rot_loc[1] - self.pos[1]) - (right_pt[1] - self.pos[1])*(rot_loc[0] - self.pos[0]))
        if right_val >= thresh: # left of line
            right_in_bound = True
        elif right_val <= -thresh: # right of line
            right_in_bound = False
        else: # on the line
            right_in_bound = True

        # check left bound
        left_val = ((left_pt[0] - self.pos[0])*(rot_loc[1] - self.pos[1]) - (left_pt[1] - self.pos[1])*(rot_loc[0] - self.pos[0]))
        if left_val >= thresh: # left of line
            left_in_bound = False
        elif left_val <= -thresh: # right of line
            left_in_bound = True
        else: # on the line
            left_in_bound = True

        return (left_in_bound and right_in_bound)

    def update(self, world_state, other_agent):
        right_pt, left_pt = self.get_vision_bound()
        self.state[0] = self.pos
        self.state[1] = int(self.ori)

        # check if objects are in vision
        print('left_pt:', left_pt, 'right_pt:', right_pt)
        if self.in_bound(world_state.timer_loc, right_pt, left_pt):
            print('Timer in bound')
            self.state[2] = world_state.timer

        if self.in_bound(other_agent.pos, right_pt, left_pt):
            print('Other agent in bound')
            self.state[3], self.state[4] = other_agent.pos


if __name__ == '__main__':
    world = World(0, 10, 10)
    rand_agent = RandHuman([0,0], world)
    h0 = ImmersiveHuman([1,1], 0, world, rand_agent) 

    for i in range(1000):
        print('Clock:', world.timer, '; Random agent =', rand_agent.pos)
        h0.update(world, rand_agent)
        print('Immersive Human Belief:', h0.get_states())

        rand_agent.update_pos()
        world.update_timer(1)   
        h0.ori = input()