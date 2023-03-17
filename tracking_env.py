import numpy as np
import os
import igibson
from igibson.objects.articulated_object import URDFObject
from igibson import object_states
import pybullet as p

class TrackingEnv():
    def __init__(self, igibsonEnv, pans, bowls, onions, robot, human):
        self.env = igibsonEnv
        self.pans = pans
        self.bowls = bowls
        self.onions = onions
        self.robot = robot
        self.human = human

    def num_onions_in_bowl(self):
        bowl = self.bowls[0]
        num_onions = 0
        for o in self.onions:
            if bowl.states[object_states.Inside].get_value(o):
                num_onions += 1
        return num_onions
    
    def is_cooked(self):
        pan = self.pans[0]
        num_onions = 0
        for o in self.onions:
            if pan.states[object_states.Inside].get_value(o):
                num_onions += 1
        return num_onions
    
    def get_temp(self):
        temp = []
        cooked = []
        for o in self.onions:
            t = o.states[object_states.Temperature].get_value()
            c = o.states[object_states.Cooked].get_value()
            temp.append(t)
            cooked.append(c)
        return temp, cooked
    
    def obj_in_robot_hand(self):
        pass

    def obj_in_human_hand(self):
        pass