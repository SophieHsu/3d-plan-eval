import numpy as np
import os
import igibson
from igibson.objects.articulated_object import URDFObject
from igibson.robots.manipulation_robot import IsGraspingState
from igibson import object_states
import pybullet as p

class TrackingEnv():
    def __init__(self, igibsonEnv, kitchen, robot, human):
        self.env = igibsonEnv
        self.kitchen = kitchen
        self.robot = robot
        self.human = human

    def get_bowl_status(self):
        data = {}
        for b in self.kitchen.bowls:
            onions = []
            for o in self.kitchen.onions:
                if o.states[object_states.Inside].get_value(b):
                    onions.append(o)
            data[b] = onions
        return data
    
    def get_pan_status(self):
        data = {}
        for p in self.kitchen.pans:
            onions = []
            for o in self.kitchen.onions:
                if o.states[object_states.Inside].get_value(p):
                    onions.append(o)
            data[p] = onions
        return data
    
    def is_pan_cooked(self, pan):
        num_cooked_onions = 0
        num_uncooked_onions = 0
        for o in self.kitchen.onions:
            if o.states[object_states.Inside].get_value(pan):
                if o.states[object_states.Cooked].get_value():
                    num_cooked_onions +=1
                else:
                    num_uncooked_onions += 1
        return [num_cooked_onions, num_uncooked_onions]

    def is_item_on_table(self, items):
        on_table_bool = []
        # on_table_bool.
        for i in items:
            val = i.states[object_states.OnTop].get_value(self.table)
            on_table_bool.append(val)
        return on_table_bool
    
    def obj_in_robot_hand(self):
        # return object in robot hand
        pass

    def obj_in_human_hand(self):
        all_objs = self.kitchen.onions + self.kitchen.pans + self.kitchen.bowls
        for obj in all_objs:
            body_id = obj.get_body_ids()[0]
            grasping = self.human.is_grasping_all_arms(body_id)
            if IsGraspingState.TRUE in grasping:
                return obj