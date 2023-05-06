import numpy as np
import os
import math
import igibson
from igibson.objects.articulated_object import URDFObject
from igibson.robots.manipulation_robot import IsGraspingState
from igibson import object_states
import pybullet as p
from utils import quat2euler

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
                if o.states[object_states.OnTop].get_value(p):
                    onions.append(o)
            data[p] = onions
        return data
    
    def is_pan_cooked(self, pan):
        num_cooked_onions = 0
        num_uncooked_onions = 0
        for o in self.kitchen.onions:
            if o.states[object_states.OnTop].get_value(pan):
                if o.states[object_states.Cooked].get_value():
                    num_cooked_onions +=1
                else:
                    num_uncooked_onions += 1
        return [num_cooked_onions, num_uncooked_onions]

    def is_item_on_table(self, items):
        on_table_bool = []
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
            
    def clear_pot(self):
        pan = self.kitchen.pans[0]
        for o in self.kitchen.onions:
            if o.states[object_states.OnTop].get_value(pan):
                o.set_position([0,0,0])
    # def open_fridge(self):
    #     fridge = self.kitchen.fridges[0]
    #     print(fridge.states)
    #     fridge.states[object_states.Open].set_value(True)

    def get_closest_onion(self, on_pan=False, position=None):
        closest_onion = None
        min_dist = 10000
        position = self.human._parts["right_hand"].get_position() if position is None else position
        closest_pan = self.get_closest_pan()
        for o in self.kitchen.onions:
            if on_pan and not o.states[object_states.OnTop].get_value(closest_pan):
                continue
            onion_position = o.get_position()
            dist = math.dist(position, onion_position)
            if dist < min_dist:
                min_dist = dist
                closest_onion = o
        return closest_onion
    
    def get_closest_pan(self):
        closest_pan = None
        min_dist = 10000
        position = self.human._parts["right_hand"].get_position()
        for p in self.kitchen.pans:
            pan_position = p.get_position()
            dist = math.dist(position, pan_position)
            if dist < min_dist:
                min_dist = dist
                closest_pan = p
        return closest_pan
    
    def get_closest_bowl(self):
        closest_bowl = None
        min_dist = 10000
        position = self.human._parts["right_hand"].get_position()
        for p in self.kitchen.bowls:
            bowl_position = p.get_position()
            dist = math.dist(position, bowl_position)
            if dist < min_dist:
                min_dist = dist
                closest_bowl = p
        return closest_bowl

    def get_human_position(self):
        return self.human.get_position()
    
    def get_human_orientation(self):
       orientation = self.human.get_orientation()
       x, y, z = quat2euler(orientation[0], orientation[1], orientation[2], orientation[3])
       return [x, y, z]
