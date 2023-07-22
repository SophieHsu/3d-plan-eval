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
            if len(onions) > 0:
                data[b] = onions
        return data
    
    def set_interact_obj(self,obj,is_interacting):
        self.kitchen.interact_objs[obj] = is_interacting

    def is_interact_obj(self, obj):
        return self.kitchen.interact_objs[obj]

    def items_in_bowl(self, bowl):
        data = []
        for o in self.kitchen.onions:
            if o.states[object_states.Inside].get_value(bowl):
                data.append(o)
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
    
    def get_empty_counters(self):
        empty_counters = []
        for c in self.kitchen.counters:
            is_empty = True
            for o in self.kitchen.onions:
                if o.states[object_states.OnTop].get_value(c):
                    is_empty = False
            
            for b in self.kitchen.bowls:
                if b.states[object_states.OnTop].get_value(c):
                    is_empty = False

            if is_empty: empty_counters.append(c)

        return empty_counters

    def dist_sort(self, objects, agent_pos):
        # agent pos is real coordinate
        sorted_positions = sorted(objects, key=lambda object: self.distance_to_bowl(object, agent_pos))
        return sorted_positions
    
    def get_pan_enum_status(self, pan):
        status = self.get_pan_status()
        onions_in_pan = len(status[pan])

        if self.kitchen.interact_objs[pan]:
            return 3
        elif onions_in_pan >= self.kitchen.onions_for_soup:
            return 2
        elif onions_in_pan == 0:
            return 1
        else:
            return 0

    def is_pan_cooked(self, pan):
        num_cooked_onions = 0
        num_uncooked_onions = 0
        for o in self.kitchen.onions:
            if o.states[object_states.OnTop].get_value(pan):
                if o.states[object_states.Cooked].get_value():
                    num_cooked_onions += 1
                else:
                    num_uncooked_onions += 1
        return (num_cooked_onions, num_uncooked_onions)

    def is_item_on_table(self, item):
        val = item.states[object_states.OnTop].get_value(self.kitchen.table)
        return val
    
    def is_item_on_counter(self, item):
        for counter in self.kitchen.counters:
            val = item.states[object_states.OnTop].get_value(counter)
            if val:
                return True
        return False

    def obj_in_robot_hand(self):
        # return object in robot hand
        return self.kitchen.in_robot_hand
    
    def is_obj_in_human_hand(self, obj):
        body_id = obj.get_body_ids()[0]
        grasping = self.human.is_grasping_all_arms(body_id)
        return IsGraspingState.TRUE in grasping
    
    def is_human_holding_bowl(self):
        for bowl in self.kitchen.bowls:
            if self.is_obj_in_human_hand(bowl):
                return True
        return False

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
                o.set_position([0, 0, 0])

    # def open_fridge(self):
    #     fridge = self.kitchen.fridges[0]
    #     print(fridge.states)
    #     fridge.states[object_states.Open].set_value(True)

    def set_in_robot_hand(self, name, obj):
        self.kitchen.in_robot_hand.append([name, obj])

    def get_table_locations(self):
        return self.kitchen.static_objs[self.kitchen.table]

    def remove_in_robot_hand(self, item, pos=None, counter=0):
        self.kitchen.in_robot_hand.remove(item)
        ori_ori = item[1].get_orientation()
        ori_pos = item[1].get_position()
        item[1].set_orientation([ori_ori[0], 0, 0, 0])
        if pos is not None:
            pos[-1] += counter * 0.01
            item[1].set_position(pos)
            item[1].set_orientation([0, 0, 0, 1])

    # def get_closest_onion(self, agent_pos=None, on_pan=False):
    #     closest_onion = None
    #     min_dist = 10000
    #     if agent_pos is None:
    #         position = self.human._parts["right_hand"].get_position()
    #     else:
    #         position = agent_pos

    def distance_to_bowl(self, bowl, position):
        bowl_position = bowl.get_position()[0:2]
        position = position[0:2]
        return math.dist(bowl_position, position)

    def get_bowls_dist_sort(self, is_human=None):
        if is_human is None or is_human is True:
            position = self.human._parts["right_hand"].get_position()
        else:
            position = self.robot.object.get_position()

        sorted_positions = sorted(self.kitchen.bowls, key=lambda bowl: self.distance_to_bowl(bowl, position))
        return sorted_positions

    def get_closest_onion(self, agent_pos=None, on_pan=False, position=None, on_table = False):
        closest_onion = None
        min_dist = 10000
        if agent_pos is None:
            position = self.human._parts["right_hand"].get_position(
            ) if position is None else position
        else:
            position = agent_pos
        closest_pan = self.get_closest_pan(agent_pos)
        for o in self.kitchen.onions:
            if on_pan and not o.states[object_states.OnTop].get_value(
                    closest_pan):
                continue
            onion_position = o.get_position()
            dist = math.dist(position, onion_position)
            if dist < min_dist:
                if on_table and not self.is_item_on_table(o): continue
                min_dist = dist
                closest_onion = o
        return closest_onion

    def get_closest_pan(self, agent_pos=None):
        closest_pan = None
        min_dist = 10000
        position = self.human._parts["right_hand"].get_position()
        if agent_pos is None:
            position = self.human._parts["right_hand"].get_position()
        else:
            position = agent_pos
        for p in self.kitchen.pans:
            pan_position = p.get_position()
            dist = math.dist(position, pan_position)
            if dist < min_dist:
                min_dist = dist
                closest_pan = p
        return closest_pan

    def get_closest_bowl(self, agent_pos=None):
        closest_bowl = None
        min_dist = 10000
        position = self.human._parts["right_hand"].get_position()
        if agent_pos is None:
            position = self.human._parts["right_hand"].get_position()
        else:
            position = agent_pos
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
        x, y, z = quat2euler(orientation[0], orientation[1], orientation[2],
                             orientation[3])
        return [x, y, z]
