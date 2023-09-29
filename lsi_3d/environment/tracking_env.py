import numpy as np
import os
import math
import igibson
from igibson.objects.articulated_object import URDFObject
from igibson.objects.multi_object_wrappers import ObjectMultiplexer
from igibson.robots.manipulation_robot import IsGraspingState
from igibson import object_states
import pybullet as p
from lsi_3d.environment.kitchen import Kitchen
from utils import quat2euler, real_to_grid_coord


class TrackingEnv():

    def __init__(self, igibsonEnv, kitchen:Kitchen, robot, human):
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
    
    def get_chopping_board_status(self):
        data = {}
        for c in self.kitchen.chopping_boards:
            onions = []
            for o in self.kitchen.onions:
                if o.name == "green_onion_multiplexer":
                    # on top not working for chopped onion
                    if o.current_index == 1 and real_to_grid_coord(c.get_position()) == real_to_grid_coord(o.current_selection().objects[0].get_position()):
                        onions.append(o)
                if o.states[object_states.OnTop].get_value(c):
                    onions.append(o)
            data[c] = onions
        return data
    
    def get_sink_status(self, same_location = False):
        data = {}
        for s in self.kitchen.sinks:
            plates = []
            for p in self.kitchen.plates:
                if p.states[object_states.Inside].get_value(s):
                    plates.append(p)

                if p.states[object_states.OnTop].get_value(s):
                    plates.append(p)

                if real_to_grid_coord(p.get_position()) == real_to_grid_coord(s.get_position()) and same_location:
                    plates.append(p)

            data[s] = plates
        return data
    
    def get_closest_hot_plate_sink(self, agent_pos):
        closest_sink = self.dist_sort(self.kitchen.ready_sinks, agent_pos)[0]
        status = self.get_sink_status()
        hot_plate = status[closest_sink][0]
        return (hot_plate, closest_sink)
    
    def is_item_in_object(self, item, object):
        if item.states[object_states.Inside].get_value(object):
            return True
        else:
            return False
        
    def set_item_in_object(self, item, object):
        item.states[object_states.Inside].set_value(object, True)
    
    def set_interact_obj(self,obj,is_interacting):
        self.kitchen.interact_objs[obj] = is_interacting

    def is_interact_obj(self, obj):
        return self.kitchen.interact_objs[obj]

    def items_in_bowl(self, bowl):
        data = []
        for o in self.kitchen.onions:
            if o.states[object_states.Inside].get_value(bowl):
                data.append(o)
        for o in self.kitchen.steaks:
            if o.states[object_states.Inside].get_value(bowl):
                data.append(o)
        return data

    def get_pan_status(self):
        data = {}
        for p in self.kitchen.pans:
            in_pan = []
            for o in self.kitchen.onions:
                if o.states[object_states.OnTop].get_value(p):
                    in_pan.append(o)

            for m in self.kitchen.meats:
                if m.states[object_states.OnTop].get_value(p):
                    in_pan.append(m)
            data[p] = in_pan

            for m in self.kitchen.steaks:
                if m.states[object_states.OnTop].get_value(p):
                    in_pan.append(m)
            data[p] = in_pan
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
        sorted_positions = sorted(objects, key=lambda object: self.distance_to_object(object, agent_pos))
        # sorted_positions.reverse()
        return sorted_positions
    
    def get_pan_enum_status(self, pan):
        status = self.get_pan_status()
        onions_in_pan = len(status[pan])

        # returns zero for almost full pan which should be chosen first
        # returns 1 for empty pan, then 2 for full pan

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

        tables = self.get_table_locations()
        if real_to_grid_coord(item.get_position()) in tables:
            return True
        return val
    
    def is_item_on_counter(self, item):
        for counter in self.kitchen.counters:
            val = item.states[object_states.OnTop].get_value(counter)
            if val:
                return True
        return False
    
    def get_closest_counter(self, agent_pos):
        counters = self.dist_sort(self.kitchen.counters, agent_pos)
        return counters[0]
    
    def set_item_on_closest_counter(self, obj):
        target_object = self.get_closest_counter(
                agent_pos=self.get_real_position(obj))
        object_position = target_object.get_position()
        # self.object_position = self.object.get_eef_position()
        self.set_position(obj, object_position + [0,0,0.7])

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
    
    def get_onions_in_human_soup(self):
        for bowl in self.kitchen.bowls:
            if self.is_obj_in_human_hand(bowl):
                ingredients = self.items_in_bowl(bowl)
                return ingredients
        return []

    def obj_in_human_hand(self):
        all_objs = self.kitchen.onions + self.kitchen.pans + self.kitchen.bowls + self.kitchen.meats + self.kitchen.plates + self.kitchen.hot_plates + self.kitchen.steaks
        for obj in all_objs:
            body_id = obj.get_body_ids()[0]
            grasping = self.human.is_grasping_all_arms(body_id)
            if IsGraspingState.TRUE in grasping:
                return obj
        return None
    
    # def get_obj_in_robot_hand(self):
    #     all_objs = self.kitchen.onions + self.kitchen.pans + self.kitchen.bowls + self.kitchen.meats + self.kitchen.plates + self.kitchen.hot_plates + self.kitchen.steaks
    #     for obj in all_objs:
    #         body_id = obj.get_body_ids()[0]
    #         grasping = self.robot.object.is_grasping_all_arms(body_id)
    #         if IsGraspingState.TRUE in grasping:
    #             return obj
    #     return None
        # return self.kitchen.in_human_hand

    def get_human_holding(self):
        for obj in self.kitchen.onions:
            body_id = obj.get_body_ids()[0]
            grasping = self.human.is_grasping_all_arms(body_id)
            if IsGraspingState.TRUE in grasping:
                return 'onion'
            
        for obj in self.kitchen.meats:
            body_id = obj.get_body_ids()[0]
            grasping = self.human.is_grasping_all_arms(body_id)
            if IsGraspingState.TRUE in grasping:
                return 'meat'
        
        for obj in self.kitchen.bowls:
            body_id = obj.get_body_ids()[0]
            grasping = self.human.is_grasping_all_arms(body_id)
            
            bowl_status = self.get_bowl_status()
            if IsGraspingState.TRUE in grasping:
                
                for onion in self.kitchen.onions:
                    if onion.states[object_states.Inside].get_value(obj):
                        return 'dish'
                for steak in self.kitchen.steaks:
                    if steak.states[object_states.Inside].get_value(obj):
                        return 'steak'
                if obj in self.kitchen.hot_plates:
                    return 'hot_plate'
                if obj in bowl_status.keys() and len(bowl_status[obj]) >= self.kitchen.onions_for_soup-1:
                    return 'soup'
                else:
                    return 'plate'
                
        for obj in self.kitchen.plates:
            body_id = obj.get_body_ids()[0]
            grasping = self.human.is_grasping_all_arms(body_id)
            if obj in self.kitchen.hot_plates:
                return 'hot_plate'
            if IsGraspingState.TRUE in grasping:
                return 'plate'
            
        return 'None'

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
        if [name, obj] not in self.kitchen.in_robot_hand:
            self.kitchen.in_robot_hand.append([name, obj])

    def get_table_locations(self):
        return self.kitchen.static_objs[self.kitchen.table]
    
    def get_closest_table_location(self, agent_pos):
        return self.dist_sort(self.get_table_locations, agent_pos)[0]

    def remove_in_robot_hand(self, item, pos=None, counter=0):
        self.kitchen.in_robot_hand.remove(item)
        ori_ori = self.get_orientation(item[1])
        ori_pos = self.get_real_position(item[1])
        self.set_orientation(item[1], [ori_ori[0], 0, 0, 0])
        if pos is not None:
            pos[-1] += counter * 0.01
            self.set_position(item[1], pos)
            self.set_orientation(item[1], [0, 0, 0, 1])

    # def get_closest_onion(self, agent_pos=None, on_pan=False):
    #     closest_onion = None
    #     min_dist = 10000
    #     if agent_pos is None:
    #         position = self.human._parts["right_hand"].get_position()
    #     else:
    #         position = agent_pos

    def distance_to_object(self, object, position):
        object_position = self.get_real_position(object)[0:2]
        position = position[0:2]
        return math.dist(object_position, position)
    
    def get_closest_plate(self, agent_pos):
        closest_plate = None
        position = agent_pos
        plates = self.dist_sort(self.kitchen.plates, agent_pos)
        for p in plates:
            for c in self.kitchen.counters:
                on_c = p.states[object_states.OnTop].get_value(c)
                if on_c:
                    closest_plate = p
                    break

        return closest_plate
    
    def get_closest_plate_in_sink(self, agent_pos):
        closest_plate = None
        position = agent_pos
        plates = self.dist_sort(self.kitchen.plates, agent_pos)
        for p in plates:
            for c in self.kitchen.sinks:
                on_c = p.states[object_states.OnTop].get_value(c)
                if on_c:
                    closest_plate = p
                    break

                on_c = p.states[object_states.Inside].get_value(c)
                if on_c:
                    closest_plate = p
                    break

                # sink_pos = real_to_grid_coord(c.get_position())[0:2]
                # plate_pos = real_to_grid_coord(p.get_position())[0:2]
                # if sink_pos == plate_pos:
                #     closest_plate = p
                #     break

        return closest_plate
    
    def get_closest_sink(self, agent_pos):
        position = agent_pos
        sinks = self.dist_sort(self.kitchen.sinks, agent_pos)

        return sinks[0]
    
    def get_closest_steak(self, agent_pos):
        if len(self.kitchen.steaks) > 0:
            return self.dist_sort(self.kitchen.steaks, agent_pos)[0]
        else:
            # for meat in self.kitchen.meats:
            #     for pan in self.kitchen.pans:
            #         if meat.state[object_states.Inside].get_value(pan):
            #             return meat
            return self.dist_sort(self.kitchen.meats, agent_pos)[0]
    
    def get_closest_knife(self, agent_pos):
        return self.dist_sort(self.kitchen.knives, agent_pos)[0]
    
    def get_closest_chopped_onion(self, agent_pos):
        chopped_onions = [o for o in self.kitchen.onions if o.current_index == 1]
        if len(chopped_onions) == 0:
            return None
    
        return self.dist_sort(chopped_onions, agent_pos)[0]
    
    def in_start_location(self, object):
        obj_position = self.get_position(object)
        if 'plate' in object.name:
            return obj_position == self.kitchen.where_grid_is('D')[0]
        if 'steak' in object.name:
            return obj_position == self.kitchen.where_grid_is('F')[0]
        if 'onion' in object.name:
            return obj_position == self.kitchen.where_grid_is('G')[0]
        else:
            return False
        
    def get_hot_plates(self, agent_pos):
        hot_plates = []
        for sink in self.kitchen.ready_sinks:
            for plate in self.kitchen.plates:
                if plate.states[object_states.Inside].get_value(sink):
                    hot_plates.append(plate)
        hot_plates = self.dist_sort(hot_plates, agent_pos)
        return hot_plates
    
    def get_closest_chopping_board(self, agent_pos):
        cs = self.dist_sort(self.kitchen.chopping_boards, agent_pos)
        return cs[0]
    
    def get_position(self, obj):
        if obj.name == 'green_onion_multiplexer':
            if obj.current_index == 0:
                return real_to_grid_coord(obj.get_position())
            else:
                return real_to_grid_coord(obj.current_selection().objects[0].get_position())
        else:
            return real_to_grid_coord(obj.get_position())
        
    def get_real_position(self, obj):
        if obj.name == 'green_onion_multiplexer':
            if obj.current_index == 0:
                return obj.get_position()
            else:
                return obj.current_selection().objects[0].get_position()
        else:
            return obj.get_position()
        
    def get_orientation(self, obj):
        if obj.name == 'green_onion_multiplexer':
            if obj.current_index == 0:
                return obj.get_orientation()
            else:
                return obj.current_selection().objects[0].get_orientation()
        else:
            return obj.get_orientation()
        
    def set_position(self, obj, pos):
        if obj.name == 'green_onion_multiplexer':
            if obj.current_index == 0:
                obj.set_position(pos)
            else:
                for sub_obj in obj.current_selection().objects:
                    sub_obj.set_position(pos)
        else:
            obj.set_position(pos)
        
    def set_orientation(self, obj, orientation):
        if obj.name == 'green_onion_multiplexer':
            if obj.current_index == 0:
                return obj.set_orientation(orientation)
            else:
                for sub_obj in obj.current_selection().objects:
                    sub_obj.set_orientation(orientation)
        else:
            return obj.set_orientation(orientation)
        

    def get_bowls_dist_sort(self, is_human=None):
        if is_human is None or is_human is True:
            position = self.human._parts["right_hand"].get_position()
        else:
            position = self.robot.object.get_position()

        sorted_positions = sorted(self.kitchen.bowls, key=lambda bowl: self.distance_to_object(bowl, position))
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
    
    def get_closest_meat(self, agent_pos=None):
        closest_meat = None
        position = agent_pos
        meats = self.dist_sort(self.kitchen.meats, agent_pos)
        for m in meats:
            for f in self.kitchen.fridges:
                on_c = m.states[object_states.OnTop].get_value(f)
                if on_c:
                    closest_meat = m
                    break

        return closest_meat
    
    def get_closest_green_onion(self, agent_pos=None, chopped=None):
        closest = None
        position = agent_pos
        objects = self.dist_sort(self.kitchen.onions, agent_pos)
        # for o in objects:
        #     for f in self.kitchen.counters:
        #         on_c = o.states[object_states.OnTop].get_value(f)
        #         if on_c:
        #             closest = o
        #             break
        closest = None
        if chopped is not None:
            for obj in objects:
                if chopped == False and obj.current_index == 0:
                    return obj
                elif chopped == True and obj.current_index == 1:
                    return obj
        closest = objects[0]

        return closest

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
            dist = math.dist(position[0:2], pan_position[0:2])
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
