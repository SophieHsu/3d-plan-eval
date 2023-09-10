import copy
import math
import numpy as np
from igibson import object_states
from lsi_3d.agents.human_agent import HumanAgent
from lsi_3d.environment.lsi_env import LsiEnv
from lsi_3d.environment.tracking_env import TrackingEnv
import pybullet as p

from utils import real_to_grid_coord

class VisionLimitHumanAgent(HumanAgent):
    def __init__(self, human, planner, motion_controller, occupancy_grid, hlp, lsi_env:LsiEnv, tracking_env:TrackingEnv, vr=False, insight_threshold=5):
        super().__init__(human, planner, motion_controller, occupancy_grid, hlp, lsi_env, tracking_env, vr, insight_threshold)
        self.knowledge_base = None
        self.vision_limit = True
        self.vision_bound = 120

    def deepcopy(self, world_state=None):
        new_human_model = VisionLimitHumanAgent(self.human, self.planner, self.motion_controller, self.occupancy_grid, self.hlp, self.lsi_env, self.tracking_env, self.vr, self.insight_threshold)
        
        for k, v in self.knowledge_base.items():
            new_human_model.knowledge_base[k] = v

        return new_human_model

    def init_knowledge_base(self):
        self.knowledge_base = {}
        self.knowledge_base['mobile'] = {}
        self.knowledge_base['immobile'] = {}
        for obj in self.env.kitchen.get_mobile_objects():
            self.knowledge_base['mobile'][obj] = {'name': obj.name, 'pos': self.tracking_env.get_position(obj)}
        for obj in self.env.kitchen.get_immobile_objects():
            self.knowledge_base['immobile'][obj] = {'name': obj.name, 'pos': self.tracking_env.get_position(obj)}

        self.knowledge_base['other_player'] = {'name': obj.name, 'holding': self.env.robot_state.holding}
    
    def bound_func(self, slope, point):
        # slope defines field of view
        x,y = point
        return y < -slope * abs(x) + 1
    
    def in_bound(self, agent_rcf, loc, half_angle = 60):
        r,c,f = agent_rcf
        r_loc, c_loc = loc
        
        if f == 'N':
            theta = np.radians(180)
        elif f == 'E':
            theta = np.radians(90)
        elif f == 'S':
            theta = 0
        elif f == 'W':
            theta = np.radians(-90)

        # translate
        x, y = -(c_loc-c), -(r_loc-r)

        # rotate
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        point = R @ np.array([[x],[y]])
        point = (point[0][0],point[1][0])
        slope = 1/np.tan(np.radians(half_angle))
        return self.bound_func(slope, point)

    def update(self, rollout_kb = None):
        agent_rcf = self.env.human_state.ml_state
        if self.knowledge_base == None:
            self.init_knowledge_base()

        if rollout_kb is not None:
            tmp_kb = copy.deepcopy(rollout_kb)
        else:
            tmp_kb = self.knowledge_base

        for obj,state in self.knowledge_base['mobile'].items():
            curr_pos = self.tracking_env.get_position(obj)
            if self.in_bound(agent_rcf, curr_pos):
               tmp_kb['mobile'][obj]['pos'] = curr_pos

        # k is x,y
        for obj, state in tmp_kb['mobile'].items():
            if self.in_bound(agent_rcf, state['pos']):
                # object at x,y location
                if 'steak' in obj.name:
                    pans_status = self.tracking_env.get_pan_status()
                    
                    # assume on counter, change if not
                    tmp_kb['mobile'][obj]['pan'] = None
                    tmp_kb['mobile'][obj]['status'] = 'on_counter'
                    for pan in pans_status:
                        if self.in_bound(agent_rcf, real_to_grid_coord(pan.get_position())):
                            if obj.states[object_states.Inside].get_value(pan) and not obj.states[object_states.Cooked].get_value():
                                tmp_kb['mobile'][obj]['pan'] = pan
                                tmp_kb['mobile'][obj]['status'] = 'cooking'
                            elif obj.states[object_states.Cooked].get_value():
                                tmp_kb['mobile'][obj]['pan'] = pan
                                tmp_kb['mobile'][obj]['status'] = 'ready'
                    
                
                elif 'green_onion' in obj.name:
                    chop_status = self.tracking_env.get_chopping_board_status()
                    tmp_kb['mobile'][obj]['chop'] = None
                    tmp_kb['mobile'][obj]['status'] = 'on_counter'
                    for chop in chop_status:
                        if self.in_bound(agent_rcf, chop.get_position()):
                            if obj in chop_status[chop]:
                                tmp_kb['mobile'][obj]['chop'] = chop
                                if obj.current_index == 0:
                                    tmp_kb['mobile'][obj]['status'] = 'whole'
                                elif obj.current_index == 1:
                                    tmp_kb['mobile'][obj]['status'] = 'chopped'

                elif 'plate' in obj.name:
                    sink_status = self.tracking_env.get_sink_status()
                    tmp_kb['mobile'][obj]['sink'] = None
                    tmp_kb['mobile'][obj]['status'] = 'on_counter'
                    for sink in sink_status:
                        if self.in_bound(agent_rcf, sink.get_position()):
                            if obj in sink_status[sink]:
                                tmp_kb[obj]['sink'] = sink
                                if obj in self.env.kitchen.ready_plates:
                                    tmp_kb[obj]['status'] = 'hot'
                                else:
                                    tmp_kb[obj]['status'] = 'not_hot'
                        
        for obj, state in tmp_kb['immobile'].items():
            if self.in_bound(agent_rcf, state['pos']):
                if obj.name == 'pan':
                    in_pan = self.tracking_env.get_pan_status()[obj]
                    if len(in_pan) == 0:
                        tmp_kb['immobile'][obj]['status'] = 'empty'
                    else:
                        for m in in_pan:
                            if m.states[object_states.Cooked].get_value():
                                tmp_kb['immobile'][obj]['status'] = 'ready'
                            else:
                                tmp_kb['immobile'][obj]['status'] = 'cooking'

                if obj.name == 'chopping_board':
                    on_chop = self.tracking_env.get_chopping_board_status()[obj]
                    if len(on_chop) == 0:
                        tmp_kb['immobile'][obj]['status'] = 'empty'
                    else:
                        for o in on_chop:
                            if o.current_index == 0:
                                tmp_kb['immobile'][obj]['status'] = 'full'
                            else:
                                tmp_kb['immobile'][obj]['status'] = 'ready'

                if obj.name == 'sink':
                    in_sink = self.tracking_env.get_sink_status()[obj]
                    if len(in_sink) == 0:
                        tmp_kb['immobile'][obj]['status'] = 'empty'
                    else:
                        for p in in_sink:
                            if p in self.kitchen.hot_plates:
                                tmp_kb['immobile'][obj]['status'] = 'ready'
                            else:
                                tmp_kb['immobile'][obj]['status'] = 'full'
                
                # if 'counter' in obj.name:
                #     in_fridge = self.tracking_env.get_fridge_status()[obj]
                #     if len(in_fridge) == 0:
                #         tmp_kb['immobile'][obj]['status'] = 'empty'
                #     else:
                #         for m in in_fridge:
                #             if p in self.kitchen.hot_plates:
                #                 tmp_kb['immobile'][obj]['status'] = 'ready'
                #             else:
                #                 tmp_kb['immobile'][obj]['status'] = 'full'
                                
        # check if other player is in vision
        other_player = self.env.robot_state
        if self.in_bound(agent_rcf, other_player.ml_state[0:2]):
            tmp_kb['other_player']['holding'] = other_player.holding

        # print out knowledge base
        # if rollout_kb is None and self.debug:
        #     for k, v in tmp_kb.items():
        #         print(k, ':', v)

        self.update_kb_world_states()
        
        return tmp_kb
    
    def get_kb_pans(self):
        pans = []
        for k,b in self.knowledge_base.items():
            if 'pan' in k.name:
                pans.append(k)
        return pans
    
    def get_kb_chops(self):
        items = []
        for k,v in self.knowledge_base.items():
            if 'chopping_board' in k.name:
                items.append(k)
        return items
    
    def get_kb_sinks(self):
        items = []
        for k,v in self.knowledge_base.items():
            if 'sink' in k.name:
                items.append(k)
        return items

    def update_kb_world_states(self):
        # pans
        self.knowledge_base['pot_states'] = {}
        self.knowledge_base['pot_states']['empty'] = []
        self.knowledge_base['pot_states']['ready'] = []
        self.knowledge_base['pot_states']['cooking'] = []
        for pan in self.get_kb_pans():
            if self.knowledge_base['immobile'][pan]['status'] == 'empty':
                self.knowledge_base['pot_states']['empty'].append(pan)
            elif self.knowledge_base['immobile'][pan]['status'] == 'cooking':
                self.knowledge_base['pot_states']['cooking'].append(pan)
            elif self.knowledge_base['immobile'][pan]['status'] == 'ready':
                self.knowledge_base['pot_states']['ready'].append(pan)

        # chopping boards
        self.knowledge_base['chop_states'] = {}
        self.knowledge_base['chop_states']['empty'] = []
        self.knowledge_base['chop_states']['ready'] = []
        self.knowledge_base['chop_states']['full'] = []
        for chop in self.get_kb_chops():
            if self.knowledge_base['immobile'][chop]['status'] == 'empty':
                self.knowledge_base['chop_states']['empty'].append(chop)
            elif self.knowledge_base['immobile'][chop]['status'] == 'full':
                self.knowledge_base['chop_states']['full'].append(chop)
            elif self.knowledge_base['immobile'][chop]['status'] == 'ready':
                self.knowledge_base['chop_states']['ready'].append(chop)

        # sinks
        self.knowledge_base['sink_states'] = {}
        self.knowledge_base['sink_states']['empty'] = []
        self.knowledge_base['sink_states']['ready'] = []
        self.knowledge_base['sink_states']['full'] = []
        for sink in self.get_kb_sinks():
            if self.knowledge_base['immobile'][sink]['status'] == 'empty':
                self.knowledge_base['sink_states']['empty'].append(sink)
            elif self.knowledge_base['immobile'][sink]['status'] == 'full':
                self.knowledge_base['sink_states']['full'].append(sink)
            elif self.knowledge_base['immobile'][sink]['status'] == 'ready':
                self.knowledge_base['sink_states']['ready'].append(sink)


        # update orders left get number of bowls on table remove from original list
        order_list = self.world_state.init_orders.copy()
        for p in self.kitchen.plates:
            if self.tracking_env.is_item_on_table(p):
                order_list.pop()
                
        self.world_state.orders = order_list

        self.update_joint_ml_state()

        # update holding
        if len(self.tracking_env.obj_in_robot_hand()) > 0:
            self.robot_state.holding = self.tracking_env.obj_in_robot_hand()[0][0]
        else:
            self.robot_state.holding = 'None'

        if self.tracking_env.obj_in_human_hand() is not None:
            self.human_state.holding = self.tracking_env.get_human_holding()
        else:
            self.human_state.holding = 'None'

        return



    def get_next_goal(self):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        agent_state = self.env.human_state
        world_state = self.env.world_state
        robot_state = self.env.robot_state

        # TODO: update the state based on kb
        self.update()
        
        # player = state.players[self.agent_index]
        # other_player = self.knowledge_base['other_player']
        # am = self.mlp.ml_action_manager

        counter_objects = self.tracking_env.kitchen.counters
        # sink_status = self.knowledge_base['sink_states']
        # chopping_board_status = self.knowledge_base['chop_states']
        # pot_states_dict = self.knowledge_base['pot_states']
        sink_status = world_state.state_dict['sink_states']
        chopping_board_status = world_state.state_dict['chop_states']
        pot_states_dict = world_state.state_dict['pot_states']
        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = world_state.orders[-1]

        if curr_order == 'any':
            ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
            cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
        else:
            empty_pot = pot_states_dict['empty']
            ready_soups = pot_states_dict['ready']
            cooking_soups = pot_states_dict['cooking']

        steak_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        other_has_dish = robot_state.holding == 'dish'
        other_has_hot_plate = robot_state.holding == 'hot_plate'
        other_has_steak = robot_state.holding == 'steak'

        garnish_ready = len(chopping_board_status['ready']) > 0
        chopping = len(chopping_board_status['full']) > 0
        board_empty = len(chopping_board_status['empty']) > 0
        hot_plate_ready = len(sink_status['ready']) > 0
        rinsing = len(sink_status['full']) > 0
        sink_empty = len(sink_status['empty']) > 0
        motion_goals = []

        ready_soups = pot_states_dict['ready']
        cooking_soups = pot_states_dict['cooking']

        steak_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        other_has_dish = robot_state.holding == 'dish'
        other_has_hot_plate = robot_state.holding == 'hot_plate'
        other_has_steak = robot_state.holding == 'steak'
        other_has_meat = robot_state.holding == 'meat'
        other_has_onion = robot_state.holding == 'onion'
        other_has_plate = robot_state.holding == 'plate'

        garnish_ready = len(chopping_board_status['ready']) > 0
        chopping = len(chopping_board_status['full']) > 0
        board_empty = len(chopping_board_status['empty']) > 0
        hot_plate_ready = len(sink_status['ready']) > 0
        rinsing = len(sink_status['full']) > 0
        sink_empty = len(sink_status['empty']) > 0

        if agent_state.holding == 'None':

            if chopping and not garnish_ready:
                # motion_goals = am.chop_onion_on_board_actions(state, knowledge_base=self.knowledge_base)
                action,object = ('chop','onion')
            elif rinsing and not hot_plate_ready:
                action,object = ('heat','hot_plate')
                # motion_goals = am.heat_plate_in_sink_actions(state, knowledge_base=self.knowledge_base)
            elif not steak_nearly_ready and len(world_state.orders) > 0 and not other_has_meat:
                action,object = ('pickup','meat')
                # motion_goals = am.pickup_meat_actions(counter_objects, knowledge_base=self.knowledge_base)
            
            elif not chopping and not garnish_ready and not other_has_onion:
                action,object = ('pickup','onion')
            elif not rinsing and not hot_plate_ready and not other_has_plate:
                action,object = ('pickup','plate')
            
                # motion_goals = am.pickup_plate_actions(counter_objects, state, knowledge_base=self.knowledge_base)
            #elif garnish_ready and hot_plate_ready and not (other_has_hot_plate or other_has_steak):
            elif (garnish_ready or other_has_onion) and hot_plate_ready and not (other_has_hot_plate or other_has_steak):
                action,object = ('pickup','hot_plate')

            elif (garnish_ready or other_has_onion) and steak_nearly_ready and other_has_plate and not (other_has_hot_plate or other_has_steak) and not hot_plate_ready:
                # motion_goals += am.pickup_hot_plate_from_sink_actions(counter_objects, state, knowledge_base=self.knowledge_base)
                action, object = ('pickup','hot_plate')
                # if len(motion_goals) == 0:
                #     if other_has_plate:
                #         motion_goals += self.knowledge_base['sink_states']['full']
                #         if len(motion_goals) == 0:
                #             motion_goals += self.knowledge_base['sink_states']['empty']
                # motion_goals = am.pickup_hot_plate_from_sink_actions(counter_objects, state, knowledge_base=self.knowledge_base)
            else:
                next_order = world_state.orders[-1]

                if next_order == 'steak': #pick up plate first since that is the first empty key object when in the plating stage
                    action,object = ('pickup', 'plate')
                    # motion_goals = am.pickup_plate_actions(counter_objects, knowledge_base=self.knowledge_base)

        else:
            player_obj = agent_state.holding

            if player_obj == 'onion':
                action, object = ('drop', 'onion')
                # motion_goals = am.put_onion_on_board_actions(state, knowledge_base=self.knowledge_base)
            
            elif player_obj == 'meat':
                action,object = ('drop', 'meat')
                # motion_goals = am.put_meat_in_pot_actions(pot_states_dict)

            elif player_obj == "plate":
                action,object = ('drop', 'plate')
                # motion_goals = am.put_plate_in_sink_actions(counter_objects, state, knowledge_base=self.knowledge_base)

            elif player_obj == 'hot_plate' and (other_has_meat or steak_nearly_ready):
                action,object = ('pickup', 'steak')
                # motion_goals = am.pickup_steak_with_hot_plate_actions(pot_states_dict, only_nearly_ready=True)

            elif player_obj == 'steak' and (garnish_ready or other_has_onion or chopping):
                action,object = ('pickup', 'garnish')
                # motion_goals = am.add_garnish_to_steak_actions(state, knowledge_base=self.knowledge_base)

            elif player_obj == 'dish':
                action,object = ('deliver', 'dish')
                # motion_goals = am.deliver_dish_actions()

            # else:
            #     motion_goals += am.place_obj_on_counter_actions(state)


        possible_motion_goals = self.env.map_action_to_location(
            (action, object), self.env.human_state.ml_state[0:2], is_human=True)
        goal = possible_motion_goals
        # self.next_hl_state = next_hl_state
        self.action_object = (action, object)
        return goal
    
    def _arrival_step(self):
        hand_pos = self.human._parts["right_hand"].get_position()
        # next_hl_state = self.next_hl_state
        action, object = action_object = self.action_object
        if action == "pickup" and object == "meat":
            if self.object_position is None:
                self.target_object = self.tracking_env.get_closest_meat(self.human.get_position())
                self.object_position = self.target_object.get_position()

            is_holding = self.tracking_env.is_obj_in_human_hand(self.target_object)
            done = self.pick(self.object_position, [0, 0, 0.03])

            if done:
                if is_holding:
                    self.completed_goal(None, action_object)
                else:
                    self.object_position = None

        elif action == "drop" and object == "meat":
            if self.object_position is None:
                self.object_position = self.tracking_env.get_closest_pan(
                ).get_position()
            done = self.drop(self.object_position, [0, -0.1, 0.25])
            if done:
                self.completed_goal(None, action_object)
        elif action == "pickup" and object == "plate":
            if self.object_position is None:
                self.target_object = self.tracking_env.get_closest_plate(hand_pos)
            self.object_position = self.target_object.get_position()

            is_holding = self.tracking_env.is_obj_in_human_hand(self.target_object)
            done = self.pick(self.object_position, [0, -0.3, 0.1]) and is_holding # for plate use : [0, -0.21, 0.03]
            if done:
                self.completed_goal(None, action_object)
        elif action == "pickup" and object == "hot_plate":
            if self.object_position is None:
                self.target_object, sink = self.tracking_env.get_closest_hot_plate_sink(hand_pos)
                sink_pos = sink.get_position()
                # orn = self.tracking_env.kitchen.orientation_map[("bowl", x, y)]
                # shift = self.tracking_env.kitchen.name2shift_map["bowl"]
                self.env.kitchen.hot_plates.append(self.target_object)
                pos = self.motion_controller.translate_loc(self.human, sink_pos, [0.3, 0, 0.5])
                # pos = [sink_pos[0], sink_pos[1], 1.2]
                self.env.nav_env.set_pos_orn_with_z_offset(self.target_object, tuple(pos), (0, 0, -1.5707))
            self.object_position = self.target_object.get_position()

            is_holding = self.tracking_env.is_obj_in_human_hand(self.target_object)
            done = self.pick(self.object_position, [0, -0.3, 0.1]) and is_holding # for plate use : [0, -0.21, 0.03]
            if done:
                self.completed_goal(None, action_object)
        elif action == "drop" and object == "plate":
            if self.object_position is None:
                sink = self.tracking_env.get_closest_sink(hand_pos
                )
                self.object_position = sink.get_position()
            done = self.drop(self.human.get_position(), [0, 0.5, 0.2])
            if done:
                sink = self.tracking_env.get_closest_sink(hand_pos)
                self.env.kitchen.rinse_sink(sink)
                self.completed_goal(None, action_object)
        elif action == "pickup" and object == "steak":
            if self.object_position is None:
                pan = self.tracking_env.get_closest_pan()
                self.tracking_env.kitchen.interact_objs[pan] = True
                self.interact_obj = pan
                self.object_position = pan.get_position()
            if self.step_index == 0:
                done = self.drop(self.object_position, [-0.4, -0.25, 0.3])
                if done:
                    self.step_index = self.step_index + 1
                    steak = self.tracking_env.get_closest_steak(self.human.get_position())
                    self.object_position = steak.get_position()
                    self.target_object = steak
            elif self.step_index == 1:
                is_holding_steak = self.tracking_env.is_obj_in_human_hand(self.target_object)
                done = self.pick(self.object_position, [0, -0.03, 0.05])
                if done:
                    if is_holding_steak:
                        self.step_index = self.step_index + 1
                        self.object_position = self.tracking_env.get_closest_bowl(
                        ).get_position()
                    else:
                        bowl = self.tracking_env.get_closest_bowl()
                        bowl_pos = bowl.get_position()
                        self.target_object.set_position(bowl_pos + [0,0,0.3])
                        self.step_index = 3
                        self.target_object = bowl
                        self.object_position = bowl_pos
            elif self.step_index == 2:
                done = self.drop(self.object_position, [0, -0.1, 0.3])
                if done:
                    self.step_index = 3
                    bowl = self.tracking_env.get_closest_bowl()
                    self.object_position = bowl.get_position()
                    self.target_object = bowl
            elif self.step_index == 3:
                is_holding_bowl = self.tracking_env.is_obj_in_human_hand(self.target_object)
                done = self.pick(self.object_position, [0, -0.3, 0.1]) and is_holding_bowl
                if done:
                    self.step_index = self.step_index + 1
            else:
                self.completed_goal(None, action_object)
                self.tracking_env.kitchen.interact_objs[self.interact_obj] = False
        elif action == "pickup" and object == "garnish":
            if self.object_position is None:
                chopped_onion = self.tracking_env.get_closest_chopped_onion(self.human.get_position())
                self.tracking_env.kitchen.interact_objs[chopped_onion] = True
                self.interact_obj = chopped_onion
                self.object_position = chopped_onion.current_selection().objects[0].get_position()
                self.target_object = chopped_onion
            if self.step_index == 0:
                done = self.drop(self.object_position, [-0.4, -0.25, 0.3])
                if done:
                    self.step_index = self.step_index + 1
            elif self.step_index == 1:
                done = self.pick(self.object_position, [0, -0.03, 0.2])
                if done:
                    bowl = self.tracking_env.get_closest_bowl()
                    bowl_pos = bowl.get_position()
                    for i,object in enumerate(self.target_object.current_selection().objects):
                        object.set_position(bowl_pos + [0,0,0.05 + i*0.05])
                    self.step_index = 2
                    self.target_object = bowl
                    self.object_position = bowl_pos
            elif self.step_index == 2:
                done = self.drop(self.human._parts["right_hand"].get_position(), [0,0,0])
                if done:
                    self.step_index = 3
            elif self.step_index == 3:
                is_holding_bowl = self.tracking_env.is_obj_in_human_hand(self.target_object)
                done = self.pick(self.object_position, [0, -0.3, 0.1]) and is_holding_bowl
                if done:
                    self.step_index = self.step_index + 1
            else:
                self.completed_goal(None, action_object)
                self.tracking_env.kitchen.interact_objs[self.interact_obj] = False
        elif action == "chop" and object == "onion":
            if self.object_position is None:
                knife = self.tracking_env.get_closest_knife(self.human.get_position())
                self.tracking_env.kitchen.interact_objs[knife] = True
                self.interact_obj = knife
                self.object_position = knife.get_position()
                self.target_object = knife
            elif self.step_index == 0:
                is_holding = self.tracking_env.is_obj_in_human_hand(self.target_object)
                done = self.pick(self.object_position, [0, -0.02, 0.03])
                if done:
                    if is_holding:
                        self.step_index = 2
                        self.target_object = self.tracking_env.get_closest_chopping_board(self.human.get_position())
                        self.object_position = self.target_object.get_position()
                    else:
                        # set onion to sliced
                        o = self.tracking_env.get_closest_green_onion(self.human.get_position())
                        o.states[object_states.Sliced].set_value(True)
                        self.completed_goal(None, action_object)
                        self.tracking_env.kitchen.interact_objs[self.interact_obj] = False
                        self.step_index = 1
            elif self.step_index == 1:
                reset_hand_ori = [-2.908, 0.229, 0]
                self.human._parts["right_hand"].set_orientation(p.getQuaternionFromEuler(reset_hand_ori))
                done = self.drop(self.human._parts["right_hand"].get_position(), [0,0,0])
                if done:
                    self.step_index = 2

            elif self.step_index == 2:
                done = self.drop(self.object_position, [0, -0.3, 0.3])
                if done:
                    o = self.tracking_env.get_closest_green_onion(self.human.get_position())
                    o.states[object_states.Sliced].set_value(True)
                    self.completed_goal(None, action_object)
                    self.tracking_env.kitchen.interact_objs[self.interact_obj] = False
                
        elif action == "pickup" and object == "onion":
            if self.object_position is None:
                self.target_object = self.tracking_env.get_closest_green_onion(hand_pos)
            self.object_position = self.target_object.get_position()
            is_holding = self.tracking_env.is_obj_in_human_hand(self.target_object)
            done = self.pick(self.object_position, [0, -0.1, 0.04])

            if done:
                if is_holding:
                    self.completed_goal(None, action_object)
                else:
                    self.target_object.set_position(self.object_position + [0,0,0.3])
                    self.object_position = None
        elif action == "drop" and object == "onion":
            if self.object_position is None:
                self.object_position = self.tracking_env.get_closest_chopping_board(hand_pos
                ).get_position()
            done = self.drop(self.object_position, [0, -0.1, 0.25])
            if done:
                self.completed_goal(None, action_object)
        elif action == "deliver" and (object == "soup" or object == "dish"):
            done = self.drop(self.human.get_position(), [0, 0.5, 0.2])
            if done:
                self.completed_goal(None, action_object)
        elif (action == "pickup" and object == "soup") or self.step_index >= 1:
            if self.object_position is None:
                pan = self.tracking_env.get_closest_pan()
                self.tracking_env.kitchen.interact_objs[pan] = True
                self.interact_obj = pan
                self.object_position = pan.get_position()
            if self.step_index == 0:
                done = self.drop(self.object_position, [-0.4, -0.25, 0.3])
                if done:
                    self.step_index = self.step_index + 1
                    onion = self.tracking_env.get_closest_onion(
                        on_pan=True)
                    self.object_position = onion.get_position()
                    self.target_object = onion
            elif self.step_index == 1:
                is_holding_onion = self.tracking_env.is_obj_in_human_hand(self.target_object)
                done = self.pick(self.object_position, [0, -0.05, 0.05]) and is_holding_onion
                if done:
                    self.step_index = self.step_index + 1
                    self.object_position = self.tracking_env.get_closest_bowl(
                    ).get_position()
            elif self.step_index == 2:
                done = self.drop(self.object_position, [0, -0.1, 0.3])
                if done:
                    num_item_in_bowl = len(self.tracking_env.get_bowl_status()[self.tracking_env.get_closest_bowl()])
                    if num_item_in_bowl < self.tracking_env.kitchen.onions_for_soup:
                        self.step_index = 1
                        onion = self.tracking_env.get_closest_onion(on_pan=True)
                        self.object_position = onion.get_position()
                        self.target_object = onion
                    else:
                        self.step_index = 3
                        bowl = self.tracking_env.get_closest_bowl()
                        self.object_position = bowl.get_position()
                        self.target_object = bowl
            elif self.step_index == 3:
                is_holding_bowl = self.tracking_env.is_obj_in_human_hand(self.target_object)
                done = self.pick(self.object_position, [0, -0.3, 0.1]) and is_holding_bowl
                if done:
                    self.step_index = self.step_index + 1
            else:
                self.completed_goal(next_hl_state, action_object)
                self.tracking_env.kitchen.interact_objs[self.interact_obj] = False
        # print(next_hl_state, action_object)