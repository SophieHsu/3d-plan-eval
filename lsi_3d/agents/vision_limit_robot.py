from ast import literal_eval
import math
import time
import numpy as np
from lsi_3d.agents.agent import Agent
from lsi_3d.agents.hl_qmdp_agent import MAX_DELAY_TIME, STUCK_TIME_LIMIT, HlQmdpPlanningAgent
from lsi_3d.agents.igibson_agent import ONE_STEP, iGibsonAgent
from lsi_3d.environment.lsi_env import LsiEnv
from lsi_3d.planners.hl_qmdp_planner import HumanSubtaskQMDPPlanner
from lsi_3d.planners.mid_level_motion import AStarMotionPlanner
from lsi_3d.planners.steak_human_subtask_qmdp_planner import SteakHumanSubtaskQMDPPlanner
from lsi_3d.utils.constants import TARGET_ORNS
from lsi_3d.utils.functions import grid_transition
import pprint

import json
import socket

import socket
import pygame

from utils import real_to_grid_coord


class UdpToPygame():

    def __init__(self):
        UDP_IP="127.0.0.1"
        UDP_PORT=15007
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(0)
        self.sock.bind((UDP_IP,UDP_PORT))

    def update(self):
        try:
            data, addr = self.sock.recvfrom(1024)
            ev = {'data': data, 'addr': addr}
            return ev
            # pygame.event.post(ev)
        except socket.error:
            return None
            pass 

class Player():
    def __init__(self, active_log = [1, 1],
                  held_object = None,
                  num_ingre_held = 0,
                  num_plate_held = 0,
                  num_served = 0,
                  orientation = (0, -1),
                  pos_and_or = ((6, 7), (0, -1)),
                  position = (6, 7),
                  stuck_log = [0, 1]) -> None:
        self.active_log = active_log
        self.held_object = held_object
        self.num_ingre_held = num_ingre_held
        self.num_plate_held = num_plate_held
        self.num_served = num_served
        self.orientation = orientation
        self.pos_and_or = pos_and_or
        self.position = position
        self.stuck_log = stuck_log

class Env():
    def __init__(self, all_objects_list = [],
                  curr_order = 'steak',
                  next_order = 'steak',
                  num_orders_remaining = 2,
                  obj_count = 0,
                  objects = {},
                  order_list = ['steak','steak'],
                  player_objects_by_type = {},
                  player_orientations = ((0, -1), (0, -1)),
                  player_positions = ((6, 7), (10, 6)),
                  players = None, 
                  ) -> None:
        self.all_objects_list = all_objects_list
        self.curr_order = curr_order
        self.next_order = next_order
        self.num_orders_remaining = num_orders_remaining
        self.obj_count = obj_count
        self.objects = objects
        self.order_list = order_list
        self.player_objects_by_type = player_objects_by_type
        self.player_orientations = player_orientations
        self.player_positions = player_positions
        self.players = (
            Player(),
            Player(active_log=[],
                    held_object=None,
                    num_ingre_held=0,
                    num_plate_held=0,
                    num_served=0,
                    orientation=(0, -1),
                    pos_and_or=((10, 6), (0, -1)),
                    position=(10, 6),
                    stuck_log=[])
        )
        self.players_pos_and_or = (
            ((6, 7), (0, -1)),
            ((10, 6), (0, -1))
        )

class EnvEncoder(json.JSONEncoder):
        def default(self, o):
            return o.__dict__


class VisionLimitRobotAgent(HlQmdpPlanningAgent):
    def __init__(self, hlp_planner: SteakHumanSubtaskQMDPPlanner, mlp: AStarMotionPlanner, hl_human_agent: Agent, env: LsiEnv, ig_robot: iGibsonAgent, log_dict={}):
        super().__init__(hlp_planner, mlp, hl_human_agent, env, ig_robot)

        self.action_obj = None
        self.stuck_plan = []
        self.dispatcher = UdpToPygame()
        self.ig_robot.name = "human_sim"
        self.awaiting_response = False
        self.current_q = []
        self.current_ovc_action = None
        self.log_dict = log_dict

    def action(self):
        # TODO: Make it so action calls hlp. and hlp takes a state and returns the best action and the next state

        robot_state_str = self.mdp_planner.get_mdp_key_from_state(self.env)
        human_holding = self.env.human_state.holding
        world_state_string =  robot_state_str + '_' + human_holding

        if self.prev_world_string != world_state_string:
            self.belief = np.full(len(self.belief), 1/len(self.belief))

        mdp_state_keys = self.mdp_planner.world_to_state_keys(
            self.env.world_state, self.env.robot_state, self.env.human_state, self.belief, self.env)
        # update the belief in a state by the result of observations
        # self.belief, self.prev_dist_to_feature = self.mdp_planner.belief_update(state, state.players[0], num_item_in_pot, state.players[1], self.belief, self.prev_dist_to_feature, greedy=self.greedy_known)
        self.belief, self.prev_dist_to_feature = self.mdp_planner.belief_update(
            self.env, self.belief, self.prev_dist_to_feature, self.hl_human_agent)
        print('belief: ', self.belief)
        print('prev_dist_to_feature ', self.prev_dist_to_feature)
        # map abstract to low-level state
        
        # compute in low-level the action and cost
        action_idx, action_object_pair, LOW_LEVEL_ACTION = self.mdp_planner.step(
            self.env, self.belief, 1)

        # TODO: state string is used for updating robot state, so pickup onion gets discarded, proper environment update will remove this code
        # state_str = f'{robot_state_str}_{human_holding}_pickup_onion'
        action, obj = action_object_pair
        # state_str = f'{robot_state_str}_{human_holding}_pickup_onion'
        state_str = self.get_first_key_starts_with_str(
            self.mdp_planner.state_idx_dict,
            f'{robot_state_str}_{human_holding}')
        state_idx = self.mdp_planner.state_idx_dict[state_str]

        # TODO: Eliminate need for this indexing
        #next_state_idx = np.where(self.mdp_planner.transition_matrix[action_idx][state_idx] == 1)[0][0]
        next_state_idx = np.argmax(
            self.mdp_planner.transition_matrix[action_idx][state_idx])  #[0][0]
        next_state = list(self.mdp_planner.state_idx_dict.keys())[list(
            self.mdp_planner.state_idx_dict.values()).index(next_state_idx)]
        # print(f"Next HL Goal State: {next_state}")
        keys = list(self.mdp_planner.action_idx_dict.keys())
        vals = list(self.mdp_planner.action_idx_dict.values())
        action_object_pair = self.mdp_planner.action_dict[keys[vals.index(
            action_idx)]]

        # map back the medium level action to low level action
        goal = self.env.map_action_to_location(action_object_pair, self.env.robot_state.ml_state[0:2])

        # if goal is not None:
        #     goal = self.transform_end_location(goal)
        
        self.prev_world_string = world_state_string
        return (next_state, goal, tuple(action_object_pair))
    
    def update_world_state(self):
        if self.ml_robot_action:
            ml_loc, ml_a = self.ml_robot_action

            self.env.update_world()

            if ml_a == 'I' and self.ig_robot.action_completed(ml_a):
                # self.env.update_world()
                # self.update_state(self.action_obj)
                self.env.kitchen.update_overcooked_robot_holding()
                self.action_obj = None
                self.human_sim_state.ml_state = self.env.human_state.ml_state
                self.take_ml_robot_step = True

    def stuck_handler(self):

        if self.stuck_time == None or self.stuck_ml_pos == None:
            # start timer
            self.stuck_time = time.time()
            self.stuck_ml_pos = self.env.robot_state.ml_state

        elapsed = time.time() - self.stuck_time
        if elapsed > STUCK_TIME_LIMIT:
            print('Resolving Stuck')
            # set a new ml goal to adjacent square and recalculate plan
            # self.take = True
            new_ml_goal = self.adjacent_empty_square(
                self.env.human_state.ml_state)
            self.robot_goal = new_ml_goal
            self.stuck_plan = self.mlp.compute_single_agent_astar_path(
                self.env.robot_state.ml_state, new_ml_goal)
            self.stuck_plan.append((None, 'D'))
            self.stuck_plan.append((None, 'D'))

            self.stuck_time = time.time()

        if not self.env.robot_state.equal_ml(self.stuck_ml_pos):
            # reset timer when robot moves
            self.stuck_ml_pos = self.env.robot_state.ml_state
            self.stuck_time = time.time()
    
    def step(self):
        self.update_world_state()
        # self.hl_human_action = self.hl_human_step()
        # self.ml_human_action = self.ml_human_step()
        # self.hl_robot_action = self.hl_robot_step()
        self.ml_robot_action = self.ml_robot_step()
        self.stuck_handler()
        self.ll_step()

    def from_overcooked_action(self, res_dict):
        action = tuple(res_dict['action'])
        print('overcooked action: ', action)
        q = res_dict['q']
        self.current_q = q
        print('overcooked q: ', q)

        if action[0] == 'i':
            a = 'I'
            self.current_ovc_action = a
            return a
        
        max_q_idx = np.argmax(q[0:4])
        
        if round(q[max_q_idx], 5) == round(q[4], 5):
            # return stay if stay value is equal to the max value
            action = (0,0)
        if action == (0,-1):
            a = 'N'
        elif action == (1,0):
            a = 'E'
        elif action == (0,1):
            a = 'S'
        elif action == (-1,0):
            a = 'W'
        elif action == (0,0):
            a = 'D'
        elif action[0] == 'i':
            a = 'I'

        print('action: ', a)
        self.current_ovc_action = a
        return a

    def robot_action_from_overcooked_api(self):

        if not self.awaiting_response:

            overcooked_state_dict = self.env.to_overcooked_state()

            all_objects_ids_dict = self.env.all_objects_ids_dict()

            transfer_dict = {'ovc_state': overcooked_state_dict, 'ids_dict': all_objects_ids_dict}
            # pprint.pprint(overcooked_state_dict)

            for i in range(1,self.log_dict['i']):
                for j, object in enumerate(self.log_dict[i]['overcooked_state_sent']['objects']):
                    if 'in_hot_plate' in object:
                        object.pop('in_hot_plate')
                        self.log_dict[i]['overcooked_state_sent']['objects'][j] = object

            for i, object in enumerate(overcooked_state_dict['objects']):
                if 'in_hot_plate' in object:
                    object.pop('in_hot_plate')
                    overcooked_state_dict['objects'][i] = object

            

            self.log_dict['i'] += 1
            counter = self.log_dict['i']
            self.log_dict[counter] = {}
            self.log_dict[counter]['overcooked_state_sent'] = overcooked_state_dict.copy()
            self.log_dict[counter]['low_level_logs'] = []
            self.log_dict['event_start_time'] = time.time()
            # print(counter)


            # try:
            filename = 'lsi_3d/logs/' + self.env.kitchen.kitchen_name + self.log_dict['log_id'] + '_log_dict.json'
            open(filename, 'w').close()
            filename = 'lsi_3d/logs/' + self.env.kitchen.kitchen_name + self.log_dict['log_id'] + '_log_dict.json'
            f = open(filename, "a")
            # json_string = json.dumps(self.log_dict)
            json.dump(self.log_dict, f)
            # f.write(json_string)
            f.close()
            # except:
            #     print('problem with logging')


            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            #while True:
            sock.sendto(json.dumps(transfer_dict, cls=EnvEncoder).encode(), ("127.0.0.1", 15006))
            self.awaiting_response = True
            self.wait_time = time.time()

        

        self.ml_robot_action = None
        # while self.ml_robot_action is None:
        data = self.dispatcher.update()

        if data is not None:
            self.awaiting_response = False
            dic = json.loads(data['data'].decode())
            # print('overcooked action: ')
            self.log_dict[self.log_dict['i']]['overcooked_recieved'] = dic
            self.ml_robot_action = self.from_overcooked_action(dic)



        elapsed = time.time()-self.wait_time
        if self.ml_robot_action == None and elapsed > 3:
            self.awaiting_response = False

        return self.ml_robot_action



    def ml_robot_step(self):
        if self.take_ml_robot_step:
            # if human is directly in front of robot then just wait
            if len(self.stuck_plan) > 0:
                self.ml_robot_action = self.stuck_plan.pop(0)
            else:
                self.ml_robot_action = None
                robot_action = self.robot_action_from_overcooked_api()
                if robot_action is not None:
                    plan = []

                    # check if action will take robot to open space
                    if robot_action == self.env.robot_state.ml_state[2]:
                        next_ml_pos = grid_transition('F', self.env.robot_state.ml_state)
                        if self.env.kitchen.grid[next_ml_pos[0]][next_ml_pos[1]] is not 'X':
                            plan.append((None,'D'))
                            print('attempted to move to non empty square so delay')
                        else:
                            plan.append((None, 'F'))
                    elif robot_action in 'DI':
                        plan.append((None, robot_action))
                    else:
                        plan.append((None, robot_action))
                        plan.append((None, 'F'))
                    self.ml_robot_action = plan.pop(0)

            if self.ml_robot_action is not None:
                self.take_ml_robot_step = False

            
            # if self.ml_robot_action and self.env.human_state.ml_state[:2] == grid_transition(self.ml_robot_action[1],
            #                                      self.env.robot_state.ml_state)[0:2]:
            #     self.ml_robot_plan = [(None, 'D')]

                self.robot_delay = self.ml_robot_action[1] == 'D'
                if self.ml_robot_action[1] == 'D':
                    self.delay_time = time.time()
                
                self.ig_robot.prepare_for_next_action(self.env.robot_state.ml_state, self.ml_robot_action[1])
            
            
            # log every ml step
                self.log_state()
                # print('In ml_robot_step:', self.ml_robot_action)
        return self.ml_robot_action
    
    def log_state(self):
        filename = 'lsi_3d/logs/' + self.env.kitchen.kitchen_name + '_log.txt'
        f = open(filename, "a")
        s = '\n'
        s += 'Plan: ' + str(self.ml_robot_plan) + '\n'
        s += 'Overcooked Q: \t' + str(self.current_q) + '\n'
        s += 'Overcooked A: \t' + str(self.current_ovc_action) + '\n'
        f.write(s)
        f.close()
    
    def get_action_object(self, ml_state):
        # gets the action object just performed for interacting at the passed ml state

        # execute forward from current state
        r,c,f = grid_transition('F', ml_state)

        robot_holding_name = None
        if self.env.kitchen.overcooked_robot_holding != None:
            robot_holding_name = self.env.kitchen.get_name(self.env.kitchen.overcooked_robot_holding)

        station_key = self.env.kitchen.grid[r][c]

        if station_key == 'D':
            action_object = ('pickup','plate')
        elif station_key == 'F':
            action_object = ('pickup', 'meat')
        elif station_key == 'W':
            plate = self.env.tracking_env.get_closest_plate_in_sink(self.ig_robot.object.get_position())
            if plate is not None:
                if self.env.kitchen.overcooked_object_states[plate]['state'] == 2:
                    action_object = ('pickup', 'hot_plate')
                else:
                    action_object = ('heat', 'plate')
            else:
                action_object = ('drop','plate')
        elif station_key == 'G':
            action_object = ('pickup','onion')
        elif station_key == 'K' and robot_holding_name == 'onion':
            action_object = ('drop','onion')
            print('dropping onion')
        elif station_key == 'K' and robot_holding_name == None and self.env.tracking_env.get_closest_green_onion(self.ig_robot.object.get_position()).current_index == 0:
            action_object = ('chop','onion')
        elif station_key == 'K' and robot_holding_name == 'steak' and self.env.tracking_env.get_closest_green_onion(self.ig_robot.object.get_position()).current_index == 1:
            action_object = ('pickup','garnish')
        elif station_key == 'F':
            action_object = ('pickup','meat')
        elif station_key == 'P' and self.env.tracking_env.obj_in_robot_hand()[0][0] == 'meat':
            action_object = ('drop','meat')
        elif station_key == 'P' and (len(self.env.tracking_env.obj_in_robot_hand()) == 0 or self.env.tracking_env.obj_in_robot_hand()[0][0] == 'hot_plate'):
            action_object = ('pickup','steak')
        elif station_key == 'C':
            action_object = ('drop', 'item')
        elif station_key == 'T':
            action_object = ('deliver', 'dish')
        else:
            action_object = None
        return action_object
        
    def update_state(self, action_obj):
        pass
        # if action_obj == ('drop','plate'):
        #     sink = self.env.tracking_env.get_closest_sink(self.ig_robot.object.get_position())
        #     plate = self.env.tracking_env.get_closest_plate_in_sink(self.ig_robot.object.get_position())
        #     self.env.kitchen.rinse_sink(sink)
            
        #     if plate is not None:
                
        #         self.env.kitchen.drop_plate(plate)

        # if action_obj == ('heat', 'plate'):
        #     plate = self.env.tracking_env.get_closest_plate_in_sink(self.ig_robot.object.get_position())
        #     self.env.kitchen.heat_plate(plate)

        
                

    def ll_step(self):
        init_action = np.zeros(self.env.nav_env.action_space.shape)
        self.ig_robot.object.apply_action(init_action)
        # self._reset_arm_position(self.ig_robot)
        if self.robot_delay:
            if time.time() - self.delay_time > MAX_DELAY_TIME:
                self.take_hl_robot_step = True
                self.robot_delay = False
                return
            else:
                return

        if self.ml_robot_action == None:
            return
        ml_goal, ml_action = self.ml_robot_action
        if ml_action == 'I':
            if self.action_obj is None:
                self.action_obj = self.get_action_object(self.env.robot_state.ml_state)

            if self.action_obj is not None:
                self.ig_robot.interact_ll_control(
                    self.action_obj, # self.hl_robot_action[-1],
                    self.env.tracking_env,
                    num_item_needed_in_dish=self.mdp_planner.mdp.num_items_for_soup
                )
            else:
                print('action object is None')
                self.take_ml_robot_step = True
            self.env.nav_env.simulator.step()
        else:
            self.continuous_motion(ml_action)
            # self.stepped_motion(ml_action)
    
    def stepped_motion(self, ml_action):
        target_x = self.ig_robot.target_x
        target_y = self.ig_robot.target_y
        target_dir = self.ig_robot.target_direction
        target_orn = TARGET_ORNS[target_dir]
        # self.env.nav_env.set_pos_orn_with_z_offset(
        #    self.ig_robot.object, [target_x, target_y, 0.6], orn=[0,0,target_orn], offset=[0, 0, 0])

        self.ig_robot.agent_set_pos_orn(target_x, target_y, target_dir)



    def continuous_motion(self, ml_action):
        # low level collision avoidance
            h_x, h_y, h_z = self.env.ig_human.object.get_position()
            r_x, r_y, _ = self.env.ig_robot.object.get_position()
            collision_radius = 1
            if math.dist([h_x, h_y], [r_x, r_y]) > collision_radius:
                self.ig_robot.agent_move_one_step(self.env.nav_env, ml_action)

            self._reset_arm_position(self.ig_robot)

            if self.ig_robot.action_completed(ml_action) or ml_action == 'D':
                self.env.update_joint_ml_state()

                if len(self.ml_robot_plan) == 0 or (len(self.ml_robot_plan) %
                                                    self.recalc_res) == 0:
                    # calculate path again
                    self.take_ml_robot_step = True