from agent import FixedMediumPlan, FixedMediumSubPlan
from igibson.external.pybullet_tools.utils import joints_from_names, set_joint_positions
from lsi_3d.agents.agent import Agent
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.mdp.lsi_env import LsiEnv
from lsi_3d.planners.mid_level_motion import AStarMotionPlanner
from lsi_3d.mdp.lsi_env import AgentState
import numpy as np
import time
from lsi_3d.utils.constants import DIRE2POSDIFF

from lsi_3d.utils.enums import Mode
from lsi_3d.utils.functions import grid_transition, get_states_in_forward_radius
from lsi_3d.planners.hl_qmdp_planner import HumanSubtaskQMDPPlanner

STUCK_TIME_LIMIT = 60


class HlQmdpPlanningAgent(Agent):

    def __init__(self, hlp_planner: HumanSubtaskQMDPPlanner,
                 mlp: AStarMotionPlanner, hl_human_agent: Agent, env: LsiEnv,
                 ig_robot: iGibsonAgent):
        self.mdp_planner = hlp_planner
        self.mlp = mlp
        self.ml_plan = []
        self.sub_path_goal_i = 0
        self.hl_human_agent = hl_human_agent
        self.env = env
        self.recalc_res = 1
        self.ig_robot = ig_robot

        # uniform distribution
        self.belief = np.full((len(self.mdp_planner.subtask_dict)),
                              1.0 / len(self.mdp_planner.subtask_dict),
                              dtype=float)
        self.prev_dist_to_feature = {}

        # # Step function state vars
        # self.a_h = None
        # self.a_r = None
        # self.next_human_hl_state = None
        # self.human_plan = None
        # self.human_goal = None
        # self.human_action_object_pair = None
        # self.human = None
        # self.next_robot_hl_state = None
        # self.robot_goal = None
        # self.robot_action_object_pair = None
        # self.robot = None
        # self.idle_last_ml_location = None

        self.human_sim_state = AgentState(None, env.human_state.ml_state)

        # for new step function
        self.ml_robot_action = None
        self.ml_robot_plan = []
        self.ml_human_action = None
        self.ml_human_plan = []
        self.take_hl_robot_step = True
        self.take_ml_robot_step = True
        self.robot_delay = False
        self.stuck_time = None
        self.stuck_ml_pos = None

    def get_pot_status(self, state):
        pot_states = self.mdp_planner.mdp.get_pot_states(state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"][
            "ready"]
        cooking_pots = ready_pots + pot_states["tomato"][
            "cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"][
            "partially_full"] + pot_states["onion"]["partially_full"]

        return nearly_ready_pots

    def get_ml_states(self, state):
        num_item_in_pot = 0
        pot_pos = []
        if state.objects is not None and len(state.objects) > 0:
            for obj_pos, obj_state in state.objects.items():
                # print(obj_state)
                if obj_state.name == 'soup' and obj_state.state[
                        1] > num_item_in_pot:
                    num_item_in_pot = obj_state.state[1]
                    pot_pos = obj_pos

        # print(pot_pos, num_item_in_pot)
        state_str = self.mdp_planner.gen_state_dict_key(
            state, state.players[1], num_item_in_pot, state.players[0])

        # print('State = ', state_str)

        return state_str

    def get_first_key_starts_with_str(self, my_dict, my_str):
        for key in my_dict:
            if key.startswith(my_str):
                return key
        # if no matching key is found
        return None

    def action(self, world_state, robot_state, human_sim_state=None):
        # TODO: Make it so action calls hlp. and hlp takes a state and returns the best action and the next state

        robot_state_str = self.mdp_planner.get_mdp_key_from_state(
            world_state, robot_state, human_sim_state)
        human_holding = self.env.human_state.holding

        # update the belief in a state by the result of observations
        # self.belief, self.prev_dist_to_feature = self.mdp_planner.belief_update(state, state.players[0], num_item_in_pot, state.players[1], self.belief, self.prev_dist_to_feature, greedy=self.greedy_known)
        self.belief, self.prev_dist_to_feature = self.mdp_planner.belief_update(
            self.env.world_state, self.env.human_state, self.env.robot_state,
            self.belief, self.prev_dist_to_feature)
        print('belief: ', self.belief)
        print('prev_dist_to_feature ', self.prev_dist_to_feature)
        # map abstract to low-level state
        mdp_state_keys = self.mdp_planner.world_to_state_keys(
            self.env.world_state, self.env.robot_state, self.env.human_state,
            self.belief)
        # compute in low-level the action and cost
        action_idx, action_object_pair, LOW_LEVEL_ACTION = self.mdp_planner.step(
            self.env.world_state, mdp_state_keys, self.belief, 1)

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
        goal = self.env.map_action_to_location(action_object_pair)

        return (next_state, goal, tuple(action_object_pair))

    def action_given_parter_current_action(self, world_state, agent_state,
                                           partner_action):

        state_str = self.get_mdp_key_from_state(world_state, agent_state)

        self.mdp_planner.get

        self.action(world_state, agent_state)

    def optimal_motion_plan(self, agent_state, goal):
        path = self.mlp.compute_single_agent_astar_path(
            agent_state.ml_state, goal)
        self.optimal_path = path
        return path

    def avoidance_motion_plan(self,
                              joint_ml_state,
                              goal,
                              avoid_path,
                              avoid_goal,
                              radius=None):
        # extract only actions for avoidance plan
        if goal == None:
            return []
        avoid_path = [s[1] for s in avoid_path]
        paths = self.mlp.compute_motion_plan(joint_ml_state,
                                             (avoid_goal, goal), avoid_path,
                                             radius)
        path = paths[1]
        return path

    def infront_of_goal(self, state, goal):
        n_x, n_y, n_f = grid_transition('F', state)
        return (n_x, n_y) == goal

    def update_world_state(self):
        if self.ml_robot_action:
            ml_loc, ml_a = self.ml_robot_action

            self.env.update_world()

            if ml_a == 'I' and self.ig_robot.action_completed(ml_a):
                # self.env.update_world()
                self.human_sim_state.ml_state = self.env.human_state.ml_state
                _, _, hl_robot_action_object = self.hl_robot_action
                self.env.update_robot_hl_state(hl_robot_action_object)
                self.take_hl_robot_step = True

    def hl_human_step(self):
        # take high level step when the human has completed an interact
        if not self.env.human_state.equal_hl(
                self.human_sim_state.hl_state
        ) or not self.env.human_state.equal_ml(self.human_sim_state.ml_state,
                                               facing=True):
            self.hl_human_action = self.hl_human_agent.action(
                self.env.world_state, self.env.human_state,
                self.env.robot_state)
            # TODO update env.human.hl_state to pull from world. currently env.human_state is not getting updated when human arrives at goal
            self.human_sim_state.hl_state = self.env.human_state.hl_state
            self.next_human_hl_state, self.human_goal, self.human_action_object = self.hl_human_action
            self.human_ml_plan = self.mlp.compute_single_agent_astar_path(
                self.human_sim_state.ml_state, self.human_goal)

            # if self.human_ml_plan != []:
            #     self.human_ml_plan.append((self.human_goal,'I'))
            self.ml_human_action = None
        return self.hl_human_action

    def ml_human_step(self):
        # if simply iterating through return next
        if (not self.env.human_state.equal_ml(self.human_sim_state.ml_state,
                                              facing=True)
                or self.ml_human_action == None) and self.human_ml_plan != []:
            self.ml_human_action = self.human_ml_plan.pop(0)
            self.human_sim_state.ml_state = self.env.human_state.ml_state
            if self.robot_delay:
                self.robot_delay = False
        return self.ml_human_action

    def hl_robot_step(self):
        # condition set during low level step
        if self.take_hl_robot_step:
            self.hl_robot_action = self.action(self.env.world_state,
                                               self.env.robot_state,
                                               self.human_sim_state)
            self.next_robot_hl_state, self.robot_goal, self.robot_action_object = self.hl_robot_action
            self.take_hl_robot_step = False
            self.recalculate_ml_plan = True

        if self.recalculate_ml_plan:
            plan = self.avoidance_motion_plan(
                (self.human_sim_state.ml_state, self.env.robot_state.ml_state),
                self.robot_goal,
                self.human_ml_plan,
                self.human_goal,
                radius=2)
            if plan == []:
                plan.append((self.env.robot_state.ml_state, 'D'))
            self.ml_robot_plan = plan
            self.take_ml_robot_step = True
            self.recalculate_ml_plan = False

        return self.hl_robot_action

    def ml_robot_step(self):
        if self.take_ml_robot_step:
            # if human is directly in front of robot then just wait
            if self.env.human_state.ml_state[:
                                             2] in get_states_in_forward_radius(
                                                 self.env.robot_state.ml_state,
                                                 0):
                self.ml_robot_plan = [(None, 'D')]

            self.ml_robot_action = self.ml_robot_plan.pop(0)
            self.take_ml_robot_step = False

            self.robot_delay = self.ml_robot_action[1] == 'D'
            self.ig_robot.prepare_for_next_action(self.ml_robot_action[1])

            # log every ml step
            self.log_state()

        return self.ml_robot_action

    def ll_step(self):
        init_action = np.zeros(self.env.nav_env.action_space.shape)
        self.ig_robot.object.apply_action(init_action)
        # self._reset_arm_position(self.ig_robot)

        if self.robot_delay:
            return

        if self.ml_robot_action == None:
            return
        ml_goal, ml_action = self.ml_robot_action
        if ml_action == 'I':
            self.ig_robot.interact_ll_control(
                self.hl_robot_action[-1],
                self.env.tracking_env,
                num_item_needed_in_dish=self.mdp_planner.mdp.num_items_for_soup
            )
        else:
            self.ig_robot.agent_move_one_step(self.env.nav_env, ml_action)
            self._reset_arm_position(self.ig_robot)

            if self.ig_robot.action_completed(ml_action) or ml_action == 'D':
                self.env.update_joint_ml_state()

                if len(self.ml_robot_plan) == 0 or (len(self.ml_robot_plan) %
                                                    self.recalc_res) == 0:
                    # calculate path again
                    self.take_hl_robot_step = True

    def log_state(self):
        filename = 'lsi_3d/test_logs/' + self.env.kitchen.kitchen_name + '_log.txt'
        f = open(filename, "a")
        s = ''
        s += 'QMDP Index: ' + str(self.env.robot_state.hl_state) + '\n'
        s += 'Plan: ' + str(self.ml_robot_plan) + '\n'
        s += 'Robot HL action,object: ' + str(self.robot_action_object) + '\n'
        s += 'Robot goal: '+ str(self.robot_goal) + '\n'
        s += 'Robot ML State: ' + str(self.env.robot_state.ml_state) + '\n'
        f.write(s)
        f.close()

    def stuck_handler(self):

        if self.stuck_time == None or self.stuck_ml_pos == None:
            # start timer
            self.stuck_time = time.time()
            self.stuck_ml_pos = self.env.robot_state.ml_state

        elapsed = time.time() - self.stuck_time
        if elapsed > STUCK_TIME_LIMIT:
            # set a new ml goal to adjacent square and recalculate plan
            self.recalculate_ml_plan = True
            new_ml_goal = self.adjacent_empty_square(
                self.human_sim_state.ml_state)
            self.robot_goal = new_ml_goal
            self.stuck_time = time.time()

        if not self.env.robot_state.equal_ml(self.stuck_ml_pos):
            # reset timer when robot moves
            self.stuck_ml_pos = self.env.robot_state.ml_state
            self.stuck_time = time.time()

    def adjacent_empty_square(self, human_ml_state):
        x, y, f = self.env.robot_state.ml_state
        for dir, add in DIRE2POSDIFF.items():
            d_x, d_y = add
            n_x, n_y = (x + d_x, y + d_y)
            if n_x > len(self.env.kitchen.grid) - 1 or n_y > len(
                    self.env.kitchen.grid) - 1:
                continue

            h_x, h_y, h_f = human_ml_state
            if self.env.kitchen.grid[n_x][n_y] == 'X' and not (n_x, n_y) == (
                    h_x, h_y):
                return (n_x, n_y)

    def step(self):
        self.update_world_state()
        self.hl_human_action = self.hl_human_step()
        self.ml_human_action = self.ml_human_step()
        self.hl_robot_action = self.hl_robot_step()
        self.ml_robot_action = self.ml_robot_step()
        # self.stuck_handler()
        self.ll_step()

    def step_old(self):
        hl_robot_agent = self
        init_action = np.zeros(self.env.nav_env.action_space.shape)
        self.ig_robot.object.apply_action(init_action)

        # update the belief in a state by the result of observations
        # self.belief, self.prev_dist_to_feature = self.mdp_planner.belief_update(state, state.players[0], num_item_in_pot, state.players[1], self.belief, self.prev_dist_to_feature, greedy=self.greedy_known)
        # self.belief, self.prev_dist_to_feature = self.mdp_planner.belief_update(self.env.world_state, self.env.human_state, self.env.robot_state, self.belief, self.prev_dist_to_feature)
        # # map abstract to low-level state
        # mdp_state_keys = self.mdp_planner.world_to_state_keys(self.env.world_state, self.env.robot_state, self.env.human_state, self.belief)
        # # compute in low-level the action and cost
        # action, action_object_pair, LOW_LEVEL_ACTION = self.mdp_planner.step(self.env.world_state, mdp_state_keys, self.belief, 1)

        if not self.env.human_state.equal_hl(self.human_sim_state.hl_state):
            '''
            human gets high level action and plans path to it. when human finishes path
            it will re-enter this state
            '''
            self.next_human_hl_state, self.human_goal, self.human_action_object_pair = self.hl_human_agent.action(
                self.env.world_state, self.env.human_state,
                self.env.robot_state)
            self.human_plan = self.mlp.compute_single_agent_astar_path(
                self.env.human_state.ml_state, self.human_goal)
            print(
                f'Executing Human High Level Action: {self.human_action_object_pair[0]} {self.human_action_object_pair[1]}'
            )
            print(f'Path is: {self.human_plan}')
            print(f'Human Holding {self.human_sim_state.holding}')
            print(
                f'Current HL State: Onions in Pot = {self.env.world_state.in_pot}, Orders = {self.env.world_state.orders}',
            )
            #human_plan.append('I')
            if len(self.human_plan) > 0:
                self.human = FixedMediumPlan(self.human_plan)
                self.human_sim_state.mode = Mode.EXEC_ML_PATH
                self.human_sim_state.hl_state = self.env.human_state.hl_state
                pos_h, self.a_h = self.human.action()
            # self.ig_human.prepare_for_next_action(self.a_h)

        if not self.env.human_state.equal_ml(self.human_sim_state.ml_state):
            self.human_sim_state.ml_state = self.env.human_state.ml_state
            self.human_plan = self.mlp.compute_single_agent_astar_path(
                self.env.human_state.ml_state, self.human_goal)
            self.human = FixedMediumPlan(self.human_plan)
            pos_h, self.a_h = self.human.action()

            if self.env.robot_state.mode == Mode.IDLE:
                self.env.robot_state.mode == Mode.CALC_HL_PATH

        if self.env.robot_state.mode == Mode.CALC_HL_PATH:
            '''
            robot gets high level action and translates into mid-level path
            when robot completes this path, it returns to this state
            '''

            self.next_robot_hl_state, self.robot_goal, self.robot_action_object_pair = hl_robot_agent.action(
                self.env.world_state, self.env.robot_state,
                self.human_sim_state)
            # optimal_plan = hl_robot_agent.optimal_motion_plan(self.env.robot_state, self.robot_goal)
            self.ml_plan = hl_robot_agent.avoidance_motion_plan(
                (self.human_sim_state.ml_state, self.env.robot_state.ml_state),
                self.robot_goal,
                self.human_plan,
                self.human_goal,
                radius=2)

            print(
                f'Executing ROBOT High Level Action: {self.robot_action_object_pair[0]} {self.robot_action_object_pair[1]}'
            )
            print(f'Human Holding {self.env.robot_state.holding}')
            print(
                f'Current HL State: Onions in Pot = {self.env.world_state.in_pot}, Orders = {self.env.world_state.orders}'
            )

            # self.optimal_plan_goal = optimal_plan[len(optimal_plan)-1]

            if self.infront_of_goal(self.env.robot_state.ml_state,
                                    self.robot_goal):
                # could not find path to goal, so idle 1 step and then recalculate
                self.ml_plan.append((self.robot_goal, 'I'))

            self.robot_plan = FixedMediumPlan(self.ml_plan)
            self.env.robot_state.mode = Mode.EXEC_ML_PATH
            self.a_r = None

            self.env.robot_state.mode = Mode.EXEC_ML_PATH

        if self.env.robot_state.mode == Mode.EXEC_ML_PATH:

            self.ig_robot.agent_move_one_step(self.env.nav_env, self.a_r)

            if self.ig_robot.action_completed(self.a_r) or self.a_r == 'D':
                self.env.update_joint_ml_state()
                if (
                        self.robot_plan.i == len(self.robot_plan.plan)
                        or self.robot_plan.i == 1 or
                    (self.robot_plan.i % self.recalc_res) == 0
                ) and self.a_r != None:  #recalc_res: #or a_r == MLAction.STAY:
                    self.env.robot_state.mode = Mode.CALC_HL_PATH
                else:
                    pos_r, self.a_r = self.robot_plan.action()

                    self.env.update_joint_ml_state()
                    self.ig_robot.prepare_for_next_action(self.a_r)

                    self._reset_arm_position(self.ig_robot)

                    if self.a_r == 'D' and self.env.robot_state.mode != Mode.IDLE:
                        self.idle_last_ml_location = self.env.human_state.ml_state
                        self.env.robot_state.mode = Mode.IDLE

                    if self.a_r == 'I':  #and env.robot_state == robot_goal:
                        self.env.robot_state.mode = Mode.INTERACT

                    if self.human_sim_state.mode == Mode.IDLE:
                        self.env.human_sim_state.mode = Mode.EXEC_ML_PATH

        if self.env.robot_state.mode == Mode.IDLE:
            h_x, h_y, h_f = self.env.human_state.ml_state
            r_x, r_y, r_f = self.env.robot_state.ml_state
            if (h_x, h_y) != (r_x, r_y):
                self.env.robot_state.mode = Mode.CALC_HL_PATH
        self.env.nav_env.simulator.step()

        if self.env.robot_state.mode == Mode.INTERACT:

            self.env.update_robot_hl_state(self.next_robot_hl_state,
                                           self.robot_action_object_pair)
            self.env.robot_state.mode = Mode.CALC_HL_PATH

        if not self.env.world_state.orders:
            print('orders complete')
            exit()

    def _get_human_sub_path(self, path, current_index, human_ml_state):
        sub_path = []
        if current_index == 1 and human_ml_state != path[current_index][0]:
            return path

        for idx, state in enumerate(path):
            if state[0] == human_ml_state:
                return path[idx + 1:len(path)]

        return sub_path

    def _reset_arm_position(self, ig_robot):
        arm_joints_names = [
            "torso_lift_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

        # arm_default_joint_positions = (
        #     0.385,
        #     0,  #-1.414019864768982,
        #     -1.2178184935241699,  #1.5178184935241699,
        #     0.08189625336474915,
        #     2.200358942909668,
        #     3.14,  #2.9631312579803466,
        #     1.9,
        #     0.0008453550418615341,
        # )

        # arm_default_joint_positions = (
        #     0.385,
        #     0,
        #     -1.0178184935241699,  #1.5178184935241699,
        #     0,  #0.8189625336474915,
        #     2.200358942909668,
        #     2.9631312579803466,
        #     -1.2862852996643066,
        #     0,  #0.0008453550418615341,
        # )

        arm_default_joint_positions = (
            0.385,
            0,  #1.3,
            -1.0178184935241699,  #1.5178184935241699,
            0,  #0.8189625336474915,
            -1.7,  #2.200358942909668,
            2.9631312579803466,
            -1.2862852996643066,
            0,  #0.0008453550418615341,
        )

        body_ids = ig_robot.object.get_body_ids()
        assert len(body_ids) == 1, "Fetch robot is expected to be single-body."
        robot_id = body_ids[0]
        arm_joint_ids = joints_from_names(robot_id, arm_joints_names)

        set_joint_positions(robot_id, arm_joint_ids,
                            arm_default_joint_positions)