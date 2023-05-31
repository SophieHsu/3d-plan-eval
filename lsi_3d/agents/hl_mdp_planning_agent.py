from agent import FixedMediumPlan, FixedMediumSubPlan
from igibson.external.pybullet_tools.utils import joints_from_names, set_joint_positions
from lsi_3d.agents.agent import Agent
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.mdp.lsi_env import LsiEnv
from lsi_3d.planners.mid_level_motion import AStarMotionPlanner
from lsi_3d.mdp.lsi_env import AgentState
import numpy as np

from lsi_3d.utils.enums import Mode
from lsi_3d.utils.functions import grid_transition

class HlMdpPlanningAgent(Agent):

    def __init__(self, hlp_planner, mlp:AStarMotionPlanner, hl_human_agent:Agent, env:LsiEnv, ig_robot:iGibsonAgent):
        self.mdp_planner = hlp_planner
        self.mlp = mlp
        self.ml_plan = []
        self.sub_path_goal_i = 0
        self.hl_human_agent = hl_human_agent
        self.env = env
        self.recalc_res = 1
        self.ig_robot = ig_robot

        # Step function vars
        self.a_h = None
        self.a_r = None
        self.next_human_hl_state = None
        self.human_plan = None
        self.human_goal = None
        self.human_action_object_pair = None
        self.human = None
        self.next_robot_hl_state = None
        self.robot_goal = None
        self.robot_action_object_pair = None
        self.robot = None
        self.idle_last_ml_location = None

        self.human_sim_state = AgentState(None, env.human_state.ml_state)

    def get_pot_status(self, state):
        pot_states = self.mdp_planner.mdp.get_pot_states(state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]

        return nearly_ready_pots

    def get_ml_states(self, state):
        num_item_in_pot = 0; pot_pos = []
        if state.objects is not None and len(state.objects) > 0:
            for obj_pos, obj_state in state.objects.items():
                # print(obj_state)
                if obj_state.name == 'soup' and obj_state.state[1] > num_item_in_pot:
                    num_item_in_pot = obj_state.state[1]
                    pot_pos = obj_pos

        # print(pot_pos, num_item_in_pot)
        state_str = self.mdp_planner.gen_state_dict_key(state, state.players[1], num_item_in_pot, state.players[0])

        # print('State = ', state_str)

        return state_str

    def action(self, world_state, robot_state, human_sim_state = None):
        # TODO: Make it so action calls hlp. and hlp takes a state and returns the best action and the next state

        state_str = self.mdp_planner.get_mdp_key_from_state(world_state, robot_state, human_sim_state)
        state_idx = self.mdp_planner.state_idx_dict[state_str]

        # retrieve high level action from policy
        action_idx = self.mdp_planner.policy_matrix[state_idx]

        # TODO: Eliminate need for this indexing
        #next_state_idx = np.where(self.mdp_planner.transition_matrix[action_idx][state_idx] == 1)[0][0]
        next_state_idx = np.argmax(self.mdp_planner.transition_matrix[action_idx][state_idx])#[0][0]
        next_state = list(self.mdp_planner.state_idx_dict.keys())[list(self.mdp_planner.state_idx_dict.values()).index(next_state_idx)]
        print(f"Next HL Goal State: {next_state}")
        keys = list(self.mdp_planner.action_idx_dict.keys())
        vals = list(self.mdp_planner.action_idx_dict.values())
        action_object_pair = self.mdp_planner.action_dict[keys[vals.index(action_idx)]]
        # print(self.mdp_planner.state_idx_dict[state_str], action_idx, action_object_pair)
        
        # map back the medium level action to low level action
        possible_motion_goals = self.mdp_planner.map_action_to_location(world_state, robot_state, action_object_pair)
        goal = possible_motion_goals[0]
        #start = ml_state[0] + ml_state[1]

        return (next_state, goal, tuple(action_object_pair))

    def action_given_parter_current_action(self, world_state, agent_state, partner_action):

        state_str = self.get_mdp_key_from_state(world_state, agent_state)

        self.mdp_planner.get

        self.action(world_state, agent_state)

    def optimal_motion_plan(self, agent_state, goal):
        path = self.mlp.compute_single_agent_astar_path(agent_state.ml_state, goal)
        self.optimal_path = path
        return path

    def avoidance_motion_plan(self, joint_ml_state, goal, avoid_path, avoid_goal, radius = None):
        # extract only actions for avoidance plan
        avoid_path = [s[1] for s in avoid_path]
        paths = self.mlp.compute_motion_plan(joint_ml_state, (avoid_goal,goal), avoid_path, radius)
        path = paths[1]
        return path

    def infront_of_goal(self, state, goal):
        n_x, n_y, n_f = grid_transition('F', state)
        return (n_x,n_y) == goal

    def step(self):
        # mdp, env:LsiEnv,
        #ig_human:iGibsonAgent,
        #ig_robot:iGibsonAgent,
        # hl_robot_agent:HlMdpPlanningAgent,
        # hl_human_agent:FixedPolicyAgent,
        #recalc_res,
        #avoid_radius):
        # """_summary_

        # Args:
        #     mdp (_type_): _description_
        #     env (LsiEnv): _description_
        #     ig_human (iGibsonAgent): _description_
        #     ig_robot (iGibsonAgent): _description_
        #     hl_robot_agent (HlMdpPlanningAgent): _description_
        #     hl_human_agent (FixedPolicyAgent): _description_
        #     bowlpans (_type_): _description_
        #     recalc_res (int): defines number of steps before the robot recalculates its path
        # """

        # print('Press enter to start...')
        # input()
        hl_robot_agent = self
        init_action = np.zeros(self.env.nav_env.action_space.shape)
        self.ig_robot.object.apply_action(init_action)

        if not self.env.human_state.equal_hl(self.human_sim_state.hl_state):
            '''
            human gets high level action and plans path to it. when human finishes path
            it will re-enter this state
            '''
            self.next_human_hl_state, self.human_goal, self.human_action_object_pair = self.hl_human_agent.action(self.env.world_state, self.env.human_state, self.env.robot_state)
            self.human_plan = self.mlp.compute_single_agent_astar_path(self.env.human_state.ml_state, self.human_goal)
            print(f'Executing Human High Level Action: {self.human_action_object_pair[0]} {self.human_action_object_pair[1]}')
            print(f'Path is: {self.human_plan}')
            print(f'Human Holding {self.human_sim_state.holding}')
            print(f'Current HL State: Onions in Pot = {self.env.world_state.in_pot}, Orders = {self.env.world_state.orders}', )
            #human_plan.append('I')
            if len(self.human_plan) > 0:
                self.human = FixedMediumPlan(self.human_plan)
                self.human_sim_state.mode = Mode.EXEC_ML_PATH
                self.human_sim_state.hl_state = self.env.human_state.hl_state
                pos_h, self.a_h = self.human.action()
            # self.ig_human.prepare_for_next_action(self.a_h)

        if not self.env.human_state.equal_ml(self.human_sim_state.ml_state):
            self.human_sim_state.ml_state = self.env.human_state.ml_state
            self.human_plan = self.mlp.compute_single_agent_astar_path(self.env.human_state.ml_state, self.human_goal)
            self.human = FixedMediumPlan(self.human_plan)
            pos_h, self.a_h = self.human.action()

            if self.env.robot_state.mode == Mode.IDLE:
                self.env.robot_state.mode == Mode.CALC_HL_PATH

        if self.env.robot_state.mode == Mode.CALC_HL_PATH:
            '''
            robot gets high level action and translates into mid-level path
            when robot completes this path, it returns to this state
            '''


            self.next_robot_hl_state, self.robot_goal, self.robot_action_object_pair = hl_robot_agent.action(self.env.world_state, self.env.robot_state, self.human_sim_state)
            # optimal_plan = hl_robot_agent.optimal_motion_plan(self.env.robot_state, self.robot_goal)
            self.ml_plan = hl_robot_agent.avoidance_motion_plan((self.human_sim_state.ml_state, self.env.robot_state.ml_state), self.robot_goal, self.human_plan, self.human_goal, radius=2)


            print(f'Executing ROBOT High Level Action: {self.robot_action_object_pair[0]} {self.robot_action_object_pair[1]}')
            print(f'Human Holding {self.env.robot_state.holding}')
            print(f'Current HL State: Onions in Pot = {self.env.world_state.in_pot}, Orders = {self.env.world_state.orders}', )

            # self.optimal_plan_goal = optimal_plan[len(optimal_plan)-1]
            
            if self.infront_of_goal(self.env.robot_state.ml_state, self.robot_goal):
                # could not find path to goal, so idle 1 step and then recalculate
                self.ml_plan.append((self.robot_goal,'I'))
            # if self.ml_plan == []:
            #     # if this is final subpath on optimal plan, the append interact at the end
            #     self.ml_plan.append((self.env.robot_state.ml_state, 'D'))
            
            self.robot_plan = FixedMediumPlan(self.ml_plan)
            self.env.robot_state.mode = Mode.EXEC_ML_PATH
            self.a_r = None
            
            self.env.robot_state.mode = Mode.EXEC_ML_PATH
            # self.robot = FixedMediumSubPlan(optimal_plan, 14)
        #     # self.next_robot_goal = self.robot.next_goal()

        # if self.env.robot_state.mode == Mode.CALC_SUB_PATH:
            

        # if self.env.robot_state.mode == Mode.CALC_SUB_PATH:
        #     self.env.update_joint_ml_state()
        #     if self.env.robot_state.ml_state == self.next_robot_goal:
        #         self.next_robot_goal = self.robot.next_goal()
                
        #     human_sub_path = self._get_human_sub_path(self.human_plan, (self.human.i), self.human_sim_state.ml_state)
        #     plan = hl_robot_agent.avoidance_motion_plan((self.human_sim_state.ml_state, self.env.robot_state.ml_state), self.next_robot_goal, human_sub_path, self.human_goal, radius=1)

            #if self.ml_plan == [] and self.optimal_plan_goal[0] == self.env.robot_state.ml_state:
                # could not find path to goal, so idle 1 step and then recalculate
            #    self.ml_plan.append((self.next_robot_goal,'I'))
            # if self.ml_plan == []:
            #     # if this is final subpath on optimal plan, the append interact at the end
            #     self.ml_plan.append((self.env.robot_state.ml_state, 'D'))
            
            # self.robot_plan = FixedMediumPlan(self.ml_plan)
            # self.env.robot_state.mode = Mode.EXEC_ML_PATH
            # self.a_r = None



        #     self.robot_plan = FixedMediumPlan(plan)
        #     self.env.robot_state.mode = Mode.EXEC_ML_PATH
        #     self.a_r = None
        #     # self.a_r = self.robot_plan.action()
        #     #self.ig_robot.prepare_for_next_action(self.a_r)

        if self.env.robot_state.mode == Mode.EXEC_ML_PATH:

            self.ig_robot.agent_move_one_step(self.env.nav_env, self.a_r)
            # env.update_joint_ml_state()
            
            if self.ig_robot.action_completed(self.a_r) or self.a_r == 'D':
                # if :
                #     env.robot_state.executing_state = ExecutingState.CALC_SUB_PATH
                self.env.update_joint_ml_state()
                if (self.robot_plan.i == len(self.robot_plan.plan) or self.robot_plan.i == 1 or (self.robot_plan.i % self.recalc_res) == 0) and self.a_r != None: #recalc_res: #or a_r == MLAction.STAY:
                    self.env.robot_state.mode = Mode.CALC_HL_PATH
                else:
                    pos_r, self.a_r = self.robot_plan.action()

                    # if self.a_r == 'D':
                    #     pos_r,self.a_r = self.robot_plan.action()
                    self.env.update_joint_ml_state()
                    self.ig_robot.prepare_for_next_action(self.a_r)

                    self._reset_arm_position(self.ig_robot)

                    if self.a_r == 'D' and self.env.robot_state.mode != Mode.IDLE:
                        self.idle_last_ml_location = self.env.human_state.ml_state
                        self.env.robot_state.mode = Mode.IDLE

                    if self.a_r == 'I': #and env.robot_state == robot_goal:
                        self.env.robot_state.mode = Mode.INTERACT

                    if self.human_sim_state.mode == Mode.IDLE:
                        self.env.human_sim_state.mode = Mode.EXEC_ML_PATH

        if self.env.robot_state.mode == Mode.IDLE:
            h_x,h_y,h_f = self.env.human_state.ml_state
            r_x,r_y,r_f = self.env.robot_state.ml_state
            if (h_x,h_y) != (r_x,r_y):
                self.env.robot_state.mode == Mode.CALC_HL_PATH

            
        # if self.env.human_sim_state.mode == Mode.EXEC_ML_PATH:
        #     if self.human.i == len(self.human.plan) or self.a_h == 'I': #or a_h == MLAction.STAY:
        #         self.env.human_sim_state.mode = Mode.INTERACT
        #     else:
        #         # if robot is in a goal state and humans next state is also this state,
        #             # then idle until robot moves
        #             # next_robot_goal == env.robot_state.ml_state and 
        #         if grid_transition(self.a_h, self.env.human_sim_state.ml_state)[0:2] != self.env.robot_state.ml_state[0:2]:
        #             self.ig_human.agent_move_one_step(self.env.nav_env, self.a_h)
        #         elif self.env.robot_state.mode == Mode.IDLE:
        #             self.env.robot_state.mode = Mode.CALC_SUB_PATH

        #             # if next step means human crashes into robot, add a delay to the plan
        #             #delay_step = human_sub_path[0]
        #             #delay_step = (delay_step[0], 'D')
        #             #human_sub_path.insert(0,delay_step)

        #         if self.ig_human.action_completed(self.a_h):
        #             # human.action() gets next FNESW medium level action to take
        #             pos_h, self.a_h = self.human.action()
        #             self.env.update_joint_ml_state()
        #             self.ig_human.prepare_for_next_action(self.a_h)

        #             if self.env.robot_state.mode == Mode.IDLE:
        #                 self.env.robot_state.mode = Mode.EXEC_ML_PATH

                    
        # for obj, pos in self.env.kitchen.bowlpans:
        #         obj.set_position(pos)
        self.env.nav_env.simulator.step()

        if self.env.robot_state.mode == Mode.INTERACT:
            self.env.update_robot_hl_state(self.next_robot_hl_state, self.robot_action_object_pair)
            self.env.robot_state.mode = Mode.CALC_HL_PATH

            #env.human_sim_state.executing_state = ExecutingState.CALC_HL_PATH
        # if self.human_sim_state.mode == Mode.INTERACT:
        #     self.env.update_human_sim_hl_state(self.next_human_hl_state, self.human_action_object_pair)
        #     #env.robot_state.executing_state = ExecutingState.CALC_HL_PATH
        #     self.env.human_sim_state.mode = Mode.CALC_HL_PATH

        if not self.env.world_state.orders:
            print('orders complete')
            exit()

    def _get_human_sub_path(self, path, current_index, human_ml_state):
        sub_path = []
        if current_index == 1 and human_ml_state != path[current_index][0]:
            return path

        for idx, state in enumerate(path):
            if state[0] == human_ml_state:
                return path[idx+1:len(path)]
        
        return sub_path

    def _reset_arm_position(self, ig_robot):
        arm_joints_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

        arm_default_joint_positions = (
            -1.414019864768982,
            1.5178184935241699,
            0.8189625336474915,
            2.200358942909668,
            2.9631312579803466,
            -1.2862852996643066,
            0.0008453550418615341,
        )

        body_ids = ig_robot.object.get_body_ids()
        assert len(body_ids) == 1, "Fetch robot is expected to be single-body."
        robot_id = body_ids[0]
        arm_joint_ids = joints_from_names(robot_id, arm_joints_names)

        set_joint_positions(robot_id, arm_joint_ids, arm_default_joint_positions)


