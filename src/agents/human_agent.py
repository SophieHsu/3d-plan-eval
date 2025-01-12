import copy
import math
import time
from datetime import datetime

from igibson.objects.articulated_object import URDFObject
from src.environment.tracking_env import TrackingEnv
from src.utils.functions import norm_cardinal_to_orn
from src.utils.helpers import quat2euler, real_to_grid_coord, grid_to_real_coord, normalize_radians


class HumanAgent:

    def __init__(self,
                 human,
                 planner,
                 motion_controller,
                 occupancy_grid,
                 hlp,
                 lsi_env,
                 tracking_env: TrackingEnv,
                 vr=False,
                 insight_threshold=5):

        self.STUCK_TIME_LIMIT = 7
        self.WAIT_COLLISION_TIME = 5

        self.human = human
        self.robot = None
        self.planner = planner
        self.motion_controller = motion_controller
        self.occupancy_grid = copy.deepcopy(occupancy_grid)
        self.hlp = hlp
        self.env = lsi_env
        self.vision_range = math.pi
        self.igibson_env = tracking_env.env
        self.tracking_env = tracking_env
        self.vr = vr
        self.insight_threshold = insight_threshold
        self.prev_end = None
        self.object_position = None
        self.arrived = False
        self.step_index = 0
        self.next_hl_state = None
        self.action_object = None
        self.avoiding_start_time = None
        self.stuck_time = None
        self.stuck_ml_pos = None
        self.interact_obj = None
        self.ingredients = []

    def change_state(self):
        # pass
        self.env.update_human_hl_state("onion_0_onion_onion", ("pickup", "onion"))
        self.env.update_human_hl_state("None_1_onion_onion", ("drop", "onion"))
        self.env.update_human_hl_state("onion_1_onion_onion", ("pickup", "onion"))
        self.env.update_human_hl_state("None_2_onion_onion", ("drop", "onion"))
        self.env.update_joint_ml_state()

    def stuck_handler(self):
        end = None
        if self.stuck_time is None or self.stuck_ml_pos is None:
            # start timer
            self.stuck_time = time.time()
            self.stuck_ml_pos = self.env.human_state.ml_state

        if not self.env.human_state.equal_ml(self.stuck_ml_pos):
            # reset timer when robot moves
            self.stuck_ml_pos = self.env.human_state.ml_state
            self.stuck_time = time.time()

        elapsed = time.time() - self.stuck_time
        if elapsed > self.STUCK_TIME_LIMIT:
            end = self.loc_to_avoid_robot()

        return end

    def set_robot(self, robot):
        self.robot = robot

    def step(self):
        if self.vr:
            self.igibson_env.simulator.switch_main_vr_robot(self.human)
            actionStep = self.igibson_env.simulator.gen_vr_robot_action()
            self.human.apply_action(actionStep)
            self.check_interact_objects()
            self.env.update_human_world_state()
        else:
            if not self.arrived:
                end = self.get_next_goal()

                if end is not None:
                    r, c, f = end
                    end = grid_to_real_coord((r, c))

                    self.interacting = False
                    orn = norm_cardinal_to_orn(f)
                    self.arrived = self._step(end, orn)
                    self.env.update_human_world_state()

            else:
                self.stuck_time = None
                self._arrival_step()
                self.interacting = True

    def _step(self, end, final_ori):
        self.update_occupancy_grid()
        x, y, z = self.human.get_position()
        robot_x, robot_y, _ = self.robot.get_position()

        # check if holding bowl collision radius is larger
        collision_radius = 0.8
        if self.tracking_env.is_human_holding_bowl():
            collision_radius = 1

        if math.dist([x, y], [robot_x, robot_y]) < collision_radius and self.avoiding_start_time is None:
            self.avoiding_start_time = datetime.now()
        if math.dist([x, y], [robot_x, robot_y]) > 1.2:
            self.avoiding_start_time = None
        stuck_goal = self.stuck_handler()
        if stuck_goal:
            asdf = 4

        if self.avoiding_start_time is not None:
            if (datetime.now() - self.avoiding_start_time).total_seconds() < self.WAIT_COLLISION_TIME:
                return False
            else:
                end = self.loc_to_avoid_robot()

        path = self.planner.find_path((x, y), end, self.occupancy_grid)

        round_prev_end = [round(p, 3) for p in self.prev_end] if self.prev_end is not None else self.prev_end
        round_end = [round(e, 3) for e in end] if end is not None else end

        is_new_end = True if round_prev_end != round_end else False

        if len(path) > 0:
            self.prev_end = end

        return self.motion_controller.step(self.human, self.robot, final_ori, path, is_new_end)

    def _arrival_step(self):
        action = self.action_object[0]
        object = self.action_object[1]
        if action == "pickup" and object == "onion":
            if self.object_position is None:
                self.target_object = self.tracking_env.get_closest_onion()
                self.object_position = self.target_object.get_position()

            is_holding_onion = self.tracking_env.is_obj_in_human_hand(self.target_object)
            done = self.pick(self.object_position, [0, 0, 0.07])

            if done:
                if is_holding_onion:
                    self.completed_goal()
                else:
                    self.object_position = None
        elif action == "drop" and object == "onion":
            if self.object_position is None:
                self.object_position = self.tracking_env.get_closest_pan().get_position()
            done = self.drop(self.object_position, [0, -0.1, 0.25])
            if done:
                self.completed_goal()
        elif action == "pickup" and object == "dish":
            if self.object_position is None:
                bowls = self.tracking_env.get_bowls_dist_sort()
                for bowl in bowls:
                    items_in_bowl = self.tracking_env.items_in_bowl(bowl)
                    if len(items_in_bowl) == 0 and self.tracking_env.is_item_on_counter(bowl):
                        self.object_position = bowl.get_position()
                        self.target_object = bowl
                        break

            is_holding_bowl = self.tracking_env.is_obj_in_human_hand(self.target_object)
            done = self.pick(self.object_position, [0, -0.25, 0.1]) and is_holding_bowl
            if done:
                self.completed_goal()
        elif action == "deliver" and object == "soup":
            done = self.drop(self.human.get_position(), [0, 0.5, 0.2])
            if done:
                self.completed_goal()
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
                    onion = self.tracking_env.get_closest_onion(on_pan=True)
                    self.object_position = onion.get_position()
                    self.target_object = onion
            elif self.step_index == 1:
                is_holding_onion = self.tracking_env.is_obj_in_human_hand(self.target_object)
                done = self.pick(self.object_position, [0, -0.05, 0.05]) and is_holding_onion
                if done:
                    self.step_index = self.step_index + 1
                    self.object_position = self.tracking_env.get_closest_bowl().get_position()
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
                self.completed_goal()
                self.tracking_env.kitchen.interact_objs[self.interact_obj] = False

    def pick(self, loc, offset=[0, 0, 0]):
        return self.motion_controller.pick(self.human, loc, offset)

    def drop(self, loc, offset=[0, 0, 0]):
        return self.motion_controller.drop(self.human, loc, offset)

    def completed_goal(self):
        self.step_index = 0
        self.object_position = None
        self.arrived = False

    def get_next_goal(self):
        agent_state = self.env.human_state
        world_state = self.env.world_state
        action, object = 'stay', agent_state.holding
        if agent_state.holding == 'None':
            if world_state.in_pot == self.tracking_env.kitchen.onions_for_soup - 1 and \
                    self.env.robot_state.holding == 'onion':
                action, object = ('pickup', 'dish')
                next_hl_state = f'dish_{world_state.in_pot}'
            elif world_state.in_pot >= self.tracking_env.kitchen.onions_for_soup and \
                    self.env.robot_state.holding != 'dish':
                action, object = ('pickup', 'dish')
                next_hl_state = f'dish_{world_state.in_pot}'
            else:
                action, object = ('pickup', 'onion')
                next_hl_state = f'onion_{world_state.in_pot}'
                agent_state.next_holding = 'onion'
        elif agent_state.holding == 'onion':
            action, object = ('drop', 'onion')
            next_hl_state = f'None_{world_state.in_pot + 1}'
            agent_state.next_holding = 'None'
        elif world_state.in_pot <= 3 and agent_state.holding == 'None':
            action, object = ('pickup', 'dish')
            next_hl_state = f'dish_{world_state.in_pot}'
            agent_state.next_holding = 'dish'
        elif agent_state.holding == 'dish' and (
                world_state.in_pot >= self.env.mdp.num_items_for_soup or self.interacting == True):
            action, object = ('pickup', 'soup')
            next_hl_state = f'soup_{world_state.in_pot}'
            agent_state.next_holding = 'soup'
        elif agent_state.holding == 'soup':
            action, object = ('deliver', 'soup')
            next_hl_state = f'None_{world_state.in_pot}'
            agent_state.next_holding = 'None'

        for order in world_state.orders:
            next_hl_state += f'_{order}'

        possible_motion_goals = self.env.map_action_to_location(
            (action, object), self.env.human_state.ml_state[0:2], is_human=True)
        goal = possible_motion_goals
        self.next_hl_state = next_hl_state
        self.action_object = (action, object)
        return goal

    def check_interact_objects(self):
        human_holding = self.tracking_env.get_human_holding()
        if human_holding == 'soup':
            onions = self.tracking_env.get_onions_in_human_soup()
            if not self.ingredients:
                self.ingredients = onions
                print(len(self.ingredients))
        elif human_holding == 'None':
            self.ingredients = []
            # make sure onions are in bowl since they sometimes bounce out

        for onion in self.ingredients:
            bowl = self.tracking_env.obj_in_human_hand()
            bowl_loc = bowl.get_position()
            if not self.tracking_env.is_item_in_object(onion, bowl):
                # spawn onion back into bowl
                x, y, z = bowl_loc
                onion.set_position([x, y, z + 0.05])

    def transform_end_location(self, loc):
        # objects = self.igibson_env.simulator.scene.get_objects()
        objects = self.tracking_env.kitchen.static_objs
        selected_object = None
        for o in objects.keys():
            if type(o) != URDFObject:
                continue

            if type(objects[o]) == list:
                for o_loc in objects[o]:
                    if o_loc == loc: selected_object = o
                    # pos = grid_to_real_coord(loc)
            elif objects[o] == loc:
                selected_object = o

        pos = list(grid_to_real_coord(loc))
        _, ori = selected_object.get_position_orientation()
        ori = quat2euler(ori[0], ori[1], ori[2], ori[3])[2]
        # absolute transformation
        ori = normalize_radians(ori - 1.57)
        pos[0] = pos[0] + math.cos(ori)
        pos[1] = pos[1] + math.sin(ori)
        return pos[0:2], normalize_radians(ori + math.pi)

    def is_at_location(self, loc, dest, tolerance):
        return (dest[0] - tolerance) < loc[0] < (dest[0] + tolerance) and \
            (dest[1] - tolerance) < loc[1] < (dest[1] + tolerance)

    def update_occupancy_grid(self):
        # if self.is_observable(self.robot):
        x, y, z = self.robot.get_position()
        loc = real_to_grid_coord((x, y))
        for i in range(len(self.occupancy_grid)):
            for j in range(len(self.occupancy_grid[0])):
                if self.occupancy_grid[i][j] == 'R':
                    self.occupancy_grid[i][j] = 'X'
        self.occupancy_grid[loc[0]][loc[1]] = 'R'

    def loc_to_avoid_robot(self):
        human_x, human_y, _ = self.human.get_position()
        human_pos_grid = real_to_grid_coord([human_x, human_y])
        robot_x, robot_y, _ = self.robot.get_position()

        relative_neighbor_locs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        farthest_neighbor = None
        dist = -1
        for n in relative_neighbor_locs:
            x_neighbor = human_pos_grid[0] + n[0]
            y_neighbor = human_pos_grid[1] + n[1]

            if 0 <= x_neighbor < 8 and 0 <= y_neighbor < 8 and self.occupancy_grid[x_neighbor][y_neighbor] == "X":
                neighbor_continuous = grid_to_real_coord(
                    [x_neighbor, y_neighbor])
                neighbor_dist = math.dist(
                    [neighbor_continuous[0], neighbor_continuous[1]],
                    [robot_x, robot_y])
                neighbor = grid_to_real_coord([x_neighbor, y_neighbor])
                path = self.planner.find_path((human_x, human_y), neighbor,
                                              self.occupancy_grid)
                if neighbor_dist > dist and len(path) > 0:
                    farthest_neighbor = neighbor
                    dist = neighbor_dist
        return farthest_neighbor

    def get_position(self):
        return self.human.get_position()
