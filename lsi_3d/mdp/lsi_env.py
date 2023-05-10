import math
from igibson.envs.igibson_env import iGibsonEnv
from kitchen import Kitchen
from lsi_3d.agents.agent import Agent
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.mdp.hl_state import AgentState, SoupState, WorldState
from lsi_3d.mdp.lsi_mdp import LsiMdp
from lsi_3d.utils.functions import orn_to_cardinal, quat2euler
from tracking_env import TrackingEnv


class LsiEnv(object):
    """
        An environment wrapper for the OvercookedGridworld Markov Decision Process.

        The environment keeps track of the current state of the agent, updates
        it as the agent takes actions, and provides rewards to the agent.

        nav_env: the simulation environment that the lsi_env wraps
    """

    def __init__(self,
                 mdp: LsiMdp,
                 nav_env: iGibsonEnv,
                 tracking_env: TrackingEnv,
                 ig_human: iGibsonAgent,
                 ig_robot: iGibsonAgent,
                 kitchen: Kitchen,
                 recalc_res=None,
                 avoid_radius=None) -> None:

        #self.joint_hl_state = mdp.hl_start_state
        self.kitchen = kitchen
        self.nav_env = nav_env
        self.tracking_env = tracking_env
        self.mdp = mdp
        self.ig_human = ig_human
        self.ig_robot = ig_robot
        self.recalc_res = recalc_res
        self.avoid_radius = avoid_radius
        self.human_state = AgentState(mdp.hl_start_state,
                                      self.mdp.start_locations[0])
        self.robot_state = AgentState(mdp.hl_start_state,
                                      self.mdp.start_locations[1])
        self.world_state = WorldState(mdp.hl_start_state)
        self.world_state.players = [self.robot_state, self.human_state]

    def update_world(self):
        pot_to_onion_dict = self.tracking_env.get_pan_status()
        onions = pot_to_onion_dict[self.kitchen.pans[0]]
        real_onions = len(onions)
        self.world_state.in_pot

        self.world_state.in_pot = real_onions + self.world_state.sim_in_pot
        return

    def update_robot_hl_state(self, action_object):
        '''
        Update hl state by updated in_pot and orders for world
        and holding for specific agent
        '''
        if action_object == (
                'pickup', 'soup'
        ) and self.world_state.in_pot == self.mdp.num_items_for_soup:
            self.tracking_env.clear_pot()

        self.world_state.update(action_object)
        # self.robot_state.update_hl_state(next_hl_state, self.world_state)
        self.robot_state.update(self.tracking_env, self.world_state)

    def update_human_hl_state(self, next_hl_state, action_object):
        '''
        Update hl state by updated in_pot and orders for world
        and holding for specific agent
        '''
        self.world_state.update(next_hl_state, action_object)
        self.human_state.update_hl_state(next_hl_state, self.world_state)

    def update_human_sim_hl_state(self, next_hl_state, action_object):
        '''
        Update hl state by updated in_pot and orders for world
        and holding for specific agent
        '''
        self.world_state.update(next_hl_state, action_object)
        self.human_sim_state.update_hl_state(next_hl_state, self.world_state)

    def get_human_hl_state(self):
        hl_state = f'{self.human_state.holding}_{self.world_state.in_pot}'
        for order in self.world_state.orders:
            hl_state += f'_{order}'

        return hl_state

    def get_robot_hl_state(self):
        hl_state = f'{self.robot_state.holding}_{self.world_state.in_pot}'
        for order in self.world_state.orders:
            hl_state += f'_{order}'

        return hl_state

    def update_joint_ml_state(self):
        human_ml_state = self.get_ml_state(self.ig_human)
        robot_ml_state = self.get_ml_state(self.ig_robot)
        # human_sim_ml_state = self.get_ml_state(self.ig_human_sim)
        self.robot_state.ml_state = robot_ml_state
        self.human_state.ml_state = human_ml_state
        # self.human_sim_state.ml_state = human_sim_ml_state
        return (human_ml_state, robot_ml_state)

    def get_ml_state(self, agent: iGibsonAgent):
        r_r, r_c, z = agent.object.get_position()
        pos_r = round(r_r + 4.5)
        pos_c = round(r_c + 4.5)

        x, y, z, w = agent.object.get_orientation()
        r, p, y = quat2euler(x, y, z, w)
        facing = orn_to_cardinal(y)

        return (pos_r, pos_c, facing)

    def update(self, new_hl_state, new_ml_state):
        prev_in_pot = self.in_pot

        human_hl, robot_hl = new_hl_state
        human_ml, robot_ml = new_ml_state

        self.human_state.update(human_hl, human_ml)
        self.robot_state.update(robot_hl, robot_ml)

        self.parse_hl_state(new_hl_state)

        new_in_pot = self.in_pot

        if new_in_pot == (prev_in_pot + 1):
            self.soup_states[0].onions_in_soup = self.in_pot

        return self

    def map_action_to_location(self, action_object):
        action, object = action_object
        if action == "pickup" and object == "onion":
            location = self.tracking_env.get_closest_onion().get_position()
        elif action == "drop" and object == "onion":
            location = self.tracking_env.get_closest_pan().get_position()
        elif action == "pickup" and object == "dish":
            location = self.tracking_env.get_closest_bowl().get_position()
        elif action == "deliver" and object == "soup":
            location = self.mdp.get_counter_locations()[0]
        elif action == "pickup" and object == "soup":
            location = self.tracking_env.get_closest_pan().get_position()

        return location