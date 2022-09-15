import math
from igibson.envs.igibson_env import iGibsonEnv
from lsi_3d.agents.igibson_agent import iGibsonAgent
from lsi_3d.mdp.hl_state import OvercookedState, SoupState
from lsi_3d.mdp.lsi_mdp import LsiMdp
from lsi_3d.utils.functions import orn_to_cardinal, quat2euler

class LsiEnv(object):
    """
        An environment wrapper for the OvercookedGridworld Markov Decision Process.

        The environment keeps track of the current state of the agent, updates
        it as the agent takes actions, and provides rewards to the agent.

        nav_env: the simulation environment that the lsi_env wraps
    """
    def __init__(self, mdp:LsiMdp, nav_env:iGibsonEnv, ig_human:iGibsonAgent, ig_robot:iGibsonAgent) -> None:
        self.hl_state = mdp.hl_start_state
        self.ml_state = mdp.start_locations
        self.nav_env = nav_env
        self.mdp = mdp
        self.ig_human = ig_human
        self.ig_robot = ig_robot
        self.human_state = OvercookedState(self.hl_state,self.ml_state,soup_locations=[mdp.get_pot_locations()])
        self.robot_state = OvercookedState(self.hl_state,self.ml_state,soup_locations=[mdp.get_pot_locations()])

    def update_robot_world_state(self, next_hl_state):
        '''
        Looks at agents place in the world and updates current state
        '''
        self.hl_state = next_hl_state # eventually replace with game logice
        self.ml_state = self.update_joint_ml_state()
        self.robot_state = self.robot_state.update(self.hl_state, self.ml_state)


    def update_joint_ml_state(self):
        # r_x,r_y,z = self.ig_robot.object.get_position()
        # pos_x = math.trunc(r_x + 1) * -1
        # pos_y = math.trunc(r_y + 1) * -1
        
        # x,y,z,w = self.ig_robot.object.get_orientation()
        # r,p,y = quat2euler(x,y,z,w)
        # facing = orn_to_cardinal(y)
        
        human_ml_state = self.get_ml_state(self.ig_human)
        robot_ml_state = self.get_ml_state(self.ig_robot)
        return (human_ml_state, robot_ml_state)

    def get_ml_state(self, agent:iGibsonAgent):
        r_r,r_c,z = agent.object.get_position()
        pos_r = round(r_r + 4.5)
        pos_c = round(r_c + 4.5)
        
        x,y,z,w = agent.object.get_orientation()
        r,p,y = quat2euler(x,y,z,w)
        facing = orn_to_cardinal(y)
        
        return (pos_r, pos_c, facing)