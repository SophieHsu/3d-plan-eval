from igibson.envs.igibson_env import iGibsonEnv
from lsi_3d.mdp.lsi_mdp import LsiMdp

class LsiEnv(object):
    """
        An environment wrapper for the OvercookedGridworld Markov Decision Process.

        The environment keeps track of the current state of the agent, updates
        it as the agent takes actions, and provides rewards to the agent.
    """
    def __init__(self, mdp:LsiMdp, nav_env:iGibsonEnv) -> None:
        self.hl_state = mdp.hl_start_state
        self.ml_state = mdp.start_locations
        self.nav_env = nav_env