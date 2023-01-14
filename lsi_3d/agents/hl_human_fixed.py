from lsi_3d.agents.agent import Agent

class HLFixedPlan(Agent):
    """
    A high level Agent with a fixed plan. Returns Stay actions once pre-defined plan has terminated.
    # NOTE: Assumes that calls to action are sequential (agent has history)
    """

    def __init__(self, plan):
        self.plan = plan
        self.i = 0
    
    def action(self):
        if self.i >= len(self.plan):
            return None
        else:
            action = self.plan[self.i]
            self.i += 1
            return action