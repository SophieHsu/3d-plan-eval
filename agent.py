from lsi_3d.utils.enums import MLAction

class Agent(object):
    def action(self, state):
        return NotImplementedError()

class FixedMediumPlan(Agent):
    """
    An Agent with a fixed plan. Returns Stay actions once pre-defined plan has terminated.
    # NOTE: Assumes that calls to action are sequential (agent has history)
    """

    def __init__(self, plan):
        self.plan = MLAction.from_strings(plan)
        self.i = 0
    
    def action(self):
        if self.i >= len(self.plan):
            return MLAction.STAY
        else:
            action = self.plan[self.i]
            self.i += 1
            return action

    def to_string(self):
        return MLAction.to_string(self.plan)
