from lsi_3d.utils.enums import MLA

class Agent(object):
    def action(self, state):
        return NotImplementedError()

class FixedMediumPlan(Agent):
    """
    An Agent with a fixed plan. Returns Stay actions once pre-defined plan has terminated.
    # NOTE: Assumes that calls to action are sequential (agent has history)
    """

    def __init__(self, plan):
        self.plan = plan #[(pos,MLAction.from_string(a)) for (pos,a) in plan]
        
        if self.plan[len(self.plan)-1][1] != MLA.INTERACT:
            self.plan.append((self.plan[len(self.plan)-1][0], MLA.INTERACT))
        self.i = 0
    
    def action(self):
        # if self.i >= len(self.plan):
        #     return MLAction.STAY
        # else:
        action = self.plan[self.i]
        self.i += 1
        return action

    def to_string(self):
        return MLA.to_string(self.plan)

class FixedMediumSubPlan(Agent):
    """
    An Agent with a fixed plan but desired execution is in chunks
    """

    def __init__(self, plan, res):
        self.plan = plan
        self.sub_plan_i = 0
        self.res = res

    def next_goal(self):
        self.sub_plan_i += self.res
        if self.sub_plan_i >= len(self.plan):
            self.sub_plan_i = len(self.plan)-1
        return self.plan[self.sub_plan_i][0]

