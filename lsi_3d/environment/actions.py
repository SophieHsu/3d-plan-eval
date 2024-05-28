from types import SimpleNamespace
from igibson import object_states

from utils import real_to_grid_coord, to_overcooked_grid

ACTION_COMMANDS = SimpleNamespace(
    DROP='drop_action',
    CLEAN='clean_action',
    CHOP='chop_action',
)


class ActionExecutor:
    def __init__(self):
        pass

    def execute(self, **kwargs):
        raise NotImplementedError


class DropCommandExecutor(ActionExecutor):
    def __init__(self):
        super().__init__()

    def execute(self, target, kitchen, **state_args):
        obj_id = kitchen.overcooked_max_id
        kitchen.overcooked_max_id += 1
        kitchen.overcooked_obj_to_id[target] = obj_id
        curr_state = kitchen.overcooked_object_states[target]['state']
        state = 0 if curr_state is None else curr_state + 1
        return {
            'id': obj_id,
            'position': to_overcooked_grid(real_to_grid_coord(target.get_position())),
            'state': state,
            **state_args
        }


class CleanCommandCommandExecutor(ActionExecutor):
    def __init__(self):
        super().__init__()

    def execute(self, target, kitchen, **state_args):
        curr_state = kitchen.overcooked_object_states[target]['state']
        state = 0 if curr_state is None else curr_state + 1
        kitchen.overcooked_object_states[target] = {
            'id': kitchen.overcooked_obj_to_id[target],
            'position': to_overcooked_grid(real_to_grid_coord(target.get_position())),
            'state': state,
            **state_args
        }

        target.states[object_states.Dusty].set_value(False)
        target.states[object_states.Stained].set_value(False)


class ChopCommandExecutor(ActionExecutor):
    def __init__(self):
        super().__init__()

    def execute(self, target, kitchen, **state_args):
        kitchen.overcooked_object_states[target] = {
            'id': kitchen.overcooked_obj_to_id[target],
            'position': to_overcooked_grid(real_to_grid_coord(target.current_selection().objects[0].get_position())),
            **state_args
        }


ACTION_EXECUTORS = {
    ACTION_COMMANDS.DROP: DropCommandExecutor(),
    ACTION_COMMANDS.HEAT: CleanCommandCommandExecutor(),
    ACTION_COMMANDS.CHOP: ChopCommandExecutor(),
}
