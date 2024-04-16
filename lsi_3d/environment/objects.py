from argparse import Namespace
from igibson.objects.articulated_object import URDFObject
from igibson import object_states


class KitchenObject:
    def __init__(self):
        self._params = None
        self._obj = None

    @property
    def obj(self):
        if self._obj is None:
            self._obj = URDFObject(
                filename=self._params.filename,
                avg_obj_dims=self._params.avg_obj_dims,
                scale=self._params.scale / self._params.scale_factor,
                model_path=self._params.model_uri,
                category=self._params.category,
                fixed_base=self._params.fixed_base,
            )
        return self._obj

    def load(self, *args):
        self._params.obj_handlers.import_obj(self._obj)


class Fridge(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self._obj)
        self._params.obj_handlers.set_pos_orn(self._obj, self._params.pos, self._params.orn)


class Onion(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self._obj)
        self._obj.states[object_states.OnTop].set_value(self._obj, True, use_ray_casting_method=True)
        self._params.obj_handlers.set_pos_orn(self._obj, self._params.pos, self._params.orn)
        self._params.obj_handlers.change_pb_dynamics(self._obj.get_body_ids()[0], -1, mass=self._params.mass)


class Steak(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self._obj)
        self._obj.states[object_states.OnTop].set_value(self._obj, True, use_ray_casting_method=True)
        self._params.obj_handlers.set_pos_orn(self._obj, self._params.pos, self._params.orn)
        self._params.obj_handlers.change_pb_dynamics(self._obj.get_body_ids()[0], -1, mass=self._params.mass)


class Plate(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self._obj)
        self._params.obj_handlers.set_pos_orn(self._obj, self._params.pos, self._params.orn)
        if self._params.dusty:
            self._obj.states[object_states.Dusty].set_value(True)
        if self._params.stained:
            self._obj.states[object_states.Stained].set_value(True)


class Stove(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self._obj)
        self._params.obj_handlers.set_pos_orn(self._obj, self._params.pos, self._params.orn)
        self._obj.states[object_states.ToggledOn].set_value(False)


class Pan(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self._obj)
        self._params.obj_handlers.set_pos_orn(self._obj, self._params.pos, self._params.orn)
