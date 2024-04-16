import os
from argparse import Namespace
from igibson.objects.articulated_object import URDFObject
from igibson import object_states
from igibson.utils.assets_utils import get_ig_model_path
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer


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
        raise NotImplementedError()


class Fridge(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self.obj)
        self._params.obj_handlers.set_pos_orn(self.obj, self._params.pos, self._params.orn)


class Onion(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self.obj)
        self.obj.states[object_states.OnTop].set_value(self.obj, True, use_ray_casting_method=True)
        self._params.obj_handlers.set_pos_orn(self.obj, self._params.pos, self._params.orn)
        self._params.obj_handlers.change_pb_dynamics(self.obj.get_body_ids()[0], -1, mass=self._params.mass)


class Steak(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self.obj)
        self.obj.states[object_states.OnTop].set_value(self.obj, True, use_ray_casting_method=True)
        self._params.obj_handlers.set_pos_orn(self.obj, self._params.pos, self._params.orn)
        self._params.obj_handlers.change_pb_dynamics(self.obj.get_body_ids()[0], -1, mass=self._params.mass)


class Plate(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self.obj)
        self._params.obj_handlers.set_pos_orn(self.obj, self._params.pos, self._params.orn)
        if self._params.dusty:
            self.obj.states[object_states.Dusty].set_value(True)
        if self._params.stained:
            self.obj.states[object_states.Stained].set_value(True)


class Stove(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self.obj)
        self._params.obj_handlers.set_pos_orn(self.obj, self._params.pos, self._params.orn)
        self.obj.states[object_states.ToggledOn].set_value(False)


class Pan(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self):
        self._params.obj_handlers.import_obj(self.obj)
        self._params.obj_handlers.set_pos_orn(self.obj, self._params.pos, self._params.orn)


class GreenOnion(KitchenObject):
    class GreenOnionPart(KitchenObject):
        def __init__(self, **kwargs):
            super().__init__()
            self._params = Namespace(**kwargs)

        def load(self):
            pass

    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)
        self._multiplexed_obj = None

    def _get_object_parts(self):
        object_parts = []
        for i, part in enumerate(self.obj.metadata['object_parts']):
            part_category = part['category']
            part_model = part['model']
            # Scale the offset accordingly
            part_pos = part['pos'] * self.obj.scale
            part_orn = part['orn']
            part_model_path = get_ig_model_path(part_category, part_model)
            part_filename = os.path.join(part_model_path, '{}.urdf'.format(part_model))
            part_obj_name = '{}_part_{}'.format(self.obj.name, i)
            part_obj = self.GreenOnionPart(
                filename=part_filename,
                name=part_obj_name,
                category=part_category,
                model_uri=part_model_path,
                scale=self.obj.scale
            )
            object_parts.append((part_obj, (part_pos, part_orn)))

        return object_parts

    @property
    def multiplexed_obj(self):
        return self._multiplexed_obj

    def load(self):
        object_parts = self._get_object_parts()
        grouped_parts_obj = ObjectGrouper(object_parts)
        self._multiplexed_obj = ObjectMultiplexer(
            '{}_multiplexer'.format(self.obj.name),
            [self.obj, grouped_parts_obj],
            0
        )

        self._params.obj_handlers.import_obj(self._multiplexed_obj)
        self.obj.set_position([100, 100, -100])
        for i, (part, _) in enumerate(object_parts):
            part.obj.set_position([101 + i, 100, -100])

        pos_z = self._params.pos[2] + .05
        self._params.obj_handlers.set_pos_orn(
            self._obj,
            [self._params.pos[0], self._params.pos[1], pos_z],
            self._params.orn
        )
        self._params.change_pb_dynamics(self._multiplexed_obj.get_body_ids()[0], -1, mass=self._params.mass)
