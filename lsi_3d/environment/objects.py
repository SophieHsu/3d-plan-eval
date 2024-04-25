import os
from argparse import Namespace
from igibson.objects.articulated_object import URDFObject
from igibson import object_states
from igibson.utils.assets_utils import get_ig_model_path
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from lsi_3d.environment.object_config import CONF_KEYS


class KitchenObject:
    def __init__(self):
        self._params = None
        self._obj = None

    @property
    def obj(self):
        if self._obj is None:
            self._obj = URDFObject(
                filename=getattr(self._params, CONF_KEYS.FILENAME),
                scale=getattr(self._params, CONF_KEYS.SCALE),
                model_path=getattr(self._params, CONF_KEYS.MODEL_PATH),
                name=getattr(self._params, CONF_KEYS.NAME),
                category=getattr(self._params, CONF_KEYS.CATEGORY),
                fixed_base=getattr(self._params, CONF_KEYS.FIXED_BASE, False),
                abilities=getattr(self._params, CONF_KEYS.ABILITIES),
                avg_obj_dims={
                    'density': getattr(self._params, CONF_KEYS.DENSITY)
                } if hasattr(self._params, CONF_KEYS.DENSITY) else None,
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

    def load(self, ref_obj):
        self._params.obj_handlers.import_obj(self.obj)
        self.obj.states[object_states.OnTop].set_value(ref_obj, True, use_ray_casting_method=True)
        self._params.obj_handlers.change_pb_dynamics(self.obj.get_body_ids()[0], -1, mass=self._params.mass)


class Steak(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = Namespace(**kwargs)

    def load(self, ref_obj):
        self._params.obj_handlers.import_obj(self.obj)
        self.obj.states[object_states.OnTop].set_value(ref_obj, True, use_ray_casting_method=True)
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
            part_pos = part['pos'] * self.obj.scale
            part_orn = part['orn']
            part_model_path = get_ig_model_path(part_category, part_model)
            part_filename = os.path.join(part_model_path, '{}.urdf'.format(part_model))
            part_obj_name = '{}_part_{}'.format(self.obj.name, i)
            part_obj = self.GreenOnionPart(
                filename=part_filename,
                name=part_obj_name,
                category=part_category,
                model_path=part_model_path,
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
        self.obj.set_position(self._params.away_pos)
        for i, (part, _) in enumerate(object_parts):
            part.obj.set_position([
                self._params.away_pos[0] + i + 1,
                self._params.away_pos[1],
                self._params.away_pos[2]
            ])

        self._params.obj_handlers.set_pos_orn(self.obj, self._params.pos, self._params.orn)
        self._params.change_pb_dynamics(self._multiplexed_obj.get_body_ids()[0], -1, mass=self._params.mass)
