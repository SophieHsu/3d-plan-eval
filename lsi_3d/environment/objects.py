from argparse import Namespace
from igibson.objects.articulated_object import URDFObject


class KitchenObject:
    def __init__(self):
        self._params = None

    def create_urdf_object(self):
        return URDFObject(
            filename=self._params.filename,
            avg_obj_dims=self._params.avg_obj_dims,
            scale=self._params.scale / self._params.scale_factor,
            model_path=self._params.model_uri,
            category=self._params.category,
            fixed_base=self._params.fixed_base,
        )

    def load(self, *args):
        raise NotImplementedError()


class Fridge(KitchenObject):
    def __init__(self, **kwargs):
        super().__init__()

        self._params = Namespace(**kwargs)

    def load(self, obj_handlers):
        obj = self.create_urdf_object()
        obj_handlers.import_obj(obj)
        obj_handlers.set_pos_orn(obj, self._params.pos, self._params.orn)
        obj_handlers.

