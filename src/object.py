import igibson
from igibson.objects.articulated_object import URDFObject
import numpy as np
import os


class Object:
    def __init__(self, name, path, scale, **kwargs):
        self.name = name
        self.path = path
        self.scale = scale
        self.kwargs = kwargs

    def load(self, handles):
        obj = URDFObject(
            filename=os.path.join(igibson.ig_dataset_path, self.path),
            scale=np.array([1.04, 0.97, 0.95]) / 1.15,
            model_path=os.path.dirname(os.path.join(igibson.ig_dataset_path, self.path)),
            fixed_base=True,
            **self.kwargs,
        )

        handles['loader'](obj)
        handles['pose_setter'](
            obj,
            self.kwargs.get('position', [0., 0., 0.]),
            self.kwargs.get('orientation', None),
            self.kwargs.get('z_offset', None),
        )

# read config
# init object from config
# load them in simulation
