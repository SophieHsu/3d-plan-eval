import igibson
import os


class Object:
    def __init__(self, name, path, **kwargs):
        self.name = name
        self.path = path
        self.kwargs = kwargs

    def load(self, handles):
        obj = handles['creator'](
            filename=os.path.join(igibson.ig_dataset_path, self.path),
            model_path=os.path.dirname(os.path.join(igibson.ig_dataset_path, self.path)),
            **self.kwargs,
        )

        handles['loader'](obj)
        handles['pose_setter'](
            obj,
            self.kwargs.get('position', [0., 0., 0.]),
            self.kwargs.get('orientation', None),
            self.kwargs.get('z_offset', None),
        )
