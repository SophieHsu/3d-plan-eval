from abc import ABC, abstractmethod
from config_provider import ConfigProvider
from igibson.envs.igibson_env import iGibsonEnv
from object import Object
from typing import Any, Dict


class Environment(ABC):
    def __init__(self, env: Any) -> None:
        super().__init__()
        self._env = env

    @abstractmethod
    def load_environment(self) -> None:
        """
        Load the environment from a given file.
        """
        raise NotImplementedError

    @abstractmethod
    def render_environment(self) -> None:
        """
        Render the loaded environment in VR.
        """
        raise NotImplementedError

    @abstractmethod
    def setup(self) -> None:
        """
        Set up the kitchen environment.
        """

        raise NotImplementedError

    @abstractmethod
    def update(self, events: Dict) -> None:
        """
        Execute updates for environment objects.

        Args:
            events (dict): A dict of named events along with their corresponding values to be updated using appropriate
            event handlers.
        """
        raise NotImplementedError


class iGibsonEnvironment(Environment):
    def __init__(self, env: iGibsonEnv, config_file: str) -> None:
        super().__init__(env)
        self._config = None
        self._objects = None
        self._object_load_handles = {
            'loader': self._env.simulator.import_object,
            'pose_setter': self._env.set_pos_orn_with_z_offset
        }
        self._config_file = config_file
        self.setup()

    def load_environment(self) -> None:
        for obj in self._objects:
            obj.load(self, self._object_load_handles)

    def render_environment(self) -> None:
        raise NotImplementedError

    def setup(self) -> None:
        self._config = ConfigProvider(self._config_file)
        self._objects = list(map(
            lambda obj_attrs: Object(**obj_attrs),
            self._config.get('object')
        ))
        self.load_environment()
        print(self._objects)

    def update(self, events: Dict) -> None:
        raise NotImplementedError


if __name__ == '__main__':
    igibson_env = iGibsonEnv(
        config_file=None,
        mode='gui_interactive',
        # action_timestep=1.0 / 15,
        # physics_timestep=1.0 / 30,  #1.0 / 30,
        action_timestep=1.0 / 30,
        physics_timestep=1.0 / 120,  # 1.0 / 30,
        use_pb_gui=True
    )

    e = iGibsonEnvironment(igibson_env, '/home/rutvik/Desktop/3d-plan-eval/src/example_conf.toml')
    e.load_environment()


