import os
import toml


class ConfigProvider:
    def __init__(self, conf_file=None):
        self._conf_file = conf_file
        self._config = None

    def _read(self):
        if os.path.exists(self._conf_file):
            self._config = toml.load(self._conf_file)
        else:
            raise FileNotFoundError(f"Configuration file '{self._conf_file}' not found.")

    @property
    def config(self):
        if self._config is None:
            self._read()
        return self._config

    def get(self, attr):
        if self._config is None:
            self._read()
        return self._config.get(attr, None)
