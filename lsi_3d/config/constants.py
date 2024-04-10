import os
from argparse import Namespace

CONF_KEYS = Namespace(
    MAP_CONFIG='map_config',
    CONF_DIRS=Namespace(
        ALGO='ALGO',
        MAP='MAP',
        AGENT='AGENT',
        EXP='EXP',
    )
)

ROOT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

CONF_DIR_NAMES = {
    CONF_KEYS.CONF_DIRS.ALGO: (ROOT_DIR_PATH, 'algorithm'),
    CONF_KEYS.CONF_DIRS.MAP: (ROOT_DIR_PATH, 'map'),
    CONF_KEYS.CONF_DIRS.AGENT: (ROOT_DIR_PATH, 'agent'),
    CONF_KEYS.CONF_DIRS.EXP: (ROOT_DIR_PATH, 'experiment'),
}
