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

CONF_DIR_NAMES = {
    CONF_KEYS.CONF_DIRS.ALGO: 'lsi_3d/config/algorithm',
    CONF_KEYS.CONF_DIRS.MAP: 'lsi_3d/config/map',
    CONF_KEYS.CONF_DIRS.AGENT: 'lsi_3d/config/agent',
    CONF_KEYS.CONF_DIRS.EXP: 'lsi_3d/config/experiment',
}
