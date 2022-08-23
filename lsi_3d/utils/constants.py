from lsi_3d.utils.enums import MLAction

TARGET_ORNS = {
    MLAction.SOUTH: 0,
    MLAction.EAST: 1.5707,
    MLAction.NORTH: 3.1415926,
    MLAction.WEST: -1.5707,
    #None: -1
}

DIRE2POSDIFF = {
    MLAction.EAST: (0, 1),
    MLAction.WEST: (0, -1),
    MLAction.SOUTH: (1, 0),
    MLAction.NORTH: (-1, 0)
}