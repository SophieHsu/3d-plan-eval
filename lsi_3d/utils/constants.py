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

NAME2DIRE = {
    "E": (0, 1),
    "W": (0, -1),
    "S": (1, 0),
    "N": (-1, 0),
}

FORWARD_RADIUS_POS = {
    # x-min, x-max, y-min, y-max
    MLAction.EAST: (0,1,-1,1),
    MLAction.WEST: (-1,0,-1,1),
    MLAction.SOUTH: (-1,1,0,1),
    MLAction.NORTH: (-1,1,-1,0)
}