from lsi_3d.utils.enums import MLA

TARGET_ORNS = {
    MLA.SOUTH: 0,
    MLA.EAST: 1.5707,
    MLA.NORTH: 3.1415926,
    MLA.WEST: -1.5707,
    #None: -1
}

DIRE2POSDIFF = {
    MLA.EAST: (0, 1),
    MLA.WEST: (0, -1),
    MLA.SOUTH: (1, 0),
    MLA.NORTH: (-1, 0)
}

NAME2DIRE = {
    "E": (0, 1),
    "W": (0, -1),
    "S": (1, 0),
    "N": (-1, 0),
}

FORWARD_RADIUS_POS = {
    # x-min, x-max, y-min, y-max
    MLA.EAST: (0,1,-1,1),
    MLA.WEST: (-1,0,-1,1),
    MLA.SOUTH: (-1,1,0,1),
    MLA.NORTH: (-1,1,-1,0)
}