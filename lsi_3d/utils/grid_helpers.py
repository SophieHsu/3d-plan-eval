class GridHelper():
    @staticmethod
    def valid_pos(grid, x, y):
        return 0 <= x and x < len(grid) and 0 <= y and y < len(grid[0]) and grid[x][y] == "X"

    @staticmethod
    def transition(r, c, f, a):

        name2dire = {
            "E": (0, 1),
            "W": (0, -1),
            "S": (1, 0),
            "N": (-1, 0),
        }
        
        if a in "EWSN":
            f = a
        elif a == "F":
            dr, dc = name2dire[f]
            r, c = r+dr, c+dc
        return r, c, f