import numpy as np


grid = open('kitchen_layouts_grid_text/steak_none_3.txt', "r").read().strip().split("\n")
sophie_grid = []
max_r = 8
max_c = 8
start_pad = 1
for i in range(10):
    sophie_grid.append([])
    for j in range(15):
        if i >= len(grid)+start_pad or j >= len(grid[0])+start_pad or i < start_pad or j < start_pad:
            sophie_grid[i].append('X')
        else:
            l = grid[i-start_pad][j-start_pad]
            if l == 'X':
                sophie_grid[i].append(' ')
            elif l == 'C':
                sophie_grid[i].append('X')
            elif l == 'K':
                sophie_grid[i].append('B')
            elif l == 'G':
                sophie_grid[i].append('O')
            elif l == 'W':
                sophie_grid[i].append('W')
            elif l == 'D':
                sophie_grid[i].append('D')
            elif l == 'P':
                sophie_grid[i].append('P')
            elif l == 'P':
                sophie_grid[i].append('P')
            elif l == 'T':
                sophie_grid[i].append('S')
            elif l == 'F':
                sophie_grid[i].append('M')
            elif l == 'B':
                sophie_grid[i].append('X')

with open('kitchen_layouts_grid_text/converted.txt', 'w') as testfile:
    for row in sophie_grid:
        testfile.write(''.join([str(a) for a in row]) + '\n')

print(sophie_grid)