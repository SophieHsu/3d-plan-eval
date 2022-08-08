from enum import Enum

class Action(Enum):
    STAY = 0
    INTERACT = 1
    NORTH = 2
    EAST = 3
    SOUTH = 4
    WEST = 5
    FORWARD = 6

    @classmethod
    def directions(self):
        return [self.NORTH, self.EAST, self.WEST, self.SOUTH]

    @classmethod
    def from_strings(self, strings):
        list = []
        for s in strings:
            if s == 'F':
                list.append(self.FORWARD)
            if s == 'STAY':
                list.append(self.STAY)
            if s == 'I':
                list.append(self.INTERACT)
            if s == 'N':
                list.append(self.NORTH)
            if s == 'E':
                list.append(self.EAST)
            if s == 'S':
                list.append(self.SOUTH)
            if s == 'W':
                list.append(self.WEST)

        return list