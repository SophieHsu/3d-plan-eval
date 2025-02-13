from enum import Enum

# # Medium Level Action
# class MLA(Enum):
#     STAY = 0
#     INTERACT = 1
#     NORTH = 2
#     EAST = 3
#     SOUTH = 4
#     WEST = 5
#     FORWARD = 6
#     IDLE = 7

#     @classmethod
#     def directions(self):
#         return [self.NORTH, self.EAST, self.WEST, self.SOUTH]

#     @classmethod
#     def from_strings(self, strings):
#         list = []
#         for s in strings:
#             list.append(self.from_string(s))

#         return list

#     @classmethod
#     def from_string(self, s):
#         if s == 'F':
#             return self.FORWARD
#         if s == 'STAY':
#             return self.STAY
#         if s == 'I':
#             return self.INTERACT
#         if s == 'N':
#             return self.NORTH
#         if s == 'E':
#             return self.EAST
#         if s == 'S':
#             return self.SOUTH
#         if s == 'W':
#             return self.WEST
#         if s == 'D':
#             return self.IDLE

#     @classmethod
#     def to_string(self, action):
#         #list = []
#         #for s in actions:
#         if action == self.FORWARD:
#             return 'F'
#         if action == self.STAY:
#             return 'STAY'
#         if action == self.INTERACT:
#             return 'I'
#         if action == self.NORTH:
#             return 'N'
#         if action == self.EAST:
#             return 'E'
#         if action == self.SOUTH:
#             return 'S'
#         if action == self.WEST:
#             return 'W'
#         if action == self.IDLE:
#             return 'D'

#         return list

# class HLAction(Enum):
#     DROP = 0
#     PICKUP = 1
#     DELIVER = 2

# class Object(Enum):
#     NONE = 0
#     ONION = 1
#     SOUP = 2

class Mode(Enum):
    EXEC_ML_PATH = 0
    CALC_HL_PATH = 1
    CALC_SUB_PATH = 2
    IDLE = 3
    INTERACT = 4
