import math
import numpy as np


ori = math.radians(270)

ori_vec_x = math.cos(ori)
ori_vec_y = math.sin(ori)

ori_vec = [ori_vec_x, ori_vec_y]
vec = [1, 1]

unit_vector_1 = ori_vec / np.linalg.norm(ori_vec)
unit_vector_2 = vec / np.linalg.norm(vec)
dot_product = np.dot(unit_vector_1, unit_vector_2)
angle = np.arccos(dot_product)

print(math.degrees(angle))