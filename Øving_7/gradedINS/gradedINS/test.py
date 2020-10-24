import numpy as np
import utils
import quaternion
import typing

v = np.array([1, 2, 3])

v.reshape(3)

S = utils.cross_product_matrix(v)

#print(S)

ql = np.array([3, 2, 0, 0])
qr = np.array([1, 1, 0, 0])

quat = quaternion.euler_to_quaternion(np.array([1,0,0]))

q_p = quaternion.quaternion_product(ql, qr)
print(quat)
R = quaternion.quaternion_to_rotation_matrix(quat)


print(R)