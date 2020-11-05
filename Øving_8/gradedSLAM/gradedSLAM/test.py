import numpy as np
from scipy import linalg as la
import utils

R = np.array([[1, 3], [4, 5]])
Fu = la.block_diag(R, 1) 

v = np.array([1, 2, 3, 4,5, 6, 7, 8, 9, 10])
v = v.reshape((-1,2)).T

x = np.array([1,4]).reshape((-1, 2)).T

v = v-x

print(v)

l = [np.arctan2(mi[1],mi[0]) for mi in v.T]

n = np.linalg.norm(v[:,1])

p = np.arctan2(v[:,1], v[:,0])

print(x[])


