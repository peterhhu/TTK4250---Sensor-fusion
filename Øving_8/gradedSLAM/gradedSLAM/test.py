import numpy as np
from scipy import linalg as la
import utils

R = np.array([[1, 3], [4, 5]])
Fu = la.block_diag(R, 1) 

v = np.array([1, 2, 3, 4,5, 6, 7, 8, 9, 10])
v = v.reshape((-1,2)).T

x = np.array([1,2]).reshape((-1, 2)).T

print(v-x)

