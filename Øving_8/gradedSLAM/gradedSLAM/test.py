import numpy as np
from scipy import linalg as la
import utils
from numpy import matlib as ml

l = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
m = np.array([2, 4])
v = l.reshape(-1,2).T
m = m.transpose()
k = v - m.reshape(2,1)

j = np.eye(2, 3)
print(j)

