import numpy as np
from scipy import linalg as la
import utils
from numpy import matlib as ml
from utils import rotmat2d


x = np.array([1, 2, np.pi/3])
u = np.array([4, 5, np.pi/4])
r = rotmat2d(x[2])

s = x[:2] + r @ u[:2]
l = x[2] + u[2]

n = np.array([*s, l])
print(n)

