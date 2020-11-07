import numpy as np
from scipy import linalg as la
import utils
from numpy import matlib as ml
from utils import rotmat2d


x = np.array([1, 2, 3, 4, 5, 6]).reshape((-1,2)).T

y = x[:,0]

m = y.T @ x

print(m)

