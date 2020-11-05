import numpy as np
from scipy import linalg as la
import utils
from numpy import matlib as ml

R = np.diag([1, 2])

t = np.diag(np.diagonal(np.matlib.repmat(R, 4, 4)))

print(t)

