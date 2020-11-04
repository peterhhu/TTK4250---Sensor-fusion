import numpy as np
from scipy import linalg as la
import utils

R = np.array([[1, 3], [4, 5]])
Fu = la.block_diag(R, 1) 


print(R)

