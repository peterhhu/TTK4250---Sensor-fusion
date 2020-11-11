import numpy as np
import time
from scipy import linalg as la
from numpy import matlib as ml

R = np.diag([0.5, 0.7]) ** 1e-3
numlmk = 1000
start_time = time.time()
#rep_kron1 = np.kron(np.eye(numlmk), R)
rep_kron2 =  np.diag(np.diagonal(ml.repmat(R, numlmk, numlmk)))

#print(rep_kron1)
#print(rep_kron2)
print(f"{time.time() - start_time:.10f}")

#%%