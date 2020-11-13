import numpy as np
import time
from scipy import linalg as la
from numpy import matlib as ml
from scipy.stats import chi2
import matplotlib.pyplot as plt

alpha = 0.1
tull = chi2.isf(alpha, 2)
plt.plot(tull)

plt.show()
#%%