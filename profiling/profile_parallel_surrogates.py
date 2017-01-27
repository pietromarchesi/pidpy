import numpy as np
from pidpy.PIDCalculator import *
import multiprocessing
import time
multiprocessing.cpu_count()

np.random.seed(seed = 123)
X = np.random.randint(2, size = [10000,4])
y = np.random.randint(5, size = 10000)

for n in [1,2,3,4,-1]:
    t0 = time.time()
    pid = PIDCalculator(X,y)
    pid.n_jobs = n
    pid.synergy(debiased=True,n=200)
    t1 = time.time()
    print t1-t0
