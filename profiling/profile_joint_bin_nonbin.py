
import numpy as np
from memory_profiler import memory_usage

from pidpy.PIDCalculator import *


'''
By manually setting pid.binary, we can force the calculator to use either
the binary or nonbinary function, and monitor the time and memory usage
of both for different values of n.

It seems as though not very much changes in memory consumption (I had expected
more, but perhaps I have fixed the bug that was causing memory crashes) and
binary is faster for all n (at some point nonbinary becomes better but
that is for larger n than you are able to compute given the constraints
on the number of samples).

But wait: now I have tested on fully random arrays, so that's why you
have no difference in memory. Even if make the matrix X a lot sparser it
doesnt seem to change much.
'''


bina, nonbina = [],[]
membin, memnonbin = [],[]

nlist = range(2,15)
nrep = 20

for n in nlist:
    X = np.random.choice(range(2), size = [10000,n],p = [0.9,0.1])
    y = np.random.randint(6, size = 10000)

    import time
    t0 = time.time()
    for i in range(nrep):
        pid = PIDCalculator(X, y)
        pid.binary = False
        pid.synergy()
    t1 = time.time()
    nonbina.append(t1-t0)

    import time
    t0 = time.time()
    for i in range(nrep):
        pid = PIDCalculator(X, y)
        pid.binary = True
        pid.synergy()
    t1 = time.time()
    bina.append(t1-t0)

    pid = PIDCalculator(X, y)
    pid.binary = True
    membin.append(np.mean(memory_usage(proc = pid.synergy)))

    pid = PIDCalculator(X, y)
    pid.binary = False
    memnonbin.append(np.mean(memory_usage(proc=pid.synergy)))

import matplotlib.pyplot as plt
f,ax = plt.subplots(1,2)
ax[0].plot(range(len(bina)),bina)
ax[0].plot(range(len(bina)),nonbina)
ax[0].set_xticks(range(len(bina)))
ax[0].set_xticklabels(nlist)

ax[1].plot(range(len(bina)),membin)
ax[1].plot(range(len(bina)),memnonbin)
ax[1].set_xticks(range(len(bina)))
ax[1].set_xticklabels(nlist)
