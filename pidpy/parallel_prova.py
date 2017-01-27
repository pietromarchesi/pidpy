

import numpy as np
from joblib import Parallel, delayed

def fun(i):
    return [i,i+1,i+2]

M = 10
arr = np.zeros([M,3],dtype = int)

for i in range(M):
    arr[i,:] = fun(i)



a = Parallel(n_jobs=2)(delayed(fun)(i) for i in range(M))
b = np.array(a)


class yo():

    def dyn(self,n):
        return n

    def par(self):
        return Parallel(n_jobs=2)(delayed(meth)(yo(),i) for i in range(10))


def meth(inst, n):
    return inst.dyn(n)

y =yo()
y.par()